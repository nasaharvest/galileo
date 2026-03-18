"""Adapter to convert Copernicus TIFs to Galileo-compatible format.

Galileo's Dataset._tif_to_array() expects a very specific band layout:

  Per timestep (18 bands = ALL_DYNAMIC_IN_TIME_BANDS):
    SPACE_TIME_BANDS (12): VV, VH, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    TIME_BANDS (6):        temperature_2m, total_precipitation_sum, def, soil, aet, avg_rad

  After all timesteps:
    SPACE_BANDS (16): elevation, slope, DW_water, DW_trees, DW_grass,
                      DW_flooded_vegetation, DW_crops, DW_shrub_and_scrub,
                      DW_built, DW_bare, DW_snow_and_ice, WC_temporarycrops,
                      WC_maize, WC_wintercereals, WC_springcereals, WC_irrigation

  At the very end:
    STATIC in TIF (1): b1 (LandScan)
    (x, y, z location bands are computed from lat/lon, not stored in TIF)

  Total: (18 × t) + 16 + 1 bands

Copernicus time series TIF provides per timestep (14 bands):
    S1 (2): VV, VH
    S2 (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12

Key mismatches to fix:
  1. Drop S2 B1 and B9 (Galileo doesn't use them)
  2. Add 6 dummy TIME_BANDS per timestep
  3. Add 16 dummy SPACE_BANDS
  4. Add 1 dummy STATIC band
"""

from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio

# Galileo band counts (from src.data.earthengine.eo)
GALILEO_S1_BANDS = 2  # VV, VH
GALILEO_S2_BANDS = 10  # B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
GALILEO_SPACE_TIME = 12  # S1 + S2
GALILEO_TIME_BANDS = 6  # temperature_2m, total_precipitation_sum, def, soil, aet, avg_rad
GALILEO_ALL_DYNAMIC = 18  # SPACE_TIME + TIME per timestep
GALILEO_SPACE_BANDS = 16  # elevation, slope, DW(9), WC(5)
GALILEO_STATIC_IN_TIF = 1  # b1 (LandScan); x,y,z computed from lat/lon

# Copernicus band layout per timestep
COPERNICUS_BANDS_PER_TIMESTEP = 14  # 2 S1 + 12 S2

# Indices of B1 and B9 within the 12 Copernicus S2 bands
# Copernicus S2 order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
# Indices to DROP:      0                                     9
S2_BANDS_TO_DROP = [0, 9]  # B1 (index 0) and B9 (index 9) within the S2 block


def copernicus_to_galileo_tif(
    copernicus_tif_path: Path,
    output_path: Path,
    dates: List[date],
    fill_missing_with_zeros: bool = True,
) -> Path:
    """Convert Copernicus time series TIF to Galileo-compatible format.

    Conversion steps:
    1. Load Copernicus TIF (14 bands x t timesteps)
    2. Drop S2 B1 and B9 (Galileo uses 10 S2 bands, not 12)
    3. Add 6 dummy time bands per timestep (ERA5, TerraClimate, VIIRS)
    4. Add 16 dummy space bands (SRTM, DW, WC)
    5. Add 1 dummy static band (LandScan)

    Result: (18 x t) + 16 + 1 bands, matching Galileo's _tif_to_array() expectation.

    Args:
        copernicus_tif_path: Path to Copernicus time series TIF
        output_path: Where to save Galileo-compatible TIF
        dates: List of acquisition dates (stored in metadata)
        fill_missing_with_zeros: If True, fill missing bands with zeros.
            If False, fill with NaN (handled by Galileo's fillna).

    Returns:
        Path to created Galileo-compatible TIF
    """
    print("Converting Copernicus TIF to Galileo format...")
    print(f"  Input:  {copernicus_tif_path}")
    print(f"  Output: {output_path}")

    with rasterio.open(copernicus_tif_path) as src:
        raw = src.read()  # (bands, H, W)
        transform = src.transform
        crs = src.crs
        H, W = src.height, src.width

    data = np.transpose(raw, (1, 2, 0))  # (H, W, bands)
    total_bands = data.shape[2]

    if total_bands % COPERNICUS_BANDS_PER_TIMESTEP != 0:
        raise ValueError(
            f"Expected multiple of {COPERNICUS_BANDS_PER_TIMESTEP} bands, "
            f"got {total_bands}. Is this a Copernicus time series TIF?"
        )

    num_timesteps = total_bands // COPERNICUS_BANDS_PER_TIMESTEP
    print(f"  Detected {num_timesteps} timesteps ({total_bands} bands)")

    if dates and len(dates) != num_timesteps:
        print(
            f"  Warning: {len(dates)} dates provided but {num_timesteps} " f"timesteps detected."
        )

    fill = 0.0 if fill_missing_with_zeros else np.nan

    # Build dynamic bands for each timestep
    galileo_dynamic = []
    for t in range(num_timesteps):
        offset = t * COPERNICUS_BANDS_PER_TIMESTEP

        # S1 bands: first 2 (VV, VH) — already correct
        s1 = data[:, :, offset : offset + 2]  # (H, W, 2)

        # S2 bands: next 12, but drop B1 (idx 0) and B9 (idx 9)
        s2_all = data[:, :, offset + 2 : offset + 14]  # (H, W, 12)
        s2_keep = np.delete(s2_all, S2_BANDS_TO_DROP, axis=2)  # (H, W, 10)

        # Combine S1 + S2 = 12 space-time bands (matches Galileo SPACE_TIME_BANDS)
        space_time = np.concatenate([s1, s2_keep], axis=-1)  # (H, W, 12)

        # Dummy time bands (ERA5, TerraClimate, VIIRS)
        time_bands = np.full((H, W, GALILEO_TIME_BANDS), fill, dtype=np.float64)

        # Per-timestep total: 12 + 6 = 18 (matches ALL_DYNAMIC_IN_TIME_BANDS)
        timestep = np.concatenate([space_time, time_bands], axis=-1)
        galileo_dynamic.append(timestep)

    all_dynamic = np.concatenate(galileo_dynamic, axis=-1)  # (H, W, 18*t)

    # Dummy space bands
    space_bands = np.full((H, W, GALILEO_SPACE_BANDS), fill, dtype=np.float64)

    # Dummy static band (LandScan b1)
    static_bands = np.full((H, W, GALILEO_STATIC_IN_TIF), fill, dtype=np.float64)

    # Final layout: [dynamic, space, static]
    final = np.concatenate([all_dynamic, space_bands, static_bands], axis=-1)

    expected = (GALILEO_ALL_DYNAMIC * num_timesteps) + GALILEO_SPACE_BANDS + GALILEO_STATIC_IN_TIF
    assert (
        final.shape[2] == expected
    ), f"Band count mismatch: got {final.shape[2]}, expected {expected}"

    print(
        f"  Output bands: {final.shape[2]} "
        f"({GALILEO_ALL_DYNAMIC}×{num_timesteps} + {GALILEO_SPACE_BANDS} + {GALILEO_STATIC_IN_TIF})"
    )

    # Write GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_raster = np.transpose(final, (2, 0, 1))  # (bands, H, W)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=final_raster.shape[0],
        dtype=rasterio.float64,
        crs=crs,
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(final_raster)
        dst.update_tags(
            source="Copernicus Data Space Ecosystem",
            conversion="copernicus_to_galileo_tif",
            timesteps=str(num_timesteps),
            dynamic_bands_per_timestep=str(GALILEO_ALL_DYNAMIC),
            space_bands=str(GALILEO_SPACE_BANDS),
            static_bands=str(GALILEO_STATIC_IN_TIF),
            dates=",".join(d.isoformat() for d in dates) if dates else "",
            note="S1+S2 from Copernicus; time/space/static bands zero-filled",
        )

    size_mb = final_raster.nbytes / 1024 / 1024
    print(f"  ✅ Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def validate_galileo_tif(tif_path: Path) -> Tuple[bool, str]:
    """Check if a TIF matches Galileo's expected band layout.

    Expected: (18 × t) + 16 + 1 bands, where t >= 1.
    """
    try:
        with rasterio.open(tif_path) as src:
            num_bands = src.count
            H, W = src.height, src.width
    except Exception as e:
        return False, f"Error reading TIF: {e}"

    overhead = GALILEO_SPACE_BANDS + GALILEO_STATIC_IN_TIF  # 17
    min_bands = GALILEO_ALL_DYNAMIC + overhead  # 35

    if num_bands < min_bands:
        return False, f"Too few bands: {num_bands} (minimum {min_bands} for 1 timestep)"

    dynamic_count = num_bands - overhead
    if dynamic_count % GALILEO_ALL_DYNAMIC != 0:
        return False, (
            f"Invalid band count: {num_bands}. "
            f"Expected ({GALILEO_ALL_DYNAMIC}×t + {overhead}). "
            f"Dynamic portion ({dynamic_count}) not divisible by {GALILEO_ALL_DYNAMIC}."
        )

    t = dynamic_count // GALILEO_ALL_DYNAMIC
    return True, (
        f"Valid Galileo TIF: {num_bands} bands = "
        f"({GALILEO_ALL_DYNAMIC}×{t} dynamic + {GALILEO_SPACE_BANDS} space + "
        f"{GALILEO_STATIC_IN_TIF} static), {H}×{W} pixels"
    )


def get_band_info(tif_path: Path) -> dict:
    """Get band structure information from a TIF file."""
    with rasterio.open(tif_path) as src:
        num_bands = src.count
        H, W = src.height, src.width

    overhead = GALILEO_SPACE_BANDS + GALILEO_STATIC_IN_TIF

    # Try Galileo format
    if num_bands >= (GALILEO_ALL_DYNAMIC + overhead):
        dynamic_count = num_bands - overhead
        if dynamic_count % GALILEO_ALL_DYNAMIC == 0:
            t = dynamic_count // GALILEO_ALL_DYNAMIC
            return {
                "format": "galileo",
                "total_bands": num_bands,
                "timesteps": t,
                "dynamic_per_timestep": GALILEO_ALL_DYNAMIC,
                "space_bands": GALILEO_SPACE_BANDS,
                "static_bands": GALILEO_STATIC_IN_TIF,
                "height": H,
                "width": W,
            }

    # Try Copernicus format
    if num_bands % COPERNICUS_BANDS_PER_TIMESTEP == 0:
        t = num_bands // COPERNICUS_BANDS_PER_TIMESTEP
        return {
            "format": "copernicus",
            "total_bands": num_bands,
            "timesteps": t,
            "bands_per_timestep": COPERNICUS_BANDS_PER_TIMESTEP,
            "height": H,
            "width": W,
            "note": "S1 (2) + S2 (12) per timestep. Needs conversion for Galileo.",
        }

    return {
        "format": "unknown",
        "total_bands": num_bands,
        "height": H,
        "width": W,
    }


def embeddings_to_geotiff(
    embeddings: np.ndarray,
    output_path: Path,
    source_tif_path: Path | None = None,
    crs: str | None = None,
    transform: rasterio.transform.Affine | None = None,
) -> Path:
    """Write Galileo model embeddings to a georeferenced GeoTIFF.

    Each embedding dimension becomes one band in the output TIF.
    Works with raw embeddings (H, W, D) or PCA-reduced (H, W, 3).

    Geo-referencing is read from ``source_tif_path`` when provided.
    You can also pass explicit ``crs`` / ``transform`` (they take precedence).

    Args:
        embeddings: Numpy array of shape (H, W, D).
        output_path: Destination file path (.tif).
        source_tif_path: Optional source TIF to copy CRS and transform from.
        crs: Optional CRS string (e.g. "EPSG:4326"). Overrides source TIF.
        transform: Optional rasterio Affine transform. Overrides source TIF.

    Returns:
        Path to the created GeoTIFF.

    Raises:
        ValueError: If embeddings shape is wrong or no geo-referencing info
            can be determined.
    """
    if embeddings.ndim != 3:
        raise ValueError(
            f"Expected embeddings with 3 dimensions (H, W, D), got shape {embeddings.shape}"
        )

    H, W, D = embeddings.shape

    # Resolve geo-referencing
    src_crs = crs
    src_transform = transform

    if source_tif_path is not None:
        with rasterio.open(source_tif_path) as src:
            if src_crs is None:
                src_crs = src.crs
            if src_transform is None:
                # If source has different spatial dims, recompute transform from bounds
                if src.height == H and src.width == W:
                    src_transform = src.transform
                else:
                    from rasterio.transform import from_bounds

                    b = src.bounds
                    src_transform = from_bounds(b.left, b.bottom, b.right, b.top, W, H)

    if src_crs is None:
        src_crs = "EPSG:4326"  # sensible default for satellite data
    if src_transform is None:
        raise ValueError(
            "No transform available. Provide source_tif_path or an explicit transform."
        )

    # (H, W, D) -> (D, H, W) for rasterio
    raster = np.transpose(embeddings, (2, 0, 1)).astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=D,
        dtype=rasterio.float32,
        crs=src_crs,
        transform=src_transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(raster)
        dst.update_tags(
            source="Galileo embeddings",
            embedding_dim=str(D),
            height=str(H),
            width=str(W),
        )

    size_mb = raster.nbytes / 1024 / 1024
    print(f"✅ Embeddings GeoTIFF saved: {output_path}")
    print(f"   {W}×{H} pixels, {D} bands (embedding dims), {size_mb:.1f} MB")
    return output_path
