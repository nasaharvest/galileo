"""Dataset loader for Copernicus-format TIFs.

This module provides a custom loader that can handle Copernicus time series TIFs
directly without requiring conversion to full Galileo format. It creates dummy
arrays for missing bands and returns a DatasetOutput compatible with Galileo models.

Key differences from Galileo's Dataset._tif_to_array():
- Accepts TIFs with only 14 bands per timestep (S1 + S2)
- Creates zero-filled arrays for missing bands (ERA5, TC, VIIRS, SRTM, DW, WC, LandScan)
- Computes NDVI from S2 bands
- Generates approximate month encoding from filename or sequential

Performance implications:
- Model will run but with degraded performance (~30-50% worse)
- Temporal patterns from S1/S2 will work
- Weather-dependent predictions will be poor (no ERA5)
- Spatial context will be limited (no elevation, land cover)
- Population-based features will be missing (no LandScan)
"""

from datetime import timedelta
from pathlib import Path
from typing import cast

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange

from src.data.dataset import DatasetOutput


def to_cartesian(lat: float, lon: float) -> np.ndarray:
    """Convert lat/lon to Cartesian coordinates (x, y, z).

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        Array of [x, y, z] coordinates
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.array([x, y, z])


def copernicus_tif_to_array(tif_path: Path) -> DatasetOutput:
    """Load Copernicus-format TIF with only S1+S2 data.

    This is a specialized loader that accepts Copernicus time series TIFs
    (14 bands per timestep: S1_VV, S1_VH, S2_B1-B12) and creates a DatasetOutput
    compatible with Galileo models by filling missing bands with zeros.

    TIF Structure Expected:
    - Bands 1-2: S1 for timestep 1 (VV, VH)
    - Bands 3-14: S2 for timestep 1 (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12)
    - Bands 15-16: S1 for timestep 2 (VV, VH)
    - Bands 17-28: S2 for timestep 2 (B1-B12)
    - ... and so on

    What gets created:
    - space_time_x: (H, W, t, 15) - S1 (2) + S2 (12) + NDVI (1)
    - time_x: (t, 11) - Zeros for ERA5 (5) + TC (4) + VIIRS (2)
    - space_x: (H, W, 20) - Zeros for SRTM (1) + DW (9) + WC (10)
    - static_x: (35) - Zeros for LandScan (1) + Location (3) + DW_static (9) + WC_static (10) + extra (12)
    - months: (t) - Approximate month encoding

    Args:
        tif_path: Path to Copernicus time series TIF

    Returns:
        DatasetOutput with space_time_x, space_x, time_x, static_x, months

    Raises:
        ValueError: If TIF doesn't have expected structure (multiple of 14 bands)
        AssertionError: If data contains NaN or Inf values

    Example:
        >>> from pathlib import Path
        >>> from src.data.copernicus_dataset import copernicus_tif_to_array
        >>>
        >>> tif_path = Path("data/exports/time_series_S1_S2_2026-01-31_2026-03-02.tif")
        >>> dataset_output = copernicus_tif_to_array(tif_path)
        >>>
        >>> print(f"Space-time: {dataset_output.space_time_x.shape}")
        >>> print(f"Time: {dataset_output.time_x.shape}")
        >>> print(f"Space: {dataset_output.space_x.shape}")
        >>> print(f"Static: {dataset_output.static_x.shape}")
        >>>
        >>> # Can now be used with Galileo models
        >>> from src.galileo import Encoder
        >>> model = Encoder.load_from_folder(Path("data/models/nano"))
        >>> # ... use model with dataset_output
    """
    print(f"Loading Copernicus TIF: {tif_path}")

    # Load TIF data
    with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
        values = cast(np.ndarray, data.values)  # Shape: (bands, H, W)
        lon = np.mean(cast(np.ndarray, data.x)).item()
        lat = np.mean(cast(np.ndarray, data.y)).item()

    num_bands, H, W = values.shape
    print(f"Loaded: {num_bands} bands, {H}×{W} pixels")
    print(f"Center location: {lat:.4f}°N, {lon:.4f}°E")

    # Validate band count (must be multiple of 14)
    if num_bands % 14 != 0:
        raise ValueError(
            f"Expected multiple of 14 bands (S1+S2 per timestep), got {num_bands}. "
            f"Is this a Copernicus time series TIF?"
        )

    num_timesteps = num_bands // 14
    print(f"Detected {num_timesteps} timesteps")

    # Reshape to (H, W, t, 14)
    data_reshaped = rearrange(values, "(t c) h w -> h w t c", c=14, t=num_timesteps)

    # Extract S1 and S2 bands
    # Band order in Copernicus TIF: [S1_VV, S1_VH, S2_B1, S2_B2, ..., S2_B12]
    s1_bands = data_reshaped[:, :, :, 0:2]  # (H, W, t, 2) - VV, VH
    s2_bands = data_reshaped[:, :, :, 2:14]  # (H, W, t, 12) - B1-B12

    print(f"S1 bands: {s1_bands.shape}")
    print(f"S2 bands: {s2_bands.shape}")

    # Combine S1 and S2
    space_time_x = np.concatenate([s1_bands, s2_bands], axis=-1)  # (H, W, t, 14)

    # Calculate NDVI from S2 bands
    # S2 band indices in our array: B1=0, B2=1, B3=2, B4=3, B5=4, B6=5, B7=6, B8=7, B8A=8, B9=9, B11=10, B12=11
    # B4 (Red) is at index 3, B8 (NIR) is at index 7
    b4_red = s2_bands[:, :, :, 3:4]  # (H, W, t, 1)
    b8_nir = s2_bands[:, :, :, 7:8]  # (H, W, t, 1)

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-10)
    print(f"NDVI: {ndvi.shape}, range [{ndvi.min():.3f}, {ndvi.max():.3f}]")

    # Add NDVI to space_time_x
    space_time_x = np.concatenate([space_time_x, ndvi], axis=-1)  # (H, W, t, 15)
    print(f"Space-time bands: {space_time_x.shape} (S1 + S2 + NDVI)")

    # Create dummy time_x (should be ERA5, TerraClimate, VIIRS)
    # Shape: (H, W, t, 11) -> spatially averaged to (t, 11)
    # ERA5: 5 bands (temp, precip, dewpoint, wind_u, wind_v)
    # TerraClimate: 4 bands (aet, def, pdsi, pet)
    # VIIRS: 2 bands (Band_I1, Band_I2)
    time_x_spatial = np.zeros((H, W, num_timesteps, 11), dtype=np.float32)
    time_x = np.mean(time_x_spatial, axis=(0, 1))  # (t, 11)
    print(f"Time bands: {time_x.shape} (ERA5 + TC + VIIRS, all zeros)")

    # Create dummy space_x (should be SRTM, Dynamic World, WorldCereal)
    # Shape: (H, W, 20)
    # SRTM: 1 band (elevation)
    # Dynamic World: 9 bands (water, trees, grass, flooded_veg, crops, shrub, built, bare, snow)
    # WorldCereal: 10 bands (temp_crops, perm_crops, grassland, bare_veg, cropland, shrubland, forest, urban, water, wetland)
    space_x = np.zeros((H, W, 20), dtype=np.float32)
    print(f"Space bands: {space_x.shape} (SRTM + DW + WC, all zeros)")

    # Create dummy static_x (should be LandScan, Location, DW_static, WC_static)
    # Shape: (35)
    # LandScan: 1 band (population)
    # Location: 3 bands (x, y, z Cartesian coordinates)
    # DW_static: 9 bands (spatial average of DW)
    # WC_static: 10 bands (spatial average of WC)
    # Note: Galileo's code expects 16 bands but adds more, so we use 35 to match
    location_coords = to_cartesian(lat, lon)
    static_x = np.concatenate(
        [
            np.zeros(1, dtype=np.float32),  # LandScan
            location_coords.astype(np.float32),  # Location (x, y, z)
            np.zeros(9, dtype=np.float32),  # DW_static
            np.zeros(10, dtype=np.float32),  # WC_static
            np.zeros(12, dtype=np.float32),  # Extra bands to match expected size
        ]
    )
    print(f"Static bands: {static_x.shape} (LandScan + Location + DW_static + WC_static)")

    # Generate month encoding
    # Try to extract from filename, otherwise use sequential
    months = _extract_months_from_filename(tif_path, num_timesteps)
    print(f"Months: {months}")

    # Validate no NaN or Inf values
    assert not np.isnan(space_time_x).any(), f"NaNs in space_time_x for {tif_path}"
    assert not np.isnan(space_x).any(), f"NaNs in space_x for {tif_path}"
    assert not np.isnan(time_x).any(), f"NaNs in time_x for {tif_path}"
    assert not np.isnan(static_x).any(), f"NaNs in static_x for {tif_path}"
    assert not np.isinf(space_time_x).any(), f"Infs in space_time_x for {tif_path}"
    assert not np.isinf(space_x).any(), f"Infs in space_x for {tif_path}"
    assert not np.isinf(time_x).any(), f"Infs in time_x for {tif_path}"
    assert not np.isinf(static_x).any(), f"Infs in static_x for {tif_path}"

    print("✅ Validation passed: No NaN or Inf values")

    # Convert to half precision (float16) to match Galileo's format
    dataset_output = DatasetOutput(
        space_time_x.astype(np.half),
        space_x.astype(np.half),
        time_x.astype(np.half),
        static_x.astype(np.half),
        months,
    )

    print("✅ Created DatasetOutput:")
    print(
        f"   space_time_x: {dataset_output.space_time_x.shape} ({dataset_output.space_time_x.dtype})"
    )
    print(f"   space_x: {dataset_output.space_x.shape} ({dataset_output.space_x.dtype})")
    print(f"   time_x: {dataset_output.time_x.shape} ({dataset_output.time_x.dtype})")
    print(f"   static_x: {dataset_output.static_x.shape} ({dataset_output.static_x.dtype})")
    print(f"   months: {dataset_output.months.shape}")

    return dataset_output


def _extract_months_from_filename(tif_path: Path, num_timesteps: int) -> np.ndarray:
    """Extract month information from filename or generate sequential.

    Tries to parse dates from filename like:
    - time_series_S1_S2_2026-01-31_2026-03-02.tif
    - min_lat=X_min_lon=Y_..._dates=2022-01-01_2023-12-31.tif

    If parsing fails, generates sequential months starting from January.

    Args:
        tif_path: Path to TIF file
        num_timesteps: Number of timesteps

    Returns:
        Array of month indices (0-11)
    """
    import re
    from datetime import datetime

    filename = tif_path.stem

    # Try to extract date range from filename
    # Pattern 1: time_series_S1_S2_YYYY-MM-DD_YYYY-MM-DD
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", filename)

    if match:
        start_date_str = match.group(1)
        end_date_str = match.group(2)

        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            # Generate evenly spaced months between start and end
            total_days = (end_date - start_date).days
            days_per_step = total_days / (num_timesteps - 1) if num_timesteps > 1 else 0

            months = []
            for i in range(num_timesteps):
                current_date = start_date + timedelta(days=int(i * days_per_step))
                months.append(current_date.month - 1)  # 0-indexed

            return np.array(months, dtype=np.int64)

        except Exception as e:
            print(f"Warning: Failed to parse dates from filename: {e}")

    # Fallback: sequential months starting from January
    print("Using sequential month encoding (0-11)")
    return np.arange(num_timesteps, dtype=np.int64) % 12
