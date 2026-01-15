"""Image processing utilities for Copernicus satellite data.

This module provides high-level functions for extracting and processing
satellite imagery from downloaded Copernicus products, including:
- Sentinel-2 optical imagery (RGB composites, false color)
- Sentinel-1 SAR imagery (radar backscatter)
"""

import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform_bounds


def extract_rgb_composite(
    zip_file_path: Path,
    bands: Optional[List[str]] = None,
    normalize: bool = True,
    bbox: Optional[List[float]] = None,
) -> Optional[Dict]:
    """Extract RGB composite from Sentinel-2 ZIP file.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bands: List of band names to extract (default: ['B04', 'B03', 'B02'] for RGB)
        normalize: Whether to apply percentile normalization for display
        bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat] to crop to
             ⚠️ IMPORTANT: This reduces MEMORY usage, not ZIP file size!
             The full ZIP is still downloaded (API limitation). Cropping happens
             AFTER extraction, reducing the returned array size by 99%+ for small areas.

             Example: Without bbox, returns 1.4 GB array (full 110km tile)
                     With bbox, returns 77 KB array (800m × 800m area)

             Use this when:
             - Processing many images (saves memory)
             - Training ML models (only need small patches)
             - Time series analysis (consistent small area)

    Returns:
        Dictionary containing:
        - 'rgb_array': RGB image array (H, W, 3)
                      Size depends on bbox: full tile or cropped area
        - 'bounds_wgs84': Geographic bounds in WGS84 coordinates
        - 'bounds_utm': Original UTM bounds
        - 'crs': Coordinate reference system
        - 'metadata': Additional metadata

        Returns None if extraction fails.
    """
    if bands is None:
        bands = ["B04", "B03", "B02"]  # Red, Green, Blue for natural color

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract ZIP file
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Find SAFE directory (Sentinel-2 format)
            safe_dirs = list(temp_path.glob("*.SAFE"))
            if not safe_dirs:
                print(f"No SAFE directory found in {zip_file_path.name}")
                return None

            safe_dir = safe_dirs[0]

            # Find IMG_DATA directory
            img_data_dir = safe_dir / "GRANULE"
            granule_dirs = list(img_data_dir.glob("*"))

            if not granule_dirs:
                print(f"No granule directories found in {zip_file_path.name}")
                return None

            granule_dir = granule_dirs[0]
            img_dir = granule_dir / "IMG_DATA"

            # Find band files
            band_files = {}
            for band in bands:
                # Try multiple naming patterns
                patterns = [
                    f"*_{band}_10m.jp2",  # Standard pattern
                    f"*_{band}.jp2",  # Alternative pattern
                    f"*{band}.jp2",  # Simple pattern
                ]

                for pattern in patterns:
                    band_matches = list(img_dir.glob(pattern))
                    if band_matches:
                        band_files[band] = band_matches[0]
                        break

            if len(band_files) < len(bands):
                print(f"Only found {len(band_files)}/{len(bands)} bands in {zip_file_path.name}")
                return None

            # Read bands and create composite
            rgb_bands = []
            bounds = None
            crs = None

            for band in bands:
                if band in band_files:
                    with rasterio.open(band_files[band]) as src:
                        band_data = src.read(1)

                        # Get geospatial info from first band
                        if bounds is None:
                            bounds = src.bounds
                            crs = src.crs

                        rgb_bands.append(band_data)

            if len(rgb_bands) != len(bands):
                return None

            # Stack bands into RGB array
            rgb_array = np.stack(rgb_bands, axis=0)

            # Apply normalization if requested
            if normalize:
                rgb_normalized = np.zeros_like(rgb_array, dtype=np.float32)

                for i in range(len(bands)):
                    band_data = rgb_array[i]
                    valid_pixels = band_data[band_data > 0]

                    if len(valid_pixels) > 0:
                        # Use percentile normalization for better contrast
                        p2, p98 = np.percentile(valid_pixels, [2, 98])
                        if p98 > p2:
                            rgb_normalized[i] = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                        else:
                            band_max = float(band_data.max()) if band_data.max() > 0 else 1.0
                            rgb_normalized[i] = band_data / band_max
                    else:
                        rgb_normalized[i] = band_data

                rgb_array = rgb_normalized

            # Convert to display format (H, W, C)
            rgb_display = np.transpose(rgb_array, (1, 2, 0))

            # Convert bounds to WGS84
            if bounds is not None:
                bounds_wgs84 = transform_bounds(
                    crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
                )
            else:
                bounds_wgs84 = None

            # Apply bbox cropping if requested
            # ⚠️ IMPORTANT: This reduces MEMORY usage, not ZIP file size!
            # The full 700MB ZIP was already downloaded. We're now extracting
            # only the pixels we need from the full tile that's in memory.
            # This saves 99%+ memory and makes processing much faster.
            if bbox is not None and bounds_wgs84 is not None:
                print(f"Cropping to bbox: {bbox}")
                rgb_display = crop_to_bbox(rgb_display, bounds_wgs84, bbox)
                if rgb_display is None:
                    print("Cropping failed, returning None")
                    return None
                # Update bounds to reflect cropped area
                bounds_wgs84 = tuple(bbox)

            return {
                "rgb_array": rgb_display,
                "bounds_wgs84": bounds_wgs84,
                "bounds_utm": bounds,
                "crs": str(crs),
                "metadata": {
                    "bands": bands,
                    "shape": rgb_display.shape,
                    "zip_file": zip_file_path.name,
                    "safe_dir": safe_dir.name,
                },
            }

    except Exception as e:
        print(f"Error extracting RGB from {zip_file_path.name}: {e}")
        return None


def get_available_bands(zip_file_path: Path) -> List[str]:
    """Get list of available bands in a Sentinel-2 ZIP file.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file

    Returns:
        List of available band names (e.g., ['B01', 'B02', 'B03', ...])
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract ZIP file
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Find SAFE directory
            safe_dirs = list(temp_path.glob("*.SAFE"))
            if not safe_dirs:
                return []

            safe_dir = safe_dirs[0]

            # Find IMG_DATA directory
            img_data_dir = safe_dir / "GRANULE"
            granule_dirs = list(img_data_dir.glob("*"))

            if not granule_dirs:
                return []

            granule_dir = granule_dirs[0]
            img_dir = granule_dir / "IMG_DATA"

            # Find all JP2 files and extract band names
            jp2_files = list(img_dir.glob("*.jp2"))
            bands = []

            for jp2_file in jp2_files:
                # Extract band name from filename (e.g., T31UGQ_20251129T103309_B02.jp2 -> B02)
                name = jp2_file.name
                if "_B" in name:
                    band_part = name.split("_B")[1]
                    band_name = "B" + band_part.split(".")[0]
                    if band_name not in bands:
                        bands.append(band_name)

            return sorted(bands)

    except Exception as e:
        print(f"Error getting bands from {zip_file_path.name}: {e}")
        return []


def create_false_color_composite(
    zip_file_path: Path, bands: Optional[List[str]] = None
) -> Optional[Dict]:
    """Create false color composite (NIR, Red, Green) for vegetation analysis.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bands: List of band names (default: ['B08', 'B04', 'B03'] for NIR-R-G)

    Returns:
        Same format as extract_rgb_composite but with false color bands
    """
    if bands is None:
        bands = ["B08", "B04", "B03"]  # NIR, Red, Green for vegetation

    return extract_rgb_composite(zip_file_path, bands=bands, normalize=True)


def get_image_statistics(rgb_data: Dict) -> Dict:
    """Calculate statistics for RGB image data.

    Args:
        rgb_data: Output from extract_rgb_composite

    Returns:
        Dictionary with image statistics
    """
    if rgb_data is None:
        return {}

    rgb_array = rgb_data["rgb_array"]

    stats = {
        "shape": rgb_array.shape,
        "dtype": str(rgb_array.dtype),
        "min_values": rgb_array.min(axis=(0, 1)).tolist(),
        "max_values": rgb_array.max(axis=(0, 1)).tolist(),
        "mean_values": rgb_array.mean(axis=(0, 1)).tolist(),
        "std_values": rgb_array.std(axis=(0, 1)).tolist(),
        "bounds_wgs84": rgb_data["bounds_wgs84"],
        "coverage_area_km2": _calculate_area_km2(rgb_data["bounds_wgs84"]),
    }

    return stats


def _calculate_area_km2(bounds_wgs84: Tuple[float, float, float, float]) -> float:
    """Calculate approximate area in km² from WGS84 bounds."""
    # Simple approximation - more accurate methods would use proper geodesic calculations
    lon_diff = bounds_wgs84[2] - bounds_wgs84[0]  # max_lon - min_lon
    lat_diff = bounds_wgs84[3] - bounds_wgs84[1]  # max_lat - min_lat

    # Approximate conversion (varies by latitude)
    avg_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
    km_per_degree_lon = 111.32 * np.cos(np.radians(avg_lat))
    km_per_degree_lat = 110.54

    area_km2 = (lon_diff * km_per_degree_lon) * (lat_diff * km_per_degree_lat)
    return area_km2


def extract_sar_composite(
    zip_file_path: Path,
    polarizations: Optional[List[str]] = None,
    to_db: bool = True,
    bbox: Optional[List[float]] = None,
) -> Optional[Dict]:
    """Extract SAR backscatter composite from Sentinel-1 ZIP file.

    WHAT IS SAR (SYNTHETIC APERTURE RADAR):
    SAR is an active radar sensor that sends microwave pulses to Earth and measures
    the reflected signal (backscatter). Unlike optical sensors (Sentinel-2), SAR:
    - Works day and night (doesn't need sunlight)
    - Penetrates clouds and rain (microwaves pass through)
    - Measures surface roughness and structure

    WHAT IS BACKSCATTER:
    Backscatter is the radar signal reflected back to the satellite. The strength depends on:
    - Surface roughness: Smooth surfaces (water) = low backscatter (dark)
                        Rough surfaces (buildings, vegetation) = high backscatter (bright)
    - Moisture content: Wet surfaces reflect more than dry surfaces
    - Viewing geometry: Angle of radar beam affects return signal

    WHAT ARE POLARIZATIONS:
    Radar can transmit and receive in different orientations:
    - VV: Vertical transmit, Vertical receive
          Good for: Water detection, urban areas, bare soil
          Sensitive to: Surface roughness, soil moisture
    - VH: Vertical transmit, Horizontal receive (cross-polarization)
          Good for: Vegetation monitoring, crop classification
          Sensitive to: Volume scattering from vegetation canopy

    WHY CONVERT TO DECIBELS (dB):
    Raw SAR data has huge dynamic range (0.0001 to 10000+). Converting to dB:
    - Compresses the range for better visualization
    - Makes values more interpretable (-30 dB to +10 dB typical range)
    - Formula: dB = 10 * log10(linear_value)

    Args:
        zip_file_path: Path to Sentinel-1 ZIP file (GRD product)
                      Example: S1A_IW_GRDH_1SDV_20220101T123456_..._.zip
        polarizations: List of polarizations to extract (default: ['VV', 'VH'])
                      Options: 'VV', 'VH', 'HH', 'HV' (availability depends on product)
        to_db: If True, convert backscatter to decibels (dB) for better visualization
              If False, keep linear scale (sigma0 values)
        bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat] to crop to
             ⚠️ IMPORTANT: This reduces MEMORY usage, not ZIP file size!
             The full 1-2GB SAR ZIP is still downloaded (API limitation). Cropping
             happens AFTER extraction, reducing the returned array size by 99%+ for
             small areas.

             Example: Without bbox, returns ~1.4 GB array (full 110km tile)
                     With bbox, returns ~77 KB array (800m × 800m area)

             Use this when:
             - Processing many SAR images (saves memory)
             - Training ML models (only need small patches)
             - Time series analysis (consistent small area)

    Returns:
        Dictionary containing:
        - 'sar_array': SAR backscatter array (H, W, num_polarizations)
                      Values in dB if to_db=True, else linear sigma0
        - 'polarizations': List of polarization names in order
        - 'bounds_wgs84': Geographic bounds [min_lon, min_lat, max_lon, max_lat]
        - 'bounds_utm': Original UTM bounds
        - 'crs': Coordinate reference system
        - 'metadata': Additional metadata (resolution, product name, etc.)

        Returns None if extraction fails.

    Example:
        >>> sar_data = extract_sar_composite(s1_zip_file)
        >>> print(sar_data['sar_array'].shape)  # (height, width, 2) for VV+VH
        >>> print(sar_data['polarizations'])    # ['VV', 'VH']
        >>> print(f"VV range: {sar_data['sar_array'][:,:,0].min():.1f} to {sar_data['sar_array'][:,:,0].max():.1f} dB")
    """
    if polarizations is None:
        polarizations = ["VV", "VH"]  # Most common dual-polarization combination

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract ZIP file
            # Sentinel-1 products are distributed as ZIP files containing:
            # - measurement/ folder with GeoTIFF files for each polarization
            # - annotation/ folder with XML metadata
            # - preview/ folder with quicklook images
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Find SAFE directory (Sentinel-1 format)
            # SAFE = Standard Archive Format for Europe
            # Directory name contains product metadata: platform, mode, type, date, etc.
            safe_dirs = list(temp_path.glob("*.SAFE"))
            if not safe_dirs:
                print(f"No SAFE directory found in {zip_file_path.name}")
                return None

            safe_dir = safe_dirs[0]

            # Find measurement directory containing the actual SAR data
            # GRD products have measurement/ folder with GeoTIFF files
            measurement_dir = safe_dir / "measurement"
            if not measurement_dir.exists():
                print(f"No measurement directory found in {zip_file_path.name}")
                return None

            # Find polarization files
            # Files are named like: s1a-iw-grd-vv-20220101t123456-...-.tiff
            pol_files = {}
            for pol in polarizations:
                # Try multiple naming patterns (lowercase and uppercase)
                patterns = [
                    f"*-{pol.lower()}-*.tiff",  # Standard pattern: ...-vv-...tiff
                    f"*-{pol.upper()}-*.tiff",  # Uppercase variant
                    f"*{pol.lower()}.tiff",  # Simple pattern
                    f"*{pol.upper()}.tiff",  # Simple uppercase
                ]

                for pattern in patterns:
                    pol_matches = list(measurement_dir.glob(pattern))
                    if pol_matches:
                        pol_files[pol] = pol_matches[0]
                        break

            if not pol_files:
                print(f"No polarization files found in {zip_file_path.name}")
                print(f"Available files: {list(measurement_dir.glob('*.tiff'))}")
                return None

            # Read polarization bands and create composite
            sar_bands = []
            bounds = None
            crs = None
            resolution = None

            for pol in polarizations:
                if pol in pol_files:
                    with rasterio.open(pol_files[pol]) as src:
                        # Read the backscatter data
                        # Values are typically in linear scale (sigma0)
                        band_data = src.read(1).astype(np.float32)

                        # Get geospatial info from first band
                        if bounds is None:
                            bounds = src.bounds
                            crs = src.crs
                            resolution = src.res  # (x_resolution, y_resolution) in meters

                        # Convert to dB if requested
                        # dB scale is more intuitive for visualization and analysis
                        if to_db:
                            # Add small epsilon to avoid log(0) = -inf
                            # Typical SAR values range from 0.0001 to 10
                            # In dB: -40 dB to +10 dB
                            band_data_db = 10 * np.log10(band_data + 1e-10)

                            # Clip extreme values for better visualization
                            # Values below -30 dB are typically noise
                            # Values above +10 dB are rare (very strong scatterers)
                            band_data_db = np.clip(band_data_db, -30, 10)
                            sar_bands.append(band_data_db)
                        else:
                            sar_bands.append(band_data)

            if not sar_bands:
                return None

            # Stack bands into multi-polarization array
            # Shape: (num_polarizations, height, width)
            sar_array = np.stack(sar_bands, axis=0)

            # Convert to display format (H, W, C)
            # This matches the format used for optical imagery
            sar_display = np.transpose(sar_array, (1, 2, 0))

            # Convert bounds to WGS84 for consistency with S2 functions
            if bounds is not None and crs is not None:
                bounds_wgs84 = transform_bounds(
                    crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
                )
            else:
                bounds_wgs84 = None

            # Apply bbox cropping if requested
            # ⚠️ IMPORTANT: This reduces MEMORY usage, not ZIP file size!
            # The full 1-2GB SAR ZIP was already downloaded. We're now extracting
            # only the pixels we need from the full tile that's in memory.
            # This saves 99%+ memory and makes processing much faster.
            if bbox is not None and bounds_wgs84 is not None:
                print(f"Cropping SAR to bbox: {bbox}")
                sar_display = crop_to_bbox(sar_display, bounds_wgs84, bbox)
                if sar_display is None:
                    print("SAR cropping failed, returning None")
                    return None
                # Update bounds to reflect cropped area
                bounds_wgs84 = tuple(bbox)

            return {
                "sar_array": sar_display,
                "polarizations": [pol for pol in polarizations if pol in pol_files],
                "bounds_wgs84": bounds_wgs84,
                "bounds_utm": bounds,
                "crs": str(crs),
                "metadata": {
                    "shape": sar_display.shape,
                    "resolution_m": resolution,
                    "zip_file": zip_file_path.name,
                    "safe_dir": safe_dir.name,
                    "scale": "dB" if to_db else "linear",
                },
            }

    except Exception as e:
        print(f"Error extracting SAR from {zip_file_path.name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def crop_to_bbox(
    image_array: np.ndarray,
    image_bounds: Tuple[float, float, float, float],
    target_bbox: List[float],
    image_crs: str = "EPSG:4326",
) -> Optional[np.ndarray]:
    """Crop satellite image to user's requested bounding box.

    ⚠️ IMPORTANT: WHAT THIS FUNCTION DOES AND DOESN'T SAVE ⚠️

    WHAT THIS SAVES:
    ✅ Memory/RAM usage (99%+ reduction)
    ✅ Processing time (operations on smaller arrays are faster)
    ✅ Downstream storage (if you save the extracted arrays)

    WHAT THIS DOESN'T SAVE:
    ❌ ZIP file download size (still downloads full 500MB-2GB file)
    ❌ Disk space for cached ZIPs (ZIPs are stored as-is)
    ❌ Download time (must download entire tile from API)

    WHEN CROPPING HAPPENS:
    1. Download full ZIP file (700 MB) ← NO CROPPING
    2. Extract bands from ZIP (load full 110km tile into memory) ← NO CROPPING
    3. Apply this crop function (extract 800m subset) ← CROPPING HAPPENS HERE
    4. Return cropped array (77 KB) ← HUGE MEMORY SAVINGS

    WHY YOU CAN'T CROP THE ZIP:
    The Copernicus API doesn't support partial tile downloads. You must download
    the entire 110km × 110km tile even if you only need 800m × 800m. This is a
    limitation of how satellite data is organized and distributed.

    SATELLITE TILE SYSTEM EXPLANATION:

    Copernicus organizes data into fixed tiles covering Earth:

      ┌─────────────────────────────────────┐
      │  Tile T31UGQ (110km × 110km)       │  ← Full tile downloaded from API
      │                                     │     (700 MB ZIP file)
      │         ┌──┐  ← User's 800m area   │  ← What user actually wants
      │         └──┘                        │     (77 KB after cropping)
      │                                     │
      └─────────────────────────────────────┘

    MEMORY SAVINGS EXAMPLE:
    - User requests 800m × 800m area (0.64 km²)
    - API returns entire 110km × 110km tile (12,100 km²)
    - That's 18,906× more data than needed!

    WITHOUT CROPPING:
    - Full tile in memory: 10,980 × 10,980 × 3 = 362 million values
    - Memory usage: ~1.4 GB for float32 array

    WITH CROPPING:
    - Cropped area: 80 × 80 × 3 = 19,200 values
    - Memory usage: ~77 KB for float32 array
    - Reduction: 18,750× smaller (99.995% less memory!)

    WHEN TO USE CROPPING:
    ✅ Training ML models (only need small patches)
    ✅ Processing many images (save memory)
    ✅ Time series analysis (consistent small area)
    ✅ Interactive applications (fast response)

    WHEN NOT TO USE CROPPING:
    ❌ Exploring data (want to see full context)
    ❌ Large area analysis (need the full tile)
    ❌ Mosaicking (combining multiple tiles)

    HOW IT WORKS:
    1. Convert geographic coordinates (lat/lon degrees) to pixel coordinates (row/col)
    2. Calculate which pixels fall within the target bounding box
    3. Extract only those pixels from the full image array
    4. Return the cropped subset

    COORDINATE SYSTEMS:
    - WGS84 (EPSG:4326): Latitude/longitude in degrees
      Example: [6.15, 49.11, 6.16, 49.12] = 800m × 800m in Luxembourg
    - UTM: Universal Transverse Mercator in meters
      Example: [293000, 5442000, 293800, 5442800] = same area in UTM zone 31N
    - Pixels: Row/column indices in image array
      Example: [5400, 2100, 5480, 2180] = 80 × 80 pixel subset

    Args:
        image_array: Full tile image data with shape:
                    - (H, W) for single band (grayscale)
                    - (H, W, C) for multi-band (RGB, multi-spectral)
                    Example: (10980, 10980, 3) for 110km tile at 10m resolution
        image_bounds: Geographic bounds of full tile [min_lon, min_lat, max_lon, max_lat]
                     in WGS84 coordinates (degrees)
                     Example: [6.0, 49.0, 7.0, 50.0] for Luxembourg region
        target_bbox: User's requested area [min_lon, min_lat, max_lon, max_lat]
                    in WGS84 coordinates (degrees)
                    Example: [6.15, 49.11, 6.16, 49.12] for 800m × 800m area
        image_crs: Coordinate reference system of the image
                  Default: "EPSG:4326" (WGS84 lat/lon)
                  Sentinel-2: Usually UTM (e.g., "EPSG:32631" for zone 31N)
                  Sentinel-1: Usually UTM

    Returns:
        Cropped image array containing only the requested area:
        - Shape: (crop_height, crop_width) or (crop_height, crop_width, C)
        - Example: (80, 80, 3) for 800m × 800m at 10m resolution
        Returns None if cropping fails (bbox outside image bounds, etc.)

    Example:
        >>> # Full Sentinel-2 tile: 110km × 110km at 10m resolution
        >>> full_image = np.random.rand(10980, 10980, 3)  # 120 million pixels
        >>> image_bounds = [6.0, 49.0, 7.0, 50.0]  # Full tile bounds
        >>> target_bbox = [6.15, 49.11, 6.16, 49.12]  # 800m × 800m area
        >>>
        >>> cropped = crop_to_bbox(full_image, image_bounds, target_bbox)
        >>> print(cropped.shape)  # (80, 80, 3) - only 6,400 pixels!
        >>> print(f"Size reduction: {full_image.size / cropped.size:.0f}×")  # 18,750×
    """
    try:
        # Extract bounds for clarity
        # Image bounds: the full tile's geographic extent
        img_min_lon, img_min_lat, img_max_lon, img_max_lat = image_bounds

        # Target bounds: the user's requested area
        tgt_min_lon, tgt_min_lat, tgt_max_lon, tgt_max_lat = target_bbox

        # Check if target bbox is within image bounds
        # If target is completely outside image, we can't crop
        if (
            tgt_max_lon < img_min_lon
            or tgt_min_lon > img_max_lon
            or tgt_max_lat < img_min_lat
            or tgt_min_lat > img_max_lat
        ):
            print("Target bbox is outside image bounds, cannot crop")
            return None

        # Get image dimensions
        # Handle both 2D (H, W) and 3D (H, W, C) arrays
        if image_array.ndim == 2:
            height, width = image_array.shape
        elif image_array.ndim == 3:
            height, width, channels = image_array.shape
        else:
            print(f"Unsupported image array dimensions: {image_array.ndim}")
            return None

        # COORDINATE CONVERSION: Geographic (lat/lon) → Pixel (row/col)
        #
        # Satellite images are stored as pixel arrays, but users specify areas
        # in geographic coordinates (latitude/longitude). We need to convert.
        #
        # The conversion formula:
        #   pixel_x = (lon - min_lon) / (max_lon - min_lon) * width
        #   pixel_y = (max_lat - lat) / (max_lat - min_lat) * height
        #
        # Note: Y-axis is flipped! In images, row 0 is at the top (north),
        # but in geographic coordinates, latitude increases upward (north).

        # Calculate pixel coordinates for target bbox corners
        # X-axis (longitude → column): increases left to right
        x_min_pixel = int((tgt_min_lon - img_min_lon) / (img_max_lon - img_min_lon) * width)
        x_max_pixel = int((tgt_max_lon - img_min_lon) / (img_max_lon - img_min_lon) * width)

        # Y-axis (latitude → row): FLIPPED! Row 0 is north (max_lat)
        # We subtract from max_lat because image rows increase downward
        y_min_pixel = int((img_max_lat - tgt_max_lat) / (img_max_lat - img_min_lat) * height)
        y_max_pixel = int((img_max_lat - tgt_min_lat) / (img_max_lat - img_min_lat) * height)

        # Clamp pixel coordinates to valid range [0, dimension)
        # This handles cases where target bbox slightly exceeds image bounds
        x_min_pixel = max(0, min(x_min_pixel, width - 1))
        x_max_pixel = max(0, min(x_max_pixel, width))
        y_min_pixel = max(0, min(y_min_pixel, height - 1))
        y_max_pixel = max(0, min(y_max_pixel, height))

        # Ensure we have a valid crop region (at least 1 pixel)
        if x_max_pixel <= x_min_pixel or y_max_pixel <= y_min_pixel:
            print("Invalid crop region: target bbox too small or outside image")
            return None

        # Extract the subset of pixels within target bbox
        # Array slicing: [row_start:row_end, col_start:col_end]
        if image_array.ndim == 2:
            cropped = image_array[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
        else:  # ndim == 3
            cropped = image_array[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel, :]

        # Log the size reduction for user feedback
        original_pixels = image_array.shape[0] * image_array.shape[1]
        cropped_pixels = cropped.shape[0] * cropped.shape[1]
        reduction_factor = original_pixels / cropped_pixels if cropped_pixels > 0 else 0

        print(
            f"Cropped from {image_array.shape[:2]} to {cropped.shape[:2]} "
            f"({reduction_factor:.1f}× reduction)"
        )

        return cropped

    except Exception as e:
        print(f"Error cropping image: {e}")
        import traceback

        traceback.print_exc()
        return None
