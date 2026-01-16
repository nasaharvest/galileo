"""Image processing utilities for Copernicus Sentinel-2 satellite data.

This module provides high-level functions for extracting and processing
Sentinel-2 optical imagery from downloaded Copernicus products, including:
- RGB composites for visualization
- False color composites for vegetation analysis
- Band extraction and statistics
- Bounding box cropping for memory optimization
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
                cropped_result = crop_to_bbox(rgb_display, bounds_wgs84, bbox)
                if cropped_result is None:
                    print("Cropping failed, returning None")
                    return None
                rgb_display = cropped_result
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

    Args:
        image_array: Full tile image data with shape:
                    - (H, W) for single band (grayscale)
                    - (H, W, C) for multi-band (RGB, multi-spectral)
        image_bounds: Geographic bounds of full tile [min_lon, min_lat, max_lon, max_lat]
                     in WGS84 coordinates (degrees)
        target_bbox: User's requested area [min_lon, min_lat, max_lon, max_lat]
                    in WGS84 coordinates (degrees)
        image_crs: Coordinate reference system of the image
                  Default: "EPSG:4326" (WGS84 lat/lon)

    Returns:
        Cropped image array containing only the requested area
        Returns None if cropping fails (bbox outside image bounds, etc.)
    """
    try:
        # Extract bounds for clarity
        img_min_lon, img_min_lat, img_max_lon, img_max_lat = image_bounds
        tgt_min_lon, tgt_min_lat, tgt_max_lon, tgt_max_lat = target_bbox

        # Check if target bbox is within image bounds
        if (
            tgt_max_lon < img_min_lon
            or tgt_min_lon > img_max_lon
            or tgt_max_lat < img_min_lat
            or tgt_min_lat > img_max_lat
        ):
            print("Target bbox is outside image bounds, cannot crop")
            return None

        # Get image dimensions
        if image_array.ndim == 2:
            height, width = image_array.shape
        elif image_array.ndim == 3:
            height, width, channels = image_array.shape
        else:
            print(f"Unsupported image array dimensions: {image_array.ndim}")
            return None

        # Calculate pixel coordinates for target bbox corners
        x_min_pixel = int((tgt_min_lon - img_min_lon) / (img_max_lon - img_min_lon) * width)
        x_max_pixel = int((tgt_max_lon - img_min_lon) / (img_max_lon - img_min_lon) * width)

        # Y-axis is flipped in images
        y_min_pixel = int((img_max_lat - tgt_max_lat) / (img_max_lat - img_min_lat) * height)
        y_max_pixel = int((img_max_lat - tgt_min_lat) / (img_max_lat - img_min_lat) * height)

        # Clamp pixel coordinates to valid range
        x_min_pixel = max(0, min(x_min_pixel, width - 1))
        x_max_pixel = max(0, min(x_max_pixel, width))
        y_min_pixel = max(0, min(y_min_pixel, height - 1))
        y_max_pixel = max(0, min(y_max_pixel, height))

        # Ensure we have a valid crop region
        if x_max_pixel <= x_min_pixel or y_max_pixel <= y_min_pixel:
            print("Invalid crop region: target bbox too small or outside image")
            return None

        # Extract the subset of pixels
        if image_array.ndim == 2:
            cropped = image_array[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
        else:  # ndim == 3
            cropped = image_array[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel, :]

        # Log the size reduction
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
