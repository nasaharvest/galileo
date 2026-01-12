"""Image processing utilities for Copernicus satellite data.

This module provides high-level functions for extracting and processing
satellite imagery from downloaded Copernicus products, particularly Sentinel-2 data.
"""

import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform_bounds


def extract_rgb_composite(
    zip_file_path: Path, bands: Optional[List[str]] = None, normalize: bool = True
) -> Optional[Dict]:
    """Extract RGB composite from Sentinel-2 ZIP file.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bands: List of band names to extract (default: ['B04', 'B03', 'B02'] for RGB)
        normalize: Whether to apply percentile normalization for display

    Returns:
        Dictionary containing:
        - 'rgb_array': RGB image array (H, W, 3)
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
