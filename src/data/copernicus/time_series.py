"""Time series utilities for creating multi-temporal satellite data stacks.

This module provides functions for stacking multiple satellite acquisitions
across time into single multi-band GeoTIFF files, compatible with the Galileo
model's expected input format.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from .image_processing import extract_all_s1_bands, extract_all_s2_bands


def generate_date_list(
    start_date: str, end_date: str, temporal_resolution: str = "weekly"
) -> List[date]:
    """Generate list of dates for time series sampling.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        temporal_resolution: One of 'daily', 'weekly', 'monthly'

    Returns:
        List of date objects
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    dates = []
    current = start

    if temporal_resolution == "daily":
        delta = timedelta(days=1)
    elif temporal_resolution == "weekly":
        delta = timedelta(days=7)
    elif temporal_resolution == "monthly":
        # Approximate - will adjust to actual month boundaries
        delta = timedelta(days=30)
    else:
        raise ValueError(f"Unknown temporal resolution: {temporal_resolution}")

    while current <= end:
        dates.append(current)

        if temporal_resolution == "monthly":
            # Move to next month (handle month boundaries properly)
            if current.month == 12:
                current = date(current.year + 1, 1, current.day)
            else:
                try:
                    current = date(current.year, current.month + 1, current.day)
                except ValueError:
                    # Handle day overflow (e.g., Jan 31 → Feb 28)
                    current = date(current.year, current.month + 1, 1)
        else:
            current += delta

    return dates


def align_arrays_to_common_grid(
    arrays: List[np.ndarray],
    bounds_list: List[Tuple[float, float, float, float]],
    target_bounds: Tuple[float, float, float, float],
    target_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """Align multiple arrays to a common geographic grid.

    This handles cases where different acquisitions have slightly different
    bounds or resolutions due to satellite orbit variations.

    Args:
        arrays: List of arrays to align (each with shape (H, W, C))
        bounds_list: List of bounds for each array
        target_bounds: Target bounds to align to
        target_shape: Target shape (height, width)

    Returns:
        List of aligned arrays, all with shape (target_shape[0], target_shape[1], C)
    """
    from scipy.ndimage import zoom

    aligned = []

    for arr, bounds in zip(arrays, bounds_list):
        if arr.shape[:2] == target_shape and bounds == target_bounds:
            # Already aligned
            aligned.append(arr)
        else:
            # Need to resample
            # Calculate zoom factors
            zoom_h = target_shape[0] / arr.shape[0]
            zoom_w = target_shape[1] / arr.shape[1]

            # Apply zoom to each channel separately
            if arr.ndim == 3:
                aligned_arr = np.zeros(
                    (target_shape[0], target_shape[1], arr.shape[2]), dtype=arr.dtype
                )
                for c in range(arr.shape[2]):
                    aligned_arr[:, :, c] = zoom(arr[:, :, c], (zoom_h, zoom_w), order=1)
            else:
                aligned_arr = zoom(arr, (zoom_h, zoom_w), order=1)

            aligned.append(aligned_arr)

    return aligned


def create_time_series_tif(
    s2_files: List[Path],
    s1_files: List[Path],
    dates: List[date],
    bbox: List[float],
    output_path: Path,
    normalize: bool = True,
) -> Path:
    """Stack multiple S1/S2 acquisitions into single multi-band TIF.

    This creates a time series TIF with the following structure (Galileo-compatible ordering):
    - Bands 1-2: S1 bands for date 1 (VV, VH)
    - Bands 3-14: S2 bands for date 1 (B1-B12)
    - Bands 15-16: S1 bands for date 2 (VV, VH)
    - Bands 17-28: S2 bands for date 2 (B1-B12)
    - ... and so on

    Total bands: (2 + 12) × num_dates = 14 × num_dates

    Note: This ordering matches Galileo's expectation where S1 comes before S2.

    Args:
        s2_files: List of S2 zip files (one per date)
        s1_files: List of S1 zip files (one per date)
        dates: List of acquisition dates
        bbox: [min_lon, min_lat, max_lon, max_lat]
        output_path: Where to save the stacked TIF
        normalize: If True, normalize to float64 [0, 1] range

    Returns:
        Path to created TIF
    """
    print(f"Creating time series TIF with {len(dates)} dates...")
    print(f"S2 files: {len(s2_files)}, S1 files: {len(s1_files)}")

    all_bands = []
    all_bounds = []
    reference_shape = None
    reference_bounds = None

    # Extract data for each date
    for i, (s2_file, s1_file, date_obj) in enumerate(zip(s2_files, s1_files, dates)):
        print(f"Processing date {i+1}/{len(dates)}: {date_obj}")

        # Extract S2 bands (12 bands)
        s2_data = extract_all_s2_bands(s2_file, bbox)
        if s2_data is None:
            print(f"Warning: Failed to extract S2 for {date_obj}, skipping")
            continue

        # Extract S1 bands (2 bands)
        s1_data = extract_all_s1_bands(s1_file, bbox, to_db=False)
        if s1_data is None:
            print(f"Warning: Failed to extract S1 for {date_obj}, skipping")
            continue

        # Set reference from first successful extraction
        if reference_shape is None:
            reference_shape = s2_data["bands_array"].shape[:2]
            reference_bounds = s2_data["bounds_wgs84"]

        # Align S1 to S2 shape if needed (they may have slightly different resolutions)
        if s1_data["bands_array"].shape[:2] != s2_data["bands_array"].shape[:2]:
            print(
                f"  Aligning S1 {s1_data['bands_array'].shape[:2]} to S2 {s2_data['bands_array'].shape[:2]}"
            )
            from scipy.ndimage import zoom

            zoom_h = s2_data["bands_array"].shape[0] / s1_data["bands_array"].shape[0]
            zoom_w = s2_data["bands_array"].shape[1] / s1_data["bands_array"].shape[1]

            s1_aligned = np.zeros(
                (s2_data["bands_array"].shape[0], s2_data["bands_array"].shape[1], 2),
                dtype=s1_data["bands_array"].dtype,
            )
            for c in range(2):
                s1_aligned[:, :, c] = zoom(
                    s1_data["bands_array"][:, :, c], (zoom_h, zoom_w), order=1
                )

            s1_data["bands_array"] = s1_aligned

        # Concatenate S1 and S2: (H, W, 2) + (H, W, 12) = (H, W, 14)
        # Order matches Galileo expectation: [S1_VV, S1_VH, S2_B1, ..., S2_B12]
        combined = np.concatenate([s1_data["bands_array"], s2_data["bands_array"]], axis=-1)

        all_bands.append(combined)
        all_bounds.append(s2_data["bounds_wgs84"])

    if not all_bands:
        raise ValueError("No data extracted for any date!")

    print(f"Successfully extracted {len(all_bands)} dates")

    # Ensure we have reference bounds and shape
    if reference_bounds is None or reference_shape is None:
        raise ValueError("No valid data extracted - all dates failed")

    # Align all arrays to common grid (handle slight variations)
    if len(set(all_bounds)) > 1 or len(set([arr.shape[:2] for arr in all_bands])) > 1:
        print("Aligning arrays to common grid...")
        all_bands = align_arrays_to_common_grid(
            all_bands, all_bounds, reference_bounds, reference_shape
        )

    # Stack all dates: (H, W, 14 × num_dates)
    stacked = np.concatenate(all_bands, axis=-1)
    print(f"Stacked shape: {stacked.shape}")
    print(f"Total bands: {stacked.shape[2]} (14 bands × {len(all_bands)} dates)")

    # Normalize if requested
    if normalize:
        print("Normalizing to float64 [0, 1] range...")
        # Convert uint16 to float64 and normalize
        if stacked.dtype == np.uint16:
            stacked = stacked.astype(np.float64) / 65535.0
        elif stacked.dtype in [np.float32, np.float64]:
            # Already float, just ensure [0, 1] range
            stacked = np.clip(stacked, 0, 1).astype(np.float64)

    # Write to GeoTIFF
    write_multiband_geotiff(stacked, reference_bounds, output_path)

    print(f"✅ Time series TIF saved to: {output_path}")
    return output_path


def write_multiband_geotiff(
    data: np.ndarray,
    bounds: Tuple[float, float, float, float],
    output_path: Path,
    crs: str = "EPSG:4326",
) -> None:
    """Write multi-band array to GeoTIFF file.

    Args:
        data: Array with shape (H, W, num_bands)
        bounds: Geographic bounds (min_lon, min_lat, max_lon, max_lat)
        output_path: Output file path
        crs: Coordinate reference system (default: WGS84)
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get dimensions
    height, width, num_bands = data.shape

    # Create transform from bounds
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    # Determine dtype
    if data.dtype == np.float64:
        dtype = rasterio.float64
    elif data.dtype == np.float32:
        dtype = rasterio.float32
    elif data.dtype == np.uint16:
        dtype = rasterio.uint16
    else:
        dtype = rasterio.float64

    # Write to file
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",  # Compress to save disk space
        tiled=True,  # Enable tiling for better performance
        blockxsize=256,
        blockysize=256,
    ) as dst:
        # Write each band
        for i in range(num_bands):
            dst.write(data[:, :, i], i + 1)

    print(f"Wrote {num_bands} bands to {output_path}")
    print(f"Shape: {height}x{width} pixels")
    print(f"Bounds: {bounds}")
    print(f"CRS: {crs}")
