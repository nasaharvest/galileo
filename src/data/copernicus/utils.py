"""Utility functions for Copernicus data fetching.

This module contains helper functions used throughout the Copernicus data fetching system:
- Input validation for bounding boxes and dates
- Cache key generation for deterministic caching
- File system utilities for safe file operations
- Coordinate system conversions for API queries
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple


def validate_bbox(bbox: List[float]) -> None:
    """Validate bounding box format and coordinate values.

    A bounding box defines a rectangular area on Earth's surface using latitude and longitude.
    This function ensures the bbox is properly formatted and contains valid coordinates.

    Args:
        bbox: List of 4 floats representing [min_longitude, min_latitude, max_longitude, max_latitude]
              All coordinates should be in WGS84 (EPSG:4326) coordinate system.
              Example: [25.6796, -27.6721, 25.6897, -27.663] represents a small area in South Africa.

    Raises:
        ValueError: If bbox format is wrong or coordinates are invalid.
                   Provides specific error messages for different validation failures.
    """
    # Check basic format: must be a list with exactly 4 numbers
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be a list of 4 floats: [min_lon, min_lat, max_lon, max_lat]")

    # Extract coordinates for readability
    min_lon, min_lat, max_lon, max_lat = bbox

    # Validate longitude bounds: must be between -180 and +180 degrees
    # Longitude lines run north-south; -180/+180 is the International Date Line
    if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180")

    # Validate latitude bounds: must be between -90 and +90 degrees
    # Latitude lines run east-west; -90 is South Pole, +90 is North Pole
    if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90")

    # Validate coordinate ordering: min values must be less than max values
    # This ensures we have a valid rectangle, not an inverted one
    if min_lon >= max_lon:
        raise ValueError("min_lon must be less than max_lon")

    if min_lat >= max_lat:
        raise ValueError("min_lat must be less than max_lat")


def validate_date(date_str: str) -> datetime:
    """Validate and parse a date string in YYYY-MM-DD format.

    This function ensures dates are in the correct format and represent valid calendar dates.

    Args:
        date_str: Date string in ISO format, e.g., "2022-01-01", "2023-12-31"

    Returns:
        datetime object representing the parsed date

    Raises:
        ValueError: If date format is invalid or represents an impossible date
                   (e.g., "2022-13-01" for month 13, or "2022-02-30" for Feb 30th)
    """
    try:
        # Use strptime to parse the date string with strict format checking
        # This will raise ValueError if format doesn't match exactly or date is invalid
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        # Re-raise with more helpful error message
        raise ValueError(f"Date must be in YYYY-MM-DD format: {e}")


def validate_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """Validate a date range ensuring start comes before end.

    This function validates both individual dates and their logical relationship.

    Args:
        start_date: Start date in YYYY-MM-DD format, e.g., "2022-01-01"
        end_date: End date in YYYY-MM-DD format, e.g., "2022-12-31"

    Returns:
        Tuple of (start_datetime, end_datetime) objects

    Raises:
        ValueError: If either date is invalid or start_date is not before end_date
    """
    # Validate each date individually first
    start_dt: datetime = validate_date(start_date)
    end_dt: datetime = validate_date(end_date)

    # Ensure logical ordering: start must come before end
    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date")

    return start_dt, end_dt


def build_cache_key(prefix: str, **params: Any) -> str:
    """Build a deterministic cache key from request parameters.

    This function creates a unique, deterministic identifier for caching purposes.
    The same parameters will always produce the same cache key, enabling reliable cache hits.

    Args:
        prefix: Cache key prefix to identify the type of data (e.g., 's1', 's2')
        **params: All parameters that affect the request result. These are hashed together
                 to create a unique identifier. Examples: bbox, start_date, end_date, resolution, etc.

    Returns:
        A cache key string in format: "{prefix}_{hash}" where hash is a 16-character hex string
        Example: "s2_abc123def456789a" for Sentinel-2 data with specific parameters
    """
    # Sort parameters by key name to ensure deterministic ordering
    # This is crucial: {"a": 1, "b": 2} and {"b": 2, "a": 1} must produce the same hash
    param_str: str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))

    # Use SHA-256 hash for cryptographic security and collision resistance
    # Take only first 16 characters for brevity while maintaining uniqueness
    hash_obj = hashlib.sha256(param_str.encode())
    return f"{prefix}_{hash_obj.hexdigest()[:16]}"


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe filesystem storage.

    This function removes or replaces characters that are invalid in filenames
    on various operating systems (Windows, macOS, Linux).

    Args:
        filename: Original filename that may contain invalid characters
                 Example: "S2A_MSIL1C_20220101T123456:789_N0400.SAFE"

    Returns:
        Sanitized filename safe for all major filesystems
        Example: "S2A_MSIL1C_20220101T123456_789_N0400.SAFE"
    """
    # Replace invalid characters with underscores
    # These characters are problematic on Windows: < > : " / \ | ? *
    # We replace them all for cross-platform compatibility
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def ensure_cache_dir(cache_dir: Path) -> None:
    """Ensure cache directory exists, creating it if necessary.

    This function safely creates the cache directory structure, including
    any parent directories that don't exist.

    Args:
        cache_dir: Path object representing the cache directory
    """
    # Create directory and all parent directories if they don't exist
    # parents=True: create parent directories as needed
    # exist_ok=True: don't raise error if directory already exists
    cache_dir.mkdir(parents=True, exist_ok=True)


def bbox_to_wkt(bbox: List[float]) -> str:
    """Convert bounding box to Well-Known Text (WKT) polygon format.

    WKT is a standard format for representing geometric shapes in spatial databases.
    The Copernicus API uses WKT polygons for spatial queries.

    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
              Example: [25.6796, -27.6721, 25.6897, -27.663]

    Returns:
        WKT polygon string representing the bounding box rectangle
        Example: "POLYGON((25.6796 -27.6721, 25.6897 -27.6721, 25.6897 -27.663, 25.6796 -27.663, 25.6796 -27.6721))"

    Note:
        The polygon is created by connecting the four corners of the rectangle in order:
        1. Bottom-left (min_lon, min_lat)
        2. Bottom-right (max_lon, min_lat)
        3. Top-right (max_lon, max_lat)
        4. Top-left (min_lon, max_lat)
        5. Back to bottom-left to close the polygon
    """
    min_lon: float = bbox[0]
    min_lat: float = bbox[1]
    max_lon: float = bbox[2]
    max_lat: float = bbox[3]

    # Create a closed polygon by listing all corner points
    # Note: WKT format is "longitude latitude" (x y), not "latitude longitude"
    return (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )
