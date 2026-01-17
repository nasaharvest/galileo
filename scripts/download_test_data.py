#!/usr/bin/env python3
"""Download sample Copernicus data for testing.

This script downloads one S1 and one S2 product to use as test fixtures.
Run once to populate test data, then use those files for all subsequent tests.

Usage:
    uv run python scripts/download_test_data.py

Requirements:
    - .env file with COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET
    - Internet connection
    - ~1-2 GB free disk space
"""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.data.copernicus.client import CopernicusClient


def main():
    """Download test data for S1 and S2."""
    # Load environment variables from .env file
    load_dotenv()

    # Setup
    test_data_dir = Path("data/test_fixtures")
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Check credentials
    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("ERROR: Missing credentials in .env file")
        print("Please set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET")
        print()
        print("Get free credentials at: https://dataspace.copernicus.eu/")
        return 1

    print("=" * 60)
    print("COPERNICUS TEST DATA DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {test_data_dir.absolute()}")
    print()

    # Initialize client
    client = CopernicusClient(client_id=client_id, client_secret=client_secret)

    # Test location: Agricultural area in Netherlands
    # Good S1/S2 coverage, flat terrain, agricultural land
    center_lat = 52.0
    center_lon = 5.5

    # Use fixed bboxes that we know have data
    s2_bbox = [5.463417359335974, 51.97747747747748, 5.536582640664026, 52.02252252252252]
    s1_bbox = [5.0, 51.5, 6.0, 52.5]  # Larger bbox that we know has S1 data

    print(f"Test location: {center_lat}°N, {center_lon}°E")
    print("S2 bounding box: ~5km x 5km")
    print("S1 bounding box: ~100km x 100km (larger for S1 coverage)")
    print(f"S2 bbox coords: {s2_bbox}")
    print(f"S1 bbox coords: {s1_bbox}")
    print()

    # Metadata to save
    metadata = {
        "downloaded_at": datetime.now().isoformat(),
        "location": {
            "lat": center_lat,
            "lon": center_lon,
            "s2_bbox": s2_bbox,
            "s1_bbox": s1_bbox,
        },
        "products": {},
    }

    # ========================================================================
    # Download Sentinel-2 product
    # ========================================================================
    print("=" * 60)
    print("DOWNLOADING SENTINEL-2 PRODUCT")
    print("=" * 60)

    try:
        s2_files = client.fetch_s2(
            bbox=s2_bbox,
            start_date="2024-07-01",
            end_date="2024-07-15",
            max_cloud_cover=20,
            download_data=True,
            interactive=False,
            max_products=2,
        )

        if not s2_files:
            print("WARNING: No S2 products found. Try different dates/location.")
        else:
            print(f"✓ Downloaded {len(s2_files)} S2 products")
            for s2_file in s2_files:
                metadata["products"][f"s2_{s2_file.stem}"] = {
                    "file": s2_file.name,
                    "size_mb": s2_file.stat().st_size / 1024**2,
                }

    except Exception as e:
        print(f"ERROR downloading S2: {e}")

    print()

    # ========================================================================
    # Download Sentinel-1 product
    # ========================================================================
    print("=" * 60)
    print("DOWNLOADING SENTINEL-1 PRODUCT")
    print("=" * 60)

    try:
        s1_files = client.fetch_s1(
            bbox=s1_bbox,
            start_date="2024-01-01",
            end_date="2024-07-31",
            product_type="GRD",
            orbit_direction="ASCENDING",
            acquisition_mode="IW",
            download_data=True,
            max_products=2,
        )

        if not s1_files:
            print("WARNING: No S1 products found. Try different dates/location.")
        else:
            print(f"✓ Downloaded {len(s1_files)} S1 products")
            for s1_file in s1_files:
                metadata["products"][f"s1_{s1_file.stem}"] = {
                    "file": s1_file.name,
                    "size_mb": s1_file.stat().st_size / 1024**2,
                }

    except Exception as e:
        print(f"ERROR downloading S1: {e}")

    print()

    # ========================================================================
    # Save metadata
    # ========================================================================
    metadata_file = test_data_dir / "test_data_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Metadata saved to: {metadata_file}")
    print()
    print("Downloaded products:")
    for sensor, info in metadata["products"].items():
        print(f"  {sensor.upper()}: {info['file']} ({info['size_mb']:.1f} MB)")
    print()
    print("You can now run tests using these files as fixtures.")

    return 0


if __name__ == "__main__":
    exit(main())
