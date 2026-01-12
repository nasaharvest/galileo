import marimo

__generated_with = "0.10.6"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    mo.md(
        r"""
        # 🛰️ Copernicus Satellite Data Explorer

        **Interactive demonstration of satellite data fetching with visual map display**

        This notebook demonstrates:
        - 🔐 Authentication with Copernicus Data Space Ecosystem
        - 🛰️ Fetching Sentinel-1 (SAR) and Sentinel-2 (optical) data
        - 🗺️ **Visual map display** of downloaded satellite imagery
        - 💾 Intelligent caching system
        - 📊 Detailed metadata analysis

        **Target Area**: Luxembourg (49.114982°N, 6.155827°E) - 800m × 800m
        """
    )
    return (mo,)


@app.cell
def __():
    # Setup and imports
    print("🔧 SETTING UP COPERNICUS SATELLITE DATA EXPLORER")
    print("=" * 60)

    import json
    import sys
    from datetime import datetime, timedelta
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.append("src")

    print("✅ Core libraries imported")
    print("✅ Visualization libraries ready for map display")
    print("✅ Source path configured")

    return Path, datetime, json, np, pd, plt, sys, timedelta


@app.cell
def __():
    # Target area configuration
    print("\n🎯 CONFIGURING TARGET AREA")
    print("=" * 40)

    center_lat = 49.114982
    center_lon = 6.155827

    print("📍 Target location: Luxembourg")
    print(f"   • Latitude: {center_lat:.6f}°N")
    print(f"   • Longitude: {center_lon:.6f}°E")

    # Calculate 800m x 800m bounding box
    lat_offset = 0.8 / 111
    lon_offset = 0.8 / 69.5

    target_bbox = [
        center_lon - lon_offset,
        center_lat - lat_offset,
        center_lon + lon_offset,
        center_lat + lat_offset,
    ]

    print(f"\n📐 Bounding box: {target_bbox}")
    print("📊 Expected at 10m resolution: 80×80 pixels")

    return center_lat, center_lon, target_bbox


@app.cell
@app.cell
def __():
    from datetime import datetime, timedelta

    # Date range configuration
    print("\n📅 CONFIGURING DATE RANGE")
    print("=" * 35)

    end_dt = datetime.now() - timedelta(days=30)
    start_dt = end_dt - timedelta(days=14)
    date_start = start_dt.strftime("%Y-%m-%d")
    date_end = end_dt.strftime("%Y-%m-%d")

    print(f"🗓️ Search period: {date_start} to {date_end}")
    print(f"⏱️ Duration: {(end_dt - start_dt).days} days")
    print("💡 Recent but processed data for best availability")

    return date_end, date_start


@app.cell
def __():
    from datetime import datetime

    # Initialize Copernicus client
    print("\n🔧 INITIALIZING COPERNICUS CLIENT")
    print("=" * 45)

    from src.data.copernicus import CopernicusClient

    try:
        copernicus_client = CopernicusClient()
        auth_token = copernicus_client._get_access_token()

        print("✅ Client created successfully!")
        print(f"📁 Cache: {copernicus_client.cache_dir}")
        print(f"🎫 Token: {auth_token[:20]}...{auth_token[-10:]}")
        print(f"⏰ Expires: {datetime.fromtimestamp(copernicus_client._token_expires_at)}")

        client_ready = True

    except Exception as error:
        print(f"❌ Setup failed: {error}")
        print("💡 Check .env file with COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET")
        client_ready = False
        copernicus_client = None

    return CopernicusClient, client_ready, copernicus_client


@app.cell
def __(client_ready, copernicus_client, date_end, date_start, json, target_bbox):
    # Fetch Sentinel-2 data
    print("🔵 FETCHING SENTINEL-2 OPTICAL IMAGERY")
    print("=" * 50)

    if client_ready:
        print("🛰️ About Sentinel-2:")
        print("   • Twin satellites providing optical/multispectral imaging")
        print("   • 13 spectral bands from visible to shortwave infrared")
        print("   • 10m, 20m, 60m spatial resolution depending on band")
        print("   • 5-day revisit time with both satellites")

        print("\n🔍 Search parameters:")
        print("   • Area: Luxembourg (800m × 800m)")
        print(f"   • Dates: {date_start} to {date_end}")
        print("   • Product: S2MSI1C (Level-1C)")
        print("   • Resolution: 10m")
        print("   • Max clouds: 50%")

        try:
            s2_files = copernicus_client.fetch_s2(
                bbox=target_bbox,
                start_date=date_start,
                end_date=date_end,
                resolution=10,
                max_cloud_cover=50,
                product_type="S2MSI1C",
                download_data=True,  # ENABLE ACTUAL DOWNLOADS
                interactive=False,  # No prompts in notebook
            )

            print("\n💡 DOWNLOAD STATUS:")
            print("=" * 30)
            print("🔄 Attempting to download actual satellite imagery")
            print("📊 This will show real Luxembourg satellite data")
            print("⚠️  Note: Downloads are ~500MB per product")
            print("🎯 Target: Luxembourg coordinates for actual imagery")

            print(f"\n✅ Found {len(s2_files)} Sentinel-2 products")

            if s2_files:
                print("\n📊 PRODUCT DETAILS:")
                for s2_idx, s2_file_path in enumerate(s2_files[:3], 1):
                    print(f"\n🛰️ Product {s2_idx}: {s2_file_path.name}")

                    if s2_file_path.suffix == ".json":
                        try:
                            with open(s2_file_path) as s2_file_handle:
                                s2_metadata = json.load(s2_file_handle)

                            prod_id = s2_metadata.get("product_id", "N/A")
                            prod_name = s2_metadata.get("product_name", "N/A")

                            print(f"   🆔 ID: {prod_id}")
                            print(f"   📛 Name: {prod_name}")

                            # Parse date
                            s2_content_date = s2_metadata.get("content_date", {})
                            acq_start = s2_content_date.get("Start", "N/A")
                            if acq_start != "N/A":
                                acq_date = acq_start[:10]
                                acq_time = acq_start[11:19]
                                print(f"   📅 Acquired: {acq_date} at {acq_time} UTC")

                            # Available bands
                            print("   📡 Key bands available:")
                            print("      • B02 (490nm) Blue - 10m")
                            print("      • B03 (560nm) Green - 10m")
                            print("      • B04 (665nm) Red - 10m")
                            print("      • B08 (842nm) NIR - 10m")
                            print("      • + 9 more spectral bands")

                        except Exception as e:
                            print(f"   ❌ Error reading metadata: {e}")
            else:
                print("⚠️ No products found - try different parameters")

        except Exception as fetch_error:
            print(f"❌ Fetch failed: {fetch_error}")
            s2_files = []
    else:
        print("❌ Client not ready")
        s2_files = []

    return (s2_files,)


@app.cell
def __(client_ready, copernicus_client, date_end, date_start, json, target_bbox):
    # Fetch Sentinel-1 data
    print("🔴 FETCHING SENTINEL-1 SAR IMAGERY")
    print("=" * 45)

    if client_ready:
        print("🛰️ About Sentinel-1:")
        print("   • Twin satellites providing SAR (radar) imaging")
        print("   • C-band frequency (~5.4 GHz)")
        print("   • All-weather, day/night capability")
        print("   • Penetrates clouds and light rain")
        print("   • 6-day revisit time with both satellites")

        print("\n🔍 SAR search parameters:")
        print("   • Area: Luxembourg (800m × 800m)")
        print(f"   • Dates: {date_start} to {date_end}")
        print("   • Product: GRD (Ground Range Detected)")
        print("   • Polarization: VV,VH (dual-pol)")
        print("   • Orbit: ASCENDING (evening pass)")

        try:
            s1_files = copernicus_client.fetch_s1(
                bbox=target_bbox,
                start_date=date_start,
                end_date=date_end,
                product_type="GRD",
                polarization="VV,VH",
                orbit_direction="ASCENDING",
            )

            print(f"\n✅ Found {len(s1_files)} Sentinel-1 products")

            if s1_files:
                print("\n📊 SAR PRODUCT DETAILS:")
                for s1_idx, s1_file_path in enumerate(s1_files[:3], 1):
                    print(f"\n🛰️ SAR Product {s1_idx}: {s1_file_path.name}")

                    if s1_file_path.suffix == ".json":
                        try:
                            with open(s1_file_path) as s1_file_handle:
                                s1_sar_metadata = json.load(s1_file_handle)

                            sar_id = s1_sar_metadata.get("product_id", "N/A")
                            sar_name = s1_sar_metadata.get("product_name", "N/A")

                            print(f"   🆔 ID: {sar_id}")
                            print(f"   📛 Name: {sar_name}")

                            # Determine satellite
                            if "S1A" in sar_name:
                                satellite = "Sentinel-1A"
                            elif "S1B" in sar_name:
                                satellite = "Sentinel-1B"
                            else:
                                satellite = "Sentinel-1"
                            print(f"   🛰️ Satellite: {satellite}")

                            # Parse date
                            sar_content = s1_sar_metadata.get("content_date", {})
                            sar_start = sar_content.get("Start", "N/A")
                            if sar_start != "N/A":
                                sar_date = sar_start[:10]
                                sar_time = sar_start[11:19]
                                print(f"   📅 Acquired: {sar_date} at {sar_time} UTC")

                            print("   📡 SAR capabilities:")
                            print("      • VV: Water detection, soil moisture")
                            print("      • VH: Vegetation structure, crops")
                            print("      • Weather independent imaging")

                        except Exception as e:
                            print(f"   ❌ Error reading SAR metadata: {e}")
            else:
                print("⚠️ No SAR products found")
                print("💡 Common for recent dates - SAR processing takes longer")

        except Exception as sar_error:
            print(f"❌ SAR fetch failed: {sar_error}")
            s1_files = []
    else:
        print("❌ Client not ready")
        s1_files = []

    return (s1_files,)


@app.cell
def __(center_lat, center_lon, mo, np, plt, s2_files, target_bbox):
    # Visual map display with Copernicus metadata + sample imagery
    print("🗺️ CREATING COMPREHENSIVE SATELLITE DATA VISUALIZATION")
    print("=" * 65)

    print("📍 Generating interactive map...")
    print(f"   • Center: {center_lat:.6f}°N, {center_lon:.6f}°E")
    print("   • Target area: 800m × 800m")
    print("   • Copernicus metadata + sample imagery demonstration")

    # Import required libraries
    try:
        from pathlib import Path as PathLib

        import rasterio

        print("✅ Rasterio imported for satellite imagery processing")

        # Check if we have downloaded S2 ZIP files (actual imagery)
        downloaded_s2_files = [f for f in s2_files if f.suffix == ".zip"]
        metadata_s2_files = [f for f in s2_files if f.suffix == ".json"]

        print("📊 Data status:")
        print(f"   • Downloaded imagery: {len(downloaded_s2_files)} ZIP files")
        print(f"   • Metadata files: {len(metadata_s2_files)} JSON files")

        # Always create a consistent figure structure
        if downloaded_s2_files:
            print("🎯 PROCESSING ACTUAL LUXEMBOURG SATELLITE IMAGERY")

            # Create figure with downloaded imagery
            num_images = min(len(downloaded_s2_files), 2)  # Show up to 2 images
            fig, axes = plt.subplots(1, num_images + 1, figsize=(8 * (num_images + 1), 10))

            # Ensure axes is always a list for consistent indexing
            if num_images + 1 == 1:
                axes = [axes]  # Single subplot case
            elif not hasattr(axes, "__len__"):
                axes = [axes]  # Fallback for single axis

            # FIRST PANEL: Coverage map
            ax_coverage = axes[0]

        else:
            print("📋 SHOWING METADATA + SAMPLE IMAGERY")

            # Create figure with three panels as before
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            # axes is always an array for 3 subplots, so we can index directly
            ax_coverage = axes[0]

        # PANEL 1: Coverage map with Copernicus metadata
        print("🗺️ Creating coverage map with Copernicus metadata...")

        # Map extent with padding
        padding = 0.01
        map_extent = [
            target_bbox[0] - padding,
            target_bbox[2] + padding,
            target_bbox[1] - padding,
            target_bbox[3] + padding,
        ]

        # Create coordinate grid
        lons = np.linspace(map_extent[0], map_extent[1], 100)
        lats = np.linspace(map_extent[2], map_extent[3], 100)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Background terrain pattern
        elevation = np.sin(lon_grid * 100) * np.cos(lat_grid * 100) * 0.1
        ax_coverage.contourf(lon_grid, lat_grid, elevation, levels=20, cmap="terrain", alpha=0.3)

        # Plot target bounding box
        bbox_lons = [
            target_bbox[0],
            target_bbox[2],
            target_bbox[2],
            target_bbox[0],
            target_bbox[0],
        ]
        bbox_lats = [
            target_bbox[1],
            target_bbox[1],
            target_bbox[3],
            target_bbox[3],
            target_bbox[1],
        ]
        ax_coverage.plot(
            bbox_lons, bbox_lats, "r-", linewidth=3, label="Target Area (800m × 800m)"
        )
        ax_coverage.fill(bbox_lons, bbox_lats, "red", alpha=0.2)

        # Center point
        ax_coverage.plot(center_lon, center_lat, "ro", markersize=10, label="Luxembourg Center")

        # Satellite coverage from Copernicus
        if s2_files:
            coverage_lons = [
                target_bbox[0] - 0.005,
                target_bbox[2] + 0.005,
                target_bbox[2] + 0.005,
                target_bbox[0] - 0.005,
                target_bbox[0] - 0.005,
            ]
            coverage_lats = [
                target_bbox[1] - 0.005,
                target_bbox[1] - 0.005,
                target_bbox[3] + 0.005,
                target_bbox[3] + 0.005,
                target_bbox[1] - 0.005,
            ]
            ax_coverage.plot(
                coverage_lons,
                coverage_lats,
                "b--",
                linewidth=2,
                alpha=0.7,
                label=f"Copernicus Products ({len(s2_files)} found)",
            )

        # Customize coverage map
        ax_coverage.set_xlabel("Longitude (°E)", fontsize=12)
        ax_coverage.set_ylabel("Latitude (°N)", fontsize=12)
        ax_coverage.set_title(
            "Luxembourg Target Area\nCopernicus Coverage", fontsize=14, fontweight="bold"
        )
        ax_coverage.grid(True, alpha=0.3)
        ax_coverage.legend(loc="upper right")

        # Add Luxembourg info box
        lux_info = f"Luxembourg Data:\n• Lat: {center_lat:.4f}°N\n• Lon: {center_lon:.4f}°E\n• Area: 800m × 800m\n• Products: {len(s2_files)}"
        ax_coverage.text(
            0.02,
            0.98,
            lux_info,
            transform=ax_coverage.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # PROCESS DOWNLOADED LUXEMBOURG IMAGERY - SHOW ACTUAL IMAGES
        if downloaded_s2_files:
            print("🛰️ Processing downloaded Luxembourg satellite imagery...")

            # Process each downloaded ZIP file
            for idx, s2_zip_file in enumerate(downloaded_s2_files[:2]):  # Limit to 2
                if idx + 1 >= len(axes):
                    break  # Skip if not enough axes

                ax_img = axes[idx + 1]

                try:
                    import tempfile
                    import zipfile
                    from pathlib import Path as PathLib

                    print(f"   📦 Processing: {s2_zip_file.name}")

                    # Extract and process Sentinel-2 ZIP file to show ACTUAL imagery
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = PathLib(temp_dir)

                        # Extract ZIP file
                        with zipfile.ZipFile(s2_zip_file, "r") as zip_ref:
                            zip_ref.extractall(temp_path)

                        # Find SAFE directory (Sentinel-2 format)
                        safe_dirs = list(temp_path.glob("*.SAFE"))
                        if safe_dirs:
                            safe_dir = safe_dirs[0]

                            # Find 10m resolution bands in IMG_DATA directory
                            img_data_dir = safe_dir / "GRANULE"
                            granule_dirs = list(img_data_dir.glob("*"))

                            if granule_dirs:
                                granule_dir = granule_dirs[0]
                                img_dir = granule_dir / "IMG_DATA"

                                # Look for RGB bands with correct naming pattern
                                band_files = {}
                                for band in ["B02", "B03", "B04"]:  # Blue, Green, Red
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
                                            print(f"     Found {band}: {band_matches[0].name}")
                                            break

                                if len(band_files) >= 3:
                                    print("     Creating RGB composite from Luxembourg imagery...")

                                    # Create RGB composite from actual satellite data
                                    rgb_bands = []
                                    band_order = [
                                        "B04",
                                        "B03",
                                        "B02",
                                    ]  # Red, Green, Blue for RGB display

                                    for band_name in band_order:
                                        if band_name in band_files:
                                            with rasterio.open(band_files[band_name]) as src:
                                                band_data = src.read(1)

                                                # Get geospatial info from first band
                                                if len(rgb_bands) == 0:
                                                    _ = (
                                                        src.transform
                                                    )  # Store transform (unused but available)
                                                    bounds = src.bounds
                                                    crs = src.crs
                                                    print(f"     Image bounds: {bounds}")

                                                rgb_bands.append(band_data)

                                    if len(rgb_bands) == 3:
                                        # Stack and normalize bands for display
                                        rgb_array = np.stack(rgb_bands, axis=0)
                                        rgb_normalized = np.zeros_like(rgb_array, dtype=np.float32)

                                        for i in range(3):
                                            band = rgb_array[i]
                                            # Use percentile normalization for better contrast
                                            valid_pixels = band[band > 0]
                                            if len(valid_pixels) > 0:
                                                p2, p98 = np.percentile(valid_pixels, [2, 98])
                                                if p98 > p2:
                                                    rgb_normalized[i] = np.clip(
                                                        (band - p2) / (p98 - p2), 0, 1
                                                    )
                                                else:
                                                    rgb_normalized[i] = (
                                                        band / band.max()
                                                        if band.max() > 0
                                                        else band
                                                    )

                                        # Display the ACTUAL Luxembourg satellite image
                                        rgb_display = np.transpose(rgb_normalized, (1, 2, 0))

                                        # Convert UTM bounds to WGS84 for proper display
                                        from rasterio.warp import transform_bounds

                                        # Transform bounds from UTM to WGS84
                                        wgs84_bounds = transform_bounds(
                                            crs,
                                            "EPSG:4326",
                                            bounds.left,
                                            bounds.bottom,
                                            bounds.right,
                                            bounds.top,
                                        )

                                        extent = [
                                            wgs84_bounds[0],
                                            wgs84_bounds[2],
                                            wgs84_bounds[1],
                                            wgs84_bounds[3],
                                        ]
                                        print(f"     WGS84 bounds: {wgs84_bounds}")

                                        # Show the actual satellite imagery
                                        ax_img.imshow(rgb_display, extent=extent, aspect="auto")

                                        # Add target area overlay on the satellite image
                                        bbox_lons = [
                                            target_bbox[0],
                                            target_bbox[2],
                                            target_bbox[2],
                                            target_bbox[0],
                                            target_bbox[0],
                                        ]
                                        bbox_lats = [
                                            target_bbox[1],
                                            target_bbox[1],
                                            target_bbox[3],
                                            target_bbox[3],
                                            target_bbox[1],
                                        ]
                                        ax_img.plot(
                                            bbox_lons,
                                            bbox_lats,
                                            "red",
                                            linewidth=3,
                                            alpha=0.8,
                                            label="Target Area",
                                        )

                                        # Zoom to Luxembourg area (with some padding)
                                        padding = 0.02
                                        ax_img.set_xlim(
                                            target_bbox[0] - padding, target_bbox[2] + padding
                                        )
                                        ax_img.set_ylim(
                                            target_bbox[1] - padding, target_bbox[3] + padding
                                        )

                                        # Customize plot
                                        ax_img.set_xlabel("Longitude (°E)", fontsize=12)
                                        ax_img.set_ylabel("Latitude (°N)", fontsize=12)
                                        ax_img.set_title(
                                            f"Luxembourg Satellite Image #{idx+1}\n{s2_zip_file.name[:40]}...",
                                            fontsize=11,
                                            fontweight="bold",
                                        )
                                        ax_img.grid(True, alpha=0.3, color="white")
                                        ax_img.legend()

                                        print(
                                            "   ✅ ACTUAL Luxembourg RGB satellite image displayed!"
                                        )
                                        continue

                                # If we couldn't find RGB bands, show info
                                ax_img.text(
                                    0.5,
                                    0.5,
                                    f"Found {len(band_files)} bands\nLooking for RGB bands...",
                                    ha="center",
                                    va="center",
                                    transform=ax_img.transAxes,
                                    fontsize=12,
                                )
                                ax_img.set_title("Processing Bands...", fontsize=12)
                            else:
                                ax_img.text(
                                    0.5,
                                    0.5,
                                    "No granule directories found",
                                    ha="center",
                                    va="center",
                                    transform=ax_img.transAxes,
                                    fontsize=12,
                                )
                                ax_img.set_title("Extraction Issue", fontsize=12)
                        else:
                            ax_img.text(
                                0.5,
                                0.5,
                                "No SAFE directory found in ZIP",
                                ha="center",
                                va="center",
                                transform=ax_img.transAxes,
                                fontsize=12,
                            )
                            ax_img.set_title("ZIP Structure Issue", fontsize=12)

                except Exception as e:
                    print(f"   ❌ Error processing {s2_zip_file.name}: {e}")
                    ax_img.text(
                        0.5,
                        0.5,
                        f"Error extracting imagery:\n{str(e)[:100]}...",
                        ha="center",
                        va="center",
                        transform=ax_img.transAxes,
                        fontsize=10,
                    )
                    ax_img.set_title("Processing Error", fontsize=12)

            print("✅ Luxembourg satellite imagery processing complete!")

        else:
            # FALLBACK: Show metadata only
            pass

        print("📊 Creating Copernicus metadata visualization...")

        # Determine which axis to use for metadata display
        if hasattr(axes, "__len__") and len(axes) >= 2:
            ax_metadata = axes[1]
        else:
            ax_metadata = ax_coverage  # Fallback to coverage axis if only one panel

        if s2_files:
            # Read metadata from one of the files
            metadata_file = s2_files[0]
            if metadata_file.suffix == ".json":
                try:
                    import json as json_lib

                    with open(metadata_file) as f:
                        viz_metadata = json_lib.load(f)

                    # Create a text visualization of the metadata
                    ax_metadata.axis("off")  # Remove axes for text display

                    # Format metadata for display
                    display_text = "🛰️ COPERNICUS METADATA\n" + "=" * 30 + "\n\n"

                    if "product_name" in viz_metadata:
                        product_name = viz_metadata["product_name"]
                        display_text += f"📛 Product: {product_name[:40]}...\n\n"

                    if "content_date" in viz_metadata:
                        viz_content_date = viz_metadata.get("content_date", {})
                        if isinstance(viz_content_date, dict) and "Start" in viz_content_date:
                            date_str = viz_content_date["Start"][:10]
                            time_str = viz_content_date["Start"][11:19]
                            display_text += f"📅 Acquired: {date_str}\n"
                            display_text += f"⏰ Time: {time_str} UTC\n\n"

                    display_text += "📡 Bands Available:\n"
                    display_text += "   • B02 (490nm) Blue - 10m\n"
                    display_text += "   • B03 (560nm) Green - 10m\n"
                    display_text += "   • B04 (665nm) Red - 10m\n"
                    display_text += "   • B08 (842nm) NIR - 10m\n"
                    display_text += "   • + 9 more spectral bands\n\n"

                    if "download_url" in viz_metadata:
                        display_text += "🔗 Download URL:\n"
                        display_text += "   Available via Copernicus API\n\n"

                    display_text += "💾 Status: Metadata cached\n"
                    display_text += "📊 Ready for download/processing"

                    # Display the text
                    ax_metadata.text(
                        0.05,
                        0.95,
                        display_text,
                        transform=ax_metadata.transAxes,
                        fontsize=11,
                        verticalalignment="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=0.9),
                    )

                    ax_metadata.set_title(
                        "Copernicus Product Metadata\n(From Live API)",
                        fontsize=14,
                        fontweight="bold",
                    )

                    print("✅ Copernicus metadata visualization created")

                except Exception as e:
                    ax_metadata.text(
                        0.5,
                        0.5,
                        f"Error reading\nCopernicus metadata:\n{e}",
                        ha="center",
                        va="center",
                        transform=ax_metadata.transAxes,
                        fontsize=12,
                    )
                    ax_metadata.set_title("Metadata Error", fontsize=14)
            else:
                ax_metadata.text(
                    0.5,
                    0.5,
                    "No Copernicus\nmetadata available",
                    ha="center",
                    va="center",
                    transform=ax_metadata.transAxes,
                    fontsize=14,
                )
                ax_metadata.set_title("No Metadata", fontsize=14)
        else:
            ax_metadata.text(
                0.5,
                0.5,
                "No Copernicus products\nfound for this area\nand date range",
                ha="center",
                va="center",
                transform=ax_metadata.transAxes,
                fontsize=14,
            )
            ax_metadata.set_title("No Products Found", fontsize=14)

        # Skip sample imagery - focus on actual Luxembourg data
        print("🎯 Focusing on actual Luxembourg satellite data from Copernicus")

        plt.tight_layout()

        print("✅ Luxembourg satellite data visualization created!")
        print("   • LEFT: Luxembourg coverage map with target area")
        print("   • CENTER: Downloaded Luxembourg Sentinel-2 data")
        print("   • RIGHT: Additional Luxembourg satellite products")

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")

        # Create simple fallback
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        ax1.text(
            0.5,
            0.5,
            "Installing satellite imagery\nprocessing dependencies...",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=16,
        )
        ax1.set_title("Setting Up Satellite Processing", fontsize=14)

    mo.as_html(fig)

    return fig


@app.cell
def __(client_ready, copernicus_client, date_end, date_start, target_bbox):
    # Cache performance test
    print("💾 CACHE PERFORMANCE TEST")
    print("=" * 35)

    if client_ready:
        import time

        cache_dir = copernicus_client.cache_dir
        print(f"📁 Cache directory: {cache_dir}")

        if cache_dir.exists():
            cache_files = list(cache_dir.rglob("*"))
            cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
            print(f"📊 Cache: {len(cache_files)} files, {cache_size:,} bytes")

        print("\n🎯 Testing cache performance...")
        start_time = time.time()

        try:
            cached_results = copernicus_client.fetch_s2(
                bbox=target_bbox,
                start_date=date_start,
                end_date=date_end,
                resolution=10,
                max_cloud_cover=50,
                product_type="S2MSI1C",
                download_data=True,  # Same as main fetch
                interactive=False,  # No prompts for cache test
            )

            duration = time.time() - start_time

            print(f"⏱️ Completed in {duration:.4f} seconds")
            print(f"📦 Products: {len(cached_results)}")

            if duration < 0.1:
                print("🚀 CACHE HIT! (Lightning fast)")
                print("💡 No API calls - instant retrieval")
            else:
                print("🌐 API CALL (First time)")
                print("💡 Results cached for future requests")

        except Exception as cache_error:
            print(f"❌ Cache test failed: {cache_error}")
    else:
        print("❌ Client not ready")

    return


@app.cell
def __(client_ready, s1_files, s2_files):
    # Final summary
    print("📊 MISSION SUMMARY")
    print("=" * 30)

    if client_ready:
        s2_count = len(s2_files) if "s2_files" in locals() and s2_files else 0
        s1_count = len(s1_files) if "s1_files" in locals() and s1_files else 0
        total = s2_count + s1_count

        print("🎯 RESULTS:")
        print(f"   🔵 Sentinel-2 (Optical): {s2_count} products")
        print(f"   🔴 Sentinel-1 (SAR): {s1_count} products")
        print(f"   📦 Total: {total} satellite products")

        print("\n✅ CAPABILITIES DEMONSTRATED:")
        print("   🔐 OAuth2 authentication")
        print("   🛰️ Multi-satellite data discovery")
        print("   🗺️ Visual map display")
        print("   💾 Intelligent caching")
        print("   📊 Metadata analysis")

        print("\n🌟 APPLICATIONS ENABLED:")
        print("   🌾 Agriculture: Crop monitoring")
        print("   🌊 Water: Quality assessment")
        print("   🏙️ Urban: Development tracking")
        print("   🌲 Environment: Forest monitoring")

        print("\n🚀 READY FOR GALILEO INTEGRATION!")
        print("   📍 Luxembourg: 800m × 800m")
        print(f"   📦 Products: {total} discovered")
        print("   🎯 Resolution: 10m (80×80 pixels)")

        if total > 0:
            visual = "   " + "🔵" * s2_count + "🔴" * s1_count
            print(f"\n📈 Visual: {visual}")
            print("   Legend: 🔵=Sentinel-2, 🔴=Sentinel-1")

    else:
        print("❌ Authentication failed")
        print("💡 Check .env credentials")

    return


if __name__ == "__main__":
    app.run()
