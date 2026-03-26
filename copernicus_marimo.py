"""Copernicus Data Space Ecosystem Explorer - Interactive GUI

This Marimo app provides an interactive interface for:
1. Configuring Copernicus API credentials
2. Searching for Sentinel-1 (SAR) and Sentinel-2 (optical) satellite data
3. Downloading and visualizing satellite imagery

The app uses the Copernicus Data Space Ecosystem API, which provides free access
to Sentinel satellite data. No credit card required - just register at:
https://dataspace.copernicus.eu/
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import required libraries for the application."""
    import os
    import traceback
    from datetime import datetime, timedelta
    from pathlib import Path

    import marimo as mo

    return Path, datetime, mo, os, timedelta, traceback


@app.cell
def _(mo):
    """Display the application header with instructions."""
    mo.md(
        """
    # Copernicus Data Space Ecosystem Explorer

    This interactive GUI allows you to:
    1. **Configure credentials** - Save your Copernicus account credentials securely
    2. **Search for data** - Find Sentinel-1 (SAR) or Sentinel-2 (optical) imagery
    3. **Download & visualize** - Automatically download and display satellite images

    ## Getting Started

    **Get free Copernicus account:**
    1. Visit: https://dataspace.copernicus.eu/
    2. Click "Register" (no credit card required)
    3. Use your account username/email and password below

    **About the satellites:**
    - **Sentinel-2 (S2)**: Optical imagery (like a camera), best for seeing colors, vegetation, water
    - **Sentinel-1 (S1)**: Radar imagery (SAR), works through clouds, day/night, good for water detection
    """
    )
    return


@app.cell
def _(Path):
    """Check if Copernicus credentials are already configured in .env file.

    This cell reads the .env file (if it exists) and checks for valid
    COPERNICUS_USERNAME and COPERNICUS_PASSWORD entries.
    """
    env_path = Path(".env")
    env_exists = env_path.exists()

    # Initialize credential flags
    has_username = False
    has_password = False

    # If .env exists, check if it has valid credentials
    if env_exists:
        with open(env_path, "r") as _f:
            content = _f.read()
            # Check for USERNAME (must have non-empty value)
            has_username = (
                "COPERNICUS_USERNAME=" in content
                and len(content.split("COPERNICUS_USERNAME=")[1].split("\n")[0].strip()) > 0
            )
            # Check for PASSWORD (must have non-empty value)
            has_password = (
                "COPERNICUS_PASSWORD=" in content
                and len(content.split("COPERNICUS_PASSWORD=")[1].split("\n")[0].strip()) > 0
            )

    # Credentials are configured only if both username and password are present
    credentials_configured = has_username and has_password

    return credentials_configured, env_path


@app.cell
def _(credentials_configured, mo):
    """Display credential input form with status message.

    Shows a green checkmark if credentials are already configured,
    or a warning if they need to be entered.
    """
    # Show appropriate status message
    if credentials_configured:
        status_msg = mo.md(
            """
            ## ✅ Credentials Configured

            Your Copernicus credentials are saved in `.env` file.
            You can proceed to search for satellite data below.

            *To update credentials, enter new values and click Save.*
            """
        )
    else:
        status_msg = mo.md(
            """
            ## ⚠️ Credentials Required

            Please enter your Copernicus account credentials below.
            These will be saved securely in a `.env` file.

            **Don't have an account yet?**
            1. Register for free at: https://dataspace.copernicus.eu/
            2. Use your account username/email and password below
            """
        )

    # Create input widgets (password type hides the values for security)
    username_input = mo.ui.text(
        label="Username/Email",
        kind="text",
        placeholder="Enter your Copernicus username or email",
    )
    password_input = mo.ui.text(
        label="Password",
        kind="password",
        placeholder="Enter your Copernicus password",
    )
    save_button = mo.ui.run_button(label="💾 Save Credentials")

    # Display the form vertically stacked
    mo.vstack([status_msg, username_input, password_input, save_button])
    return password_input, save_button, username_input


@app.cell
def _(
    env_path,
    mo,
    os,
    password_input,
    save_button,
    traceback,
    username_input,
):
    """Save credentials to .env file when Save button is clicked.

    This cell:
    1. Validates that both fields are filled
    2. Preserves any existing .env variables (doesn't overwrite other settings)
    3. Writes COPERNICUS_USERNAME and COPERNICUS_PASSWORD
    4. Sets credentials in current environment for immediate use
    """
    save_result = ""

    # Only process if save button was clicked
    if save_button.value:
        # Get values from input fields
        username = username_input.value
        password = password_input.value

        # Validate that both fields have values
        if not username or not password:
            save_result = """
            ❌ **Error: Both fields are required**

            Please enter both your username/email and password.
            """
        else:
            try:
                # Read existing .env content to preserve other variables
                existing_content = ""
                if env_path.exists():
                    with open(env_path, "r") as _f:
                        lines = _f.readlines()
                        # Keep all lines except COPERNICUS credentials
                        # (we'll add updated ones below)
                        existing_content = "".join(
                            [
                                line
                                for line in lines
                                if not line.startswith("COPERNICUS_USERNAME=")
                                and not line.startswith("COPERNICUS_PASSWORD=")
                            ]
                        )

                # Write the updated .env file
                with open(env_path, "w") as _f:
                    _f.write(existing_content)
                    # Ensure newline before adding credentials
                    if existing_content and not existing_content.endswith("\n"):
                        _f.write("\n")
                    _f.write(f"COPERNICUS_USERNAME={username}\n")
                    _f.write(f"COPERNICUS_PASSWORD={password}\n")

                # Also set in current environment for immediate use
                # (so you don't need to restart the app)
                os.environ["COPERNICUS_USERNAME"] = username
                os.environ["COPERNICUS_PASSWORD"] = password

                save_result = """
                ✅ **Credentials saved successfully!**

                Your credentials are now saved in `.env` file and ready to use.
                You can proceed to search for satellite data below.
                """

            except Exception as e:
                save_result = f"""
                ❌ **Error saving credentials**

                {str(e)}

                Please check file permissions and try again.
                """

    # Display result message if there is one
    mo.md(save_result) if save_result else None
    return


@app.cell
def _(mo):
    """Display section header for search parameters."""
    mo.md(
        """
    ---
    ## 🛰️ Search Parameters

    Configure your search below. The default values show a small area in Luxembourg
    as an example. You can modify any parameter to search for data in your area of interest.
    """
    )
    return


@app.cell
def _(datetime, mo, timedelta):
    """Create search parameter input widgets.

    Default values:
    - Location: Bloemhof Dam area, South Africa (25.68-25.69°E, -27.67--27.66°S)
    - Date range: Last 30 days
    - Satellite: Sentinel-2 (optical)
    - Max products: 2 (to keep download size manageable)
    - Crop to bbox: False (show full context by default)
    """
    # Calculate default date range (last 30 days)
    default_end_date = datetime.now().strftime("%Y-%m-%d")
    default_start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Coordinate inputs (bounding box)
    # Format: [min_lon, min_lat, max_lon, max_lat]
    min_lon = mo.ui.number(
        start=-180,
        stop=180,
        step=0.01,
        value=25.6796,
        label="Min Longitude (°E)",
    )
    min_lat = mo.ui.number(
        start=-90,
        stop=90,
        step=0.01,
        value=-27.6721,
        label="Min Latitude (°N)",
    )
    max_lon = mo.ui.number(
        start=-180,
        stop=180,
        step=0.01,
        value=25.6897,
        label="Max Longitude (°E)",
    )
    max_lat = mo.ui.number(
        start=-90,
        stop=90,
        step=0.01,
        value=-27.663,
        label="Max Latitude (°N)",
    )

    # Satellite type selection
    # S2 = Sentinel-2 (optical/camera-like imagery)
    # S1 = Sentinel-1 (radar/SAR imagery, works through clouds)
    # S1+S2 = Both (required for Galileo time series export)
    satellite_type = mo.ui.dropdown(
        options=["S2", "S1", "S1+S2"],
        value="S2",
        label="Satellite Type (S2=Optical, S1=Radar, S1+S2=Both)",
    )

    # Date range inputs
    start_date = mo.ui.date(value=default_start_date, label="Start Date")
    end_date = mo.ui.date(value=default_end_date, label="End Date")

    # Max products to download (limited to prevent excessive downloads)
    max_products = mo.ui.number(
        start=1,
        stop=50,
        step=1,
        value=2,
        label="Max Products",
    )

    # Crop to bbox option
    # When enabled: Only shows target area (faster, less memory)
    # When disabled: Shows full tile with context (slower, more memory)
    crop_to_bbox = mo.ui.checkbox(
        value=False,
        label="Crop to target area only (faster, less memory)",
    )

    # Search button triggers the search and download
    search_button = mo.ui.run_button(label="🔍 Search & Download")

    return (
        crop_to_bbox,
        end_date,
        max_lat,
        max_lon,
        max_products,
        min_lat,
        min_lon,
        satellite_type,
        search_button,
        start_date,
    )


@app.cell
def _(
    crop_to_bbox,
    end_date,
    max_lat,
    max_lon,
    max_products,
    min_lat,
    min_lon,
    mo,
    satellite_type,
    search_button,
    start_date,
):
    """Display search parameter widgets in a nice layout."""
    mo.vstack(
        [
            mo.md(
                """
                **Bounding Box**: Define the geographic area to search.
                Coordinates are in decimal degrees (WGS84).
                """
            ),
            mo.hstack([min_lon, min_lat, max_lon, max_lat]),
            mo.md(
                """
                **Satellite & Time Range**: Choose satellite type and date range.
                - **S2 (Sentinel-2)**: Optical imagery, 10m resolution, affected by clouds
                - **S1 (Sentinel-1)**: Radar imagery, 10m resolution, works through clouds

                **Max Products**: Number of images to download (1-50). Start with 2-5 for testing.
                """
            ),
            mo.hstack([satellite_type, start_date, end_date, max_products]),
            mo.md(
                """
                **Visualization Options**:
                """
            ),
            crop_to_bbox,
            mo.callout(
                mo.md(
                    """
                    **Tip**: Unchecked (default) shows full satellite tile with context around your target area.
                    Check the box if you only want to see the target area (uses less memory, good for many images).
                    """
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    """
                    **Note**: After changing any parameters above, click "Search & Download" again to apply the changes.
                    """
                ),
                kind="warn",
            ),
            search_button,
        ]
    )
    return


@app.cell
def _(
    credentials_configured,
    end_date,
    max_lat,
    max_lon,
    max_products,
    min_lat,
    min_lon,
    mo,
    satellite_type,
    search_button,
    start_date,
    traceback,
):
    """Search for and download satellite data when Search button is clicked.

    This cell:
    1. Validates that credentials are configured
    2. Initializes the Copernicus client
    3. Searches for products matching the criteria
    4. Downloads the products to data/cache/copernicus/
    5. Returns list of downloaded file paths for visualization

    The search uses different parameters for S1 vs S2:
    - S2: Filters by cloud cover (<30%), uses L2A processing level
    - S1: Filters by polarization (VV+VH), uses GRD product type
    """
    download_result = ""
    downloaded_files = []

    # Only process if search button was clicked
    if search_button.value:
        # First check: Do we have credentials?
        if not credentials_configured:
            download_result = """
            ❌ **Credentials Required**

            Please configure your Copernicus credentials above before searching.
            """
        else:
            # Show spinner while processing (can take 30s-2min for downloads)
            with mo.status.spinner(title="Searching and downloading...") as _spinner:
                try:
                    # Import the Copernicus client
                    _spinner.update(title="Initializing Copernicus client...")
                    from src.data.copernicus import CopernicusClient

                    # Get search parameters from widgets
                    _bbox = [min_lon.value, min_lat.value, max_lon.value, max_lat.value]

                    # Validate bounding box
                    if _bbox[0] >= _bbox[2] or _bbox[1] >= _bbox[3]:
                        download_result = """
                        ❌ **Invalid Bounding Box**

                        Min longitude must be less than max longitude.
                        Min latitude must be less than max latitude.
                        """
                        raise ValueError("Invalid bounding box")

                    # Initialize the client (handles OAuth2 authentication)
                    client = CopernicusClient()

                    # Build initial result message
                    download_result = f"""
                    ### 🔍 Searching for {satellite_type.value} products

                    **Area**: {_bbox[0]:.3f}°E to {_bbox[2]:.3f}°E, {_bbox[1]:.3f}°N to {_bbox[3]:.3f}°N
                    **Dates**: {start_date.value} to {end_date.value}
                    **Max products**: {max_products.value}

                    """

                    # Capture stdout to get the "Found X products" message
                    import io
                    import sys

                    _stdout_capture = io.StringIO()
                    _old_stdout = sys.stdout
                    sys.stdout = _stdout_capture

                    try:
                        # Call appropriate fetch method based on satellite type
                        if satellite_type.value == "S2":
                            _spinner.update(title="Searching for Sentinel-2 products...")
                            # Sentinel-2 parameters:
                            # - resolution: 10m (highest resolution for RGB bands)
                            # - max_cloud_cover: 30% (filter out very cloudy images)
                            # - product_type: S2MSI2A (Level-2A = atmospherically corrected)
                            downloaded_files = client.fetch_s2(
                                bbox=_bbox,
                                start_date=str(start_date.value),
                                end_date=str(end_date.value),
                                resolution=10,
                                max_cloud_cover=30,
                                product_type="S2MSI2A",
                                download_data=True,
                                interactive=False,
                                max_products=max_products.value,
                            )
                        elif satellite_type.value == "S1":
                            _spinner.update(title="Searching for Sentinel-1 products...")
                            # Sentinel-1 parameters:
                            # - product_type: GRD (Ground Range Detected = processed SAR)
                            # - polarization: VV,VH (dual-pol for better feature detection)
                            # - orbit_direction: ASCENDING (consistent viewing geometry)
                            downloaded_files = client.fetch_s1(
                                bbox=_bbox,
                                start_date=str(start_date.value),
                                end_date=str(end_date.value),
                                product_type="GRD",
                                polarization="VV,VH",
                                orbit_direction="ASCENDING",
                                download_data=True,
                                max_products=max_products.value,
                            )
                        else:  # S1+S2
                            _spinner.update(title="Searching for Sentinel-2 products...")
                            # Download S2 first
                            _s2_files = client.fetch_s2(
                                bbox=_bbox,
                                start_date=str(start_date.value),
                                end_date=str(end_date.value),
                                resolution=10,
                                max_cloud_cover=30,
                                product_type="S2MSI2A",
                                download_data=True,
                                interactive=False,
                                max_products=max_products.value,
                            )

                            _spinner.update(title="Searching for Sentinel-1 products...")
                            # Download S1 second
                            _s1_files = client.fetch_s1(
                                bbox=_bbox,
                                start_date=str(start_date.value),
                                end_date=str(end_date.value),
                                product_type="GRD",
                                polarization="VV,VH",
                                orbit_direction="ASCENDING",
                                download_data=True,
                                max_products=max_products.value,
                            )

                            # Store both as a dict for later use
                            downloaded_files = {"s2": _s2_files, "s1": _s1_files}
                    finally:
                        # Restore stdout
                        sys.stdout = _old_stdout
                        _captured_output = _stdout_capture.getvalue()

                    # Try to get total available from cache file
                    # The cache stores all products found, not just the ones downloaded
                    _total_available = None
                    try:
                        import json

                        from src.data.copernicus.utils import build_cache_key

                        _cache_key = build_cache_key(
                            "s2" if satellite_type.value == "S2" else "s1",
                            bbox=_bbox,
                            start_date=str(start_date.value),
                            end_date=str(end_date.value),
                            resolution=10 if satellite_type.value == "S2" else None,
                            max_cloud_cover=30 if satellite_type.value == "S2" else None,
                            product_type="S2MSI2A" if satellite_type.value == "S2" else "GRD",
                            download_data=True,
                            max_products=max_products.value,
                            polarization="VV,VH" if satellite_type.value == "S1" else None,
                            orbit_direction="ASCENDING" if satellite_type.value == "S1" else None,
                        )
                        _cache_file = client.cache_dir / f"{_cache_key}.json"

                        if _cache_file.exists():
                            with open(_cache_file) as _f:
                                _cache_data = json.load(_f)
                                _all_products = _cache_data.get("products", [])
                                _total_available = len(_all_products)
                    except Exception as _e:
                        # If cache reading fails, try parsing stdout
                        for _line in _captured_output.split("\n"):
                            if "Found" in _line and "products" in _line:
                                import re as _re

                                _match = _re.search(r"Found (\d+)", _line)
                                if _match:
                                    _total_available = int(_match.group(1))
                                    break

                    # Check if we got any files
                    if satellite_type.value == "S1+S2":
                        # Handle S1+S2 case
                        _s2_count = (
                            len(downloaded_files.get("s2", []))
                            if isinstance(downloaded_files, dict)
                            else 0
                        )
                        _s1_count = (
                            len(downloaded_files.get("s1", []))
                            if isinstance(downloaded_files, dict)
                            else 0
                        )

                        if _s2_count > 0 or _s1_count > 0:
                            _spinner.update(title="Download complete!")
                            download_result += f"""
                            ✅ **Downloaded S1+S2 products!**

                            - **S2 (Optical)**: {_s2_count} product(s)
                            - **S1 (Radar)**: {_s1_count} product(s)

                            Files are cached in `data/cache/copernicus/` for future use.

                            *Note: Visualization will show S2 images. Use "Export Time Series for Galileo" to export both.*
                            """
                        else:
                            download_result += """
                            ⚠️ **No products found**

                            Try adjusting your search parameters:
                            - Expand the date range
                            - Try a different location
                            - For S2, try a different time of year (less clouds)
                            """
                    elif downloaded_files and len(downloaded_files) > 0:
                        _spinner.update(title="Download complete!")

                        # Show total available vs downloaded
                        if _total_available is not None and _total_available > len(
                            downloaded_files
                        ):
                            download_result += f"""
                            ✅ **Downloaded {len(downloaded_files)} of {_total_available} available product(s)!**

                            *Note: {_total_available - len(downloaded_files)} additional product(s) available. Increase "Max Products" to download more.*

                            Files are cached in `data/cache/copernicus/` for future use.

                            **Downloaded files:**
                            """
                        elif _total_available is not None:
                            download_result += f"""
                            ✅ **Downloaded all {len(downloaded_files)} available product(s)!**

                            Files are cached in `data/cache/copernicus/` for future use.

                            **Downloaded files:**
                            """
                        else:
                            download_result += f"""
                            ✅ **Downloaded {len(downloaded_files)} product(s)!**

                            Files are cached in `data/cache/copernicus/` for future use.

                            **Downloaded files:**
                            """
                        for _f in downloaded_files:
                            # Show just the filename, not full path
                            download_result += f"\n- `{_f.name}`"
                    else:
                        download_result += """
                        ⚠️ **No products found**

                        Try adjusting your search parameters:
                        - Expand the date range
                        - Try a different location
                        - For S2, try a different time of year (less clouds)
                        """

                except ValueError:
                    # Validation error (already set download_result above)
                    pass
                except Exception as e:
                    # Handle any other errors
                    _error_msg = str(e)
                    download_result = f"""
                    ❌ **Error during search/download**

                    {_error_msg}

                    **Common issues:**
                    - Check your internet connection
                    - Verify your credentials are correct
                    - Try a smaller area or fewer products
                    - Check console for detailed error trace
                    """
                    # Print full traceback to console for debugging
                    print("Error details:")
                    print(traceback.format_exc())

    # Display the result message
    mo.md(download_result) if download_result else None
    return (downloaded_files,)


@app.cell
def _(
    crop_to_bbox,
    datetime,
    downloaded_files,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    mo,
    satellite_type,
):
    """Create time slider and pre-process all images for fast rendering.

    This cell:
    1. Extracts dates from filenames and creates metadata
    2. Pre-processes ALL images into memory (cached)
    3. Creates slider widget for navigation

    Pre-processing happens once, making slider interactions instant.
    The crop_to_bbox option controls whether to show full context or just target area.

    For S1+S2 mode: Visualizes S2 images (optical), but both S1 and S2 are available for export.
    """
    time_slider = None
    file_metadata = []
    cached_images = []

    # Determine which files to visualize
    _files_to_visualize = []
    _viz_satellite_type = satellite_type.value

    if satellite_type.value == "S1+S2":
        # For S1+S2, visualize S2 images (more intuitive for users)
        if isinstance(downloaded_files, dict) and "s2" in downloaded_files:
            _files_to_visualize = downloaded_files["s2"]
            _viz_satellite_type = "S2"
    elif downloaded_files and not isinstance(downloaded_files, dict):
        # Single satellite type (S1 or S2)
        _files_to_visualize = downloaded_files

    if _files_to_visualize and len(_files_to_visualize) > 0:
        # Show progress while pre-processing
        with mo.status.spinner(title="Pre-processing images for fast slider...") as _spinner:
            import re

            # Get bbox for optional cropping
            _bbox = [min_lon.value, min_lat.value, max_lon.value, max_lat.value]
            _use_bbox = crop_to_bbox.value  # User's choice

            # Extract dates and pre-process images
            for _idx, _file_path in enumerate(_files_to_visualize):
                _spinner.update(title=f"Processing image {_idx + 1}/{len(_files_to_visualize)}...")

                _filename = _file_path.name

                # Extract date from filename
                _date_match = re.search(r"(\d{8}T\d{6})", _filename)
                if _date_match:
                    _date_str = _date_match.group(1)
                    _date_obj = datetime.strptime(_date_str, "%Y%m%dT%H%M%S")
                    _date_display = _date_obj.strftime("%Y-%m-%d %H:%M")
                else:
                    _date_obj = None
                    _date_display = "Unknown date"

                # Pre-process the image based on satellite type
                _processed_data = None
                try:
                    if _viz_satellite_type == "S2":
                        # Sentinel-2: Extract RGB composite
                        from src.data.copernicus.image_processing import extract_rgb_composite

                        _processed_data = extract_rgb_composite(
                            _file_path, bbox=_bbox if _use_bbox else None
                        )
                    else:
                        # Sentinel-1: Extract SAR data (VV polarization)
                        from src.data.copernicus.image_processing import extract_sar_composite

                        _processed_data = extract_sar_composite(
                            _file_path,
                            polarizations=["VV"],
                            bbox=_bbox if _use_bbox else None,
                        )

                    if (
                        _processed_data is not None
                        and _processed_data.get("bounds_wgs84") is not None
                    ):
                        # Store metadata and cached image data
                        file_metadata.append(
                            {
                                "path": _file_path,
                                "date": _date_obj,
                                "date_str": _date_display,
                                "filename": _filename,
                            }
                        )
                        cached_images.append(_processed_data)
                    else:
                        print(f"Warning: Failed to process {_filename} (missing bounds or data)")

                except Exception as _e:
                    print(f"Error processing {_filename}: {_e}")
                    import traceback as _tb

                    _tb.print_exc()
                    continue

            _spinner.update(title="Processing complete!")

        # Sort by date (oldest first) - sort both lists together
        if file_metadata:
            _sorted_pairs = sorted(
                zip(file_metadata, cached_images),
                key=lambda x: x[0]["date"] if x[0]["date"] else datetime.min,
            )
            file_metadata, cached_images = (
                [x[0] for x in _sorted_pairs],
                [x[1] for x in _sorted_pairs],
            )

            # Create slider if we have multiple files
            if len(file_metadata) > 1:
                time_slider = mo.ui.slider(
                    start=0,
                    stop=len(file_metadata) - 1,
                    step=1,
                    value=0,
                    label=f"Time Step (1 of {len(file_metadata)})",
                    show_value=False,
                )

    return cached_images, file_metadata, time_slider


@app.cell
def _(cached_images, mo, satellite_type):
    """Create band recipe selector (independent of time slider to maintain state)."""
    band_recipe_selector = None

    if cached_images and len(cached_images) > 0:
        from src.data.copernicus.band_recipes import get_available_recipes

        _recipes = get_available_recipes()

        # Filter recipes based on satellite type
        if satellite_type.value == "S2":
            # S2: Show optical recipes only
            _recipe_names = [r.name for rid, r in _recipes.items() if not rid.startswith("sar_")]
        elif satellite_type.value == "S1":
            # S1: Show SAR recipes only
            _recipe_names = [r.name for rid, r in _recipes.items() if rid.startswith("sar_")]
        else:  # S1+S2
            # S1+S2: Show optical recipes (visualizing S2)
            _recipe_names = [r.name for rid, r in _recipes.items() if not rid.startswith("sar_")]

        if _recipe_names:
            band_recipe_selector = mo.ui.dropdown(
                options=_recipe_names,
                label="Band Recipe",
            )

    return (band_recipe_selector,)


@app.cell
def _(band_recipe_selector, cached_images, file_metadata, mo, time_slider):
    """Display time slider and adjustment sliders in a compact side-by-side layout."""
    slider_display = None
    contrast_slider = None
    brightness_slider = None
    gamma_slider = None

    if cached_images and len(cached_images) > 0:
        # Create adjustment sliders (0-100 range, will be normalized in visualization)
        contrast_slider = mo.ui.slider(
            start=0,
            stop=100,
            step=1,
            value=50,  # 50 = 1.0x (no change)
            label="Contrast",
            show_value=True,
        )

        brightness_slider = mo.ui.slider(
            start=0,
            stop=100,
            step=1,
            value=50,  # 50 = 0.0 (no change)
            label="Brightness",
            show_value=True,
        )

        gamma_slider = mo.ui.slider(
            start=0,
            stop=100,
            step=1,
            value=50,  # 50 = 1.0 (no change)
            label="Gamma",
            show_value=True,
        )

        # Build the layout based on whether we have time slider
        if time_slider is not None and len(file_metadata) > 1:
            # Get current selection
            _current_idx = time_slider.value
            _current_meta = file_metadata[_current_idx]

            # Build adjustment controls
            _adjustment_controls = [
                mo.md("**Fine-tune image appearance**"),
                contrast_slider,
                brightness_slider,
                gamma_slider,
            ]

            # Add band recipe selector if available
            if band_recipe_selector is not None:
                _adjustment_controls.insert(1, band_recipe_selector)

            # Create two-column layout: time slider on left, adjustments on right
            slider_display = mo.vstack(
                [
                    mo.md(
                        """
                        ---
                        ## 📅 Time Series & Image Adjustments
                        """
                    ),
                    mo.hstack(
                        [
                            # Left column: Time slider
                            mo.vstack(
                                [
                                    mo.md(f"**Navigate through {len(file_metadata)} images**"),
                                    time_slider,
                                    mo.md(
                                        f"""
                                        **Image {_current_idx + 1} of {len(file_metadata)}**
                                        Date: {_current_meta['date_str']}
                                        File: `{_current_meta['filename'][:40]}...`
                                        """
                                    ),
                                ],
                                align="start",
                            ),
                            # Right column: Adjustment sliders and band recipe
                            mo.vstack(_adjustment_controls, align="start"),
                        ],
                        justify="start",
                    ),
                ]
            )
        elif file_metadata and len(file_metadata) == 1:
            # Build adjustment controls
            _adjustment_controls = [
                contrast_slider,
                brightness_slider,
                gamma_slider,
            ]

            # Add band recipe selector if available
            if band_recipe_selector is not None:
                _adjustment_controls.insert(0, band_recipe_selector)

            # Single image - just show adjustments compactly
            slider_display = mo.vstack(
                [
                    mo.md(
                        f"""
                        ---
                        ## 📅 Satellite Image & Adjustments

                        **Date**: {file_metadata[0]['date_str']} | **File**: `{file_metadata[0]['filename'][:50]}...`
                        """
                    ),
                    mo.hstack(_adjustment_controls, justify="start"),
                ]
            )

    slider_display
    return (
        brightness_slider,
        contrast_slider,
        gamma_slider,
    )


@app.cell
def _():
    """Apply image adjustments (contrast, brightness, gamma) to image data.

    This function applies non-destructive image adjustments in the correct order:
    1. Brightness adjustment (add/subtract)
    2. Contrast adjustment (scale around midpoint)
    3. Gamma correction (power transformation)

    Args:
        image_array: Input image array (H, W) or (H, W, C), values in [0, 1] range
        contrast: Contrast multiplier (0-100 slider → 0.5-3.0 actual)
        brightness: Brightness offset (0-100 slider → -0.5 to +0.5 actual)
        gamma: Gamma exponent (0-100 slider → 0.3-3.0 actual)

    Returns:
        Adjusted image array, clipped to [0, 1] range
    """
    import numpy as _np

    def apply_image_adjustments(image_array, contrast, brightness, gamma):
        """Apply contrast, brightness, and gamma adjustments to image."""
        # Convert slider values (0-100) to actual adjustment values
        # Contrast: 0→0.5x, 50→1.0x, 100→3.0x
        contrast_actual = 0.5 + (contrast / 100.0) * 2.5

        # Brightness: 0→-0.5, 50→0.0, 100→+0.5
        brightness_actual = (brightness - 50) / 100.0

        # Gamma: 0→0.3, 50→1.0, 100→3.0
        gamma_actual = 0.3 + (gamma / 100.0) * 2.7

        # Make a copy to avoid modifying cached data
        adjusted = image_array.copy()

        # Step 1: Apply brightness (additive)
        adjusted = adjusted + brightness_actual

        # Step 2: Apply contrast (multiplicative around midpoint)
        # Formula: (pixel - 0.5) * contrast + 0.5
        adjusted = (adjusted - 0.5) * contrast_actual + 0.5

        # Step 3: Apply gamma correction (power transformation)
        # Gamma < 1: brightens shadows, compresses highlights
        # Gamma > 1: darkens shadows, expands highlights
        # Ensure non-negative values before applying power
        adjusted = _np.clip(adjusted, 0, 1)
        adjusted = _np.power(adjusted, gamma_actual)

        # Final clipping to valid range
        adjusted = _np.clip(adjusted, 0, 1)

        return adjusted

    return (apply_image_adjustments,)


@app.cell
def _(
    apply_image_adjustments,
    band_recipe_selector,
    brightness_slider,
    cached_images,
    contrast_slider,
    crop_to_bbox,
    file_metadata,
    gamma_slider,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    mo,
    satellite_type,
    time_slider,
    traceback,
):
    """Visualize the selected satellite image using cached data with band recipe support.

    This cell displays pre-processed images from cache, making slider
    interactions nearly instant (no disk I/O or processing needed).

    Visualization details:
    - S2: Supports multiple band recipes (True Color, False Color, Agriculture, NDVI, NDWI)
    - S1: VV polarization (grayscale) with adaptive contrast
    - Both: Already cropped to target bbox during pre-processing
    - Image adjustments applied in real-time based on slider values
    """
    viz_result = None

    # Only visualize if we have cached images
    if cached_images and len(cached_images) > 0:
        try:
            # Import visualization libraries
            import matplotlib.pyplot as plt
            import numpy as np

            # Get bbox for overlay
            _viz_bbox = [min_lon.value, min_lat.value, max_lon.value, max_lat.value]

            # Determine which image to display
            if time_slider is not None:
                # Use slider value
                _selected_idx = time_slider.value
            else:
                # No slider (single image)
                _selected_idx = 0

            # Get the cached image data (already processed!)
            _image_data = cached_images[_selected_idx]
            _metadata = file_metadata[_selected_idx]

            # Get adjustment values from sliders (default to 50 if sliders don't exist)
            _contrast = contrast_slider.value if contrast_slider is not None else 50
            _brightness = brightness_slider.value if brightness_slider is not None else 50
            _gamma = gamma_slider.value if gamma_slider is not None else 50

            # Get selected band recipe
            _selected_recipe = None
            if band_recipe_selector is not None:
                _selected_recipe = band_recipe_selector.value

            # Apply band recipe if needed
            _display_data = _image_data
            _needs_recipe_processing = False

            # Determine if we need to reprocess with recipe
            if _selected_recipe is not None:
                if satellite_type.value in ["S2", "S1+S2"]:
                    # S2: Reprocess if not True Color
                    _needs_recipe_processing = _selected_recipe != "True Color (RGB)"
                elif satellite_type.value == "S1":
                    # S1: Reprocess if not default SAR VV
                    _needs_recipe_processing = _selected_recipe != "SAR VV (Surface)"

            if _needs_recipe_processing:
                # Need to reprocess with the selected recipe
                from src.data.copernicus.band_recipes import apply_band_recipe

                # Only pass bbox if crop_to_bbox is enabled
                _recipe_bbox = _viz_bbox if crop_to_bbox.value else None

                _recipe_result = apply_band_recipe(
                    _metadata["path"], _selected_recipe, bbox=_recipe_bbox
                )
                if _recipe_result is not None:
                    _display_data = _recipe_result
                else:
                    # Fall back to cached data if recipe fails
                    print(f"Failed to apply recipe '{_selected_recipe}', using cached data")

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Display the image
            if _display_data is not None and _display_data.get("bounds_wgs84") is not None:
                _bounds = _display_data["bounds_wgs84"]
                _extent = (
                    _bounds[0],
                    _bounds[2],
                    _bounds[1],
                    _bounds[3],
                )  # (min_lon, max_lon, min_lat, max_lat)

                # PERFORMANCE FIX: If image is not cropped, crop it now for visualization
                # This prevents matplotlib from rendering millions of pixels that won't be visible
                _display_array = None
                _display_extent = _extent

                # Calculate zoom area with padding
                _padding = 0.02
                _zoom_bbox = [
                    _viz_bbox[0] - _padding,
                    _viz_bbox[1] - _padding,
                    _viz_bbox[2] + _padding,
                    _viz_bbox[3] + _padding,
                ]

                # Check if we need to crop for visualization (image is larger than zoom area)
                _needs_crop = (
                    _bounds[2] - _bounds[0] > _zoom_bbox[2] - _zoom_bbox[0]
                    or _bounds[3] - _bounds[1] > _zoom_bbox[3] - _zoom_bbox[1]
                )

                if _needs_crop:
                    # Crop the image to the zoom area for faster rendering
                    from src.data.copernicus.image_processing import crop_to_bbox as _crop_fn

                    # Check if we have RGB, index, or SAR data
                    if "rgb_array" in _display_data:
                        _display_array = _crop_fn(_display_data["rgb_array"], _bounds, _zoom_bbox)
                    elif "index_array" in _display_data:
                        _display_array = _crop_fn(
                            _display_data["index_array"], _bounds, _zoom_bbox
                        )
                    elif "sar_array" in _display_data:
                        _sar_data = _display_data["sar_array"]
                        if _sar_data.ndim == 3:
                            _sar_data = _sar_data[:, :, 0]  # Extract first polarization
                        _display_array = _crop_fn(_sar_data, _bounds, _zoom_bbox)
                    else:
                        _display_array = None

                    if _display_array is not None:
                        # Update extent to match cropped area
                        _display_extent = (
                            _zoom_bbox[0],
                            _zoom_bbox[2],
                            _zoom_bbox[1],
                            _zoom_bbox[3],
                        )
                    else:
                        # Cropping failed, fall back to full image
                        _needs_crop = False

                # Display the image array (already normalized and ready)
                # Check what type of data we have (RGB, index, or SAR)
                if "rgb_array" in _display_data:
                    # RGB image (H, W, 3) - for S2 or S1+S2 mode
                    if _needs_crop and _display_array is not None:
                        # Apply image adjustments
                        _adjusted_data = apply_image_adjustments(
                            _display_array, _contrast, _brightness, _gamma
                        )
                        ax.imshow(_adjusted_data, extent=_display_extent, aspect="auto")
                    else:
                        # Apply image adjustments
                        _adjusted_data = apply_image_adjustments(
                            _display_data["rgb_array"], _contrast, _brightness, _gamma
                        )
                        ax.imshow(_adjusted_data, extent=_extent, aspect="auto")
                elif "index_array" in _display_data:
                    # Spectral index or SAR data (H, W) - for NDVI, NDWI, SAR VV/VH, etc.
                    _index_array = (
                        _display_array
                        if _needs_crop and _display_array is not None
                        else _display_data["index_array"]
                    )

                    # Check if this is SAR data (needs different normalization)
                    _is_sar = _display_data.get("metadata", {}).get("type") == "sar"

                    if _is_sar:
                        # SAR data: Use percentile-based normalization (dB values vary)
                        _vmin, _vmax = np.percentile(_index_array, [2, 98])
                        _normalized = np.clip(
                            (_index_array - _vmin) / (_vmax - _vmin + 1e-10), 0, 1
                        )
                    else:
                        # Spectral indices: Use fixed value range
                        _vmin, _vmax = _display_data.get("value_range", (-1, 1))
                        _normalized = np.clip(
                            (_index_array - _vmin) / (_vmax - _vmin + 1e-10), 0, 1
                        )

                    # Apply image adjustments
                    _adjusted_data = apply_image_adjustments(
                        _normalized, _contrast, _brightness, _gamma
                    )

                    # Display with appropriate colormap
                    _cmap = _display_data.get("colormap", "RdYlGn")
                    _extent_to_use = _display_extent if _needs_crop else _extent
                    im = ax.imshow(
                        _adjusted_data,
                        extent=_extent_to_use,
                        aspect="auto",
                        cmap=_cmap,
                        vmin=0,
                        vmax=1,
                    )
                    # Add colorbar for index visualization
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    # Label colorbar appropriately
                    if _is_sar:
                        _pol_name = _display_data.get("metadata", {}).get("polarization", "SAR")
                        cbar.set_label(
                            f"{_pol_name} Backscatter (dB)",
                            rotation=270,
                            labelpad=15,
                        )
                    else:
                        cbar.set_label(
                            _display_data.get("metadata", {}).get("index", "Index Value"),
                            rotation=270,
                            labelpad=15,
                        )
                elif "sar_array" in _display_data:
                    # SAR grayscale image (H, W, 1) - squeeze to (H, W) - for S1 mode
                    if _needs_crop and _display_array is not None:
                        # Normalize to 0-1 range for adjustments
                        _vmin, _vmax = np.percentile(_display_array, [2, 98])
                        _normalized = np.clip(
                            (_display_array - _vmin) / (_vmax - _vmin + 1e-10), 0, 1
                        )

                        # Apply image adjustments
                        _adjusted_data = apply_image_adjustments(
                            _normalized, _contrast, _brightness, _gamma
                        )

                        ax.imshow(
                            _adjusted_data,
                            extent=_display_extent,
                            aspect="auto",
                            cmap="gray",
                            vmin=0,
                            vmax=1,
                        )
                    else:
                        _sar_data = _display_data["sar_array"]
                        if _sar_data.ndim == 3:
                            _sar_data = _sar_data[:, :, 0]  # Extract first polarization

                        # Normalize to 0-1 range for adjustments
                        _vmin, _vmax = np.percentile(_sar_data, [2, 98])
                        _normalized = np.clip((_sar_data - _vmin) / (_vmax - _vmin + 1e-10), 0, 1)

                        # Apply image adjustments
                        _adjusted_data = apply_image_adjustments(
                            _normalized, _contrast, _brightness, _gamma
                        )

                        ax.imshow(
                            _adjusted_data,
                            extent=_extent,
                            aspect="auto",
                            cmap="gray",
                            vmin=0,
                            vmax=1,
                        )

                # Add target area overlay
                _bbox_lons = [
                    _viz_bbox[0],
                    _viz_bbox[2],
                    _viz_bbox[2],
                    _viz_bbox[0],
                    _viz_bbox[0],
                ]
                _bbox_lats = [
                    _viz_bbox[1],
                    _viz_bbox[1],
                    _viz_bbox[3],
                    _viz_bbox[3],
                    _viz_bbox[1],
                ]
                ax.plot(
                    _bbox_lons,
                    _bbox_lats,
                    "red",
                    linewidth=3,
                    alpha=0.8,
                    label="Target Area",
                )

                # Zoom to target area with padding
                _padding = 0.02
                ax.set_xlim(_viz_bbox[0] - _padding, _viz_bbox[2] + _padding)
                ax.set_ylim(_viz_bbox[1] - _padding, _viz_bbox[3] + _padding)

                # Customize plot
                ax.set_xlabel("Longitude (°E)", fontsize=12)
                ax.set_ylabel("Latitude (°N)", fontsize=12)

                # Update title to show recipe name
                _recipe_name = _selected_recipe if _selected_recipe else satellite_type.value
                _title = (
                    f"{_recipe_name} - {_metadata['date_str']}\n{_metadata['filename'][:50]}..."
                )
                ax.set_title(_title, fontsize=11, fontweight="bold")

                ax.grid(True, alpha=0.3, color="white")
                ax.legend()
            else:
                # Visualization failed
                ax.text(
                    0.5,
                    0.5,
                    "⚠️ Image data unavailable\n\nProcessing may have failed",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title("Error")

            # Adjust layout
            plt.tight_layout()

            # Set result to figure for Marimo to display
            viz_result = fig

        except Exception as e:
            # Handle visualization errors gracefully
            _error_msg = str(e)
            viz_result = mo.md(
                f"""
                ## ❌ Visualization Error

                Failed to visualize the cached imagery.

                **Error**: {_error_msg}

                **Possible causes:**
                - Cached data format issue
                - Insufficient memory for images

                Check the console for detailed error information.
                """
            )
            # Print full traceback to console for debugging
            print("Visualization error details:")
            print(traceback.format_exc())

    # Return the figure for Marimo to display
    viz_result
    return


@app.cell
def _(
    cached_images,
    downloaded_files,
    file_metadata,
    mo,
    satellite_type,
    time_slider,
):
    """Display export buttons and handle GeoTIFF export."""
    export_button = None
    export_time_series_button = None
    export_display = None

    # Only show export buttons if we have images
    if cached_images and len(cached_images) > 0:
        export_button = mo.ui.run_button(label="📥 Export Current Image (RGB/SAR)")

        # Only show time series button for S1+S2 mode with multiple images
        if satellite_type.value == "S1+S2" and len(cached_images) > 1:
            export_time_series_button = mo.ui.run_button(
                label="🚀 Export Time Series for Galileo (All Bands)"
            )

            # Display both buttons
            export_display = mo.vstack(
                [
                    mo.md("---"),
                    mo.md("## 💾 Export Options"),
                    mo.md(
                        """
                        **Export Current Image**: Exports the currently displayed S2 image as RGB.
                        Good for visualization and sharing.

                        **Export Time Series for Galileo**: Exports ALL S2 and S1 images with the correct
                        band layout for Galileo's `_tif_to_array()`. Drops S2 B1/B9, adds zero-filled
                        placeholder bands for missing data (ERA5, SRTM, etc).
                        """
                    ),
                    mo.hstack([export_button, export_time_series_button], justify="start"),
                ]
            )
        else:
            # Single satellite type or single image - only show simple export
            _note = ""
            if satellite_type.value != "S1+S2":
                _note = """
                *Note: Time series export for Galileo requires S1+S2 mode.
                Select "S1+S2" in Satellite Type above to enable it.*
                """
            elif len(cached_images) == 1:
                _note = """
                *Note: Time series export requires multiple images.
                Adjust your search to download more products.*
                """

            export_display = mo.vstack(
                [
                    mo.md("---"),
                    mo.md("## 💾 Export"),
                    mo.md(
                        f"""
                        Export the current image as a GeoTIFF file.

                        {_note}
                        """
                    ),
                    export_button,
                ]
            )

    export_display
    return (export_button, export_time_series_button)


@app.cell
def _(
    Path,
    cached_images,
    downloaded_files,
    end_date,
    export_button,
    export_time_series_button,
    file_metadata,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    mo,
    satellite_type,
    start_date,
    time_slider,
    traceback,
):
    """Handle GeoTIFF export when buttons are clicked."""
    export_result = ""

    # Handle single image export
    if export_button is not None and export_button.value:
        try:
            # Determine which image to export
            if time_slider is not None:
                _export_idx = time_slider.value
            else:
                _export_idx = 0

            # Get the cached image data
            _export_data = cached_images[_export_idx]
            _export_metadata = file_metadata[_export_idx]

            # Create output filename based on original filename
            _original_name = _export_metadata["filename"]
            _output_name = _original_name.replace(".zip", ".tif").replace(".SAFE", ".tif")
            _output_path = Path("data/exports") / _output_name

            # Ensure exports directory exists
            _output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export using the client
            from src.data.copernicus import CopernicusClient as _CopernicusClient

            _client = _CopernicusClient()
            _result_path = _client.export_to_geotiff(
                _export_data, _output_path, satellite_type.value
            )

            export_result = f"""
            ✅ **Single Image Export Successful!**

            File saved to: `{_result_path}`

            **Image Details:**
            - Date: {_export_metadata['date_str']}
            - Satellite: {satellite_type.value}
            - Bands: 3 (RGB) for S2, or 1-2 (SAR) for S1
            - Bounds: {_export_data['bounds_wgs84']}

            You can now open this GeoTIFF in QGIS, ArcGIS, or any GIS software.
            """

        except Exception as e:
            export_result = f"""
            ❌ **Export Failed**

            Error: {str(e)}

            Make sure rasterio is installed: `pip install rasterio`

            Check console for detailed error information.
            """
            print("Export error details:")
            print(traceback.format_exc())

    # Handle time series export
    elif export_time_series_button is not None and export_time_series_button.value:
        try:
            # Get bbox from widgets
            _bbox = [min_lon.value, min_lat.value, max_lon.value, max_lat.value]

            # Create output filename
            _output_name = f"time_series_S1_S2_{start_date.value}_{end_date.value}.tif"
            _output_path = Path("data/exports") / _output_name

            # Ensure exports directory exists
            _output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if we have both S1 and S2 files
            if (
                not isinstance(downloaded_files, dict)
                or "s2" not in downloaded_files
                or "s1" not in downloaded_files
            ):
                export_result = """
                ❌ **Missing S1 or S2 Data**

                Time series export requires both S1 and S2 data.
                Please ensure you selected "S1+S2" mode and both types were downloaded successfully.
                """
            elif len(downloaded_files["s2"]) == 0 or len(downloaded_files["s1"]) == 0:
                export_result = f"""
                ❌ **Insufficient Data**

                - S2 files: {len(downloaded_files.get('s2', []))}
                - S1 files: {len(downloaded_files.get('s1', []))}

                Both S1 and S2 data are required for time series export.
                """
            else:
                # Show progress
                with mo.status.spinner(title="Exporting time series...") as _spinner:
                    _spinner.update(title="Creating multi-band time series TIF...")

                    # Import time series functions
                    from src.data.copernicus.time_series import create_time_series_tif

                    # Extract dates from file metadata
                    _dates = [
                        _meta["date"] for _meta in file_metadata if _meta["date"] is not None
                    ]

                    if len(_dates) < 2:
                        export_result = """
                        ⚠️ **Not Enough Images**

                        Time series export requires at least 2 images.
                        Please download more products by increasing "Max Products" in the search parameters.
                        """
                    else:
                        # Step 1: Create raw Copernicus time series TIF
                        _raw_path = create_time_series_tif(
                            s2_files=downloaded_files["s2"],
                            s1_files=downloaded_files["s1"],
                            dates=_dates,
                            bbox=_bbox,
                            output_path=_output_path,
                            normalize=False,  # Keep raw values for Galileo normalization
                        )

                        _spinner.update(title="Converting to Galileo-compatible format...")

                        # Step 2: Convert to Galileo format
                        from src.data.copernicus.galileo_adapter import copernicus_to_galileo_tif

                        _galileo_name = (
                            f"galileo_S1_S2_dates={start_date.value}_{end_date.value}.tif"
                        )
                        _galileo_path = Path("data/exports") / _galileo_name

                        _result_path = copernicus_to_galileo_tif(
                            copernicus_tif_path=_raw_path,
                            output_path=_galileo_path,
                            dates=_dates,
                            fill_missing_with_zeros=True,
                        )

                        _galileo_bands = (18 * len(_dates)) + 16 + 1

                        # Success message
                        export_result = f"""
                        ✅ **Galileo-Compatible Export Successful!**

                        Galileo TIF: `{_result_path}`
                        Raw TIF: `{_raw_path}`

                        **Galileo format details:**
                        - Dates: {len(_dates)} timesteps
                        - Total bands: {_galileo_bands} (18×{len(_dates)} dynamic + 16 space + 1 static)
                        - S1 bands: VV, VH (2 per timestep)
                        - S2 bands: B2-B8, B8A, B11, B12 (10 per timestep, B1/B9 dropped)
                        - Time bands: 6 per timestep (zero-filled: ERA5, TerraClimate, VIIRS)
                        - Space bands: 16 (zero-filled: SRTM, DW, WC)
                        - Static bands: 1 (zero-filled: LandScan)

                        **Compatibility:**
                        ✅ Band ordering matches Galileo's `_tif_to_array()`
                        ✅ S2 B1/B9 removed (Galileo doesn't use them)
                        ✅ Structure: `(18×t) + 16 + 1` bands
                        ⚠️ Time/space/static bands are zero-filled (missing source data)
                        """

        except Exception as e:
            export_result = f"""
            ❌ **Time Series Export Failed**

            Error: {str(e)}

            Check console for detailed error information.
            """
            print("Time series export error details:")
            print(traceback.format_exc())

    mo.md(export_result) if export_result else None
    return


@app.cell
def _(mo):
    """Section header for Galileo embeddings."""
    mo.md(
        """
    ---
    ## 🧠 Galileo Embedding Generation

    This section loads a Galileo-compatible TIF from `data/exports/`,
    runs it through the Galileo encoder to produce per-pixel embeddings,
    visualises them (PCA + K-means), and lets you export the result as a
    new GeoTIFF.
    """
    )
    return


@app.cell
def _(mo):
    """Scan data/exports/ and data/tifs/ for Galileo-compatible TIFs and let the user pick one or type a custom path."""
    from pathlib import Path as _Path

    _exports_dir = _Path("data/exports")
    _tifs_dir = _Path("data/tifs")

    # Collect all candidate TIFs
    _found_tifs = {}
    # Galileo exports (galileo_*.tif)
    if _exports_dir.exists():
        for _p in sorted(_exports_dir.glob("galileo_*.tif")):
            _found_tifs[f"[exports] {_p.name}"] = str(_p)
    # Training TIFs in data/tifs/
    if _tifs_dir.exists():
        for _p in sorted(_tifs_dir.glob("*.tif")):
            _found_tifs[f"[tifs] {_p.name}"] = str(_p)

    galileo_tif_selector = None
    galileo_tif_custom_path = mo.ui.text(
        label="Or enter a custom TIF path",
        placeholder="e.g. data/exports/my_file.tif",
    )
    generate_embeddings_button = mo.ui.run_button(label="🧠 Generate Embeddings")

    _ui_elements = []

    if _found_tifs:
        galileo_tif_selector = mo.ui.dropdown(
            options=_found_tifs,
            label="Select a TIF",
        )
        _ui_elements.append(
            mo.md(f"Found **{len(_found_tifs)}** TIF(s) in `data/exports/` and `data/tifs/`.")
        )
        _ui_elements.append(galileo_tif_selector)
    else:
        _ui_elements.append(
            mo.callout(
                mo.md(
                    "No TIFs found in `data/exports/` or `data/tifs/`. "
                    "You can still enter a path manually below, or use the Copernicus workflow above to create one."
                ),
                kind="warn",
            )
        )

    _ui_elements.append(galileo_tif_custom_path)
    _ui_elements.append(generate_embeddings_button)

    mo.output.replace(mo.vstack(_ui_elements))
    return galileo_tif_selector, galileo_tif_custom_path, generate_embeddings_button


@app.cell
def _(galileo_tif_selector, galileo_tif_custom_path, generate_embeddings_button, mo):
    """Generate Galileo embeddings from the selected TIF."""
    import traceback as _traceback
    from pathlib import Path as _Path

    embedding_result = None
    embeddings_arr = None
    embeddings_flat_arr = None
    embedding_labels = None
    embeddings_pca = None
    selected_galileo_tif_path = None

    if generate_embeddings_button is not None and generate_embeddings_button.value:
        # Resolve which TIF to use: custom path takes priority over dropdown
        _custom = (
            galileo_tif_custom_path.value.strip() if galileo_tif_custom_path is not None else ""
        )
        _dropdown = galileo_tif_selector.value if galileo_tif_selector is not None else None

        if _custom:
            selected_galileo_tif_path = _Path(_custom)
        elif _dropdown:
            selected_galileo_tif_path = _Path(_dropdown)

        if selected_galileo_tif_path is None or not selected_galileo_tif_path.exists():
            embedding_result = f"""
            ❌ **TIF not found**

            Path: `{selected_galileo_tif_path}`

            Select a TIF from the dropdown or enter a valid path.
            """
        else:
            with mo.status.spinner(
                title="Loading Galileo model and generating embeddings..."
            ) as _spinner:
                try:
                    import numpy as _np
                    import torch as _torch
                    from einops import rearrange as _rearrange
                    from sklearn.cluster import KMeans as _KMeans
                    from sklearn.decomposition import PCA as _PCA
                    from tqdm import tqdm as _tqdm

                    from src.data.config import NORMALIZATION_DICT_FILENAME as _NORM_FILENAME
                    from src.data.dataset import Dataset as _Dataset
                    from src.data.dataset import Normalizer as _Normalizer
                    from src.galileo import Encoder as _Encoder
                    from src.masking import MaskedOutput as _MaskedOutput
                    from src.utils import config_dir as _config_dir

                    _DATA_FOLDER = _Path("data")

                    # --- Load & normalise the TIF ---
                    _spinner.update(title="Loading and normalising TIF...")
                    _normalizing_dict = _Dataset.load_normalization_values(
                        path=_config_dir / _NORM_FILENAME
                    )
                    _normalizer = _Normalizer(std=True, normalizing_dicts=_normalizing_dict)
                    _dataset_output = _Dataset._tif_to_array(selected_galileo_tif_path).normalize(
                        _normalizer
                    )

                    # --- Load model ---
                    _spinner.update(title="Loading Galileo nano encoder...")
                    _model = _Encoder.load_from_folder(_DATA_FOLDER / "models/nano")
                    _model.eval()

                    # --- Generate embeddings ---
                    _spinner.update(title="Generating embeddings (this may take a while)...")
                    _device = _torch.device("cpu")
                    _output_list = []
                    _batch_count = 0
                    for i in _tqdm(
                        _dataset_output.in_pixel_batches(batch_size=128, window_size=1)
                    ):
                        _batch_count += 1
                        _masked = _MaskedOutput.from_datasetoutput(i, device=_device)
                        with _torch.no_grad():
                            _model_out = _model(
                                _masked.space_time_x.float(),
                                _masked.space_x.float(),
                                _masked.time_x.float(),
                                _masked.static_x.float(),
                                _masked.space_time_mask,
                                _masked.space_mask,
                                _torch.ones_like(_masked.time_mask),
                                _torch.ones_like(_masked.static_mask),
                                _masked.months.long(),
                                patch_size=1,
                            )
                            _output_list.append(
                                _model.average_tokens(*_model_out[:-1]).cpu().numpy()
                            )

                    _all = _np.concatenate(_output_list, axis=0)
                    _h_b = _dataset_output.space_time_x.shape[0]
                    _w_b = _dataset_output.space_time_x.shape[1]
                    embeddings_arr = _rearrange(_all, "(h w) d -> h w d", h=_h_b, w=_w_b)
                    embeddings_flat_arr = _rearrange(embeddings_arr, "h w d -> (h w) d")

                    # --- K-means ---
                    _spinner.update(title="Clustering embeddings (K-means)...")
                    _kmeans = _KMeans(n_clusters=3)
                    _labels_flat = _kmeans.fit_predict(embeddings_flat_arr)
                    embedding_labels = _rearrange(
                        _labels_flat,
                        "(h w) -> h w",
                        h=embeddings_arr.shape[0],
                        w=embeddings_arr.shape[1],
                    )

                    # --- PCA ---
                    _spinner.update(title="Reducing dimensions (PCA)...")
                    _pca = _PCA(n_components=3)
                    _pca_flat = _pca.fit_transform(embeddings_flat_arr)
                    embeddings_pca = _rearrange(
                        _pca_flat,
                        "(h w) d -> h w d",
                        h=embeddings_arr.shape[0],
                        w=embeddings_arr.shape[1],
                    )

                    embedding_result = f"""
                    ✅ **Embeddings generated!**

                    - Shape: `{embeddings_arr.shape}` (H × W × D)
                    - Pixels: {embeddings_flat_arr.shape[0]:,}
                    - Embedding dim: {embeddings_flat_arr.shape[1]}
                    - PCA explained variance: {_pca.explained_variance_ratio_.sum():.1%}
                    """

                except Exception as e:
                    embedding_result = f"""
                    ❌ **Embedding generation failed**

                    {str(e)}

                    Make sure the Galileo model is available at `data/models/nano/`
                    and all dependencies are installed (torch, einops, sklearn).
                    """
                    print("Embedding error:")
                    print(_traceback.format_exc())

    mo.output.replace(mo.md(embedding_result)) if embedding_result else None
    return (
        embeddings_arr,
        embeddings_flat_arr,
        embedding_labels,
        embeddings_pca,
        selected_galileo_tif_path,
    )


@app.cell
def _(embedding_labels, embeddings_pca, mo):
    """Visualise embeddings: K-means clusters and PCA RGB."""
    if embeddings_pca is not None and embedding_labels is not None:
        import matplotlib.pyplot as _plt

        _fig, _axes = _plt.subplots(1, 2, figsize=(12, 5))

        # K-means
        _axes[0].imshow(embedding_labels, cmap="tab10")
        _axes[0].set_title("K-means Clustering (3 clusters)")

        # PCA as RGB
        _pca_norm = (embeddings_pca - embeddings_pca.min()) / (
            embeddings_pca.max() - embeddings_pca.min() + 1e-10
        )
        _axes[1].imshow(_pca_norm)
        _axes[1].set_title("PCA-reduced Embeddings (RGB)")

        _plt.tight_layout()
        mo.output.replace(mo.vstack([_fig]))
    return


@app.cell
def _(embeddings_arr, mo, selected_galileo_tif_path):
    """Show export button for embedding GeoTIFF."""
    export_embeddings_button = None

    if embeddings_arr is not None:
        export_embeddings_button = mo.ui.run_button(label="💾 Export Embeddings as GeoTIFF")
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        f"Embeddings ready: `{embeddings_arr.shape[0]}×{embeddings_arr.shape[1]}` pixels, "
                        f"`{embeddings_arr.shape[2]}` dimensions."
                    ),
                    export_embeddings_button,
                ]
            )
        )
    return (export_embeddings_button,)


@app.cell
def _(
    embeddings_arr,
    export_embeddings_button,
    mo,
    selected_galileo_tif_path,
):
    """Export embeddings to GeoTIFF when button is clicked."""
    if (
        export_embeddings_button is not None
        and export_embeddings_button.value
        and embeddings_arr is not None
    ):
        try:
            from pathlib import Path as _Path

            from src.data.copernicus.galileo_adapter import embeddings_to_geotiff

            _src_name = (
                selected_galileo_tif_path.stem if selected_galileo_tif_path else "embeddings"
            )
            _out_path = _Path("data/exports") / f"{_src_name}_embeddings.tif"

            _result = embeddings_to_geotiff(
                embeddings=embeddings_arr,
                output_path=_out_path,
                source_tif_path=selected_galileo_tif_path,
            )

            mo.output.replace(
                mo.md(
                    f"""
                    ✅ **Embedding GeoTIFF exported!**

                    Saved to: `{_result}`

                    - Bands: {embeddings_arr.shape[2]} (one per embedding dimension)
                    - Size: {embeddings_arr.shape[1]}×{embeddings_arr.shape[0]} pixels
                    - CRS and transform copied from source TIF
                    """
                )
            )
        except Exception as e:
            import traceback as _traceback

            mo.output.replace(
                mo.md(
                    f"""
                    ❌ **Export failed**

                    {str(e)}
                    """
                )
            )
            print("Embedding export error:")
            print(_traceback.format_exc())
    return


if __name__ == "__main__":
    app.run()
