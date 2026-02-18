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
    1. **Configure credentials** - Save your Copernicus API credentials securely
    2. **Search for data** - Find Sentinel-1 (SAR) or Sentinel-2 (optical) imagery
    3. **Download & visualize** - Automatically download and display satellite images

    ## Getting Started

    **Get free Copernicus credentials:**
    1. Visit: https://dataspace.copernicus.eu/
    2. Click "Register" (no credit card required)
    3. After registration, go to your account settings at:
       **https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings**
    4. Copy your Client ID and Client Secret below

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
    COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET entries.
    """
    env_path = Path(".env")
    env_exists = env_path.exists()

    # Initialize credential flags
    has_client_id = False
    has_client_secret = False

    # If .env exists, check if it has valid credentials
    if env_exists:
        with open(env_path, "r") as _f:
            content = _f.read()
            # Check for CLIENT_ID (must have non-empty value)
            has_client_id = (
                "COPERNICUS_CLIENT_ID=" in content
                and len(content.split("COPERNICUS_CLIENT_ID=")[1].split("\n")[0].strip()) > 0
            )
            # Check for CLIENT_SECRET (must have non-empty value)
            has_client_secret = (
                "COPERNICUS_CLIENT_SECRET=" in content
                and len(content.split("COPERNICUS_CLIENT_SECRET=")[1].split("\n")[0].strip()) > 0
            )

    # Credentials are configured only if both are present and non-empty
    credentials_configured = has_client_id and has_client_secret

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

            Please enter your Copernicus API credentials below.
            These will be saved securely in a `.env` file.

            **Don't have credentials yet?**
            1. Register for free at: https://dataspace.copernicus.eu/
            2. Get your credentials at: https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
            """
        )

    # Create input widgets (password type hides the values for security)
    client_id_input = mo.ui.text(
        label="Client ID",
        kind="password",
        placeholder="Enter your Client ID from Copernicus",
    )
    client_secret_input = mo.ui.text(
        label="Client Secret",
        kind="password",
        placeholder="Enter your Client Secret from Copernicus",
    )
    save_button = mo.ui.run_button(label="💾 Save Credentials")

    # Display the form vertically stacked
    mo.vstack([status_msg, client_id_input, client_secret_input, save_button])
    return client_id_input, client_secret_input, save_button


@app.cell
def _(
    client_id_input,
    client_secret_input,
    env_path,
    mo,
    os,
    save_button,
    traceback,
):
    """Save credentials to .env file when Save button is clicked.

    This cell:
    1. Validates that both fields are filled
    2. Preserves any existing .env variables (doesn't overwrite other settings)
    3. Writes COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET
    4. Sets credentials in current environment for immediate use
    """
    save_result = ""

    # Only process if save button was clicked
    if save_button.value:
        # Get values from input fields
        client_id = client_id_input.value
        client_secret = client_secret_input.value

        # Validate that both fields have values
        if not client_id or not client_secret:
            save_result = """
            ❌ **Error: Both fields are required**

            Please enter both your Client ID and Client Secret.
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
                                if not line.startswith("COPERNICUS_CLIENT_ID=")
                                and not line.startswith("COPERNICUS_CLIENT_SECRET=")
                            ]
                        )

                # Write the updated .env file
                with open(env_path, "w") as _f:
                    _f.write(existing_content)
                    # Ensure newline before adding credentials
                    if existing_content and not existing_content.endswith("\n"):
                        _f.write("\n")
                    _f.write(f"COPERNICUS_CLIENT_ID={client_id}\n")
                    _f.write(f"COPERNICUS_CLIENT_SECRET={client_secret}\n")

                # Also set in current environment for immediate use
                # (so you don't need to restart the app)
                os.environ["COPERNICUS_CLIENT_ID"] = client_id
                os.environ["COPERNICUS_CLIENT_SECRET"] = client_secret

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
    - Location: Small area in Luxembourg (6.15-6.16°E, 49.11-49.12°N)
    - Date range: Last 30 days
    - Satellite: Sentinel-2 (optical)
    - Max products: 2 (to keep download size manageable)
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
        value=6.15,
        label="Min Longitude (°E)",
    )
    min_lat = mo.ui.number(
        start=-90,
        stop=90,
        step=0.01,
        value=49.11,
        label="Min Latitude (°N)",
    )
    max_lon = mo.ui.number(
        start=-180,
        stop=180,
        step=0.01,
        value=6.16,
        label="Max Longitude (°E)",
    )
    max_lat = mo.ui.number(
        start=-90,
        stop=90,
        step=0.01,
        value=49.12,
        label="Max Latitude (°N)",
    )

    # Satellite type selection
    # S2 = Sentinel-2 (optical/camera-like imagery)
    # S1 = Sentinel-1 (radar/SAR imagery, works through clouds)
    satellite_type = mo.ui.dropdown(
        options=["S2", "S1"],
        value="S2",
        label="Satellite Type (S2=Optical, S1=Radar)",
    )

    # Date range inputs
    start_date = mo.ui.date(value=default_start_date, label="Start Date")
    end_date = mo.ui.date(value=default_end_date, label="End Date")

    # Max products to download (limited to prevent excessive downloads)
    max_products = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=2,
        label="Max Products (1-10)",
    )

    # Search button triggers the search and download
    search_button = mo.ui.run_button(label="🔍 Search & Download")

    return (
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
                """
            ),
            mo.hstack([satellite_type, start_date, end_date, max_products]),
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
                    else:
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

                    # Check if we got any files
                    if downloaded_files and len(downloaded_files) > 0:
                        _spinner.update(title="Download complete!")
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
def _(datetime, downloaded_files, mo):
    """Create time slider for navigating through downloaded images.

    This cell creates a slider widget that allows users to navigate
    through multiple satellite images ordered by acquisition time.
    """
    time_slider = None
    file_metadata = []

    if downloaded_files and len(downloaded_files) > 0:
        # Extract dates from filenames and create metadata
        # Sentinel filenames contain acquisition date in format: YYYYMMDDTHHMMSS
        import re

        for _file_path in downloaded_files:
            _filename = _file_path.name
            # Try to extract date from filename
            # S2 format: S2A_MSIL2A_YYYYMMDDTHHMMSS_...
            # S1 format: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_...
            _date_match = re.search(r"(\d{8}T\d{6})", _filename)
            if _date_match:
                _date_str = _date_match.group(1)
                # Parse to readable format
                _date_obj = datetime.strptime(_date_str, "%Y%m%dT%H%M%S")
                file_metadata.append(
                    {
                        "path": _file_path,
                        "date": _date_obj,
                        "date_str": _date_obj.strftime("%Y-%m-%d %H:%M"),
                        "filename": _filename,
                    }
                )
            else:
                # Fallback if date not found
                file_metadata.append(
                    {
                        "path": _file_path,
                        "date": None,
                        "date_str": "Unknown date",
                        "filename": _filename,
                    }
                )

        # Sort by date (oldest first)
        file_metadata.sort(key=lambda x: x["date"] if x["date"] else datetime.min)

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

    return file_metadata, time_slider


@app.cell
def _(file_metadata, mo, time_slider):
    """Display time slider controls and current image info."""
    slider_display = None

    if time_slider is not None and len(file_metadata) > 1:
        # Get current selection
        _current_idx = time_slider.value
        _current_meta = file_metadata[_current_idx]

        # Display slider with metadata
        slider_display = mo.vstack(
            [
                mo.md(
                    f"""
                    ---
                    ## 📅 Time Series Visualization

                    Navigate through {len(file_metadata)} images using the slider below.
                    """
                ),
                time_slider,
                mo.md(
                    f"""
                    **Image {_current_idx + 1} of {len(file_metadata)}**
                    - **Date**: {_current_meta['date_str']}
                    - **File**: `{_current_meta['filename']}`
                    """
                ),
            ]
        )
    elif file_metadata and len(file_metadata) == 1:
        # Single image - no slider needed
        slider_display = mo.md(
            f"""
            ---
            ## 📅 Satellite Image

            **Date**: {file_metadata[0]['date_str']}
            **File**: `{file_metadata[0]['filename']}`
            """
        )

    slider_display
    return


@app.cell
def _(
    file_metadata,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    mo,
    satellite_type,
    time_slider,
    traceback,
):
    """Visualize the selected satellite image based on slider position.

    This cell:
    1. Gets the currently selected image from the slider
    2. Imports visualization functions
    3. Creates a matplotlib figure for the selected image
    4. Calls appropriate visualization function (S2=RGB, S1=SAR)
    5. Displays the resulting figure

    Visualization details:
    - S2: RGB composite (natural color) with target bbox overlay
    - S1: VV polarization (grayscale) with adaptive contrast
    - Both: Cropped to target bbox for focused view
    """
    viz_result = None

    # Only visualize if we have files
    if file_metadata and len(file_metadata) > 0:
        try:
            # Import visualization libraries
            import matplotlib.pyplot as plt

            from src.data.copernicus import display_sar_image, display_satellite_image

            # Get bbox for cropping and overlay
            _viz_bbox = [min_lon.value, min_lat.value, max_lon.value, max_lat.value]

            # Determine which image to display
            if time_slider is not None:
                # Use slider value
                _selected_idx = time_slider.value
            else:
                # No slider (single image)
                _selected_idx = 0

            # Get the file to display
            _selected_file = file_metadata[_selected_idx]["path"]

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Render the image
            if satellite_type.value == "S2":
                # Sentinel-2: Display RGB composite
                result_ax = display_satellite_image(_selected_file, _viz_bbox, ax=ax)

                if result_ax is None:
                    # Visualization failed
                    ax.text(
                        0.5,
                        0.5,
                        "⚠️ Image extraction failed\n\nFile may be corrupt or incomplete",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )
                    ax.set_title("Error")
            else:
                # Sentinel-1: Display SAR image (VV polarization)
                result_ax = display_sar_image(_selected_file, _viz_bbox, ax=ax, polarization="VV")

                if result_ax is None:
                    # Visualization failed
                    ax.text(
                        0.5,
                        0.5,
                        "⚠️ SAR extraction failed\n\nFile may be corrupt or incomplete",
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

                Failed to visualize the downloaded imagery.

                **Error**: {_error_msg}

                **Possible causes:**
                - Corrupt or incomplete download
                - Missing required bands in the product
                - Insufficient memory for large images

                Check the console for detailed error information.
                """
            )
            # Print full traceback to console for debugging
            print("Visualization error details:")
            print(traceback.format_exc())

    # Return the figure for Marimo to display
    viz_result
    return


if __name__ == "__main__":
    app.run()
