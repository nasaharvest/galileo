import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    print("=" * 80)
    print("🚀 MARIMO APP STARTING - Importing libraries...")
    print("=" * 80)

    import os
    import traceback
    from datetime import datetime, timedelta
    from pathlib import Path

    import marimo as mo

    print("✅ All imports successful")
    print()
    return Path, datetime, mo, os, timedelta, traceback


@app.cell
def _(mo):
    print("📝 Rendering header markdown...")
    mo.md(
        """
    # Copernicus Data Space Ecosystem Explorer

    This interactive GUI allows you to:
    1. Configure your Copernicus credentials
    2. Search for Sentinel-1 (SAR) or Sentinel-2 (optical) satellite data
    3. Download and visualize satellite imagery

    ## Getting Started

    **Get free Copernicus credentials:**
    - Visit: https://dataspace.copernicus.eu/
    - Click "Register" (no credit card required)
    - After registration, go to "User Settings" → "API Credentials"
    - Copy your Client ID and Client Secret
    """
    )
    return


@app.cell
def _(Path):
    print("🔍 CHECKING FOR EXISTING CREDENTIALS...")

    # Check if .env file exists
    env_path = Path(".env")
    env_exists = env_path.exists()
    print(f"  → .env file exists: {env_exists}")

    # Initialize credential flags
    has_client_id = False
    has_client_secret = False

    # If .env exists, check if it has valid credentials
    if env_exists:
        print("  → Reading .env file...")
        with open(env_path, "r") as _f:
            content = _f.read()
            # Check for CLIENT_ID
            has_client_id = (
                "COPERNICUS_CLIENT_ID=" in content
                and len(content.split("COPERNICUS_CLIENT_ID=")[1].split("\n")[0].strip()) > 0
            )
            # Check for CLIENT_SECRET
            has_client_secret = (
                "COPERNICUS_CLIENT_SECRET=" in content
                and len(content.split("COPERNICUS_CLIENT_SECRET=")[1].split("\n")[0].strip()) > 0
            )
            print(f"  → Has CLIENT_ID: {has_client_id}")
            print(f"  → Has CLIENT_SECRET: {has_client_secret}")
    else:
        print("  → .env file does not exist")

    # Determine if credentials are fully configured
    credentials_configured = has_client_id and has_client_secret
    print(f"  → ✅ Credentials configured: {credentials_configured}")
    print()
    return credentials_configured, env_path


@app.cell
def _(credentials_configured, mo):
    print("🎨 Rendering credential input form...")

    # Show different status message based on credential state
    if credentials_configured:
        print("  → Status: Credentials already configured")
        status_msg = mo.md(
            "## ✅ Credentials Configured\n\nYour Copernicus credentials are set in `.env` file."
        )
    else:
        print("  → Status: Credentials NOT configured")
        status_msg = mo.md(
            "## ⚠️ Credentials Not Configured\n\nPlease enter your Copernicus credentials below."
        )

    # Create input widgets (password type hides the values)
    client_id_input = mo.ui.text(
        label="Client ID", kind="password", placeholder="Enter your Client ID"
    )
    client_secret_input = mo.ui.text(
        label="Client Secret", kind="password", placeholder="Enter your Client Secret"
    )
    save_button = mo.ui.run_button(label="💾 Save Credentials")

    print("  → Form widgets created")
    print()

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
    print("🔄 CREDENTIAL SAVE CELL - Checking if save button was clicked...")
    print(f"  → save_button.value = {save_button.value}")

    # Initialize result message
    save_result = ""

    # Check if the save button was clicked (value > 0)
    if save_button.value:
        print("\n" + "=" * 80)
        print("💾 SAVE BUTTON CLICKED!")
        print("=" * 80)

        # Get values from input fields
        client_id = client_id_input.value
        client_secret = client_secret_input.value

        print(f"  → Client ID length: {len(client_id) if client_id else 0}")
        print(f"  → Client Secret length: {len(client_secret) if client_secret else 0}")

        # Validate that both fields have values
        if not client_id or not client_secret:
            save_result = "❌ Error: Both fields are required"
            print("  → ❌ Validation failed: One or both fields are empty")
        else:
            print("  → ✅ Validation passed: Both fields have values")
            try:
                print(f"  → 📝 Writing credentials to {env_path}...")

                # Read existing .env content (preserve other variables)
                existing_content = ""
                if env_path.exists():
                    print("  → Reading existing .env file...")
                    with open(env_path, "r") as _f:
                        lines = _f.readlines()
                        # Keep all lines except COPERNICUS credentials
                        existing_content = "".join(
                            [
                                line
                                for line in lines
                                if not line.startswith("COPERNICUS_CLIENT_ID=")
                                and not line.startswith("COPERNICUS_CLIENT_SECRET=")
                            ]
                        )
                    print("  → Existing content preserved")
                else:
                    print("  → No existing .env file")

                # Write the updated .env file
                print("  → Writing new .env file...")
                with open(env_path, "w") as _f:
                    _f.write(existing_content)
                    if existing_content and not existing_content.endswith("\n"):
                        _f.write("\n")
                    _f.write(f"COPERNICUS_CLIENT_ID={client_id}\n")
                    _f.write(f"COPERNICUS_CLIENT_SECRET={client_secret}\n")

                # Also set in current environment for immediate use
                print("  → Setting credentials in current environment...")
                os.environ["COPERNICUS_CLIENT_ID"] = client_id
                os.environ["COPERNICUS_CLIENT_SECRET"] = client_secret
                print("  → ✅ Credentials set in current environment")

                save_result = "✅ Credentials saved successfully! You can now search for data."
                print("  → ✅ Credentials saved to .env file")

            except Exception as e:
                save_result = f"❌ Error: {str(e)}"
                print("  → ❌ Error saving credentials:")
                print(traceback.format_exc())

        print("=" * 80)
        print()
    else:
        print("  → Save button not clicked (value is 0)")
        print()

    # Display result message if there is one
    mo.md(save_result) if save_result else None
    return


@app.cell
def _(mo):
    print("📝 Rendering search parameters header...")
    mo.md(
        """
    ---
    ## 🛰️ Search Parameters
    """
    )
    return


@app.cell
def _(datetime, mo, timedelta):
    print("🎨 Creating search parameter widgets...")

    # Calculate default date range (last 30 days)
    default_end_date = datetime.now().strftime("%Y-%m-%d")
    default_start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"  → Default date range: {default_start_date} to {default_end_date}")

    # Create coordinate input widgets (small area in Luxembourg by default)
    min_lon = mo.ui.number(start=-180, stop=180, step=0.01, value=6.15, label="Min Longitude")
    min_lat = mo.ui.number(start=-90, stop=90, step=0.01, value=49.11, label="Min Latitude")
    max_lon = mo.ui.number(start=-180, stop=180, step=0.01, value=6.16, label="Max Longitude")
    max_lat = mo.ui.number(start=-90, stop=90, step=0.01, value=49.12, label="Max Latitude")
    print("  → Coordinate widgets created")

    # Create satellite and date input widgets
    satellite_type = mo.ui.dropdown(options=["S2", "S1"], value="S2", label="Satellite Type")
    start_date = mo.ui.date(value=default_start_date, label="Start Date")
    end_date = mo.ui.date(value=default_end_date, label="End Date")
    max_products = mo.ui.number(start=1, stop=10, step=1, value=2, label="Max Products")
    print("  → Satellite and date widgets created")

    # Create the search button
    search_button = mo.ui.run_button(label="🔍 Search & Download")

    print("  → Search button created")
    print()

    # Return the widgets so other cells can use them
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
    print("🎨 Displaying search parameter widgets...")

    # Display all widgets in a nice layout
    mo.vstack(
        [
            mo.md("### Coordinates"),
            mo.hstack([min_lon, min_lat, max_lon, max_lat]),
            mo.md("### Satellite & Dates"),
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
    print("🔄 SEARCH CELL - Checking if search button was clicked...")
    print(f"  → search_button.value = {search_button.value}")

    # Initialize variables
    download_result = ""
    downloaded_files = []

    # Check if search button was clicked (value > 0)
    if search_button.value:
        print("\n" + "=" * 80)
        print("🔍 SEARCH BUTTON CLICKED - Starting search process")
        print("=" * 80)
        print(f"  → Credentials configured: {credentials_configured}")

        # First check: Do we have credentials?
        if not credentials_configured:
            download_result = "❌ Please configure credentials first!"
            print("  → ❌ Credentials not configured - aborting search")
            print("=" * 80)
            print()
        else:
            print("  → ✅ Credentials are configured, proceeding...")

            try:
                # Import the Copernicus client
                print("\n  → 📦 Importing CopernicusClient...")
                from src.data.copernicus import CopernicusClient

                print("  → ✅ Import successful")

                # Get search parameters
                _bbox = [min_lon.value, min_lat.value, max_lon.value, max_lat.value]
                print(f"\n  → 📍 BBox: {_bbox}")
                print(f"  → 🛰️  Satellite: {satellite_type.value}")
                print(f"  → 📅 Date range: {start_date.value} to {end_date.value}")
                print(f"  → 🔢 Max products: {max_products.value}")

                # Initialize the client
                print("\n  → 🔐 Initializing CopernicusClient...")
                client = CopernicusClient()
                print("  → ✅ Client initialized successfully")

                # Build initial result message
                download_result = f"🔍 Searching for {satellite_type.value} products...\n"
                download_result += f"📍 Area: {_bbox}\n"
                download_result += f"📅 {start_date.value} to {end_date.value}\n\n"

                # Call appropriate fetch method based on satellite type
                if satellite_type.value == "S2":
                    print("\n  → 🛰️  Calling fetch_s2()...")
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
                    print(
                        f"  → ✅ fetch_s2() returned {len(downloaded_files) if downloaded_files else 0} files"
                    )
                else:
                    print("\n  → 🛰️  Calling fetch_s1()...")
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
                    print(
                        f"  → ✅ fetch_s1() returned {len(downloaded_files) if downloaded_files else 0} files"
                    )

                # Check if we got any files
                if downloaded_files:
                    print(f"\n  → ✅ SUCCESS: Downloaded {len(downloaded_files)} products")
                    download_result += f"✅ Downloaded {len(downloaded_files)} products!\n\n"
                    download_result += "📁 Files:\n"
                    for _f in downloaded_files:
                        download_result += f"  • {_f}\n"
                        print(f"     - {_f}")
                else:
                    print("\n  → ⚠️  No products found for this search")
                    download_result += "⚠️ No products found."

            except Exception as e:
                # Handle any errors that occurred
                error_details = traceback.format_exc()
                download_result = f"❌ Error: {str(e)}\n\nSee console for details."
                print("\n  → ❌ ERROR occurred:")
                print(error_details)

        print("=" * 80)
        print()
    else:
        print("  → Search button not clicked (value is 0 or None)")
        print()

    # Display the result message
    mo.md(download_result) if download_result else None

    return (downloaded_files,)


@app.cell
def _(
    downloaded_files,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    mo,
    satellite_type,
    traceback,
):
    print("=" * 80)
    print("🔄 VISUALIZATION CELL - Checking if there are files to visualize...")
    print("=" * 80)
    print(f"  → Number of downloaded files: {len(downloaded_files) if downloaded_files else 0}")

    if downloaded_files:
        print("  → Downloaded files list:")
        for idx, _f in enumerate(downloaded_files):
            print(f"     [{idx}] {_f}")
            print(f"         Type: {type(_f)}")
            print(f"         Exists: {_f.exists() if hasattr(_f, 'exists') else 'N/A'}")

    # Initialize result
    viz_result = None

    # Only visualize if we have downloaded files
    if downloaded_files:
        print(f"\n  → 🎨 Starting visualization for {len(downloaded_files)} files...")

        try:
            # Import visualization libraries
            print("  → Importing matplotlib and visualization functions...")
            import matplotlib.pyplot as plt

            from src.data.copernicus import display_sar_image, display_satellite_image

            print("  → ✅ Imports successful")

            # Get bbox for cropping
            _viz_bbox = [min_lon.value, min_lat.value, max_lon.value, max_lat.value]
            num_files = min(len(downloaded_files), 2)  # Max 2 images
            print(f"  → Visualizing {num_files} files")
            print(f"  → Target bbox: {_viz_bbox}")
            print(f"  → Satellite type: {satellite_type.value}")

            # Create subplot grid
            print("\n  → Creating matplotlib figure...")
            fig, axes = plt.subplots(1, num_files, figsize=(12, 6))
            if num_files == 1:
                axes = [axes]  # Make it a list for consistency
            print(f"  → ✅ Figure created with {num_files} subplot(s)")
            print(f"  → Figure size: {fig.get_size_inches()}")
            print(f"  → Axes type: {type(axes)}, length: {len(axes)}")

            # Render each file
            for idx, file_path in enumerate(downloaded_files[:num_files]):
                print(f"\n  → {'='*60}")
                print(f"  → Rendering file {idx+1}/{num_files}")
                print(f"  → {'='*60}")
                print(f"  → File path: {file_path}")
                print(f"  → File type: {type(file_path)}")
                print(f"  → Axes[{idx}]: {axes[idx]}")

                if satellite_type.value == "S2":
                    print("    → Calling display_satellite_image (RGB)...")
                    print(f"    → Parameters: file={file_path}, bbox={_viz_bbox}, ax={axes[idx]}")

                    result_ax = display_satellite_image(file_path, _viz_bbox, ax=axes[idx])

                    print(f"    → display_satellite_image returned: {result_ax}")
                    print(f"    → Return type: {type(result_ax)}")

                    if result_ax is None:
                        print("    → ⚠️  WARNING: display_satellite_image returned None!")
                        print("    → This means RGB extraction likely failed")
                        # Add a text message to the plot
                        axes[idx].text(
                            0.5,
                            0.5,
                            "Image extraction failed",
                            ha="center",
                            va="center",
                            transform=axes[idx].transAxes,
                        )
                    else:
                        print(f"    → ✅ Image rendered successfully on axes {idx}")
                else:
                    print("    → Calling display_sar_image (VV polarization)...")
                    print(f"    → Parameters: file={file_path}, bbox={_viz_bbox}, ax={axes[idx]}")

                    result_ax = display_sar_image(
                        file_path, _viz_bbox, ax=axes[idx], polarization="VV"
                    )

                    print(f"    → display_sar_image returned: {result_ax}")
                    print(f"    → Return type: {type(result_ax)}")

                    if result_ax is None:
                        print("    → ⚠️  WARNING: display_sar_image returned None!")
                        print("    → This means SAR extraction likely failed")
                        axes[idx].text(
                            0.5,
                            0.5,
                            "SAR extraction failed",
                            ha="center",
                            va="center",
                            transform=axes[idx].transAxes,
                        )
                    else:
                        print(f"    → ✅ SAR image rendered successfully on axes {idx}")

            print("\n  → Applying tight_layout...")
            plt.tight_layout()

            print("  → Setting viz_result to figure...")
            viz_result = fig
            print(f"  → viz_result type: {type(viz_result)}")
            print(f"  → viz_result value: {viz_result}")
            print("  → ✅ Visualization complete")

        except Exception as e:
            # Handle visualization errors
            error_msg = f"Visualization error: {str(e)}"
            print("\n  → ❌ EXCEPTION CAUGHT:")
            print(f"  → Error message: {error_msg}")
            print("  → Full traceback:")
            print(traceback.format_exc())
            viz_result = mo.md(
                f"## ❌ Visualization Error\n\n```\n{error_msg}\n\n{traceback.format_exc()}\n```"
            )
    else:
        print("  → No files to visualize (downloaded_files is empty or None)")

    print("=" * 80)
    print()

    # Return the figure for Marimo to display
    viz_result
    return


if __name__ == "__main__":
    app.run()
