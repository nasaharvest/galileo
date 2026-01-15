"""Sentinel-2 data fetching logic.

This module handles the specific details of searching for and fetching Sentinel-2 optical imagery.
Sentinel-2 provides high-resolution multispectral imagery useful for land monitoring, agriculture,
and environmental applications.

Key features:
- Searches Copernicus catalog using OData API queries
- Filters by cloud cover, product type, and spatial/temporal criteria
- Creates metadata files for discovered products (actual download to be implemented later)
- Implements caching to avoid repeated API calls for the same requests
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .utils import bbox_to_wkt, build_cache_key, sanitize_filename

# Use TYPE_CHECKING to avoid circular imports while still getting type hints
if TYPE_CHECKING:
    from .client import CopernicusClient


def fetch_s2_products(
    client: "CopernicusClient",
    bbox: List[float],
    start_date: str,
    end_date: str,
    resolution: int,
    max_cloud_cover: float,
    product_type: str,
    download_data: bool = True,
    interactive: bool = True,
    max_products: int = 3,
) -> List[Path]:
    """Fetch Sentinel-2 products for given parameters.

    This is the main entry point for Sentinel-2 data fetching. It handles the complete workflow:
    1. Check if results are already cached
    2. If not cached, search the Copernicus catalog for matching products
    3. Optionally prompt user for download confirmation
    4. Download actual satellite imagery or create metadata files
    5. Cache the results for future requests

    Args:
        client: CopernicusClient instance providing authentication and caching infrastructure
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84 coordinate system
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        resolution: Spatial resolution in meters (10, 20, or 60)
        max_cloud_cover: Maximum cloud cover percentage (0-100)
        product_type: Product type ("S2MSI1C" for Level-1C or "S2MSI2A" for Level-2A)
        download_data: If True, download actual satellite imagery. If False, only metadata.
        interactive: If True, prompt user for download confirmation when products are found.
        max_products: Maximum number of products to download/process
                     Default: 3 (prevents accidental huge downloads)
                     Set to None for unlimited (use with caution!)

                     WHY LIMIT:
                     - Each S2 product is 500MB-1GB
                     - 10 products = 5-10GB disk space
                     - Downloads can take hours

                     Example: For 1 year of data over small area,
                     you might get 70+ products (one every 5 days).
                     Default limit prevents overwhelming your system.

    Returns:
        List of Path objects pointing to downloaded imagery files or metadata files.
    """
    # Build a unique cache key based on all parameters that affect the result
    # This ensures that requests with identical parameters will hit the cache
    cache_key = build_cache_key(
        "s2",  # Prefix to identify this as Sentinel-2 data
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        max_cloud_cover=max_cloud_cover,
        product_type=product_type,
        download_data=download_data,  # Include download mode in cache key
    )

    # Cache file stores both the search results and file paths
    cache_file = client.cache_dir / f"{cache_key}.json"

    # Check if we already have cached results for this exact request
    if cache_file.exists():
        print(f"Loading S2 products from cache: {cache_file}")
        with open(cache_file) as f:
            cached_data: Dict[str, Any] = json.load(f)

        # Verify that all cached files still exist on disk
        cached_paths: List[Path] = [Path(p) for p in cached_data["file_paths"]]
        if all(p.exists() for p in cached_paths):
            return cached_paths  # Cache hit - return existing results
        else:
            print("Some cached files missing, re-downloading...")
            # Fall through to re-fetch the data

    # Also check for cache with different download_data setting
    # This allows us to reuse product searches but change download behavior
    alt_cache_key = build_cache_key(
        "s2",
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        max_cloud_cover=max_cloud_cover,
        product_type=product_type,
        download_data=not download_data,  # Check opposite setting
    )
    alt_cache_file = client.cache_dir / f"{alt_cache_key}.json"

    if alt_cache_file.exists():
        print(f"Found products in alternate cache: {alt_cache_file}")
        with open(alt_cache_file) as f:
            alt_cached_data: Dict[str, Any] = json.load(f)

        # Reuse the product list but process with current download_data setting
        products = alt_cached_data.get("products", [])
        if products:
            print(f"Reusing {len(products)} products from alternate cache")
        else:
            # Search the Copernicus catalog for products matching our criteria
            products = _search_s2_products(
                client, bbox, start_date, end_date, max_cloud_cover, product_type
            )
    else:
        # Search the Copernicus catalog for products matching our criteria
        products = _search_s2_products(
            client, bbox, start_date, end_date, max_cloud_cover, product_type
        )

    # Handle case where no products were found
    if not products:
        print(f"No S2 products found for bbox={bbox}, dates={start_date} to {end_date}")
        return []

    print(f"Found {len(products)} S2 products")

    # Apply max_products limit if specified
    # This prevents accidental huge downloads (each product is 500MB-1GB)
    products_to_process = products
    if max_products is not None and len(products) > max_products:
        print(f"⚠️  Limiting to first {max_products} products (found {len(products)} total)")
        print("   To download more, use max_products parameter:")
        print(f"   client.fetch_s2(..., max_products={len(products)})")
        products_to_process = products[:max_products]

    # Interactive user confirmation if requested
    if interactive and products_to_process:
        print("\n🛰️ DOWNLOAD CONFIRMATION")
        print("=" * 40)
        print(f"Found {len(products)} Sentinel-2 products:")

        for i, product in enumerate(products_to_process[:5], 1):  # Show first 5
            name = product.get("Name", "Unknown")
            size_mb = product.get("ContentLength", 0) / (1024 * 1024)
            print(f"  {i}. {name} ({size_mb:.1f} MB)")

        if len(products_to_process) > 5:
            print(f"  ... and {len(products_to_process) - 5} more products")

        total_size_gb = sum(p.get("ContentLength", 0) for p in products_to_process) / (1024**3)
        print(f"\nTotal size: {total_size_gb:.2f} GB")

        if download_data:
            print("Mode: Download actual satellite imagery")
            response = (
                input(f"\nDownload {len(products_to_process)} products? [Y/n]: ").strip().lower()
            )
            if response and response not in ["y", "yes"]:
                print("Download cancelled by user")
                return []
        else:
            print("Mode: Metadata only (no actual imagery download)")

    # Process products (download or create metadata)
    downloaded_paths: List[Path] = []

    if download_data:
        print("\n📥 DOWNLOADING SATELLITE IMAGERY")
        print("=" * 45)

        for i, product in enumerate(products_to_process, 1):
            print(f"\n🛰️ Downloading product {i}/{len(products_to_process)}")

            downloaded_file = _download_s2_product(client, product, resolution, i - 1)
            if downloaded_file:
                downloaded_paths.append(downloaded_file)
                print(f"✅ Downloaded: {downloaded_file.name}")
            else:
                print(f"❌ Failed to download product {i}")
    else:
        print("\n📋 CREATING METADATA FILES")
        print("=" * 35)

        # Create metadata files for the found products
        for i, product in enumerate(products_to_process):
            metadata_file: Optional[Path] = _create_product_metadata(
                client, product, resolution, i
            )
            if metadata_file:
                downloaded_paths.append(metadata_file)

    # Cache the results for future requests
    cache_data: Dict[str, Any] = {
        "parameters": {
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "resolution": resolution,
            "max_cloud_cover": max_cloud_cover,
            "product_type": product_type,
            "download_data": download_data,
        },
        "products": products,  # Full product metadata from API
        "file_paths": [str(p) for p in downloaded_paths],  # Paths to created files
    }

    # Write cache data to disk
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

    action = "Downloaded" if download_data else "Created metadata for"
    print(f"\n✅ {action} {len(downloaded_paths)} S2 products, cached to {cache_file}")
    return downloaded_paths


def _search_s2_products(
    client: "CopernicusClient",
    bbox: List[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float,
    product_type: str,
) -> List[Dict[str, Any]]:
    """Search for Sentinel-2 products using the Copernicus OData API.

    This function constructs and executes a search query against the Copernicus catalog.
    The catalog uses OData (Open Data Protocol) for structured queries.

    Args:
        client: CopernicusClient for making authenticated API requests
        bbox: Bounding box coordinates
        start_date: Start date for temporal filtering
        end_date: End date for temporal filtering
        max_cloud_cover: Maximum acceptable cloud cover percentage
        product_type: Sentinel-2 product type to search for

    Returns:
        List of product dictionaries containing metadata for each found product.
        Each dictionary includes product ID, name, dates, attributes, etc.
    """
    # Convert bounding box to WKT (Well-Known Text) format required by the API
    wkt_geometry: str = bbox_to_wkt(bbox)

    # Build OData query filter with multiple conditions
    # All conditions must be true (AND logic) for a product to match
    filter_parts: List[str] = [
        # Filter by collection: only Sentinel-2 products
        "Collection/Name eq 'SENTINEL-2'",
        # Filter by date range: product acquisition date must be within our range
        f"ContentDate/Start ge {start_date}T00:00:00.000Z",  # Greater than or equal to start
        f"ContentDate/Start le {end_date}T23:59:59.999Z",  # Less than or equal to end
        # Filter by spatial intersection: product footprint must overlap our bounding box
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_geometry}')",
    ]

    # Add product type filter based on processing level
    if product_type == "S2MSI1C":
        # Level-1C: Top-of-atmosphere reflectance (not atmospherically corrected)
        filter_parts.append("contains(Name,'MSIL1C')")
    elif product_type == "S2MSI2A":
        # Level-2A: Bottom-of-atmosphere reflectance (atmospherically corrected)
        filter_parts.append("contains(Name,'MSIL2A')")

    # Combine all filter conditions with AND logic
    filter_query: str = " and ".join(filter_parts)

    # Set up query parameters for the OData API
    params: Dict[str, Any] = {
        "$filter": filter_query,  # The filter conditions we built above
        "$orderby": "ContentDate/Start asc",  # Sort by acquisition date (oldest first)
        "$top": 100,  # Limit results to 100 products (reduced for testing)
    }

    # Construct the full API URL
    url: str = f"{client.BASE_URL}/Products"

    print(f"Searching S2 products with filter: {filter_query}")

    # Make the authenticated API request
    response = client._make_request(url, params=params)
    data: Dict[str, Any] = response.json()

    # Extract the list of products from the API response
    products: List[Dict[str, Any]] = data.get("value", [])

    # Apply cloud cover filtering
    # Note: Not all products have cloud cover metadata, so we filter what we can
    filtered_products: List[Dict[str, Any]] = []
    for product in products:
        # Try to extract cloud cover percentage from product attributes
        cloud_cover: Optional[float] = _extract_cloud_cover(product)

        # Include product if:
        # 1. No cloud cover info available (cloud_cover is None), OR
        # 2. Cloud cover is within acceptable range
        if cloud_cover is None or cloud_cover <= max_cloud_cover:
            filtered_products.append(product)

    return filtered_products


def _extract_cloud_cover(product: Dict[str, Any]) -> Optional[float]:
    """Extract cloud cover percentage from product metadata.

    Cloud cover information is stored in the product's Attributes array.
    Not all products have this information available.

    Args:
        product: Product dictionary from the API response

    Returns:
        Cloud cover percentage as float (0-100), or None if not available
    """
    # Look through the product's attributes for cloud cover information
    attributes: List[Dict[str, Any]] = product.get("Attributes", [])
    for attr in attributes:
        if attr.get("Name") == "cloudCover":
            try:
                # Convert the string value to float
                return float(attr.get("Value", 0))
            except (ValueError, TypeError):
                # If conversion fails, treat as missing data
                pass

    # Return None if cloud cover information is not available
    return None


def _create_product_metadata(
    client: "CopernicusClient",
    product: Dict[str, Any],
    resolution: int,
    index: int,
) -> Optional[Path]:
    """Create a metadata file for a Sentinel-2 product instead of downloading the full product.

    This function creates a JSON file containing all the important information about
    a Sentinel-2 product. This serves as a placeholder until full download functionality
    is implemented, and provides all the information needed for future processing.

    Args:
        client: CopernicusClient for accessing cache directory
        product: Product dictionary from the API search results
        resolution: Requested resolution (used in filename)
        index: Product index (used as fallback for naming)

    Returns:
        Path to the created metadata file, or None if creation failed
    """
    # Extract product identifiers, with fallbacks for missing data
    product_id: str = product.get("Id", f"unknown_{index}")
    product_name: str = product.get("Name", f"S2_product_{index}")

    # Create a safe filename by sanitizing the product name
    # Add resolution and metadata suffix to make purpose clear
    safe_name: str = sanitize_filename(product_name)
    filename: str = f"{safe_name}_R{resolution}m_metadata.json"

    # Determine file path within the cache directory
    # Use s2/ subdirectory to organize by satellite type
    file_path: Path = client.cache_dir / "s2" / filename

    # Create the subdirectory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if metadata file already exists
    if file_path.exists():
        print(f"S2 metadata already cached: {filename}")
        return file_path

    print(f"Creating S2 metadata: {filename}")

    # Create comprehensive metadata dictionary
    # This includes all information needed for future processing
    metadata: Dict[str, Any] = {
        "product_id": product_id,  # Unique identifier for API requests
        "product_name": product_name,  # Human-readable product name
        "resolution": resolution,  # Requested spatial resolution
        "content_date": product.get("ContentDate", {}),  # Acquisition date/time
        "attributes": product.get("Attributes", []),  # All product attributes (cloud cover, etc.)
        "footprint": product.get("Footprint", ""),  # Geographic footprint as WKT
        "download_url": f"{client.BASE_URL}/Products({product_id})/$value",  # Direct download URL
        "note": "This is metadata only. Actual product download not implemented yet.",
    }

    try:
        # Write metadata to JSON file with pretty formatting
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Created metadata: {filename}")
        return file_path

    except Exception as e:
        print(f"Failed to create metadata {filename}: {e}")
        return None


def _download_s2_product(
    client: "CopernicusClient",
    product: Dict[str, Any],
    resolution: int,
    index: int,
) -> Optional[Path]:
    """Download actual Sentinel-2 satellite imagery.

    This function downloads the complete satellite product from Copernicus Data Space Ecosystem.
    Products are typically 500MB-1GB in size and contain multiple spectral bands.

    Args:
        client: CopernicusClient for authentication and cache directory
        product: Product dictionary from the API search results
        resolution: Requested resolution (used in filename)
        index: Product index (used as fallback for naming)

    Returns:
        Path to the downloaded file, or None if download failed
    """
    from .download_utils import download_with_retry

    # Extract product identifiers
    product_id: str = product.get("Id", f"unknown_{index}")
    product_name: str = product.get("Name", f"S2_product_{index}")
    content_length: int = product.get("ContentLength", 0)

    # Create safe filename
    safe_name: str = sanitize_filename(product_name)
    filename: str = f"{safe_name}_R{resolution}m.zip"

    # Determine file path within cache directory
    file_path: Path = client.cache_dir / "s2" / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if file_path.exists() and file_path.stat().st_size >= content_length:
        print(f"✅ Already downloaded: {filename}")
        return file_path

    # Construct download URL - use the correct download endpoint
    download_url = (
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    )

    print(f"📥 Downloading: {product_name}")
    print(f"   Size: {content_length / (1024*1024):.1f} MB")
    print(f"   URL: {download_url}")

    # Use robust download with retry and token refresh
    success = download_with_retry(
        client=client,
        url=download_url,
        output_path=file_path,
        total_size=content_length,
        max_retries=3,
    )

    if success:
        return file_path
    else:
        # Clean up failed download
        if file_path.exists():
            file_path.unlink()
        return None
