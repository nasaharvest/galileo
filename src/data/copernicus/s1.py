"""Sentinel-1 data fetching logic.

This module handles the specific details of searching for and fetching Sentinel-1 SAR imagery.
Sentinel-1 provides Synthetic Aperture Radar (SAR) data that works day/night and through clouds,
making it complementary to optical imagery from Sentinel-2.

Key features:
- Searches Copernicus catalog for SAR products with specific polarizations and orbit directions
- Filters by product type (GRD, SLC, OCN) and radar polarization modes
- Creates metadata files for discovered products (actual download to be implemented later)
- Implements caching to avoid repeated API calls for the same requests

SAR Background:
- SAR uses microwave radiation to image the Earth's surface
- Different polarizations (VV, VH, HH, HV) reveal different surface properties
- Orbit direction affects the viewing angle and shadow patterns
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .utils import bbox_to_wkt, build_cache_key, sanitize_filename

# Use TYPE_CHECKING to avoid circular imports while still getting type hints
if TYPE_CHECKING:
    from .client import CopernicusClient


def fetch_s1_products(
    client: "CopernicusClient",
    bbox: List[float],
    start_date: str,
    end_date: str,
    product_type: str,
    polarization: str,
    orbit_direction: str,
) -> List[Path]:
    """Fetch Sentinel-1 products for given parameters.

    This is the main entry point for Sentinel-1 SAR data fetching. It handles the complete workflow:
    1. Check if results are already cached
    2. If not cached, search the Copernicus catalog for matching SAR products
    3. Create metadata files for found products (actual download to be implemented later)
    4. Cache the results for future requests

    Args:
        client: CopernicusClient instance providing authentication and caching infrastructure
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84 coordinate system
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        product_type: SAR product type:
                     - "GRD": Ground Range Detected (most common, preprocessed)
                     - "SLC": Single Look Complex (raw data, requires more processing)
                     - "OCN": Ocean products (specialized for ocean analysis)
        polarization: Radar polarization modes (e.g., "VV,VH"):
                     - "VV": Vertical transmit, Vertical receive
                     - "VH": Vertical transmit, Horizontal receive
                     - "HH": Horizontal transmit, Horizontal receive
                     - "HV": Horizontal transmit, Vertical receive
        orbit_direction: Satellite orbit direction:
                        - "ASCENDING": Moving from south to north
                        - "DESCENDING": Moving from north to south

    Returns:
        List of Path objects pointing to metadata files for discovered products.
        Each file contains product information including download URLs and metadata.

    Note:
        Currently creates metadata files instead of downloading full products.
        This allows the system to work while full download functionality is developed.
    """
    # Build a unique cache key based on all parameters that affect the result
    # SAR products have different parameters than optical imagery
    cache_key = build_cache_key(
        "s1",  # Prefix to identify this as Sentinel-1 data
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
        polarization=polarization,
        orbit_direction=orbit_direction,
    )

    # Cache file stores both the search results and file paths
    cache_file = client.cache_dir / f"{cache_key}.json"

    # Check if we already have cached results for this exact request
    if cache_file.exists():
        print(f"Loading S1 products from cache: {cache_file}")
        with open(cache_file) as f:
            cached_data: Dict[str, Any] = json.load(f)

        # Verify that all cached files still exist on disk
        # If files were deleted, we need to re-create them
        cached_paths: List[Path] = [Path(p) for p in cached_data["file_paths"]]
        if all(p.exists() for p in cached_paths):
            return cached_paths  # Cache hit - return existing results
        else:
            print("Some cached files missing, re-downloading...")
            # Fall through to re-fetch the data

    # Search the Copernicus catalog for SAR products matching our criteria
    products: List[Dict[str, Any]] = _search_s1_products(
        client, bbox, start_date, end_date, product_type, polarization, orbit_direction
    )

    # Handle case where no products were found
    if not products:
        print(f"No S1 products found for bbox={bbox}, dates={start_date} to {end_date}")
        return []

    print(f"Found {len(products)} S1 products")

    # Create metadata files for the found products
    # We limit to first 3 products for testing to avoid overwhelming the system
    downloaded_paths: List[Path] = []
    for i, product in enumerate(products[:3]):  # Limit to first 3 for testing
        # Create a metadata file instead of downloading the full product (multi-GB files)
        metadata_file: Optional[Path] = _create_product_metadata(client, product, i)
        if metadata_file:
            downloaded_paths.append(metadata_file)

    # Cache the results for future requests
    # Store both the original search parameters and the resulting file paths
    cache_data: Dict[str, Any] = {
        "parameters": {
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "product_type": product_type,
            "polarization": polarization,
            "orbit_direction": orbit_direction,
        },
        "products": products,  # Full product metadata from API
        "file_paths": [str(p) for p in downloaded_paths],  # Paths to created files
    }

    # Write cache data to disk
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"Created metadata for {len(downloaded_paths)} S1 products, cached to {cache_file}")
    return downloaded_paths


def _search_s1_products(
    client: "CopernicusClient",
    bbox: List[float],
    start_date: str,
    end_date: str,
    product_type: str,
    polarization: str,
    orbit_direction: str,
) -> List[Dict[str, Any]]:
    """Search for Sentinel-1 products using the Copernicus OData API.

    This function constructs and executes a search query against the Copernicus catalog
    specifically for Sentinel-1 SAR products. SAR products have different metadata
    and filtering requirements compared to optical imagery.

    Args:
        client: CopernicusClient for making authenticated API requests
        bbox: Bounding box coordinates
        start_date: Start date for temporal filtering
        end_date: End date for temporal filtering
        product_type: SAR product type (GRD, SLC, OCN)
        polarization: Radar polarization modes
        orbit_direction: Satellite orbit direction

    Returns:
        List of product dictionaries containing metadata for each found SAR product.
        Each dictionary includes product ID, name, dates, attributes, polarization info, etc.
    """
    # Convert bounding box to WKT (Well-Known Text) format required by the API
    wkt_geometry: str = bbox_to_wkt(bbox)

    # Build OData query filter with multiple conditions
    # All conditions must be true (AND logic) for a product to match
    filter_parts: List[str] = [
        # Filter by collection: only Sentinel-1 products
        "Collection/Name eq 'SENTINEL-1'",
        # Filter by date range: product acquisition date must be within our range
        f"ContentDate/Start ge {start_date}T00:00:00.000Z",  # Greater than or equal to start
        f"ContentDate/Start le {end_date}T23:59:59.999Z",  # Less than or equal to end
        # Filter by spatial intersection: product footprint must overlap our bounding box
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_geometry}')",
    ]

    # Add product type filter based on SAR processing level
    if product_type == "GRD":
        # Ground Range Detected: Most common, preprocessed and geocoded
        filter_parts.append("contains(Name,'GRD')")
    elif product_type == "SLC":
        # Single Look Complex: Raw SAR data in slant range geometry
        filter_parts.append("contains(Name,'SLC')")
    elif product_type == "OCN":
        # Ocean products: Specialized products for ocean wind and wave analysis
        filter_parts.append("contains(Name,'OCN')")

    # Add orbit direction filter
    # This affects the viewing geometry and shadow patterns
    filter_parts.append(f"contains(Name,'{orbit_direction}')")

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

    print(f"Searching S1 products with filter: {filter_query}")

    # Make the authenticated API request
    response = client._make_request(url, params=params)
    data: Dict[str, Any] = response.json()

    # Extract the list of products from the API response
    products: List[Dict[str, Any]] = data.get("value", [])

    # Apply polarization filtering
    # SAR products can have different polarization combinations
    filtered_products: List[Dict[str, Any]] = []
    requested_pols: Set[str] = set(pol.strip() for pol in polarization.split(","))

    for product in products:
        # Extract available polarizations from product metadata
        product_pols: Set[str] = _extract_polarization(product)

        # Include product if:
        # 1. No specific polarizations requested (empty set), OR
        # 2. Product has all requested polarizations
        if not requested_pols or requested_pols.issubset(product_pols):
            filtered_products.append(product)

    return filtered_products


def _extract_polarization(product: Dict[str, Any]) -> Set[str]:
    """Extract polarization information from SAR product metadata.

    SAR products can have different polarization combinations (VV, VH, HH, HV).
    This information is stored in the product attributes or can be inferred from the name.

    Args:
        product: Product dictionary from the API response

    Returns:
        Set of polarization strings (e.g., {"VV", "VH"})
    """
    # First, try to extract from product Attributes (most reliable)
    attributes: List[Dict[str, Any]] = product.get("Attributes", [])
    for attr in attributes:
        if attr.get("Name") == "polarisation":
            pol_value: str = attr.get("Value", "")
            # Parse polarization string - can be "VV VH" or "VV,VH" format
            pols_list: List[str] = pol_value.replace(",", " ").split()
            return set(pols_list)

    # Fallback: try to extract from product name
    # SAR product names often contain polarization information
    name: str = product.get("Name", "")
    pols: Set[str] = set()

    # Check for each possible polarization in the product name
    for pol in ["VV", "VH", "HH", "HV"]:
        if pol in name:
            pols.add(pol)

    return pols


def _create_product_metadata(
    client: "CopernicusClient",
    product: Dict[str, Any],
    index: int,
) -> Optional[Path]:
    """Create a metadata file for a Sentinel-1 product instead of downloading the full product.

    This function creates a JSON file containing all the important information about
    a Sentinel-1 SAR product. This serves as a placeholder until full download functionality
    is implemented, and provides all the information needed for future processing.

    Args:
        client: CopernicusClient for accessing cache directory
        product: Product dictionary from the API search results
        index: Product index (used as fallback for naming)

    Returns:
        Path to the created metadata file, or None if creation failed
    """
    # Extract product identifiers, with fallbacks for missing data
    product_id: str = product.get("Id", f"unknown_{index}")
    product_name: str = product.get("Name", f"S1_product_{index}")

    # Create a safe filename by sanitizing the product name
    # Add metadata suffix to make purpose clear
    safe_name: str = sanitize_filename(product_name)
    filename: str = f"{safe_name}_metadata.json"

    # Determine file path within the cache directory
    # Use s1/ subdirectory to organize by satellite type
    file_path: Path = client.cache_dir / "s1" / filename

    # Create the subdirectory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if metadata file already exists
    if file_path.exists():
        print(f"S1 metadata already cached: {filename}")
        return file_path

    print(f"Creating S1 metadata: {filename}")

    # Create comprehensive metadata dictionary
    # This includes all information needed for future processing
    metadata: Dict[str, Any] = {
        "product_id": product_id,  # Unique identifier for API requests
        "product_name": product_name,  # Human-readable product name
        "content_date": product.get("ContentDate", {}),  # Acquisition date/time
        "attributes": product.get("Attributes", []),  # All product attributes (polarization, etc.)
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
