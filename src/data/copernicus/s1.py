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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from .common import (
    apply_product_limit,
    check_cache,
    process_products,
    save_cache,
)
from .enums import S1AcquisitionMode, S1OrbitDirection, S1Polarization, S1ProductType
from .utils import bbox_to_wkt, build_cache_key, sanitize_filename

# Use TYPE_CHECKING to avoid circular imports while still getting type hints
if TYPE_CHECKING:
    from .client import CopernicusClient


def fetch_s1_products(
    client: "CopernicusClient",
    bbox: List[float],
    start_date: str,
    end_date: str,
    product_type: Union[str, S1ProductType],
    polarization: Union[str, S1Polarization],
    orbit_direction: Union[str, S1OrbitDirection],
    acquisition_mode: Union[str, S1AcquisitionMode] = "IW",
    download_data: bool = True,
    max_products: int = 3,
) -> List[Path]:
    """Fetch Sentinel-1 products for given parameters.

    This is the main entry point for Sentinel-1 SAR data fetching. It handles the complete workflow:
    1. Check if results are already cached
    2. If not cached, search the Copernicus catalog for matching SAR products
    3. Download actual SAR imagery or create metadata files
    4. Cache the results for future requests

    Args:
        client: CopernicusClient instance providing authentication and caching infrastructure
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84 coordinate system
              Example: [6.15, 49.11, 6.16, 49.12] for 800m x 800m area in Luxembourg
        start_date: Start date in YYYY-MM-DD format
                   Example: "2024-01-01"
        end_date: End date in YYYY-MM-DD format
                 Example: "2024-01-31"
        product_type: SAR product type (string or S1ProductType enum):
                     - "GRD" or S1ProductType.GRD: Ground Range Detected (most common)
                     - "SLC" or S1ProductType.SLC: Single Look Complex (raw data)
                     - "OCN" or S1ProductType.OCN: Ocean products
        polarization: Radar polarization modes (string or S1Polarization enum):
                     - "VV,VH" or S1Polarization.dual_pol_vv_vh(): Dual pol (most common)
                     - "VV" or S1Polarization.VV: Single vertical polarization
                     - "VH" or S1Polarization.VH: Cross polarization
        orbit_direction: Satellite orbit direction (string or S1OrbitDirection enum):
                        - "ASCENDING" or S1OrbitDirection.ASCENDING: South to north
                        - "DESCENDING" or S1OrbitDirection.DESCENDING: North to south
        acquisition_mode: SAR acquisition mode (string or S1AcquisitionMode enum, default: "IW")
                         - "IW" or S1AcquisitionMode.IW: Interferometric Wide (default)
                         - "EW" or S1AcquisitionMode.EW: Extra Wide
                         - "SM" or S1AcquisitionMode.SM: Strip Map
                         - "WV" or S1AcquisitionMode.WV: Wave Mode

                         WHAT IS ACQUISITION MODE:
                         Sentinel-1 SAR can operate in different imaging modes,
                         like a camera with different lenses. Each mode trades
                         off between coverage area and resolution.

                         AVAILABLE MODES:
                         - "IW" (Interferometric Wide Swath): DEFAULT
                           Coverage: 250km wide, Resolution: 10m
                           Use: General land monitoring (95% of cases)
                           Best for: Agriculture, forests, urban areas, most ML applications

                         - "EW" (Extra Wide Swath):
                           Coverage: 400km wide, Resolution: 40m
                           Use: Ocean monitoring, polar regions, wide area surveillance
                           Best for: Maritime surveillance, ice sheets, large-scale monitoring

                         - "SM" (Strip Map):
                           Coverage: 80km wide, Resolution: 5m
                           Use: Emergency response, detailed monitoring
                           Best for: Disasters, high-detail urban mapping, infrastructure

                         - "WV" (Wave Mode):
                           Coverage: 20km samples, Resolution: 5m
                           Use: Ocean wave studies (very specialized)
                           Best for: Ocean wave height/direction analysis

                         WHAT CHANGES WITH MODE:
                         ✅ Resolution (how detailed the image is)
                            IW=10m, EW=40m, SM=5m, WV=5m
                         ✅ Coverage area (how wide the swath is)
                            IW=250km, EW=400km, SM=80km, WV=20km samples
                         ✅ Image size (number of pixels)
                            Higher resolution = more pixels for same area

                         WHAT DOESN'T CHANGE:
                         ❌ Polarizations (always VV, VH or HH, HV)
                         ❌ Data format (always 2-channel array)
                         ❌ Visualization (same grayscale SAR display)
                         ❌ Processing code (same functions work for all modes)
                         ❌ Backscatter values (same dB range -30 to 0)

                         VISUAL COMPARISON:
                         Satellite flying →

                         EW Mode:  ████████████████████████████████  (400km, lower res)
                         IW Mode:  ████████████████████              (250km, good res) ← DEFAULT
                         SM Mode:  ██████████                        (80km, high res)
                         WV Mode:  ██  ██  ██  ██                    (samples only)

                         FOR GALILEO ML:
                         - Use IW mode (default) for consistency across your dataset
                         - Don't mix modes in the same training set (different resolutions)
                         - IW provides the best balance of coverage and resolution
                         - Most Sentinel-1 data available is IW mode (95%+ of acquisitions)

                         WHEN TO USE EACH MODE:
                         - IW: Default choice, works for 95% of use cases
                               Land monitoring, agriculture, forestry, urban areas
                         - EW: When you need very wide coverage and resolution isn't critical
                               Ocean monitoring, polar ice, maritime surveillance
                         - SM: When you need maximum detail in a smaller area
                               Emergency response, disaster mapping, detailed infrastructure
                         - WV: Only for specialized ocean wave analysis
                               Rarely used for general remote sensing

                         Example:
                         >>> # Default (IW) - works for 95% of cases
                         >>> s1_files = client.fetch_s1(bbox, start_date, end_date)
                         >>>
                         >>> # Ocean monitoring - use EW for wide coverage
                         >>> s1_files = client.fetch_s1(bbox, start_date, end_date,
                         ...                            acquisition_mode="EW")
                         >>>
                         >>> # Disaster response - use SM for high detail
                         >>> s1_files = client.fetch_s1(bbox, start_date, end_date,
                         ...                            acquisition_mode="SM")
                         >>>
                         >>> # Ocean wave analysis - use WV (specialized)
                         >>> s1_files = client.fetch_s1(bbox, start_date, end_date,
                         ...                            acquisition_mode="WV")
        download_data: If True, download actual SAR imagery (1-2GB per product)
                      If False, only create metadata files (few KB per product)
                      Default: True
        max_products: Maximum number of products to download/process
                     Default: 3 (prevents accidental huge downloads)
                     Set to None for unlimited (use with caution!)

                     WHY LIMIT:
                     - Each S1 product is 1-2GB
                     - 10 products = 10-20GB disk space
                     - Downloads can take hours

                     Example: For 1 year of data over small area,
                     you might get 70+ products (one every 5 days).
                     Default limit prevents overwhelming your system.

    Returns:
        List of Path objects pointing to downloaded ZIP files or metadata JSON files.
        Each ZIP contains SAR backscatter data in GeoTIFF format.

    Example:
        >>> client = CopernicusClient()
        >>> files = client.fetch_s1(
        ...     bbox=[6.15, 49.11, 6.16, 49.12],
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     download_data=True
        ... )
        >>> print(f"Downloaded {len(files)} SAR products")
    """
    # Convert enums to strings for internal use
    product_type_str = str(product_type.value if hasattr(product_type, "value") else product_type)
    polarization_str = str(polarization.value if hasattr(polarization, "value") else polarization)
    orbit_direction_str = str(
        orbit_direction.value if hasattr(orbit_direction, "value") else orbit_direction
    )
    acquisition_mode_str = str(
        acquisition_mode.value if hasattr(acquisition_mode, "value") else acquisition_mode
    )

    # Build a unique cache key based on all parameters that affect the result
    # SAR products have different parameters than optical imagery
    cache_key = build_cache_key(
        "s1",  # Prefix to identify this as Sentinel-1 data
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type_str,
        polarization=polarization_str,
        orbit_direction=orbit_direction_str,
        acquisition_mode=acquisition_mode_str,  # Include acquisition mode in cache key
        download_data=download_data,  # Include download mode in cache key
        max_products=max_products,  # Include max_products to avoid cache conflicts
    )

    # Cache file stores both the search results and file paths
    cache_file = client.cache_dir / f"{cache_key}.json"

    # Check if we already have cached results for this exact request
    cached_paths = check_cache(cache_file)
    if cached_paths is not None:
        return cached_paths

    # Search the Copernicus catalog for SAR products matching our criteria
    products: List[Dict[str, Any]] = _search_s1_products(
        client,
        bbox,
        start_date,
        end_date,
        product_type_str,
        polarization_str,
        orbit_direction_str,
        acquisition_mode_str,
    )

    # Handle case where no products were found
    if not products:
        print(f"No S1 products found for bbox={bbox}, dates={start_date} to {end_date}")
        return []

    print(f"Found {len(products)} S1 products")

    # Apply max_products limit if specified
    products = apply_product_limit(products, max_products, "S1")

    # Process products (download or create metadata)
    downloaded_paths = process_products(
        client=client,
        products=products,
        download_data=download_data,
        satellite="SENTINEL-1",
        download_func=_download_s1_product,
        metadata_func=_create_product_metadata,
    )

    # Cache the results for future requests
    save_cache(
        cache_file=cache_file,
        parameters={
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "product_type": product_type,
            "polarization": polarization,
            "orbit_direction": orbit_direction,
            "acquisition_mode": acquisition_mode,
            "download_data": download_data,
        },
        products=products,
        file_paths=downloaded_paths,
    )

    action = "Downloaded" if download_data else "Created metadata for"
    print(f"\n✅ {action} {len(downloaded_paths)} S1 products, cached to {cache_file}")
    return downloaded_paths


def _search_s1_products(
    client: "CopernicusClient",
    bbox: List[float],
    start_date: str,
    end_date: str,
    product_type: str,
    polarization: str,
    orbit_direction: str,
    acquisition_mode: str,
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
        acquisition_mode: SAR acquisition mode (IW, EW, SM, WV)

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
    # Note: Orbit direction is NOT in the product name, it's in the Attributes
    # We'll filter this after getting results, not in the OData query
    # filter_parts.append(f"contains(Name,'{orbit_direction}')")  # This doesn't work!

    # Add acquisition mode filter
    # S1 product names contain the mode: S1A_IW_GRDH_... or S1A_EW_GRDH_...
    # This filters products by their imaging mode (IW, EW, SM, WV)
    # Example product name: S1A_IW_GRDH_1SDV_20220101T123456_20220101T123521_041234_04E567_1234
    #                           ^^
    #                           acquisition mode appears here
    # Note: Some products have the mode without underscore separator
    if acquisition_mode:
        filter_parts.append(f"contains(Name,'{acquisition_mode}')")

    # Combine all filter conditions with AND logic
    filter_query: str = " and ".join(filter_parts)

    # Set up query parameters for the OData API
    params: Dict[str, Any] = {
        "$filter": filter_query,  # The filter conditions we built above
        "$orderby": "ContentDate/Start asc",  # Sort by acquisition date (oldest first)
        "$top": 100,  # Limit results to 100 products (reduced for testing)
        "$expand": "Attributes",  # CRITICAL: Expand Attributes to get polarization, orbit, etc.
    }

    # Construct the full API URL
    url: str = f"{client.BASE_URL}/Products"

    print(f"Searching S1 products with filter: {filter_query}")
    print(f"Query params: {params}")

    # Make the authenticated API request
    response = client._make_request(url, params=params)
    data: Dict[str, Any] = response.json()

    # Debug: print response
    print(f"API Response status: {response.status_code}")
    print(f"API Response URL: {response.url}")  # Check if $expand is in URL
    print(f"API Response keys: {list(data.keys())}")
    if "value" in data:
        print(f"Number of products in response: {len(data['value'])}")
        if len(data["value"]) > 0:
            first_product = data["value"][0]
            print(f"First product has Attributes: {'Attributes' in first_product}")
            if "Attributes" in first_product:
                print(f"Number of Attributes: {len(first_product['Attributes'])}")
    if "error" in data:
        print(f"API Error: {data['error']}")

    # Extract the list of products from the API response
    products: List[Dict[str, Any]] = data.get("value", [])

    # Apply polarization filtering
    # SAR products can have different polarization combinations
    filtered_products: List[Dict[str, Any]] = []
    requested_pols: Set[str] = set(pol.strip() for pol in polarization.split(","))

    for product in products:
        # Extract available polarizations from product metadata
        product_pols: Set[str] = _extract_polarization(product)

        # Check orbit direction (if specified)
        # Orbit direction is in Attributes, not in the product name
        if orbit_direction:
            product_orbit = _extract_orbit_direction(product)
            if product_orbit and product_orbit.upper() != orbit_direction.upper():
                continue  # Skip products with wrong orbit direction

        # Include product if:
        # 1. No specific polarizations requested (empty set), OR
        # 2. Product has all requested polarizations
        if not requested_pols or requested_pols.issubset(product_pols):
            filtered_products.append(product)

    return filtered_products


def _extract_orbit_direction(product: Dict[str, Any]) -> Optional[str]:
    """Extract orbit direction from SAR product metadata.

    Args:
        product: Product dictionary from the API response

    Returns:
        Orbit direction string ("ASCENDING" or "DESCENDING"), or None if not found
    """
    # Check Attributes for orbitDirection
    attributes: List[Dict[str, Any]] = product.get("Attributes", [])
    for attr in attributes:
        if attr.get("Name") == "orbitDirection":
            return attr.get("Value", "").upper()

    # Fallback: check if it's in the product name (some products have it)
    name = product.get("Name", "")
    if "ASCENDING" in name.upper():
        return "ASCENDING"
    if "DESCENDING" in name.upper():
        return "DESCENDING"

    return None


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
        attr_name = attr.get("Name", "")
        # Check for various polarization attribute names
        if attr_name in ["polarisation", "polarisationChannels", "polarizationChannels"]:
            pol_value: str = attr.get("Value", "")
            # Parse polarization string - can be "VV VH", "VV,VH", or "VV&VH" format
            pols_list: List[str] = pol_value.replace(",", " ").replace("&", " ").split()
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

    # Create a safe filename with product ID embedded for deduplication
    # Format: {product_id}__{safe_name}_metadata.json
    safe_name: str = sanitize_filename(product_name)
    filename: str = f"{product_id}__{safe_name}_metadata.json"

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


def _download_s1_product(
    client: "CopernicusClient",
    product: Dict[str, Any],
    index: int,
) -> Optional[Path]:
    """Download actual Sentinel-1 SAR satellite imagery.

    This function downloads the complete SAR product from Copernicus Data Space Ecosystem.
    Sentinel-1 products are typically 1-2GB in size and contain radar backscatter data.

    WHAT IS SENTINEL-1:
    Sentinel-1 is a radar (SAR - Synthetic Aperture Radar) satellite that:
    - Works day and night (doesn't need sunlight like optical satellites)
    - Penetrates clouds (radar waves pass through clouds and rain)
    - Measures surface roughness (smooth surfaces = dark, rough surfaces = bright)
    - Provides all-weather monitoring capability

    WHAT'S IN A SENTINEL-1 PRODUCT:
    - Radar backscatter images in different polarizations (VV, VH)
    - Calibration data for converting to physical units (sigma0, gamma0)
    - Metadata about acquisition geometry and processing
    - Typically 1-2GB per product (larger than Sentinel-2)

    HOW S1 DIFFERS FROM S2:
    - S1 = Radar (active sensor, sends and receives microwaves)
    - S2 = Optical (passive sensor, measures reflected sunlight)
    - S1 has polarizations (VV, VH) not spectral bands (B01-B12)
    - S1 file structure uses .tiff files, not .jp2 files
    - S1 products are larger (1-2GB vs 500MB-1GB for S2)

    Args:
        client: CopernicusClient for authentication and cache directory
        product: Product dictionary from the API search results containing:
                - Id: Unique product identifier for download
                - Name: Product name (e.g., S1A_IW_GRDH_1SDV_20220101T123456...)
                - ContentLength: File size in bytes
        index: Product index (used as fallback for naming if product info missing)

    Returns:
        Path to the downloaded ZIP file, or None if download failed

    Example:
        >>> product = {"Id": "abc123", "Name": "S1A_IW_GRDH_...", "ContentLength": 1500000000}
        >>> path = _download_s1_product(client, product, 0)
        >>> print(path)  # data/cache/copernicus/s1/S1A_IW_GRDH_....zip
    """
    # Extract product identifiers from API response
    # These uniquely identify the SAR product we want to download
    product_id: str = product.get("Id", f"unknown_{index}")
    product_name: str = product.get("Name", f"S1_product_{index}")
    content_length: int = product.get("ContentLength", 0)  # File size in bytes

    # Create safe filename with product ID embedded for deduplication
    # Format: {product_id}__{safe_name}.zip
    safe_name: str = sanitize_filename(product_name)
    filename: str = f"{product_id}__{safe_name}.zip"

    # Determine file path within cache directory
    # Organize by satellite type: s1/ for Sentinel-1, s2/ for Sentinel-2
    file_path: Path = client.cache_dir / "s1" / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Create s1/ directory if needed

    # Check if file already exists and has content
    # This avoids re-downloading multi-GB files unnecessarily
    if file_path.exists() and file_path.stat().st_size >= content_length:
        print(f"✅ Already downloaded: {filename}")
        return file_path

    # Construct download URL using the Copernicus download endpoint
    # The /$value suffix tells the API to return the actual file content
    # instead of just metadata about the product
    download_url = (
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    )

    print(f"📥 Downloading: {product_name}")
    print(f"   Size: {content_length / (1024*1024):.1f} MB")
    print("   Type: Sentinel-1 SAR (Synthetic Aperture Radar)")
    print(f"   URL: {download_url}")

    # Use client's download method with retry and token refresh
    success = client.download_product(
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
