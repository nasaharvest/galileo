"""Main Copernicus Data Space Ecosystem client.

This module provides the main CopernicusClient class that handles:
- OAuth2 authentication with Copernicus Data Space Ecosystem
- Caching of downloaded products and metadata
- Coordination between Sentinel-1 and Sentinel-2 specific modules
- Error handling and retry logic for API requests
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

# Import the specific handlers for each satellite type
from .s1 import fetch_s1_products
from .s2 import fetch_s2_products
from .utils import ensure_cache_dir, validate_bbox, validate_date_range


class CopernicusClient:
    """Main client for fetching Sentinel-1 and Sentinel-2 data from Copernicus Data Space Ecosystem.

    This client handles:
    1. OAuth2 authentication with automatic token refresh
    2. Caching of search results and product metadata
    3. Input validation for bounding boxes and date ranges
    4. Coordination between S1 and S2 specific fetch operations

    The client uses the free Copernicus Data Space Ecosystem API, which replaced
    the old Copernicus Open Access Hub in 2023.
    """

    # API endpoints for Copernicus Data Space Ecosystem
    BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"  # Main catalog API
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"  # OAuth endpoint

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cache/copernicus",
        load_dotenv_file: bool = True,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        """Initialize the Copernicus client with authentication and caching setup.

        Args:
            cache_dir: Directory where downloaded files and metadata will be cached.
                      Defaults to "data/cache/copernicus" relative to current directory.
            load_dotenv_file: Whether to automatically load credentials from .env file.
                             Set to False if you're providing credentials directly.
            client_id: OAuth client ID. If None, will try to load from COPERNICUS_CLIENT_ID env var.
            client_secret: OAuth client secret. If None, will try to load from COPERNICUS_CLIENT_SECRET env var.

        Raises:
            ValueError: If credentials cannot be found in environment variables or parameters.
        """
        # Convert cache directory to Path object and ensure it exists
        self.cache_dir: Path = Path(cache_dir)
        ensure_cache_dir(self.cache_dir)

        # Load environment variables from .env file if requested
        # This looks for a .env file in the current directory
        if load_dotenv_file:
            load_dotenv()

        # Get credentials from parameters or environment variables
        # Parameters take precedence over environment variables
        client_id_from_env: Optional[str] = client_id or os.getenv("COPERNICUS_CLIENT_ID")
        client_secret_from_env: Optional[str] = client_secret or os.getenv(
            "COPERNICUS_CLIENT_SECRET"
        )

        # Validate that we have the required credentials
        if not client_id_from_env or not client_secret_from_env:
            raise ValueError(
                "Copernicus credentials not found. Set COPERNICUS_CLIENT_ID and "
                "COPERNICUS_CLIENT_SECRET environment variables or pass them directly."
            )

        self.client_id: str = client_id_from_env
        self.client_secret: str = client_secret_from_env

        # OAuth token management - these will be set when we first authenticate
        self._access_token: Optional[str] = None  # The actual Bearer token
        self._token_expires_at: float = 0  # Unix timestamp when token expires

        # Create a persistent HTTP session for connection pooling and cookie management
        self.session: requests.Session = requests.Session()

    def _get_access_token(self) -> str:
        """Get a valid OAuth access token, refreshing if necessary.

        This method implements token caching - it only requests a new token if:
        1. We don't have a token yet, OR
        2. The current token is about to expire (within 5 minutes)

        The Copernicus API uses OAuth2 "client credentials" flow, which means
        we exchange our client_id and client_secret for a temporary access token.

        Returns:
            A valid Bearer token string that can be used in API requests.

        Raises:
            requests.HTTPError: If the token request fails (e.g., invalid credentials).
        """
        # Check if we already have a valid token that hasn't expired
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        # Request a new token using OAuth2 client credentials flow
        # This is the standard way to authenticate machine-to-machine API access
        data = {
            "grant_type": "client_credentials",  # OAuth2 flow type
            "client_id": self.client_id,  # Your registered client ID
            "client_secret": self.client_secret,  # Your registered client secret
        }

        # Make the token request
        response = self.session.post(self.TOKEN_URL, data=data)
        response.raise_for_status()  # Raise exception if request failed

        # Parse the response to get token information
        token_data = response.json()
        self._access_token = token_data["access_token"]  # The actual token string

        # Calculate when this token expires and set expiry with 5 minute buffer
        # This prevents us from using a token that might expire during a request
        expires_in = token_data.get("expires_in", 3600)  # Default to 1 hour if not specified
        self._token_expires_at = time.time() + expires_in - 300  # Subtract 5 minutes (300 seconds)

        # At this point we're guaranteed to have a valid token
        assert self._access_token is not None
        return self._access_token

    def _make_request(
        self, url: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> requests.Response:
        """Make an authenticated HTTP request with automatic retry logic.

        This method handles:
        1. Adding the OAuth Bearer token to request headers
        2. Automatic token refresh if we get a 401 Unauthorized response
        3. Exponential backoff retry logic for transient failures

        Args:
            url: The full URL to make the request to
            params: Query parameters to include in the request
            **kwargs: Additional arguments passed to requests.get()

        Returns:
            The HTTP response object

        Raises:
            requests.HTTPError: If the request fails after all retries
            RuntimeError: If maximum retries are exceeded
        """
        # Prepare headers, ensuring we don't overwrite any existing headers
        headers: Dict[str, str] = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._get_access_token()}"

        # Retry logic with exponential backoff
        max_retries: int = 3
        for attempt in range(max_retries):
            try:
                # Make the actual HTTP request
                response: requests.Response = self.session.get(
                    url, params=params, headers=headers, **kwargs
                )

                # Handle token expiration - if we get 401, refresh token and retry
                if response.status_code == 401:  # Unauthorized - token likely expired
                    print("Token expired, refreshing...")
                    self._access_token = None  # Force token refresh
                    headers["Authorization"] = f"Bearer {self._get_access_token()}"
                    continue  # Retry the request with new token

                # Raise exception for other HTTP errors (4xx, 5xx)
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                # If this was our last attempt, give up
                if attempt == max_retries - 1:
                    raise

                # Calculate wait time with exponential backoff: 2^attempt seconds
                wait_time: int = 2**attempt  # 1s, 2s, 4s for attempts 0, 1, 2
                print(
                    f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

        # This should never be reached due to the raise in the except block above
        raise RuntimeError("Max retries exceeded")

    def fetch_s2(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        *,
        resolution: int = 10,
        max_cloud_cover: float = 100.0,
        product_type: str = "S2MSI1C",
        download_data: bool = True,
        interactive: bool = True,
        max_products: int = 3,
    ) -> List[Path]:
        """Fetch Sentinel-2 products for a given area and time period.

        This method searches for Sentinel-2 satellite imagery that covers the specified
        bounding box during the given date range. It handles caching automatically,
        so repeated requests with the same parameters will return cached results.

        Args:
            bbox: Bounding box as [min_longitude, min_latitude, max_longitude, max_latitude]
                  in WGS84 coordinate system (EPSG:4326). Example: [25.6796, -27.6721, 25.6897, -27.663]
            start_date: Start date in YYYY-MM-DD format, e.g., "2022-01-01"
            end_date: End date in YYYY-MM-DD format, e.g., "2022-01-31"
            resolution: Spatial resolution in meters. Options are:
                       - 10: Highest resolution (10m per pixel) - good for detailed analysis
                       - 20: Medium resolution (20m per pixel) - good balance of detail and coverage
                       - 60: Lowest resolution (60m per pixel) - good for large area analysis
            max_cloud_cover: Maximum acceptable cloud cover percentage (0-100).
                           Lower values = clearer images but fewer results.
                           100 = accept any cloud cover level.
            product_type: Type of Sentinel-2 product:
                         - "S2MSI1C": Level-1C (top-of-atmosphere reflectance, not atmospherically corrected)
                         - "S2MSI2A": Level-2A (bottom-of-atmosphere reflectance, atmospherically corrected)
            download_data: If True, download actual satellite imagery. If False, only fetch metadata.
            interactive: If True, prompt user for download confirmation when products are found.
            max_products: Maximum number of products to download/process.
                         Default: 3 (prevents accidental huge downloads)
                         Set to higher value or None for unlimited
                         Example: max_products=10 for 10 products

        Returns:
            List of Path objects pointing to downloaded imagery files or metadata files.

        Raises:
            ValueError: If input parameters are invalid (e.g., invalid bbox coordinates,
                       unsupported resolution, invalid date format, etc.)
        """
        # Validate all input parameters before proceeding
        # This catches common errors early and provides helpful error messages
        validate_bbox(bbox)  # Check bbox format and coordinate bounds
        validate_date_range(start_date, end_date)  # Check date format and ordering

        # Validate resolution parameter
        if resolution not in [10, 20, 60]:
            raise ValueError("resolution must be 10, 20, or 60 meters")

        # Validate cloud cover parameter
        if not 0 <= max_cloud_cover <= 100:
            raise ValueError("max_cloud_cover must be between 0 and 100")

        # Delegate the actual work to the S2-specific module
        # This keeps the client class focused on coordination and validation
        return fetch_s2_products(
            client=self,  # Pass self so S2 module can use our authentication and caching
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
            max_cloud_cover=max_cloud_cover,
            product_type=product_type,
            download_data=download_data,
            interactive=interactive,
            max_products=max_products,
        )

    def fetch_s1(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        *,
        product_type: str = "GRD",
        polarization: str = "VV,VH",
        orbit_direction: str = "ASCENDING",
        acquisition_mode: str = "IW",
        download_data: bool = True,
        max_products: int = 3,
    ) -> List[Path]:
        """Fetch Sentinel-1 products for a given area and time period.

        This method searches for Sentinel-1 SAR (Synthetic Aperture Radar) imagery
        that covers the specified bounding box during the given date range.
        SAR imagery works day/night and through clouds, making it complementary to optical imagery.

        Args:
            bbox: Bounding box as [min_longitude, min_latitude, max_longitude, max_latitude]
                  in WGS84 coordinate system (EPSG:4326). Example: [25.6796, -27.6721, 25.6897, -27.663]
            start_date: Start date in YYYY-MM-DD format, e.g., "2022-01-01"
            end_date: End date in YYYY-MM-DD format, e.g., "2022-01-31"
            product_type: Type of Sentinel-1 product:
                         - "GRD": Ground Range Detected (most common, preprocessed and ready to use)
                         - "SLC": Single Look Complex (raw data, requires more processing)
                         - "OCN": Ocean products (specialized for ocean analysis)
            polarization: Radar polarization modes to include:
                         - "VV": Vertical transmit, Vertical receive
                         - "VH": Vertical transmit, Horizontal receive
                         - "HH": Horizontal transmit, Horizontal receive
                         - "HV": Horizontal transmit, Vertical receive
                         - "VV,VH": Both VV and VH (most common for land applications)
                         Different polarizations reveal different surface properties.
            orbit_direction: Satellite orbit direction:
                           - "ASCENDING": Satellite moving from south to north
                           - "DESCENDING": Satellite moving from north to south
                           Different directions can show different aspects of terrain.
            acquisition_mode: SAR acquisition mode (default: "IW")

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
            download_data: If True, download actual SAR imagery (1-2GB per product).
                          If False, only fetch metadata (few KB per product).
                          Default: True
            max_products: Maximum number of products to download/process.
                         Default: 3 (prevents accidental huge downloads)
                         Set to higher value or None for unlimited
                         Example: max_products=10 for 10 products

        Returns:
            List of Path objects pointing to downloaded ZIP files or metadata JSON files.

        Raises:
            ValueError: If input parameters are invalid (e.g., invalid bbox coordinates,
                       unsupported product type, invalid polarization, etc.)
        """
        # Validate all input parameters before proceeding
        validate_bbox(bbox)  # Check bbox format and coordinate bounds
        validate_date_range(start_date, end_date)  # Check date format and ordering

        # Validate product type parameter
        if product_type not in ["GRD", "SLC", "OCN"]:
            raise ValueError("product_type must be GRD, SLC, or OCN")

        # Validate polarization parameter
        # Split by comma and check each polarization mode
        valid_pols: set[str] = {"VV", "VH", "HH", "HV"}
        requested_pols: set[str] = set(pol.strip() for pol in polarization.split(","))
        if not requested_pols.issubset(valid_pols):
            raise ValueError(f"Invalid polarization. Must be subset of {valid_pols}")

        # Validate orbit direction parameter
        if orbit_direction not in ["ASCENDING", "DESCENDING"]:
            raise ValueError("orbit_direction must be ASCENDING or DESCENDING")

        # Validate acquisition mode parameter
        if acquisition_mode not in ["IW", "EW", "SM", "WV"]:
            raise ValueError(
                f"acquisition_mode must be IW, EW, SM, or WV. Got: {acquisition_mode}\n"
                "\n"
                "Available modes:\n"
                "  IW (default): 250km coverage, 10m resolution - general land monitoring\n"
                "                Best for agriculture, forests, urban areas (95% of use cases)\n"
                "  EW: 400km coverage, 40m resolution - ocean/polar regions\n"
                "      Best for maritime surveillance, ice sheets, wide area monitoring\n"
                "  SM: 80km coverage, 5m resolution - emergency response\n"
                "      Best for disasters, high-detail urban mapping, infrastructure\n"
                "  WV: 20km samples, 5m resolution - ocean waves only\n"
                "      Specialized for ocean wave height/direction analysis\n"
            )

        # Delegate the actual work to the S1-specific module
        # This keeps the client class focused on coordination and validation
        return fetch_s1_products(
            client=self,  # Pass self so S1 module can use our authentication and caching
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            product_type=product_type,
            polarization=polarization,
            orbit_direction=orbit_direction,
            acquisition_mode=acquisition_mode,
            download_data=download_data,
            max_products=max_products,
        )
