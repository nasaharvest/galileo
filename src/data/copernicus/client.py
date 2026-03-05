"""Main Copernicus Data Space Ecosystem client.

This module provides the main CopernicusClient class that handles:
- OAuth2 authentication with Copernicus Data Space Ecosystem
- Caching of downloaded products and metadata
- Coordination for Sentinel-2 data fetching
- Error handling and retry logic for API requests
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

# Import the specific handler for Sentinel-2
from .s2 import fetch_s2_products
from .utils import ensure_cache_dir, validate_bbox, validate_date_range


class CopernicusClient:
    """Main client for fetching Sentinel-2 data from Copernicus Data Space Ecosystem.

    This client handles:
    1. OAuth2 authentication with automatic token refresh
    2. Caching of search results and product metadata
    3. Input validation for bounding boxes and date ranges
    4. Coordination of S2 fetch operations

    The client uses the free Copernicus Data Space Ecosystem API, which replaced
    the old Copernicus Open Access Hub in 2023.

    The client can be used as a context manager to ensure proper cleanup of HTTP sessions:

    Example:
        >>> with CopernicusClient() as client:
        ...     products = client.fetch_s2(
        ...         bbox=[25.6796, -27.6721, 25.6897, -27.663],
        ...         start_date="2022-01-01",
        ...         end_date="2022-01-31"
        ...     )

    Or used directly (remember to call close() when done):

    Example:
        >>> client = CopernicusClient()
        >>> products = client.fetch_s2(bbox, start_date, end_date)
        >>> client.close()  # Clean up when done
    """

    # API endpoints for Copernicus Data Space Ecosystem
    BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"  # Main catalog API
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"  # OAuth endpoint

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cache/copernicus",
        load_dotenv_file: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Initialize the Copernicus client with authentication and caching setup.

        The Copernicus Data Space Ecosystem uses username/password authentication for
        the OData catalog and download APIs.

        Args:
            cache_dir: Directory where downloaded files and metadata will be cached.
                      Defaults to "data/cache/copernicus" relative to current directory.
            load_dotenv_file: Whether to automatically load credentials from .env file.
                             Set to False if you're providing credentials directly.
            username: Copernicus account username/email. If None, loads from COPERNICUS_USERNAME env var.
            password: Copernicus account password. If None, loads from COPERNICUS_PASSWORD env var.

        Raises:
            ValueError: If username/password credentials cannot be found.
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
        username_from_env: Optional[str] = username or os.getenv("COPERNICUS_USERNAME")
        password_from_env: Optional[str] = password or os.getenv("COPERNICUS_PASSWORD")

        # Validate that we have the required username/password credentials
        if not username_from_env or not password_from_env:
            raise ValueError(
                "Copernicus credentials not found. Set COPERNICUS_USERNAME and "
                "COPERNICUS_PASSWORD environment variables or pass them directly. "
                "Get an account at: https://dataspace.copernicus.eu/"
            )

        self.username: str = username_from_env
        self.password: str = password_from_env

        # OAuth token management - these will be set when we first authenticate
        self._access_token: Optional[str] = None  # The actual Bearer token
        self._token_expires_at: float = 0  # Unix timestamp when token expires
        self._refresh_token: Optional[str] = None  # Refresh token for extending session

        # Create a persistent HTTP session for connection pooling and cookie management
        self.session: requests.Session = requests.Session()

    def _get_access_token(self) -> str:
        """Get a valid OAuth access token, refreshing if necessary.

        This method implements token caching - it only requests a new token if:
        1. We don't have a token yet, OR
        2. The current token is about to expire (within 5 minutes)

        The Copernicus OData/Download API uses OAuth2 "password" grant flow,
        which means we exchange username and password for a temporary access token.

        Returns:
            A valid Bearer token string that can be used in API requests.

        Raises:
            requests.HTTPError: If the token request fails (e.g., invalid credentials).
        """
        # Check if we already have a valid token that hasn't expired
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        # Try to use refresh token first if we have one
        if self._refresh_token:
            try:
                return self._refresh_access_token()
            except Exception as e:
                print(f"⚠️  Refresh token failed: {e}, requesting new token...")
                self._refresh_token = None  # Clear invalid refresh token

        # Request a new token using OAuth2 password grant flow
        # This is the correct method for OData catalog and download APIs
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public",  # Public client ID for Copernicus Data Space
        }

        try:
            # Make the token request
            response = self.session.post(self.TOKEN_URL, data=data, timeout=30)

            # Debug: Show what we're requesting
            print("🔑 Requesting token with grant_type=password")

            response.raise_for_status()  # Raise exception if request failed

            # Parse the response to get token information
            token_data = response.json()

            # Debug: Show token details
            if "access_token" in token_data:
                self._access_token = token_data["access_token"]  # The actual token string
                self._refresh_token = token_data.get(
                    "refresh_token", None
                )  # Save refresh token (optional)

                # Validate we got a non-empty token
                if not self._access_token:
                    raise ValueError("Received empty access token from API")

                # Calculate when this token expires and set expiry with 5 minute buffer
                expires_in = token_data.get("expires_in", 600)  # Default to 10 minutes
                self._token_expires_at = (
                    time.time() + expires_in - 300
                )  # Subtract 5 minutes (300 seconds)

                # Check token scope/audience
                token_scope = token_data.get("scope", "not specified")
                token_type = token_data.get("token_type", "Bearer")

                print("✓ Got new access token")
                print(f"  - Type: {token_type}")
                print(f"  - Expires in: {expires_in}s (~{expires_in//60} minutes)")
                print(f"  - Scope: {token_scope}")
                print(f"  - Has refresh token: {self._refresh_token is not None}")

                return self._access_token
            else:
                print("❌ Token response missing 'access_token' field")
                print(f"   Response: {token_data}")
                raise ValueError("Invalid token response")

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get access token: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"   Response status: {e.response.status_code}")
                print(f"   Response body: {e.response.text[:200]}")
            raise

    def _refresh_access_token(self) -> str:
        """Refresh the access token using the refresh token.

        This extends the session without requiring username/password again.
        Refresh tokens are valid for 60 minutes.

        Returns:
            A valid Bearer token string.

        Raises:
            requests.HTTPError: If the refresh request fails.
        """
        if not self._refresh_token:
            raise ValueError("No refresh token available")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": "cdse-public",
        }

        try:
            response = self.session.post(self.TOKEN_URL, data=data, timeout=30)
            response.raise_for_status()

            token_data = response.json()

            if "access_token" in token_data:
                self._access_token = token_data["access_token"]
                self._refresh_token = token_data.get("refresh_token", self._refresh_token)

                # Validate we got a non-empty token
                if not self._access_token:
                    raise ValueError("Received empty access token from refresh")

                expires_in = token_data.get("expires_in", 600)
                self._token_expires_at = time.time() + expires_in - 300

                print(f"🔄 Refreshed access token (expires in {expires_in}s)")

                return self._access_token
            else:
                raise ValueError("Invalid refresh token response")

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to refresh token: {e}")
            raise

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
        from .s1 import fetch_s1_products

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

    def download_product(
        self,
        url: str,
        output_path: Path,
        total_size: int,
        max_retries: int = 3,
        chunk_size: int = 8192,
    ) -> bool:
        """Download a product file with automatic retry and token refresh.

        This method handles large file downloads (1-10 GB) that may take longer
        than the OAuth token lifetime (~10 minutes). It uses HTTP Range requests
        to resume downloads and refreshes tokens between retry attempts.

        Key features:
        - Fresh token for each download attempt
        - Automatic retry with exponential backoff on failures
        - Resume partial downloads using HTTP Range requests
        - Progress bar for user feedback
        - Unlimited token refreshes (token expiration doesn't count as a retry)

        IMPORTANT: For downloads longer than token lifetime (~10 min), the download
        will fail and automatically retry with a fresh token, resuming from where
        it left off using HTTP Range requests.

        Args:
            url: Download URL (typically ends with /$value)
            output_path: Where to save the downloaded file
            total_size: Expected file size in bytes
            max_retries: Maximum number of retry attempts for actual failures (default: 3)
                        Note: Token refreshes don't count toward this limit
            chunk_size: Download chunk size in bytes (default: 8KB)

        Returns:
            True if download succeeded, False otherwise

        Example:
            >>> client = CopernicusClient()
            >>> success = client.download_product(
            ...     download_url, Path("product.zip"), 1500000000
            ... )
            >>> if success:
            ...     print("Download complete!")
        """
        import time

        from tqdm import tqdm

        # Track download progress
        bytes_downloaded = 0
        retry_count = 0
        token_refresh_count = 0  # Track token refreshes separately

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if partial download exists
        if output_path.exists():
            bytes_downloaded = output_path.stat().st_size
            if bytes_downloaded >= total_size:
                print(f"✅ Already downloaded: {output_path.name}")
                return True
            print(f"📥 Resuming download from {bytes_downloaded / 1024**2:.1f} MB")

        while retry_count <= max_retries:
            try:
                # Always get a fresh token for each attempt
                token = self._get_access_token()
                headers = {
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Galileo-Copernicus-Client/1.0",
                }

                # Add Range header for resume capability
                if bytes_downloaded > 0:
                    headers["Range"] = f"bytes={bytes_downloaded}-"

                # Debug: Log the request details
                if token_refresh_count == 0:
                    print("📡 Making download request...")
                    print(f"   URL: {url}")
                    print(f"   Token (first 20 chars): {token[:20]}...")

                # Start download with streaming
                response = self.session.get(url, headers=headers, stream=True, timeout=300)

                # Debug: Log response status
                if response.status_code != 200 and response.status_code != 206:
                    print(f"⚠️  Response status: {response.status_code}")

                    # Check for rate limiting headers
                    session_limit = response.headers.get("x-cf-sessionlimit-limit")
                    session_remaining = response.headers.get("x-cf-sessionlimit-remaining")
                    if session_limit or session_remaining:
                        print(f"   Session limit: {session_remaining}/{session_limit}")

                    print(f"   Response headers: {dict(response.headers)}")
                    if response.text:
                        print(f"   Response body (first 200 chars): {response.text[:200]}")

                # Handle resume responses
                if response.status_code == 206:  # Partial Content (resume)
                    print(f"✓ Server supports resume, continuing from byte {bytes_downloaded}")
                elif response.status_code == 200:  # Full content
                    if bytes_downloaded > 0:
                        print("⚠️  Server doesn't support resume, restarting download")
                        bytes_downloaded = 0
                        if output_path.exists():
                            output_path.unlink()
                elif response.status_code == 401:  # Token expired
                    token_refresh_count += 1

                    # Check if we're stuck in a loop (no progress after multiple refreshes)
                    if token_refresh_count > 10 and bytes_downloaded == 0:
                        print(
                            "❌ Token keeps expiring without making progress. "
                            "This might indicate invalid credentials or API issues."
                        )
                        return False

                    print(
                        f"🔄 Token expired, refreshing and retrying... "
                        f"(refresh #{token_refresh_count})"
                    )
                    self._access_token = None  # Force token refresh
                    time.sleep(2)  # Longer pause to avoid rate limiting
                    continue  # Don't increment retry_count for token expiration
                else:
                    response.raise_for_status()

                # Open file in append mode if resuming, write mode otherwise
                mode = "ab" if bytes_downloaded > 0 else "wb"

                # Download with progress bar
                with open(output_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=bytes_downloaded,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {output_path.name[:40]}",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                chunk_len = len(chunk)
                                bytes_downloaded += chunk_len
                                pbar.update(chunk_len)

                # Verify download completed
                if bytes_downloaded >= total_size:
                    print(f"✅ Download complete: {output_path.name}")
                    return True
                else:
                    print(f"⚠️  Download incomplete: {bytes_downloaded}/{total_size} bytes")
                    retry_count += 1
                    continue

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"❌ Download failed after {max_retries} retries: {e}")
                    return False

                # Exponential backoff
                wait_time = 2**retry_count
                print(f"⚠️  Download error: {e}")
                print(
                    f"🔄 Retrying in {wait_time} seconds... "
                    f"(attempt {retry_count}/{max_retries})"
                )
                time.sleep(wait_time)
                continue

            except Exception as e:
                print(f"❌ Unexpected error during download: {e}")
                return False

        print(f"❌ Download failed after {max_retries} retries")
        return False

    def export_to_geotiff(
        self,
        image_data: Dict[str, Any],
        output_path: Union[str, Path],
        satellite_type: str = "S2",
    ) -> Path:
        """Export processed image data to a GeoTIFF file.

        This method takes pre-processed image data (from extract_rgb_composite or
        extract_sar_composite) and exports it as a georeferenced GeoTIFF file.

        Args:
            image_data: Dictionary containing image array and bounds from image_processing functions.
                       Expected keys:
                       - 'rgb_array' or 'sar_array': The image data (H, W, C) or (H, W)
                       - 'bounds_wgs84': Bounding box [min_lon, min_lat, max_lon, max_lat]
            output_path: Where to save the GeoTIFF file (e.g., "output.tif")
            satellite_type: Type of satellite data ("S2" for Sentinel-2, "S1" for Sentinel-1)

        Returns:
            Path object pointing to the created GeoTIFF file

        Raises:
            ValueError: If image_data is missing required keys or has invalid format
            ImportError: If rasterio is not installed

        Example:
            >>> from src.data.copernicus.image_processing import extract_rgb_composite
            >>> client = CopernicusClient()
            >>> image_data = extract_rgb_composite(product_path, bbox=[6.15, 49.11, 6.16, 49.12])
            >>> geotiff_path = client.export_to_geotiff(image_data, "output.tif", "S2")
            >>> print(f"Exported to {geotiff_path}")
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError(
                "rasterio is required for GeoTIFF export. Install with: pip install rasterio"
            )

        import numpy as np

        # Validate input data
        if not isinstance(image_data, dict):
            raise ValueError("image_data must be a dictionary")

        if "bounds_wgs84" not in image_data:
            raise ValueError("image_data must contain 'bounds_wgs84' key")

        # Get the image array based on satellite type
        if satellite_type == "S2":
            if "rgb_array" not in image_data:
                raise ValueError("image_data must contain 'rgb_array' key for S2 data")
            array = image_data["rgb_array"]
        else:  # S1
            if "sar_array" not in image_data:
                raise ValueError("image_data must contain 'sar_array' key for S1 data")
            array = image_data["sar_array"]

        # Get bounds
        bounds = image_data["bounds_wgs84"]
        if len(bounds) != 4:
            raise ValueError("bounds_wgs84 must be [min_lon, min_lat, max_lon, max_lat]")

        # Convert output path to Path object
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare array for export
        # Rasterio expects (bands, height, width) format
        if array.ndim == 2:
            # Single band (grayscale SAR)
            array = array[np.newaxis, :, :]  # Add band dimension
            count = 1
        elif array.ndim == 3:
            # Multi-band (RGB or multi-polarization)
            # Convert from (H, W, C) to (C, H, W)
            array = np.transpose(array, (2, 0, 1))
            count = array.shape[0]
        else:
            raise ValueError(f"Unexpected array dimensions: {array.ndim}")

        height, width = array.shape[1], array.shape[2]

        # Validate array is not empty
        if array.size == 0:
            raise ValueError("Cannot export empty array to GeoTIFF")

        # Create affine transform from bounds
        transform = from_bounds(
            bounds[0],  # west (min_lon)
            bounds[1],  # south (min_lat)
            bounds[2],  # east (max_lon)
            bounds[3],  # north (max_lat)
            width,
            height,
        )

        # Handle NaN values if present
        if np.isnan(array).any():
            print("⚠️ Warning: Array contains NaN values, filling with 0")
            array = np.nan_to_num(array, nan=0.0)

        # Convert to appropriate data type for GeoTIFF
        # If data is in [0, 1] range (normalized), scale to uint16
        if array.max() <= 1.0 and array.min() >= 0.0:
            array = (array * 65535).astype(np.uint16)
            dtype = rasterio.uint16
        else:
            # Keep as float32 for raw values
            array = array.astype(np.float32)
            dtype = rasterio.float32

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs="EPSG:4326",  # WGS84
            transform=transform,
            compress="lzw",  # Compress to save space
        ) as dst:
            dst.write(array)

        print(f"✅ Exported GeoTIFF: {output_path}")
        print(f"   Size: {width}x{height} pixels, {count} band(s)")
        print(f"   Bounds: {bounds}")

        return output_path

    def close(self) -> None:
        """Close the HTTP session and clean up resources.

        This method should be called when you're done using the client to ensure
        the HTTP session is properly closed and connections are released.

        It's automatically called when using the client as a context manager.
        """
        if self.session:
            self.session.close()

    def __enter__(self) -> "CopernicusClient":
        """Enter the context manager.

        Returns:
            self: The client instance for use in the with block.

        Example:
            >>> with CopernicusClient() as client:
            ...     products = client.fetch_s2(bbox, start_date, end_date)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and clean up resources.

        This ensures the HTTP session is properly closed even if an exception occurs.

        Args:
            exc_type: Exception type if an exception was raised, None otherwise
            exc_val: Exception value if an exception was raised, None otherwise
            exc_tb: Exception traceback if an exception was raised, None otherwise
        """
        self.close()

    def export_time_series_for_galileo(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        output_path: Path,
        temporal_resolution: str = "weekly",
        max_products_per_date: int = 1,
    ) -> Path:
        """Export Galileo-compatible time series TIF with S1 and S2 data.

        This method downloads Sentinel-1 and Sentinel-2 data for a date range,
        extracts all spectral bands, and stacks them into a single multi-band
        GeoTIFF file compatible with the Galileo model.

        ⚠️ IMPORTANT LIMITATIONS:
        - This exports ONLY S1 and S2 bands (14 bands per timestep)
        - Galileo expects 449 bands including ERA5, SRTM, VIIRS, etc.
        - This is Phase 1 - additional data sources needed for full compatibility

        Output Format:
        - Bands 1-12: S2 spectral bands (B01-B12) for date 1
        - Bands 13-14: S1 polarizations (VV, VH) for date 1
        - Bands 15-26: S2 spectral bands for date 2
        - Bands 27-28: S1 polarizations for date 2
        - ... and so on
        - Total: 14 bands × num_dates

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_path: Where to save the time series TIF
            temporal_resolution: One of 'daily', 'weekly', 'monthly'
            max_products_per_date: Max products to download per date (default: 1)

        Returns:
            Path to created GeoTIFF file

        Example:
            >>> client = CopernicusClient()
            >>> result = client.export_time_series_for_galileo(
            ...     bbox=[6.15, 49.11, 6.16, 49.12],
            ...     start_date="2023-01-01",
            ...     end_date="2023-12-31",
            ...     output_path=Path("data/exports/time_series.tif"),
            ...     temporal_resolution="weekly"
            ... )
            >>> print(f"Exported {result}")
        """
        from .time_series import create_time_series_tif, generate_date_list

        print("=" * 80)
        print("EXPORTING TIME SERIES FOR GALILEO")
        print("=" * 80)
        print(f"Bbox: {bbox}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Temporal resolution: {temporal_resolution}")
        print()

        # Generate date list
        dates = generate_date_list(start_date, end_date, temporal_resolution)
        print(f"Generated {len(dates)} dates to process")
        print()

        # Download S2 for each date
        print("📥 DOWNLOADING SENTINEL-2 DATA")
        print("-" * 80)
        s2_files: List[Optional[Path]] = []
        for i, date_obj in enumerate(dates):
            date_str = date_obj.strftime("%Y-%m-%d")
            print(f"[{i+1}/{len(dates)}] Downloading S2 for {date_str}...")

            try:
                files = self.fetch_s2(
                    bbox=bbox,
                    start_date=date_str,
                    end_date=date_str,
                    max_products=max_products_per_date,
                    download_data=True,
                    interactive=False,
                )

                if files and len(files) > 0:
                    s2_files.append(files[0])
                    print(f"  ✅ Downloaded: {files[0].name}")
                else:
                    print(f"  ⚠️  No S2 data available for {date_str}")
                    # Add None placeholder to maintain date alignment
                    s2_files.append(None)
            except Exception as e:
                print(f"  ❌ Error downloading S2 for {date_str}: {e}")
                s2_files.append(None)

        print()
        print(
            f"Successfully downloaded {sum(1 for f in s2_files if f is not None)}/{len(dates)} S2 products"
        )
        print()

        # Download S1 for each date
        print("📥 DOWNLOADING SENTINEL-1 DATA")
        print("-" * 80)
        s1_files: List[Optional[Path]] = []
        for i, date_obj in enumerate(dates):
            date_str = date_obj.strftime("%Y-%m-%d")
            print(f"[{i+1}/{len(dates)}] Downloading S1 for {date_str}...")

            try:
                files = self.fetch_s1(
                    bbox=bbox,
                    start_date=date_str,
                    end_date=date_str,
                    max_products=max_products_per_date,
                    download_data=True,
                )

                if files and len(files) > 0:
                    s1_files.append(files[0])
                    print(f"  ✅ Downloaded: {files[0].name}")
                else:
                    print(f"  ⚠️  No S1 data available for {date_str}")
                    s1_files.append(None)
            except Exception as e:
                print(f"  ❌ Error downloading S1 for {date_str}: {e}")
                s1_files.append(None)

        print()
        print(
            f"Successfully downloaded {sum(1 for f in s1_files if f is not None)}/{len(dates)} S1 products"
        )
        print()

        # Filter out dates where both S1 and S2 are missing
        valid_indices = [
            i for i in range(len(dates)) if s2_files[i] is not None and s1_files[i] is not None
        ]

        if not valid_indices:
            raise ValueError("No dates with both S1 and S2 data available!")

        # Type assertion: we know these are all non-None after filtering
        filtered_s2: List[Path] = [s2_files[i] for i in valid_indices]  # type: ignore[misc]
        filtered_s1: List[Path] = [s1_files[i] for i in valid_indices]  # type: ignore[misc]
        filtered_dates = [dates[i] for i in valid_indices]

        print(f"Proceeding with {len(filtered_dates)} dates that have both S1 and S2 data")
        print()

        # Create time series TIF
        print("🔄 CREATING TIME SERIES TIF")
        print("-" * 80)
        result = create_time_series_tif(
            s2_files=filtered_s2,
            s1_files=filtered_s1,
            dates=filtered_dates,
            bbox=bbox,
            output_path=output_path,
            normalize=True,
        )

        print()
        print("=" * 80)
        print("✅ TIME SERIES EXPORT COMPLETE")
        print("=" * 80)
        print(f"Output file: {result}")
        print(f"Dates processed: {len(filtered_dates)}")
        print(f"Total bands: {14 * len(filtered_dates)} (14 bands × {len(filtered_dates)} dates)")
        print()
        print("⚠️  IMPORTANT: This TIF contains ONLY S1+S2 bands")
        print("   Galileo expects 449 bands including:")
        print("   - ERA5 weather data")
        print("   - SRTM elevation")
        print("   - VIIRS nighttime lights")
        print("   - Dynamic World land cover")
        print("   - WorldCereal crop types")
        print("   - Landscan population")
        print()
        print("   Next steps:")
        print("   1. Test if Galileo works with partial bands")
        print("   2. If needed, add Phase 2 data sources (ERA5, SRTM, VIIRS)")
        print("   3. Or train new model with only S1+S2 data")
        print()

        return result
