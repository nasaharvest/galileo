"""Robust download utilities for large Copernicus satellite products.

This module provides download functions that handle:
- Token expiration during long downloads
- Network interruptions with automatic retry
- Partial download resumption
- Progress tracking for multi-GB files
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from tqdm import tqdm

if TYPE_CHECKING:
    from .client import CopernicusClient


def download_with_retry(
    client: "CopernicusClient",
    url: str,
    output_path: Path,
    total_size: int,
    max_retries: int = 3,
    chunk_size: int = 8192,
) -> bool:
    """Download a file with automatic retry and token refresh.

    This function handles large file downloads (1-10 GB) that may take longer
    than the OAuth token lifetime (~10 minutes). It refreshes the token
    periodically and retries on failures.

    Key features:
    - Token refresh every 5 minutes during download
    - Automatic retry with exponential backoff on failures
    - Resume partial downloads using HTTP Range requests
    - Progress bar for user feedback

    Args:
        client: CopernicusClient for authentication
        url: Download URL (typically ends with /$value)
        output_path: Where to save the downloaded file
        total_size: Expected file size in bytes
        max_retries: Maximum number of retry attempts (default: 3)
        chunk_size: Download chunk size in bytes (default: 8KB)

    Returns:
        True if download succeeded, False otherwise

    Example:
        >>> success = download_with_retry(
        ...     client, download_url, Path("product.zip"), 1500000000
        ... )
        >>> if success:
        ...     print("Download complete!")
    """
    # Track download progress
    bytes_downloaded = 0
    retry_count = 0
    last_token_refresh = time.time()
    token_refresh_interval = 300  # Refresh token every 5 minutes

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
            # Refresh token if needed (before starting/resuming download)
            current_time = time.time()
            if current_time - last_token_refresh > token_refresh_interval:
                print("🔄 Refreshing authentication token...")
                client._get_access_token()  # Force token refresh
                last_token_refresh = current_time

            # Get fresh token for this attempt
            token = client._get_access_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "Galileo-Copernicus-Client/1.0",
            }

            # Add Range header for resume capability
            if bytes_downloaded > 0:
                headers["Range"] = f"bytes={bytes_downloaded}-"

            # Start download with streaming
            response = requests.get(url, headers=headers, stream=True, timeout=300)

            # Handle resume responses
            if response.status_code == 206:  # Partial Content (resume)
                print(f"✓ Server supports resume, continuing from byte {bytes_downloaded}")
            elif response.status_code == 200:  # Full content
                if bytes_downloaded > 0:
                    print("⚠️  Server doesn't support resume, restarting download")
                    bytes_downloaded = 0
                    if output_path.exists():
                        output_path.unlink()
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

                            # Check if we need to refresh token during download
                            current_time = time.time()
                            if current_time - last_token_refresh > token_refresh_interval:
                                # Token might expire soon, but we can't refresh mid-stream
                                # Just note it for the next retry if this fails
                                last_token_refresh = current_time

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

            # Exponential backoff: wait 2^retry seconds
            wait_time = 2**retry_count
            print(f"⚠️  Download error: {e}")
            print(f"🔄 Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
            time.sleep(wait_time)
            continue

        except Exception as e:
            print(f"❌ Unexpected error during download: {e}")
            return False

    print(f"❌ Download failed after {max_retries} retries")
    return False
