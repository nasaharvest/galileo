"""Shared utilities for Sentinel product fetching.

This module contains common logic used by both S1 and S2 fetching modules,
reducing code duplication and providing a consistent interface.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .client import CopernicusClient


def check_cache(
    cache_file: Path,
) -> Optional[List[Path]]:
    """Check if cached results exist and are valid.

    Args:
        cache_file: Path to the cache file

    Returns:
        List of cached file paths if cache is valid, None otherwise
    """
    if not cache_file.exists():
        return None

    print(f"Loading products from cache: {cache_file}")
    with open(cache_file) as f:
        cached_data: Dict[str, Any] = json.load(f)

    # Verify that all cached files still exist on disk
    cached_paths: List[Path] = [Path(p) for p in cached_data["file_paths"]]
    if all(p.exists() for p in cached_paths):
        return cached_paths  # Cache hit - return existing results
    else:
        print("Some cached files missing, re-fetching...")
        return None


def save_cache(
    cache_file: Path,
    parameters: Dict[str, Any],
    products: List[Dict[str, Any]],
    file_paths: List[Path],
) -> None:
    """Save search results and file paths to cache.

    Args:
        cache_file: Path to the cache file
        parameters: Search parameters used
        products: Product metadata from API
        file_paths: Paths to downloaded/created files
    """
    cache_data: Dict[str, Any] = {
        "parameters": parameters,
        "products": products,
        "file_paths": [str(p) for p in file_paths],
    }

    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def apply_product_limit(
    products: List[Dict[str, Any]],
    max_products: Optional[int],
    satellite: str,
) -> List[Dict[str, Any]]:
    """Apply max_products limit with user-friendly messaging.

    Args:
        products: List of products to limit
        max_products: Maximum number of products (None for unlimited)
        satellite: Satellite name for messaging (S1, S2, etc.)

    Returns:
        Limited list of products
    """
    if max_products is None or len(products) <= max_products:
        return products

    print(f"⚠️  Limiting to first {max_products} products (found {len(products)} total)")
    print("   To download more, use max_products parameter:")
    print(f"   fetch_{satellite.lower()}(..., max_products={len(products)})")
    return products[:max_products]


def show_download_confirmation(
    products: List[Dict[str, Any]],
    download_data: bool,
    satellite: str,
) -> bool:
    """Show interactive download confirmation prompt.

    Args:
        products: List of products to download
        download_data: Whether downloading actual data or just metadata
        satellite: Satellite name for messaging

    Returns:
        True if user confirms, False if cancelled
    """
    if not products:
        return True

    print("\n🛰️ DOWNLOAD CONFIRMATION")
    print("=" * 40)
    print(f"Found {len(products)} {satellite} products:")

    for i, product in enumerate(products[:5], 1):  # Show first 5
        name = product.get("Name", "Unknown")
        size_mb = product.get("ContentLength", 0) / (1024 * 1024)
        print(f"  {i}. {name} ({size_mb:.1f} MB)")

    if len(products) > 5:
        print(f"  ... and {len(products) - 5} more products")

    total_size_gb = sum(p.get("ContentLength", 0) for p in products) / (1024**3)
    print(f"\nTotal size: {total_size_gb:.2f} GB")

    if download_data:
        print("Mode: Download actual satellite imagery")
        response = input(f"\nDownload {len(products)} products? [Y/n]: ").strip().lower()
        if response and response not in ["y", "yes"]:
            print("Download cancelled by user")
            return False
    else:
        print("Mode: Metadata only (no actual imagery download)")

    return True


def process_products(
    client: "CopernicusClient",
    products: List[Dict[str, Any]],
    download_data: bool,
    satellite: str,
    download_func: Callable,
    metadata_func: Callable,
    **kwargs: Any,
) -> List[Path]:
    """Process products by downloading or creating metadata.

    Args:
        client: CopernicusClient instance
        products: List of products to process
        download_data: Whether to download actual data
        satellite: Satellite name for messaging
        download_func: Function to download a product
        metadata_func: Function to create metadata
        **kwargs: Additional arguments passed to download/metadata functions

    Returns:
        List of paths to downloaded/created files
    """
    downloaded_paths: List[Path] = []

    if download_data:
        print(f"\n📥 DOWNLOADING {satellite} IMAGERY")
        print("=" * 45)

        for i, product in enumerate(products, 1):
            print(f"\n🛰️ Downloading product {i}/{len(products)}")

            downloaded_file = download_func(client, product, i - 1, **kwargs)
            if downloaded_file:
                downloaded_paths.append(downloaded_file)
                print(f"✅ Downloaded: {downloaded_file.name}")
            else:
                print(f"❌ Failed to download product {i}")
    else:
        print(f"\n📋 CREATING {satellite} METADATA FILES")
        print("=" * 35)

        for i, product in enumerate(products):
            metadata_file = metadata_func(client, product, i, **kwargs)
            if metadata_file:
                downloaded_paths.append(metadata_file)

    return downloaded_paths
