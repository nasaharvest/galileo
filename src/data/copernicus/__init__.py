"""Copernicus Data Space Ecosystem client for fetching Sentinel-1 and Sentinel-2 data."""

from .client import CopernicusClient
from .image_processing import (
    create_false_color_composite,
    extract_rgb_composite,
    get_available_bands,
    get_image_statistics,
)
from .visualization import (
    create_band_analysis_plot,
    create_comparison_plot,
    create_coverage_map,
    create_metadata_summary,
    display_satellite_image,
)

__all__ = [
    "CopernicusClient",
    # Image processing
    "extract_rgb_composite",
    "get_available_bands",
    "create_false_color_composite",
    "get_image_statistics",
    # Visualization
    "create_coverage_map",
    "display_satellite_image",
    "create_comparison_plot",
    "create_metadata_summary",
    "create_band_analysis_plot",
]
