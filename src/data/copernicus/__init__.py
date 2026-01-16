"""Copernicus Data Space Ecosystem client for fetching Sentinel-2 data."""

from .client import CopernicusClient
from .image_processing import (
    create_false_color_composite,
    crop_to_bbox,
    extract_rgb_composite,
    get_available_bands,
    get_image_statistics,
)
from .indices import (
    calculate_evi,
    calculate_nbr,
    calculate_ndvi,
    calculate_ndwi,
    calculate_savi,
)
from .quality import apply_cloud_mask_to_image, extract_cloud_mask
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
    "crop_to_bbox",
    "get_available_bands",
    "create_false_color_composite",
    "get_image_statistics",
    # Quality control
    "extract_cloud_mask",
    "apply_cloud_mask_to_image",
    # Spectral indices
    "calculate_ndvi",
    "calculate_ndwi",
    "calculate_evi",
    "calculate_savi",
    "calculate_nbr",
    # Visualization
    "create_coverage_map",
    "display_satellite_image",
    "create_comparison_plot",
    "create_metadata_summary",
    "create_band_analysis_plot",
]
