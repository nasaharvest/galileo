"""Copernicus Data Space Ecosystem client for fetching Sentinel-1 and Sentinel-2 data."""

from .client import CopernicusClient
from .enums import S1AcquisitionMode, S1Polarization, S1ProductType, S2Band
from .image_processing import (
    create_false_color_composite,
    crop_to_bbox,
    extract_all_s1_bands,
    extract_all_s2_bands,
    extract_rgb_composite,
    extract_sar_composite,
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
from .time_series import (
    create_time_series_tif,
    generate_date_list,
    write_multiband_geotiff,
)
from .utils import create_validated_bbox, find_granule_directory
from .visualization import (
    create_band_analysis_plot,
    create_comparison_plot,
    create_coverage_map,
    create_metadata_summary,
    create_sar_comparison_plot,
    display_sar_image,
    display_satellite_image,
)

__all__ = [
    "CopernicusClient",
    # Enums
    "S2Band",
    "S1ProductType",
    "S1Polarization",
    "S1AcquisitionMode",
    # Image processing
    "extract_rgb_composite",
    "extract_sar_composite",
    "extract_all_s2_bands",
    "extract_all_s1_bands",
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
    # Utilities
    "create_validated_bbox",
    "find_granule_directory",
    # Time series
    "create_time_series_tif",
    "generate_date_list",
    "write_multiband_geotiff",
    # Visualization
    "create_coverage_map",
    "display_satellite_image",
    "display_sar_image",
    "create_comparison_plot",
    "create_sar_comparison_plot",
    "create_metadata_summary",
    "create_band_analysis_plot",
]
