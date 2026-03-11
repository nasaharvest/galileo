"""Band recipes for Sentinel-2 visualization.

This module provides predefined band combinations optimized for different
analysis tasks. Band recipes help users quickly visualize specific aspects
of satellite data without needing to know which bands to use.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .image_processing import extract_rgb_composite, extract_sar_composite
from .indices import calculate_ndvi, calculate_ndwi


class BandRecipe:
    """Definition of a band recipe for visualization."""

    def __init__(
        self,
        name: str,
        description: str,
        bands: Optional[List[str]] = None,
        index_function: Optional[Callable] = None,
        colormap: str = "viridis",
        value_range: Tuple[float, float] = (0, 1),
    ):
        """Initialize a band recipe.

        Args:
            name: Display name of the recipe
            description: What this recipe shows and when to use it
            bands: List of band names for RGB composite (e.g., ['B04', 'B03', 'B02'])
            index_function: Function to calculate spectral index (e.g., calculate_ndvi)
            colormap: Matplotlib colormap for index visualization
            value_range: Expected value range for the index
        """
        self.name = name
        self.description = description
        self.bands = bands
        self.index_function = index_function
        self.colormap = colormap
        self.value_range = value_range
        self.is_index = index_function is not None

    def __repr__(self) -> str:
        return f"BandRecipe(name='{self.name}', is_index={self.is_index})"


# Define available band recipes
BAND_RECIPES = {
    # Sentinel-2 (Optical) Recipes
    "true_color": BandRecipe(
        name="True Color (RGB)",
        description="Natural color composite - shows the world as human eyes see it. "
        "Good for general visualization, identifying features, and sharing with non-technical audiences.",
        bands=["B04", "B03", "B02"],  # Red, Green, Blue
        colormap="viridis",  # Not used for RGB
    ),
    "false_color": BandRecipe(
        name="False Color (NIR-R-G)",
        description="Near-infrared false color - vegetation appears bright red, water appears dark. "
        "Excellent for vegetation analysis, crop health, and distinguishing vegetation from soil. "
        "Healthy vegetation = bright red, stressed vegetation = dull red/brown.",
        bands=["B08", "B04", "B03"],  # NIR, Red, Green
        colormap="viridis",  # Not used for RGB
    ),
    "agriculture": BandRecipe(
        name="Agriculture (SWIR-NIR-B)",
        description="SWIR composite optimized for agriculture - shows crop types, soil moisture, and field boundaries. "
        "Different crops appear in different colors. Wet soil = dark, dry soil = bright. "
        "Useful for: crop classification, irrigation monitoring, harvest planning.",
        bands=["B11", "B08", "B02"],  # SWIR1, NIR, Blue
        colormap="viridis",  # Not used for RGB
    ),
    "ndvi": BandRecipe(
        name="NDVI (Vegetation Index)",
        description="Normalized Difference Vegetation Index - quantifies vegetation health and density. "
        "Green = healthy vegetation, yellow = moderate, red = sparse/stressed. "
        "Values: -1 to 1 (higher = more/healthier vegetation). "
        "Best for: crop monitoring, drought detection, vegetation mapping.",
        bands=None,
        index_function=calculate_ndvi,
        colormap="RdYlGn",  # Red-Yellow-Green
        value_range=(-1, 1),
    ),
    "ndwi": BandRecipe(
        name="NDWI (Water Index)",
        description="Normalized Difference Water Index - highlights water bodies and moisture. "
        "Blue = water/wet areas, yellow/red = dry land. "
        "Values: -1 to 1 (higher = more water). "
        "Best for: water detection, flood mapping, irrigation monitoring, wetland analysis.",
        bands=None,
        index_function=calculate_ndwi,
        colormap="RdYlBu",  # Red-Yellow-Blue
        value_range=(-1, 1),
    ),
    # Sentinel-1 (SAR) Recipes
    "sar_vv": BandRecipe(
        name="SAR VV (Surface)",
        description="VV polarization - emphasizes surface scattering. "
        "Bright = rough surfaces (buildings, rocks), dark = smooth surfaces (water, roads). "
        "Best for: water detection, urban mapping, surface roughness analysis.",
        bands=None,
        index_function=lambda path, bbox=None: _extract_sar_polarization(path, "VV", bbox),
        colormap="gray",
        value_range=(-30, 0),  # dB range
    ),
    "sar_vh": BandRecipe(
        name="SAR VH (Volume)",
        description="VH polarization - emphasizes volume scattering from vegetation. "
        "Bright = vegetation/forests, dark = bare soil/water. "
        "Best for: vegetation monitoring, forest mapping, crop type classification.",
        bands=None,
        index_function=lambda path, bbox=None: _extract_sar_polarization(path, "VH", bbox),
        colormap="gray",
        value_range=(-30, 0),  # dB range
    ),
    "sar_ratio": BandRecipe(
        name="SAR VH/VV Ratio",
        description="Cross-polarization ratio (VH/VV) - distinguishes surface types. "
        "High values = vegetation/volume scattering, low values = surface scattering. "
        "Best for: land cover classification, crop vs bare soil, vegetation structure.",
        bands=None,
        index_function=lambda path, bbox=None: _calculate_sar_ratio(path, bbox),
        colormap="viridis",
        value_range=(0, 1),
    ),
}


def get_available_recipes() -> Dict[str, BandRecipe]:
    """Get all available band recipes.

    Returns:
        Dictionary mapping recipe IDs to BandRecipe objects
    """
    return BAND_RECIPES


def get_recipe_names() -> List[str]:
    """Get list of recipe display names for UI dropdowns.

    Returns:
        List of recipe names in display order
    """
    return [recipe.name for recipe in BAND_RECIPES.values()]


def get_recipe_by_name(display_name: str) -> Optional[BandRecipe]:
    """Get recipe by its display name.

    Args:
        display_name: The display name shown in UI (e.g., "True Color (RGB)")

    Returns:
        BandRecipe object or None if not found
    """
    for recipe in BAND_RECIPES.values():
        if recipe.name == display_name:
            return recipe
    return None


def apply_band_recipe(
    zip_file_path: Path,
    recipe_name: str,
    bbox: Optional[List[float]] = None,
) -> Optional[Dict]:
    """Apply a band recipe to a Sentinel-2 image.

    This function extracts the appropriate bands or calculates indices
    based on the selected recipe.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        recipe_name: Display name of the recipe (e.g., "True Color (RGB)")
        bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat] to crop

    Returns:
        Dictionary containing:
        - For RGB recipes: {'rgb_array': ndarray, 'bounds_wgs84': tuple, 'metadata': dict}
        - For index recipes: {'index_array': ndarray, 'bounds_wgs84': tuple, 'metadata': dict,
                              'colormap': str, 'value_range': tuple}
        Returns None if processing fails.

    Example:
        >>> # Apply true color recipe
        >>> result = apply_band_recipe(s2_file, "True Color (RGB)", bbox)
        >>> plt.imshow(result['rgb_array'])
        >>>
        >>> # Apply NDVI recipe
        >>> result = apply_band_recipe(s2_file, "NDVI (Vegetation Index)", bbox)
        >>> plt.imshow(result['index_array'], cmap=result['colormap'])
    """
    # Get the recipe
    recipe = get_recipe_by_name(recipe_name)
    if recipe is None:
        print(f"Recipe not found: {recipe_name}")
        return None

    try:
        if recipe.is_index:
            # Calculate spectral index or SAR metric
            if recipe.index_function is None:
                print(f"Recipe {recipe_name} has no index function")
                return None

            index_result = recipe.index_function(zip_file_path, bbox=bbox)
            if index_result is None:
                return None

            # Extract the index array (key varies by index type)
            # For S2: 'ndvi', 'ndwi', etc.
            # For S1: 'vv', 'vh', 'ratio', etc.
            index_keys = [k for k in index_result.keys() if k not in ["bounds_wgs84", "metadata"]]
            if not index_keys:
                print("No data key found in index result")
                return None

            index_key = index_keys[0]
            index_array = index_result[index_key]

            # Get bounds from the index result (now included)
            bounds_wgs84 = index_result.get("bounds_wgs84")

            return {
                "index_array": index_array,
                "bounds_wgs84": bounds_wgs84,
                "metadata": {
                    "recipe": recipe.name,
                    "type": "index",
                    **index_result.get("metadata", {}),
                },
                "colormap": recipe.colormap,
                "value_range": recipe.value_range,
            }
        else:
            # Extract RGB composite with specified bands
            rgb_result = extract_rgb_composite(zip_file_path, bands=recipe.bands, bbox=bbox)
            if rgb_result is None:
                return None

            return {
                "rgb_array": rgb_result["rgb_array"],
                "bounds_wgs84": rgb_result["bounds_wgs84"],
                "metadata": {
                    "recipe": recipe.name,
                    "type": "rgb_composite",
                    "bands": recipe.bands,
                    **rgb_result.get("metadata", {}),
                },
            }

    except Exception as e:
        print(f"Error applying recipe '{recipe_name}': {e}")
        import traceback

        traceback.print_exc()
        return None


def apply_recipe_to_cached_data(
    cached_data: Dict,
    recipe_name: str,
    zip_file_path: Optional[Path] = None,
    bbox: Optional[List[float]] = None,
) -> Optional[Dict]:
    """Apply a band recipe to already-cached image data.

    This is an optimization for the marimo UI - if we already have RGB data cached,
    we can reuse it for True Color. For other recipes, we need to reprocess.

    Args:
        cached_data: Previously cached image data (from pre-processing)
        recipe_name: Display name of the recipe
        zip_file_path: Path to original ZIP file (needed for reprocessing)
        bbox: Optional bounding box for cropping

    Returns:
        Recipe result dictionary or None if processing fails
    """
    recipe = get_recipe_by_name(recipe_name)
    if recipe is None:
        return None

    # If requesting True Color and we have RGB cached, return it directly
    if recipe_name == "True Color (RGB)" and "rgb_array" in cached_data:
        return cached_data

    # For other recipes, we need to reprocess from the ZIP file
    if zip_file_path is None:
        print("ZIP file path required for non-True Color recipes")
        return None

    return apply_band_recipe(zip_file_path, recipe_name, bbox=bbox)


def _extract_sar_polarization(
    zip_file_path: Path,
    polarization: str,
    bbox: Optional[List[float]] = None,
) -> Optional[Dict]:
    """Extract a single SAR polarization (VV or VH).

    Args:
        zip_file_path: Path to Sentinel-1 ZIP file
        polarization: 'VV' or 'VH'
        bbox: Optional bounding box to crop

    Returns:
        Dictionary with SAR data and metadata
    """
    try:
        # Extract SAR composite with the requested polarization
        sar_result = extract_sar_composite(
            zip_file_path,
            polarizations=[polarization],
            bbox=bbox,
        )

        if sar_result is None:
            return None

        # Extract the single polarization
        sar_array = sar_result["sar_array"][:, :, 0]  # (H, W, 1) -> (H, W)

        return {
            polarization.lower(): sar_array,
            "bounds_wgs84": sar_result.get("bounds_wgs84"),
            "metadata": {
                "polarization": polarization,
                "type": "sar",
                **sar_result.get("metadata", {}),
            },
        }
    except Exception as e:
        print(f"Error extracting SAR {polarization}: {e}")
        import traceback

        traceback.print_exc()
        return None


def _calculate_sar_ratio(
    zip_file_path: Path,
    bbox: Optional[List[float]] = None,
) -> Optional[Dict]:
    """Calculate VH/VV cross-polarization ratio.

    This ratio helps distinguish surface types:
    - High values: Volume scattering (vegetation, forests)
    - Low values: Surface scattering (water, bare soil)

    Args:
        zip_file_path: Path to Sentinel-1 ZIP file
        bbox: Optional bounding box to crop

    Returns:
        Dictionary with ratio data and metadata
    """
    try:
        # Extract both polarizations
        sar_result = extract_sar_composite(
            zip_file_path,
            polarizations=["VV", "VH"],
            bbox=bbox,
        )

        if sar_result is None:
            return None

        # Extract VV and VH
        vv = sar_result["sar_array"][:, :, 0]
        vh = sar_result["sar_array"][:, :, 1]

        # Convert from dB to linear for ratio calculation
        vv_linear = np.power(10, vv / 10)
        vh_linear = np.power(10, vh / 10)

        # Calculate ratio (VH/VV)
        ratio = vh_linear / (vv_linear + 1e-10)

        # Normalize to 0-1 range for visualization
        ratio_normalized = np.clip(ratio, 0, 1)

        return {
            "ratio": ratio_normalized,
            "bounds_wgs84": sar_result.get("bounds_wgs84"),
            "metadata": {
                "index": "VH/VV Ratio",
                "formula": "VH / VV (linear scale)",
                "type": "sar_ratio",  # Not "sar" - this is a ratio, use fixed range
                "range": "[0, 1]",
                "shape": ratio_normalized.shape,
            },
        }
    except Exception as e:
        print(f"Error calculating SAR ratio: {e}")
        import traceback

        traceback.print_exc()
        return None
