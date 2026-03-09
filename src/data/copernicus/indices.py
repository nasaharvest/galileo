"""Spectral indices calculation for Sentinel-2 data.

This module provides functions to calculate common vegetation and water indices
from Sentinel-2 multispectral imagery. These indices enhance specific features
and are widely used in remote sensing applications.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio


def _extract_band(
    zip_file_path: Path, band_name: str, bbox: Optional[list] = None
) -> Optional[Tuple[np.ndarray, Optional[Tuple]]]:
    """Extract a single band from Sentinel-2 ZIP file.

    Helper function to read individual spectral bands with optional bbox cropping.
    Uses selective extraction (only extracts the needed band file, not the entire ZIP).

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        band_name: Band identifier (e.g., 'B04', 'B08', 'B11')
        bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat] to crop

    Returns:
        Tuple of (band_data, bounds_wgs84) or (None, None) if extraction fails
        - band_data: Band data as numpy array
        - bounds_wgs84: Geographic bounds in WGS84 (min_lon, min_lat, max_lon, max_lat)
    """
    import tempfile
    import zipfile

    from rasterio.warp import transform_bounds

    from .image_processing import crop_to_bbox

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Selective extraction: only extract the band file we need
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                all_files = zip_ref.namelist()

                # Find the band file
                band_file_path = None
                patterns = [
                    f"_{band_name}_10m.jp2",
                    f"_{band_name}_20m.jp2",
                    f"_{band_name}.jp2",
                    f"{band_name}.jp2",
                ]

                for file_path in all_files:
                    filename = file_path.split("/")[-1]
                    for pattern in patterns:
                        if pattern in filename:
                            band_file_path = file_path
                            break
                    if band_file_path:
                        break

                if not band_file_path:
                    print(f"Band {band_name} not found in {zip_file_path.name}")
                    return None, None

                # Extract only this one file
                zip_ref.extract(band_file_path, temp_path)

            # Open and read the band
            extracted_file = temp_path / band_file_path
            if not extracted_file.exists():
                print(f"Extracted file not found: {extracted_file}")
                return None, None

            with rasterio.open(extracted_file) as src:
                band_data = src.read(1).astype(np.float32)

                # Get bounds
                bounds = src.bounds
                crs = src.crs

                # Convert bounds to WGS84
                bounds_wgs84 = transform_bounds(
                    crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
                )

                # Apply bbox cropping if requested
                if bbox is not None:
                    cropped = crop_to_bbox(band_data, bounds_wgs84, bbox)
                    if cropped is not None:
                        band_data = cropped
                        bounds_wgs84 = tuple(bbox)

                return band_data, bounds_wgs84

    except Exception as e:
        print(f"Error extracting band {band_name}: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def calculate_ndvi(zip_file_path: Path, bbox: Optional[list] = None) -> Optional[Dict]:
    """Calculate NDVI (Normalized Difference Vegetation Index).

    WHAT IS NDVI:
    NDVI measures vegetation health and density using the difference between
    near-infrared (NIR) and red light reflectance.

    WHY IT WORKS:
    - Healthy vegetation reflects a lot of NIR light (invisible to humans)
      This is because plant cells scatter NIR to avoid overheating
    - Healthy vegetation absorbs red light (for photosynthesis via chlorophyll)
    - Dead/stressed vegetation reflects less NIR and more red

    FORMULA:
    NDVI = (NIR - Red) / (NIR + Red)

    For Sentinel-2:
    - NIR = Band 8 (B08, 842nm wavelength, 10m resolution)
    - Red = Band 4 (B04, 665nm wavelength, 10m resolution)

    VALUE INTERPRETATION:
    - -1 to 0: Water, snow, clouds, bare soil, rock
    - 0 to 0.2: Sparse vegetation, bare soil, urban areas
    - 0.2 to 0.5: Moderate vegetation (grassland, shrubs, crops)
    - 0.5 to 0.8: Dense vegetation (forests, mature crops)
    - 0.8 to 1.0: Very dense vegetation (tropical rainforest)

    APPLICATIONS:
    - Crop health monitoring
    - Vegetation mapping
    - Drought detection
    - Deforestation monitoring
    - Agricultural yield prediction

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat] to crop result

    Returns:
        Dictionary containing:
        - 'ndvi': NDVI array with values from -1 to 1
        - 'metadata': Information about calculation
    """
    # Extract NIR band (B08)
    nir_result = _extract_band(zip_file_path, "B08", bbox)
    if nir_result is None:
        print("Failed to extract NIR band (B08)")
        return None
    nir, bounds_wgs84 = nir_result

    # Extract Red band (B04)
    red_result = _extract_band(zip_file_path, "B04", bbox)
    if red_result is None:
        print("Failed to extract Red band (B04)")
        return None
    red, _ = red_result

    # Resample if shapes don't match
    if red.shape != nir.shape:
        from scipy.ndimage import zoom

        zoom_factor = (nir.shape[0] / red.shape[0], nir.shape[1] / red.shape[1])
        red = zoom(red, zoom_factor, order=1)

    # Calculate NDVI using normalized difference
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Clip to valid range
    ndvi = np.clip(ndvi, -1, 1)

    return {
        "ndvi": ndvi,
        "bounds_wgs84": bounds_wgs84,
        "metadata": {
            "index": "NDVI",
            "formula": "(NIR - Red) / (NIR + Red)",
            "bands_used": ["B08 (NIR)", "B04 (Red)"],
            "range": "[-1, 1]",
            "shape": ndvi.shape,
        },
    }


def calculate_ndwi(zip_file_path: Path, bbox: Optional[list] = None) -> Optional[Dict]:
    """Calculate NDWI (Normalized Difference Water Index).

    WHAT IS NDWI:
    NDWI enhances water features and suppresses vegetation and soil.
    It's used to detect and monitor water bodies, wetlands, and soil moisture.

    WHY IT WORKS:
    - Water strongly absorbs NIR light (appears dark in NIR)
    - Water reflects green light (why water looks blue/green)
    - This creates strong contrast between water and land

    FORMULA:
    NDWI = (Green - NIR) / (Green + NIR)

    For Sentinel-2:
    - Green = Band 3 (B03, 560nm wavelength, 10m resolution)
    - NIR = Band 8 (B08, 842nm wavelength, 10m resolution)

    VALUE INTERPRETATION:
    - > 0.3: Water bodies (lakes, rivers, ocean)
    - 0.0 to 0.3: Wetlands, moist soil
    - -0.3 to 0.0: Dry soil, sparse vegetation
    - < -0.3: Dense vegetation, built-up areas

    APPLICATIONS:
    - Water body mapping
    - Flood extent monitoring
    - Wetland detection
    - Irrigation monitoring
    - Drought assessment

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bbox: Optional bounding box to crop result

    Returns:
        Dictionary with 'ndwi' array and metadata
    """
    # Extract Green band (B03)
    green_result = _extract_band(zip_file_path, "B03", bbox)
    if green_result is None:
        print("Failed to extract Green band (B03)")
        return None
    green, bounds_wgs84 = green_result

    # Extract NIR band (B08)
    nir_result = _extract_band(zip_file_path, "B08", bbox)
    if nir_result is None:
        print("Failed to extract NIR band (B08)")
        return None
    nir, _ = nir_result

    # Resample if shapes don't match
    if green.shape != nir.shape:
        from scipy.ndimage import zoom

        zoom_factor = (nir.shape[0] / green.shape[0], nir.shape[1] / green.shape[1])
        green = zoom(green, zoom_factor, order=1)

    # Calculate NDWI
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi = np.clip(ndwi, -1, 1)

    return {
        "ndwi": ndwi,
        "bounds_wgs84": bounds_wgs84,
        "metadata": {
            "index": "NDWI",
            "formula": "(Green - NIR) / (Green + NIR)",
            "bands_used": ["B03 (Green)", "B08 (NIR)"],
            "range": "[-1, 1]",
            "shape": ndwi.shape,
        },
    }


def calculate_evi(zip_file_path: Path, bbox: Optional[list] = None) -> Optional[Dict]:
    """Calculate EVI (Enhanced Vegetation Index).

    WHAT IS EVI:
    EVI is an improved version of NDVI that:
    - Reduces atmospheric effects (haze, aerosols)
    - Reduces soil background effects
    - Is more sensitive in high biomass regions

    WHY IT'S BETTER THAN NDVI:
    - NDVI saturates in dense vegetation (all values near 1.0)
    - EVI continues to increase with vegetation density
    - EVI works better in tropical regions with dense canopy

    FORMULA:
    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

    For Sentinel-2:
    - NIR = Band 8 (B08, 842nm)
    - Red = Band 4 (B04, 665nm)
    - Blue = Band 2 (B02, 490nm)

    VALUE INTERPRETATION:
    - < 0.2: Bare soil, water, snow
    - 0.2 to 0.4: Sparse vegetation
    - 0.4 to 0.6: Moderate vegetation
    - > 0.6: Dense vegetation

    APPLICATIONS:
    - Tropical forest monitoring
    - High biomass vegetation studies
    - Agricultural monitoring in dense crops
    - Global vegetation monitoring (MODIS EVI product)

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bbox: Optional bounding box to crop result

    Returns:
        Dictionary with 'evi' array and metadata
    """
    # Extract required bands
    nir_result = _extract_band(zip_file_path, "B08", bbox)
    red_result = _extract_band(zip_file_path, "B04", bbox)
    blue_result = _extract_band(zip_file_path, "B02", bbox)

    if nir_result is None or red_result is None or blue_result is None:
        print("Failed to extract required bands for EVI")
        return None

    nir, bounds_wgs84 = nir_result
    red, _ = red_result
    blue, _ = blue_result

    # Resample if shapes don't match
    if red.shape != nir.shape:
        from scipy.ndimage import zoom

        zoom_factor = (nir.shape[0] / red.shape[0], nir.shape[1] / red.shape[1])
        red = zoom(red, zoom_factor, order=1)
    if blue.shape != nir.shape:
        from scipy.ndimage import zoom

        zoom_factor = (nir.shape[0] / blue.shape[0], nir.shape[1] / blue.shape[1])
        blue = zoom(blue, zoom_factor, order=1)

    # Calculate EVI with standard coefficients
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)

    # EVI typically ranges from -1 to 1
    evi = np.clip(evi, -1, 1)

    return {
        "evi": evi,
        "bounds_wgs84": bounds_wgs84,
        "metadata": {
            "index": "EVI",
            "formula": "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)",
            "bands_used": ["B08 (NIR)", "B04 (Red)", "B02 (Blue)"],
            "range": "[-1, 1]",
            "shape": evi.shape,
        },
    }


def calculate_savi(
    zip_file_path: Path, L: float = 0.5, bbox: Optional[list] = None
) -> Optional[Dict]:
    """Calculate SAVI (Soil Adjusted Vegetation Index).

    WHAT IS SAVI:
    SAVI minimizes soil brightness influences on vegetation indices.
    It's particularly useful in areas with sparse vegetation where
    soil background significantly affects the signal.

    WHY IT'S NEEDED:
    - NDVI is affected by soil color and brightness
    - In sparse vegetation, soil dominates the pixel
    - SAVI adds a soil brightness correction factor (L)

    FORMULA:
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    For Sentinel-2:
    - NIR = Band 8 (B08, 842nm)
    - Red = Band 4 (B04, 665nm)
    - L = Soil brightness correction factor

    L PARAMETER:
    - L = 0: Equivalent to NDVI (high vegetation cover)
    - L = 0.5: Intermediate (default, works for most cases)
    - L = 1: Maximum soil adjustment (very sparse vegetation)

    VALUE INTERPRETATION:
    Similar to NDVI but less affected by soil:
    - < 0.2: Bare soil, water
    - 0.2 to 0.4: Sparse vegetation
    - 0.4 to 0.6: Moderate vegetation
    - > 0.6: Dense vegetation

    APPLICATIONS:
    - Arid and semi-arid regions
    - Early crop growth monitoring
    - Rangeland assessment
    - Areas with exposed soil

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        L: Soil brightness correction factor (0 to 1, default 0.5)
        bbox: Optional bounding box to crop result

    Returns:
        Dictionary with 'savi' array and metadata
    """
    # Extract required bands
    nir_result = _extract_band(zip_file_path, "B08", bbox)
    red_result = _extract_band(zip_file_path, "B04", bbox)

    if nir_result is None or red_result is None:
        print("Failed to extract required bands for SAVI")
        return None

    nir, bounds_wgs84 = nir_result
    red, _ = red_result

    # Resample if shapes don't match
    if red.shape != nir.shape:
        from scipy.ndimage import zoom

        zoom_factor = (nir.shape[0] / red.shape[0], nir.shape[1] / red.shape[1])
        red = zoom(red, zoom_factor, order=1)

    # Calculate SAVI with soil adjustment factor
    savi = ((nir - red) / (nir + red + L + 1e-8)) * (1 + L)
    savi = np.clip(savi, -1, 1)

    return {
        "savi": savi,
        "bounds_wgs84": bounds_wgs84,
        "metadata": {
            "index": "SAVI",
            "formula": f"((NIR - Red) / (NIR + Red + {L})) * (1 + {L})",
            "bands_used": ["B08 (NIR)", "B04 (Red)"],
            "L_factor": L,
            "range": "[-1, 1]",
            "shape": savi.shape,
        },
    }


def calculate_nbr(zip_file_path: Path, bbox: Optional[list] = None) -> Optional[Dict]:
    """Calculate NBR (Normalized Burn Ratio).

    WHAT IS NBR:
    NBR is designed to highlight burned areas and assess burn severity.
    It uses the difference between NIR and SWIR bands.

    WHY IT WORKS:
    - Healthy vegetation: High NIR reflectance, low SWIR reflectance
    - Burned areas: Low NIR reflectance, high SWIR reflectance
    - This creates strong contrast between burned and unburned areas

    FORMULA:
    NBR = (NIR - SWIR) / (NIR + SWIR)

    For Sentinel-2:
    - NIR = Band 8 (B08, 842nm, 10m resolution)
    - SWIR = Band 12 (B12, 2190nm, 20m resolution)

    VALUE INTERPRETATION:
    - > 0.4: Healthy vegetation
    - 0.1 to 0.4: Moderate vegetation
    - -0.1 to 0.1: Recently burned or bare soil
    - < -0.1: Severely burned areas

    BURN SEVERITY (using dNBR = pre-fire NBR - post-fire NBR):
    - dNBR > 0.66: High severity
    - dNBR 0.44-0.66: Moderate-high severity
    - dNBR 0.27-0.44: Moderate-low severity
    - dNBR 0.10-0.27: Low severity
    - dNBR < 0.10: Unburned

    APPLICATIONS:
    - Wildfire mapping
    - Burn severity assessment
    - Post-fire recovery monitoring
    - Fire damage estimation

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bbox: Optional bounding box to crop result

    Returns:
        Dictionary with 'nbr' array and metadata
    """
    # Extract NIR band (B08)
    nir_result = _extract_band(zip_file_path, "B08", bbox)
    if nir_result is None:
        print("Failed to extract NIR band (B08)")
        return None
    nir, bounds_wgs84 = nir_result

    # Extract SWIR band (B12)
    swir_result = _extract_band(zip_file_path, "B12", bbox)
    if swir_result is None:
        print("Failed to extract SWIR band (B12)")
        return None
    swir, _ = swir_result

    # Resample SWIR to match NIR resolution if needed
    if swir.shape != nir.shape:
        from scipy.ndimage import zoom

        zoom_factor = (nir.shape[0] / swir.shape[0], nir.shape[1] / swir.shape[1])
        swir = zoom(swir, zoom_factor, order=1)  # Bilinear interpolation

    # Calculate NBR
    nbr = (nir - swir) / (nir + swir + 1e-8)
    nbr = np.clip(nbr, -1, 1)

    return {
        "nbr": nbr,
        "bounds_wgs84": bounds_wgs84,
        "metadata": {
            "index": "NBR",
            "formula": "(NIR - SWIR) / (NIR + SWIR)",
            "bands_used": ["B08 (NIR)", "B12 (SWIR)"],
            "range": "[-1, 1]",
            "shape": nbr.shape,
        },
    }
