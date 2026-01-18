"""Spectral indices calculation for Sentinel-2 data.

This module provides functions to calculate common vegetation and water indices
from Sentinel-2 multispectral imagery. These indices enhance specific features
and are widely used in remote sensing applications.
"""

import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio


def _extract_band(zip_file_path: Path, band_name: str) -> Optional[np.ndarray]:
    """Extract a single band from Sentinel-2 ZIP file.

    Helper function to read individual spectral bands.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        band_name: Band identifier (e.g., 'B04', 'B08', 'B11')

    Returns:
        Band data as numpy array, or None if extraction fails
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            safe_dirs = list(temp_path.glob("*.SAFE"))
            if not safe_dirs:
                return None

            safe_dir = safe_dirs[0]
            img_data_dir = safe_dir / "GRANULE"
            granule_dirs = list(img_data_dir.glob("*"))

            if not granule_dirs:
                return None

            granule_dir = granule_dirs[0]
            img_dir = granule_dir / "IMG_DATA"

            # Try multiple naming patterns
            patterns = [
                f"*_{band_name}_10m.jp2",
                f"*_{band_name}_20m.jp2",
                f"*_{band_name}.jp2",
                f"*{band_name}.jp2",
            ]

            for pattern in patterns:
                band_matches = list(img_dir.glob(pattern))
                if band_matches:
                    with rasterio.open(band_matches[0]) as src:
                        return src.read(1).astype(np.float32)

            return None

    except Exception as e:
        print(f"Error extracting band {band_name}: {e}")
        return None


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

    Example:
        >>> ndvi_data = calculate_ndvi(s2_file)
        >>> ndvi = ndvi_data['ndvi']
        >>> print(f"NDVI range: {ndvi.min():.2f} to {ndvi.max():.2f}")
        >>> print(f"Mean vegetation: {ndvi.mean():.2f}")
        >>> # Classify vegetation density
        >>> dense_veg = (ndvi > 0.5).sum() / ndvi.size * 100
        >>> print(f"Dense vegetation: {dense_veg:.1f}% of area")
    """
    # Extract NIR band (B08)
    # NIR = Near-Infrared, wavelength ~842nm, 10m resolution
    # Healthy plants reflect ~50% of NIR light to avoid overheating
    nir = _extract_band(zip_file_path, "B08")
    if nir is None:
        print("Failed to extract NIR band (B08)")
        return None

    # Extract Red band (B04)
    # Red light, wavelength ~665nm, 10m resolution
    # Plants absorb red light for photosynthesis (chlorophyll absorption peak)
    red = _extract_band(zip_file_path, "B04")
    if red is None:
        print("Failed to extract Red band (B04)")
        return None

    # Calculate NDVI using normalized difference
    # Normalization (dividing by sum) keeps values between -1 and 1
    # This makes NDVI comparable across different sensors and dates
    # Add small epsilon to avoid division by zero
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Clip to valid range (handles numerical errors)
    ndvi = np.clip(ndvi, -1, 1)

    return {
        "ndvi": ndvi,
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

    Example:
        >>> ndwi_data = calculate_ndwi(s2_file)
        >>> ndwi = ndwi_data['ndwi']
        >>> # Identify water pixels
        >>> water_mask = ndwi > 0.3
        >>> water_area_pct = water_mask.sum() / water_mask.size * 100
        >>> print(f"Water coverage: {water_area_pct:.1f}%")
    """
    # Extract Green band (B03)
    green = _extract_band(zip_file_path, "B03")
    if green is None:
        print("Failed to extract Green band (B03)")
        return None

    # Extract NIR band (B08)
    nir = _extract_band(zip_file_path, "B08")
    if nir is None:
        print("Failed to extract NIR band (B08)")
        return None

    # Calculate NDWI
    # Note: Formula is (Green - NIR), opposite of NDVI
    # This makes water positive and vegetation negative
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi = np.clip(ndwi, -1, 1)

    return {
        "ndwi": ndwi,
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
    nir = _extract_band(zip_file_path, "B08")
    red = _extract_band(zip_file_path, "B04")
    blue = _extract_band(zip_file_path, "B02")

    if nir is None or red is None or blue is None:
        print("Failed to extract required bands for EVI")
        return None

    # Calculate EVI with standard coefficients
    # Coefficients: G=2.5, C1=6, C2=7.5, L=1
    # These are empirically derived for optimal performance
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)

    # EVI typically ranges from -1 to 1, but can exceed in rare cases
    evi = np.clip(evi, -1, 1)

    return {
        "evi": evi,
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
    nir = _extract_band(zip_file_path, "B08")
    red = _extract_band(zip_file_path, "B04")

    if nir is None or red is None:
        print("Failed to extract required bands for SAVI")
        return None

    # Calculate SAVI with soil adjustment factor
    savi = ((nir - red) / (nir + red + L + 1e-8)) * (1 + L)
    savi = np.clip(savi, -1, 1)

    return {
        "savi": savi,
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

    Example:
        >>> # Calculate NBR for pre-fire and post-fire images
        >>> nbr_pre = calculate_nbr(s2_file_before_fire)
        >>> nbr_post = calculate_nbr(s2_file_after_fire)
        >>> # Calculate difference (dNBR) to assess burn severity
        >>> dnbr = nbr_pre['nbr'] - nbr_post['nbr']
        >>> high_severity = (dnbr > 0.66).sum() / dnbr.size * 100
        >>> print(f"High severity burn: {high_severity:.1f}% of area")
    """
    # Extract NIR band (B08)
    nir = _extract_band(zip_file_path, "B08")
    if nir is None:
        print("Failed to extract NIR band (B08)")
        return None

    # Extract SWIR band (B12)
    # SWIR = Shortwave Infrared, wavelength ~2190nm, 20m resolution
    # Sensitive to moisture content and burned areas
    swir = _extract_band(zip_file_path, "B12")
    if swir is None:
        print("Failed to extract SWIR band (B12)")
        return None

    # Resample SWIR to match NIR resolution if needed
    # B12 is 20m, B08 is 10m - need to upsample B12
    if swir.shape != nir.shape:
        from scipy.ndimage import zoom

        zoom_factor = (nir.shape[0] / swir.shape[0], nir.shape[1] / swir.shape[1])
        swir = zoom(swir, zoom_factor, order=1)  # Bilinear interpolation

    # Calculate NBR
    nbr = (nir - swir) / (nir + swir + 1e-8)
    nbr = np.clip(nbr, -1, 1)

    return {
        "nbr": nbr,
        "metadata": {
            "index": "NBR",
            "formula": "(NIR - SWIR) / (NIR + SWIR)",
            "bands_used": ["B08 (NIR)", "B12 (SWIR)"],
            "range": "[-1, 1]",
            "shape": nbr.shape,
        },
    }
