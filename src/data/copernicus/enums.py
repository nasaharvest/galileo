"""Enums for Copernicus Sentinel-1 and Sentinel-2 parameters.

This module provides type-safe enums for common Sentinel parameters,
improving API ergonomics and preventing typos in band names and product types.
"""

from enum import Enum


class S2Band(str, Enum):
    """Sentinel-2 band identifiers.

    Sentinel-2 has 13 spectral bands covering visible, near-infrared, and
    short-wave infrared wavelengths at 10m, 20m, and 60m spatial resolutions.

    Common band combinations:
    - True color RGB: RED, GREEN, BLUE (B04, B03, B02)
    - False color (vegetation): NIR, RED, GREEN (B08, B04, B03)
    - Agriculture: SWIR1, NIR, BLUE (B11, B08, B02)
    """

    # 10m resolution bands (highest detail)
    BLUE = "B02"  # 490nm - Blue, good for water/atmosphere
    GREEN = "B03"  # 560nm - Green, peak vegetation reflectance
    RED = "B04"  # 665nm - Red, chlorophyll absorption
    NIR = "B08"  # 842nm - Near-infrared, vegetation structure

    # 20m resolution bands
    RED_EDGE_1 = "B05"  # 705nm - Vegetation red edge
    RED_EDGE_2 = "B06"  # 740nm - Vegetation red edge
    RED_EDGE_3 = "B07"  # 783nm - Vegetation red edge
    NIR_NARROW = "B8A"  # 865nm - Narrow NIR for water vapor
    SWIR1 = "B11"  # 1610nm - Snow/ice/cloud discrimination
    SWIR2 = "B12"  # 2190nm - Improved snow/ice/cloud, vegetation moisture

    # 60m resolution bands (atmospheric/cirrus)
    COASTAL_AEROSOL = "B01"  # 443nm - Coastal and aerosol studies
    WATER_VAPOR = "B09"  # 945nm - Water vapor absorption
    CIRRUS = "B10"  # 1375nm - Cirrus cloud detection

    @classmethod
    def rgb_bands(cls) -> list[str]:
        """Get standard RGB bands for true color composite."""
        return [cls.RED.value, cls.GREEN.value, cls.BLUE.value]

    @classmethod
    def false_color_bands(cls) -> list[str]:
        """Get false color bands for vegetation analysis (NIR-R-G)."""
        return [cls.NIR.value, cls.RED.value, cls.GREEN.value]

    @classmethod
    def all_10m_bands(cls) -> list[str]:
        """Get all 10m resolution bands."""
        return [cls.BLUE.value, cls.GREEN.value, cls.RED.value, cls.NIR.value]


class S1ProductType(str, Enum):
    """Sentinel-1 product types.

    Different processing levels for SAR data, from raw to geocoded products.
    """

    GRD = "GRD"  # Ground Range Detected - most common, preprocessed and geocoded
    SLC = "SLC"  # Single Look Complex - raw data in slant range geometry
    OCN = "OCN"  # Ocean products - specialized for ocean wind/wave analysis


class S1Polarization(str, Enum):
    """Sentinel-1 radar polarization modes.

    Polarization refers to the orientation of the radar wave's electric field.
    Different polarizations are sensitive to different surface properties.
    """

    VV = "VV"  # Vertical transmit, Vertical receive - good for water, urban
    VH = "VH"  # Vertical transmit, Horizontal receive - good for vegetation
    HH = "HH"  # Horizontal transmit, Horizontal receive - less common
    HV = "HV"  # Horizontal transmit, Vertical receive - less common

    @classmethod
    def dual_pol_vv_vh(cls) -> str:
        """Get dual polarization string for VV+VH (most common for land)."""
        return f"{cls.VV.value},{cls.VH.value}"

    @classmethod
    def dual_pol_hh_hv(cls) -> str:
        """Get dual polarization string for HH+HV."""
        return f"{cls.HH.value},{cls.HV.value}"


class S1AcquisitionMode(str, Enum):
    """Sentinel-1 acquisition modes.

    Different imaging modes with different swath widths and resolutions.
    """

    IW = "IW"  # Interferometric Wide swath - most common, 250km swath
    EW = "EW"  # Extra Wide swath - 400km swath, lower resolution
    SM = "SM"  # Stripmap - high resolution, narrow swath
    WV = "WV"  # Wave mode - ocean wave spectra


class S1OrbitDirection(str, Enum):
    """Sentinel-1 orbit direction.

    The satellite's orbit direction affects viewing geometry and shadow patterns.
    """

    ASCENDING = "ASCENDING"  # Moving south to north (evening pass)
    DESCENDING = "DESCENDING"  # Moving north to south (morning pass)
