from datetime import date

import ee

image_collection = "USGS/SRTMGL1_003"
SRTM_BANDS = ["elevation", "slope"]
# visually gauged 90th percentile from
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]


def get_single_srtm_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    elevation = ee.Image(image_collection).clip(region).select(SRTM_BANDS[0])
    slope = ee.Terrain.slope(elevation)  # type: ignore
    together = ee.Image.cat([elevation, slope]).toDouble()  # type: ignore

    return together
