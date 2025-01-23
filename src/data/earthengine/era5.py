from datetime import date

import ee

from .utils import get_monthly_data

image_collection = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
# for temperature, shift to celcius and then divide by 35 based on notebook (ranges from)
# 37 to -22 degrees celcius
# For rainfall, based on
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]


def get_single_era5_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    return get_monthly_data(image_collection, ERA5_BANDS, region, start_date)
