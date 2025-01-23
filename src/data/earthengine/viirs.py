import warnings
from datetime import date

import ee

from .utils import get_monthly_data

VIIRS_URL = (
    "https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG"
)
image_collection = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
VIIRS_BANDS = ["avg_rad"]
VIIRS_SHIFT_VALUES = [0.0]
# visually checked - this seems much more reasonable than
# the GEE estimate
VIIRS_DIV_VALUES = [100]

# last date on GEE is 2024-06-04 on 28 October 2024
LATEST_START_DATE = date(2024, 5, 4)


def get_single_viirs_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    if (start_date.year == 2023) & (start_date.month == 10):
        # for some reason, VIIRS data for October 2023 is missing
        # so we replace it with November 2023 data
        start_date = date(start_date.year, 11, 1)
    elif start_date > LATEST_START_DATE:
        warnings.warn(
            f"No data for {start_date} (check {VIIRS_URL} to see if this can be updated). "
            f"Defaulting to latest date of {LATEST_START_DATE} checked on October 28 2024"
        )
        start_date = LATEST_START_DATE

    return get_monthly_data(image_collection, VIIRS_BANDS, region, start_date, unmask=True)
