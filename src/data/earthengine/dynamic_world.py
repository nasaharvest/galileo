from datetime import date

import ee

ORIGINAL_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_BANDS = [f"DW_{band}" for band in ORIGINAL_BANDS]
DW_SHIFT_VALUES = [0] * len(DW_BANDS)
DW_DIV_VALUES = [1] * len(DW_BANDS)


def get_single_dw_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    dw_collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(ee.DateRange(str(start_date), str(end_date)))
        .select(ORIGINAL_BANDS, DW_BANDS)
        .mean()
    )

    return dw_collection
