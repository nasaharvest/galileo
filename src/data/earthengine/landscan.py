import warnings
from datetime import date

import ee

LANDSCAN_BANDS = ["b1"]
# LANDSCAN values range from approximately 0 to 185000 in 2022: https://code.earthengine.google.com/?scriptPath=users/sat-io/awesome-gee-catalog-examples:population-socioeconomics/LANDSCAN-GLOBAL
LANDSCAN_SHIFT_VALUES = [92500]
LANDSCAN_DIV_VALUES = [92500]
image_collection = "projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL"

landscan_minmum_date = date(2022, 1, 1)


def get_single_landscan_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    if start_date > landscan_minmum_date:
        warnings.warn(
            f"Minimum landscan date is later than {landscan_minmum_date}. "
            f"The exported data will be from {landscan_minmum_date.year}"
        )
        start_date = landscan_minmum_date
    ls_collection = (
        ee.ImageCollection("projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL")
        .filterDate(ee.DateRange(str(start_date), str(end_date)))
        .select(LANDSCAN_BANDS)
        # reduce to a single image
        .mean()
        # unmask ocean values
        .unmask(0)
    )
    return ls_collection.toDouble()
