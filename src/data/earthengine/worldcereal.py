from datetime import date

import ee

ORIGINAL_BANDS = [
    "temporarycrops",
    "maize",
    "wintercereals",
    "springcereals",
    "irrigation",
]

WC_BANDS = [f"WC_{band}" for band in ORIGINAL_BANDS]
WC_SHIFT_VALUES = [0] * len(WC_BANDS)
WC_DIV_VALUES = [100] * len(WC_BANDS)


def get_single_wc_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    # we start by getting all the data for the range
    wc_collection = ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
    composite = (
        ee.Image.cat(
            [
                wc_collection.filter(f"product == '{product}'")
                .mosaic()
                .unmask(0)
                .select("classification")
                .rename(product)
                for product in ORIGINAL_BANDS
            ]
        )
        .clip(region)
        .toDouble()  # type: ignore
    )

    return composite
