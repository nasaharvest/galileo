from datetime import date, timedelta
from typing import List, Union

import ee


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")


def get_closest_dates(mid_date: date, imcol: ee.ImageCollection) -> ee.ImageCollection:
    fifteen_days_in_ms = 1296000000

    mid_date_ee = ee.Date(date_to_string(mid_date))
    # first, order by distance from mid_date
    from_mid_date = imcol.map(
        lambda image: image.set(
            "dateDist",
            ee.Number(image.get("system:time_start"))
            .subtract(mid_date_ee.millis())  # type: ignore
            .abs(),
        )
    )
    from_mid_date = from_mid_date.sort("dateDist", opt_ascending=True)

    # no matter what, we take the first element in the image collection
    # and we add 1 to ensure the less_than condition triggers
    max_diff = ee.Number(from_mid_date.first().get("dateDist")).max(  # type: ignore
        ee.Number(fifteen_days_in_ms)
    )

    kept_images = from_mid_date.filterMetadata("dateDist", "not_greater_than", max_diff)
    return kept_images


def get_monthly_data(
    collection: str, bands: List[str], region: ee.Geometry, start_date: date, unmask: bool = False
) -> ee.Image:
    # This only really works with the values currently in config.
    # What happens is that the images are associated with the first day of the month,
    # so if we just use the given start_date and end_date, then we will often get
    # the image from the following month (e.g. the start and end dates of
    # 2016-02-07, 2016-03-08 respectively return data from March 2016, even though
    # February 2016 has a much higher overlap). It also means that the final month
    # timestep, with range 2017-01-02 to 2017-02-01 was returning no data (but we'd)
    # like it to return data for January
    # TODO: in the future, this could be updated to an overlapping_month function, similar
    # to what happens with the Plant Village labels
    month, year = start_date.month, start_date.year
    start = date(year, month, 1)
    # first day of next month
    end = (date(year, month, 1) + timedelta(days=32)).replace(day=1)

    if (date.today().replace(day=1) - end) < timedelta(days=32):
        raise ValueError(
            f"Cannot get data for range {start} - {end}, please set an earlier end date"
        )
    dates = ee.DateRange(date_to_string(start), date_to_string(end))
    startDate = ee.DateRange(dates).start()  # type: ignore
    endDate = ee.DateRange(dates).end()  # type: ignore

    imcol = (
        ee.ImageCollection(collection)
        .filterDate(startDate, endDate)
        .filterBounds(region)
        .select(bands)
    )
    if unmask:
        imcol = imcol.map(lambda x: x.unmask(0))

    # there should only be one timestep per daterange, so a mean shouldn't change the values
    return imcol.mean().toDouble()
