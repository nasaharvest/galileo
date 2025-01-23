# https://github.com/nasaharvest/openmapflow/blob/main/openmapflow/ee_exporter.py
import json
import os
import shutil
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any, List, Optional, Union

import ee
import numpy as np
import pandas as pd
import requests
from pandas.compat._optional import import_optional_dependency
from tqdm import tqdm

from ..config import (
    DAYS_PER_TIMESTEP,
    EE_BUCKET_TIFS,
    EE_FOLDER_TIFS,
    EE_PROJECT,
    END_YEAR,
    EXPORTED_HEIGHT_WIDTH_METRES,
    START_YEAR,
    TIFS_FOLDER,
)
from .dynamic_world import (
    DW_BANDS,
    DW_DIV_VALUES,
    DW_SHIFT_VALUES,
    get_single_dw_image,
)
from .ee_bbox import EEBoundingBox
from .era5 import ERA5_BANDS, ERA5_DIV_VALUES, ERA5_SHIFT_VALUES, get_single_era5_image
from .landscan import (
    LANDSCAN_BANDS,
    LANDSCAN_DIV_VALUES,
    LANDSCAN_SHIFT_VALUES,
    get_single_landscan_image,
)
from .s1 import (
    S1_BANDS,
    S1_DIV_VALUES,
    S1_SHIFT_VALUES,
    get_s1_image_collection,
    get_single_s1_image,
)
from .s2 import S2_BANDS, S2_DIV_VALUES, S2_SHIFT_VALUES, get_single_s2_image
from .srtm import SRTM_BANDS, SRTM_DIV_VALUES, SRTM_SHIFT_VALUES, get_single_srtm_image
from .terraclimate import (
    TC_BANDS,
    TC_DIV_VALUES,
    TC_SHIFT_VALUES,
    get_single_terraclimate_image,
    get_terraclim_image_collection,
)
from .viirs import VIIRS_BANDS, VIIRS_DIV_VALUES, VIIRS_SHIFT_VALUES, get_single_viirs_image
from .worldcereal import WC_BANDS, WC_DIV_VALUES, WC_SHIFT_VALUES, get_single_wc_image

# dataframe constants when exporting the labels
LAT = "lat"
LON = "lon"
START_DATE = date(START_YEAR, 1, 1)
END_DATE = date(END_YEAR, 12, 31)

TIME_IMAGE_FUNCTIONS = [
    get_single_s2_image,
    get_single_era5_image,
    "terraclim",
    get_single_viirs_image,
]
SPACE_TIME_BANDS = S1_BANDS + S2_BANDS
SPACE_TIME_SHIFT_VALUES = np.array(S1_SHIFT_VALUES + S2_SHIFT_VALUES)
SPACE_TIME_DIV_VALUES = np.array(S1_DIV_VALUES + S2_DIV_VALUES)

TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
TIME_SHIFT_VALUES = np.array(ERA5_SHIFT_VALUES + TC_SHIFT_VALUES + VIIRS_SHIFT_VALUES)
TIME_DIV_VALUES = np.array(ERA5_DIV_VALUES + TC_DIV_VALUES + VIIRS_DIV_VALUES)

ALL_DYNAMIC_IN_TIME_BANDS = SPACE_TIME_BANDS + TIME_BANDS

SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
SPACE_IMAGE_FUNCTIONS = [get_single_srtm_image, get_single_dw_image, get_single_wc_image]
SPACE_SHIFT_VALUES = np.array(SRTM_SHIFT_VALUES + DW_SHIFT_VALUES + WC_SHIFT_VALUES)
SPACE_DIV_VALUES = np.array(SRTM_DIV_VALUES + DW_DIV_VALUES + WC_DIV_VALUES)

STATIC_IMAGE_FUNCTIONS = [get_single_landscan_image]
# we will add latlons in dataset.py function
LOCATION_BANDS = ["x", "y", "z"]
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS
STATIC_SHIFT_VALUES = np.array(LANDSCAN_SHIFT_VALUES + [0, 0, 0])
STATIC_DIV_VALUES = np.array(LANDSCAN_DIV_VALUES + [1, 1, 1])

GeoJsonType = dict[str, Any]


def get_ee_task_list(key: str = "description") -> List[str]:
    """Gets a list of all active tasks in the EE task list."""
    task_list = ee.data.getTaskList()
    return [
        task[key]
        for task in tqdm(task_list, desc="Loading Earth Engine tasks")
        if task["state"] in ["READY", "RUNNING", "FAILED"]
    ]


def get_ee_task_amount(prefix: Optional[str] = None) -> int:
    """
    Gets amount of active tasks in Earth Engine.
    Args:
        prefix: Prefix to filter tasks.
    Returns:
        Amount of active tasks.
    """
    ee_prefix = None if prefix is None else ee_safe_str(prefix)
    amount = 0
    task_list = ee.data.getTaskList()
    for t in tqdm(task_list):
        valid_state = t["state"] in ["READY", "RUNNING"]
        if valid_state and (ee_prefix is None or t["description"].startswith(ee_prefix)):
            amount += 1
    return amount


def get_cloud_tif_list(
    dest_bucket: str, prefix: str = EE_FOLDER_TIFS, region: str = "us-central1"
) -> List[str]:
    """Gets a list of all cloud-free TIFs in a bucket."""
    storage = import_optional_dependency("google.cloud.storage")
    cloud_tif_list_iterator = storage.Client().list_blobs(dest_bucket, prefix=prefix)
    try:
        tif_list = [
            blob.name
            for blob in tqdm(cloud_tif_list_iterator, desc="Loading tifs already on Google Cloud")
        ]
    except Exception as e:
        raise Exception(
            f"{e}\nPlease create the Google Cloud bucket: {dest_bucket}"
            + f"\nCommand: gsutil mb -l {region} gs://{dest_bucket}"
        )
    print(f"Found {len(tif_list)} already exported tifs")
    return tif_list


def make_combine_bands_function(bands: List[str]):
    def combine_bands(current, previous):
        # Transforms an Image Collection with 1 band per Image into a single
        # Image with items as bands
        # Author: Jamie Vleeshouwer

        # Rename the band
        previous = ee.Image(previous)
        current = current.select(bands)
        # Append it to the result (Note: only return current item on first
        # element/iteration)
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(previous, None),
            current,
            previous.addBands(ee.Image(current)),
        )

    return combine_bands


def ee_safe_str(s: str):
    """Earth Engine descriptions only allow certain characters"""
    return s.replace(".", "-").replace("=", "-").replace("/", "-")[:100]


def create_ee_image(
    polygon: ee.Geometry,
    start_date: date,
    end_date: date,
    days_per_timestep: int = DAYS_PER_TIMESTEP,
) -> ee.Image:
    """
    Returns an ee.Image which we can then export.
    This image will contain S1, S2, ERA5 and Dynamic World data
    between start_date and end_date, in intervals of
    days_per_timestep. Each timestep will be a different channel in the
    image (e.g. if I have 3 timesteps, then I'll have VV, VV_1, VV_2 for the
    S1 VV bands). The static in time SRTM bands will also be in the image.
    """
    image_collection_list: List[ee.Image] = []
    cur_date = start_date
    cur_end_date = cur_date + timedelta(days=days_per_timestep)

    # We get all the S1 images in an exaggerated date range. We do this because
    # S1 data is sparser, so we will pull from outside the days_per_timestep
    # range if we are missing data within that range
    vv_imcol, vh_imcol = get_s1_image_collection(
        polygon, start_date - timedelta(days=31), end_date + timedelta(days=31)
    )
    tc_imcol = get_terraclim_image_collection(polygon, start_date, end_date)

    while cur_end_date <= end_date:
        image_list: List[ee.Image] = []

        # first, the S1 image which gets the entire s1 collection
        image_list.append(
            get_single_s1_image(
                region=polygon,
                start_date=cur_date,
                end_date=cur_end_date,
                vv_imcol=vv_imcol,
                vh_imcol=vh_imcol,
            )
        )
        for image_function in TIME_IMAGE_FUNCTIONS:
            if image_function == "terraclim":
                image_list.append(
                    get_single_terraclimate_image(
                        region=polygon,
                        start_date=cur_date,
                        end_date=cur_end_date,
                        tc_imcol=tc_imcol,
                    )
                )
            else:
                assert callable(image_function)
                image_list.append(
                    image_function(region=polygon, start_date=cur_date, end_date=cur_end_date)
                )

        image_collection_list.append(ee.Image.cat(image_list))
        cur_date += timedelta(days=days_per_timestep)
        cur_end_date += timedelta(days=days_per_timestep)

    # now, we want to take our image collection and append the bands into a single image
    imcoll = ee.ImageCollection(image_collection_list)
    combine_bands_function = make_combine_bands_function(ALL_DYNAMIC_IN_TIME_BANDS)
    img = ee.Image(imcoll.iterate(combine_bands_function))

    # finally, we add the static in time images
    total_image_list: List[ee.Image] = [img]
    for space_image_function in SPACE_IMAGE_FUNCTIONS:
        total_image_list.append(
            space_image_function(
                region=polygon,
                start_date=start_date - timedelta(days=31),
                end_date=end_date + timedelta(days=31),
            )
        )
    for static_image_function in STATIC_IMAGE_FUNCTIONS:
        total_image_list.append(
            static_image_function(
                region=polygon,
                start_date=start_date - timedelta(days=31),
                end_date=end_date + timedelta(days=31),
            )
        )

    return ee.Image.cat(total_image_list)


def get_ee_credentials():
    gcp_sa_key = os.environ.get("GCP_SA_KEY")
    if gcp_sa_key is not None:
        gcp_sa_email = json.loads(gcp_sa_key)["client_email"]
        print(f"Logging into EarthEngine with {gcp_sa_email}")
        return ee.ServiceAccountCredentials(gcp_sa_email, key_data=gcp_sa_key)
    else:
        print("Logging into EarthEngine with default credentials")
        return "persistent"


class EarthEngineExporter:
    """
    Export satellite data from Earth engine. It's called using the following
    script:
    ```
    from src.data import EarthEngineExporter
    EarthEngineExporter(dest_bucket="bucket_name").export_for_labels(df)
    ```
    :param check_ee: Whether to check Earth Engine before exporting
    :param check_gcp: Whether to check Google Cloud Storage before exporting,
        google-cloud-storage must be installed.
    :param credentials: The credentials to use for the export. If not specified,
        the default credentials will be used
    :param dest_bucket: The bucket to export to, google-cloud-storage must be installed.
    """

    def __init__(
        self,
        dest_bucket: str = EE_BUCKET_TIFS,
        check_ee: bool = False,
        check_gcp: bool = False,
        credentials=None,
        mode: str = "batch",
        local_tifs_folder: Path = TIFS_FOLDER,
        gcloud_tifs_folder: str = EE_FOLDER_TIFS,
    ) -> None:
        assert mode in ["batch", "url"]
        self.mode = mode
        self.local_tifs_folder = local_tifs_folder
        self.gcloud_tifs_folder = gcloud_tifs_folder
        if mode == "url":
            print(
                f"Mode: url. Files will be saved to {self.local_tifs_folder} and rsynced to google cloud"
            )
        self.surrounding_metres = EXPORTED_HEIGHT_WIDTH_METRES / 2
        self.dest_bucket = dest_bucket
        initialize_args = {
            "credentials": credentials if credentials else get_ee_credentials(),
            "project": EE_PROJECT,
        }
        if mode == "url":
            initialize_args["opt_url"] = "https://earthengine-highvolume.googleapis.com"
        ee.Initialize(**initialize_args)
        self.check_ee = check_ee
        self.ee_task_list = get_ee_task_list() if self.check_ee else []
        self.check_gcp = check_gcp
        self.cloud_tif_list = (
            get_cloud_tif_list(dest_bucket, self.gcloud_tifs_folder) if self.check_gcp else []
        )
        self.local_tif_list = [x.name for x in self.local_tifs_folder.glob("*.tif*")]

    def sync_local_and_gcloud(self):
        os.system(
            f"gcloud storage rsync -r {self.local_tifs_folder} gs://{self.dest_bucket}/{self.gcloud_tifs_folder}"
        )

    def _export_for_polygon(
        self,
        polygon: ee.Geometry,
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        file_dimensions: Optional[int] = None,
    ) -> bool:
        cloud_filename = f"{self.gcloud_tifs_folder}/{str(polygon_identifier)}"
        local_filename = f"{str(polygon_identifier).replace('/', '_')}.tif"

        # Description of the export cannot contain certrain characters
        description = ee_safe_str(cloud_filename)

        if f"{cloud_filename}.tif" in self.cloud_tif_list:
            # checks that we haven't already exported this file
            print(f"{cloud_filename}.tif already in cloud_tif_files")
            return False

        if local_filename in self.local_tif_list:
            # checks that we haven't already exported this file
            print(f"{local_filename} already in local_tif_files, but not in the cloud")
            return False

        # Check if task is already started in EarthEngine
        if description in self.ee_task_list:
            print(f"{description} already in ee task list")
            return False

        if len(self.ee_task_list) >= 3000:
            # we can only have 3000 running exports at once
            print("3000 exports started")
            return False

        img = create_ee_image(polygon, start_date, end_date)

        if self.mode == "batch":
            try:
                ee.batch.Export.image.toCloudStorage(
                    bucket=self.dest_bucket,
                    fileNamePrefix=cloud_filename,
                    image=img.clip(polygon),
                    description=description,
                    scale=10,
                    region=polygon,
                    maxPixels=1e13,
                    fileDimensions=file_dimensions,
                ).start()
                self.ee_task_list.append(description)
            except ee.ee_exception.EEException as e:
                print(f"Task not started! Got exception {e}")
                return False
        elif self.mode == "url":
            try:
                url = img.getDownloadURL(
                    {
                        "region": polygon,
                        "scale": 10,
                        "filePerBand": False,
                        "format": "GEO_TIFF",
                    }
                )
                r = requests.get(url, stream=True)
            except ee.ee_exception.EEException as e:
                print(f"Task not started! Got exception {e}", flush=True)
                return False
            if r.status_code != 200:
                print(f"Task failed with status {r.status_code}", flush=True)
                return False
            else:
                local_path = Path(self.local_tifs_folder / local_filename)
                with local_path.open("wb") as f:
                    shutil.copyfileobj(r.raw, f)
        return True

    def export_for_latlons(
        self,
        latlons: pd.DataFrame,
        num_exports_to_start: int = 3000,
    ) -> None:
        """
        Export boxes with length and width EXPORTED_HEIGHT_WIDTH_METRES
        for the points in latlons (where latlons is a dataframe with
        the columns "lat" and "lon")
        """
        for expected_column in [LAT, LON]:
            assert expected_column in latlons

        exports_started = 0
        print(f"Exporting {len(latlons)} latlons: ")

        for _, row in tqdm(latlons.iterrows(), desc="Exporting", total=len(latlons)):
            ee_bbox = EEBoundingBox.from_centre(
                # worldstrat points are strings
                mid_lat=float(row[LAT]),
                mid_lon=float(row[LON]),
                surrounding_metres=int(self.surrounding_metres),
            )

            export_started = self._export_for_polygon(
                polygon=ee_bbox.to_ee_polygon(),
                polygon_identifier=ee_bbox.get_identifier(START_DATE, END_DATE),
                start_date=START_DATE,
                end_date=END_DATE,
            )
            if export_started:
                exports_started += 1
                if num_exports_to_start is not None and exports_started >= num_exports_to_start:
                    print(f"Started {exports_started} exports. Ending export")
                    return None
        if self.mode == "url":
            print("Export finished. Syncing to google cloud")
            self.sync_local_and_gcloud()
            print("Finished sync")

    def export_for_bbox(
        self,
        bbox: EEBoundingBox,
        start_date: date,
        end_date: date,
        identifier: Optional[str] = None,
    ):
        polygon_identifier = (
            identifier if identifier is not None else bbox.get_identifier(start_date, end_date)
        )

        self._export_for_geometry(
            geometry=bbox.to_ee_polygon(),
            start_date=start_date,
            end_date=end_date,
            identifier=polygon_identifier,
        )

    def export_for_geo_json(
        self,
        geo_json: GeoJsonType,
        start_date: date,
        end_date: date,
        identifier: str,
    ):
        self._export_for_geometry(
            geometry=ee.Geometry(geo_json),
            start_date=start_date,
            end_date=end_date,
            identifier=identifier,
        )

    def _export_for_geometry(
        self,
        geometry: ee.Geometry,
        start_date: date,
        end_date: date,
        identifier: str,
    ):
        if self.mode == "url":
            warnings.warn(
                "Downloading a polygon in url mode. "
                "It is likely you will come up against GEE's filesize limits; "
                "batch mode may be better"
            )

        export_started = self._export_for_polygon(
            polygon=geometry,
            polygon_identifier=identifier,
            start_date=start_date,
            end_date=end_date,
        )
        if not export_started:
            warnings.warn("Export failed")

        if self.mode == "url":
            print("Export finished. Syncing to google cloud")
            self.sync_local_and_gcloud()
            print("Finished sync")
