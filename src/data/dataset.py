import json
import logging
import math
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from random import sample
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union, cast
from typing import OrderedDict as OrderedDictType

import h5py
import numpy as np
import rioxarray
import torch
import xarray as xr
from einops import rearrange, repeat
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from .config import (
    DATASET_OUTPUT_HW,
    EE_BUCKET_TIFS,
    EE_FOLDER_H5PYS,
    EE_FOLDER_TIFS,
    NUM_TIMESTEPS,
)
from .earthengine.eo import (
    ALL_DYNAMIC_IN_TIME_BANDS,
    DW_BANDS,
    DW_DIV_VALUES,
    DW_SHIFT_VALUES,
    ERA5_BANDS,
    LANDSCAN_BANDS,
    LOCATION_BANDS,
    S1_BANDS,
    SPACE_BANDS,
    SPACE_DIV_VALUES,
    SPACE_SHIFT_VALUES,
    SRTM_BANDS,
    TC_BANDS,
    TIME_BANDS,
    TIME_DIV_VALUES,
    TIME_SHIFT_VALUES,
    VIIRS_BANDS,
    WC_BANDS,
    WC_DIV_VALUES,
    WC_SHIFT_VALUES,
)
from .earthengine.eo import SPACE_TIME_BANDS as EO_SPACE_TIME_BANDS
from .earthengine.eo import SPACE_TIME_DIV_VALUES as EO_SPACE_TIME_DIV_VALUES
from .earthengine.eo import SPACE_TIME_SHIFT_VALUES as EO_SPACE_TIME_SHIFT_VALUES
from .earthengine.eo import STATIC_BANDS as EO_STATIC_BANDS
from .earthengine.eo import STATIC_DIV_VALUES as EO_STATIC_DIV_VALUES
from .earthengine.eo import STATIC_SHIFT_VALUES as EO_STATIC_SHIFT_VALUES

logger = logging.getLogger("__main__")

EO_DYNAMIC_IN_TIME_BANDS_NP = np.array(EO_SPACE_TIME_BANDS + TIME_BANDS)
SPACE_TIME_BANDS = EO_SPACE_TIME_BANDS + ["NDVI"]
SPACE_TIME_SHIFT_VALUES = np.append(EO_SPACE_TIME_SHIFT_VALUES, [0])
SPACE_TIME_DIV_VALUES = np.append(EO_SPACE_TIME_DIV_VALUES, [1])

SPACE_TIME_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

TIME_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
        "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
        "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]
STATIC_BANDS = EO_STATIC_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS
STATIC_DIV_VALUES = np.append(EO_STATIC_DIV_VALUES, (DW_DIV_VALUES + WC_DIV_VALUES))
STATIC_SHIFT_VALUES = np.append(EO_STATIC_SHIFT_VALUES, (DW_SHIFT_VALUES + WC_SHIFT_VALUES))

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
        "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
        "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
    }
)


# if this changes the normalizer will need to index against something else
assert len(SPACE_TIME_BANDS) != len(SPACE_BANDS) != len(TIME_BANDS) != len(STATIC_BANDS)


class Normalizer:
    # these are the bands we will replace with the 2*std computation
    # if std = True
    std_bands: Dict[int, list] = {
        len(SPACE_TIME_BANDS): [b for b in SPACE_TIME_BANDS if b != "NDVI"],
        len(SPACE_BANDS): SRTM_BANDS,
        len(TIME_BANDS): TIME_BANDS,
        len(STATIC_BANDS): LANDSCAN_BANDS,
    }

    def __init__(
        self, std: bool = True, normalizing_dicts: Optional[Dict] = None, std_multiplier: float = 2
    ):
        self.shift_div_dict = {
            len(SPACE_TIME_BANDS): {
                "shift": deepcopy(SPACE_TIME_SHIFT_VALUES),
                "div": deepcopy(SPACE_TIME_DIV_VALUES),
            },
            len(SPACE_BANDS): {
                "shift": deepcopy(SPACE_SHIFT_VALUES),
                "div": deepcopy(SPACE_DIV_VALUES),
            },
            len(TIME_BANDS): {
                "shift": deepcopy(TIME_SHIFT_VALUES),
                "div": deepcopy(TIME_DIV_VALUES),
            },
            len(STATIC_BANDS): {
                "shift": deepcopy(STATIC_SHIFT_VALUES),
                "div": deepcopy(STATIC_DIV_VALUES),
            },
        }
        self.normalizing_dicts = normalizing_dicts
        if std:
            name_to_bands = {
                len(SPACE_TIME_BANDS): SPACE_TIME_BANDS,
                len(SPACE_BANDS): SPACE_BANDS,
                len(TIME_BANDS): TIME_BANDS,
                len(STATIC_BANDS): STATIC_BANDS,
            }
            assert normalizing_dicts is not None
            for key, val in normalizing_dicts.items():
                if isinstance(key, str):
                    continue
                bands_to_replace = self.std_bands[key]
                for band in bands_to_replace:
                    band_idx = name_to_bands[key].index(band)
                    mean = val["mean"][band_idx]
                    std = val["std"][band_idx]
                    min_value = mean - (std_multiplier * std)
                    max_value = mean + (std_multiplier * std)
                    div = max_value - min_value
                    if div == 0:
                        raise ValueError(f"{band} has div value of 0")
                    self.shift_div_dict[key]["shift"][band_idx] = min_value
                    self.shift_div_dict[key]["div"][band_idx] = div

    @staticmethod
    def _normalize(x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray) -> np.ndarray:
        x = (x - shift_values) / div_values
        return x

    def __call__(self, x: np.ndarray):
        div_values = self.shift_div_dict[x.shape[-1]]["div"]
        return self._normalize(x, self.shift_div_dict[x.shape[-1]]["shift"], div_values)


class DatasetOutput(NamedTuple):
    space_time_x: np.ndarray
    space_x: np.ndarray
    time_x: np.ndarray
    static_x: np.ndarray
    months: np.ndarray

    @classmethod
    def concatenate(cls, datasetoutputs: Sequence["DatasetOutput"]) -> "DatasetOutput":
        s_t_x = np.stack([o.space_time_x for o in datasetoutputs], axis=0)
        sp_x = np.stack([o.space_x for o in datasetoutputs], axis=0)
        t_x = np.stack([o.time_x for o in datasetoutputs], axis=0)
        st_x = np.stack([o.static_x for o in datasetoutputs], axis=0)
        months = np.stack([o.months for o in datasetoutputs], axis=0)
        return cls(s_t_x, sp_x, t_x, st_x, months)

    def normalize(self, normalizer: Optional[Normalizer]) -> "DatasetOutput":
        if normalizer is None:
            return self
        return DatasetOutput(
            normalizer(self.space_time_x).astype(np.half),
            normalizer(self.space_x).astype(np.half),
            normalizer(self.time_x).astype(np.half),
            normalizer(self.static_x).astype(np.half),
            self.months,
        )

    def in_pixel_batches(self, batch_size: int, window_size: int) -> Iterator["DatasetOutput"]:
        if self.space_time_x.shape[0] % window_size != 0:
            raise ValueError("DatasetOutput height must be divisible by the patch size")
        if self.space_time_x.shape[1] % window_size != 0:
            raise ValueError("DatasetOutput width must be divisible by the patch size")

        # how many batches from the height dimension, how many from the width dimension?
        h_b = self.space_time_x.shape[0] // window_size
        w_b = self.space_time_x.shape[1] // window_size
        flat_s_t_x = rearrange(
            self.space_time_x,
            "(h_b h) (w_b w) t d -> (h_b w_b) h w t d",
            h=window_size,
            w=window_size,
            h_b=h_b,
            w_b=w_b,
        )
        flat_sp_x = rearrange(
            self.space_x,
            "(h_b h) (w_b w) d -> (h_b w_b) h w d",
            h=window_size,
            w=window_size,
            h_b=h_b,
            w_b=w_b,
        )

        # static in space modalities will just get repeated per batch
        cur_idx = 0
        while cur_idx < flat_s_t_x.shape[0]:
            cur_idx_s_t_x = flat_s_t_x[cur_idx : cur_idx + batch_size].copy()
            b = cur_idx_s_t_x.shape[0]
            yield DatasetOutput(
                space_time_x=cur_idx_s_t_x,
                space_x=flat_sp_x[cur_idx : cur_idx + batch_size].copy(),
                time_x=repeat(self.time_x, "t d -> b t d", b=b),
                static_x=repeat(self.static_x, "d -> b d", b=b),
                months=repeat(self.months, "t -> b t", b=b),
            )
            cur_idx += batch_size


class ListOfDatasetOutputs(NamedTuple):
    space_time_x: List[np.ndarray]
    space_x: List[np.ndarray]
    time_x: List[np.ndarray]
    static_x: List[np.ndarray]
    months: List[np.ndarray]

    def to_datasetoutput(self) -> DatasetOutput:
        return DatasetOutput(
            np.stack(self.space_time_x, axis=0),
            np.stack(self.space_x, axis=0),
            np.stack(self.time_x, axis=0),
            np.stack(self.static_x, axis=0),
            np.stack(self.months, axis=0),
        )


def to_cartesian(
    lat: Union[float, np.ndarray, torch.Tensor], lon: Union[float, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(lat, float):
        assert -90 <= lat <= 90, f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        assert -180 <= lon <= 180, f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        assert isinstance(lon, float), f"Expected float got {type(lon)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return np.array([x, y, z])
    elif isinstance(lon, np.ndarray):
        assert -90 <= lat.min(), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert 90 >= lat.max(), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert -180 <= lon.min(), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert 180 >= lon.max(), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_np = np.cos(lat) * np.cos(lon)
        y_np = np.cos(lat) * np.sin(lon)
        z_np = np.sin(lat)
        return np.stack([x_np, y_np, z_np], axis=-1)
    elif isinstance(lon, torch.Tensor):
        assert -90 <= lat.min(), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert 90 >= lat.max(), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert -180 <= lon.min(), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert 180 >= lon.max(), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, torch.Tensor), f"Expected torch.Tensor got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_t = torch.cos(lat) * torch.cos(lon)
        y_t = torch.cos(lat) * torch.sin(lon)
        z_t = torch.sin(lat)
        return torch.stack([x_t, y_t, z_t], dim=-1)
    else:
        raise AssertionError(f"Unexpected input type {type(lon)}")


class Dataset(PyTorchDataset):
    def __init__(
        self,
        data_folder: Path,
        download: bool = True,
        h5py_folder: Optional[Path] = None,
        h5pys_only: bool = False,
        output_hw: int = DATASET_OUTPUT_HW,
        output_timesteps: int = NUM_TIMESTEPS,
        normalizer: Optional[Normalizer] = None,
    ):
        self.data_folder = data_folder
        self.h5pys_only = h5pys_only
        self.h5py_folder = h5py_folder
        self.cache = False
        self.normalizer = normalizer

        if h5py_folder is not None:
            self.cache = True
        if h5pys_only:
            assert h5py_folder is not None, "Can't use h5pys only if there is no cache folder"
            self.tifs: List[Path] = []
            if download:
                self.download_h5pys(h5py_folder)
            self.h5pys = list(h5py_folder.glob("*.h5"))
        else:
            if download:
                self.download_tifs(data_folder)
            self.tifs = []
            tifs = list(data_folder.glob("*.tif")) + list(data_folder.glob("*.tiff"))
            for tif in tifs:
                try:
                    _ = self.start_month_from_file(tif)
                    self.tifs.append(tif)
                except IndexError:
                    warnings.warn(f"IndexError for {tif}")
            self.h5pys = []

        self.output_hw = output_hw
        self.output_timesteps = output_timesteps

    def __len__(self) -> int:
        if self.h5pys_only:
            return len(self.h5pys)
        return len(self.tifs)

    @staticmethod
    def download_tifs(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/{EE_FOLDER_TIFS} {data_folder}")

    @staticmethod
    def download_h5pys(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/{EE_FOLDER_H5PYS} {data_folder}")

    @staticmethod
    def return_subset_indices(
        total_h,
        total_w,
        total_t,
        size: int,
        num_timesteps: int,
    ) -> Tuple[int, int, int]:
        """
        space_time_x: array of shape [H, W, T, D]
        space_x: array of shape [H, W, D]
        time_x: array of shape [T, D]
        static_x: array of shape [D]

        size must be greater or equal to H & W
        """
        possible_h = total_h - size
        possible_w = total_w - size
        assert (possible_h >= 0) & (possible_w >= 0)
        possible_t = total_t - num_timesteps
        assert possible_t >= 0

        if possible_h > 0:
            start_h = np.random.choice(possible_h)
        else:
            start_h = possible_h

        if possible_w > 0:
            start_w = np.random.choice(possible_w)
        else:
            start_w = possible_w

        if possible_t > 0:
            start_t = np.random.choice(possible_t)
        else:
            start_t = possible_t

        return start_h, start_w, start_t

    @staticmethod
    def subset_image(
        space_time_x: np.ndarray,
        space_x: np.ndarray,
        time_x: np.ndarray,
        static_x: np.ndarray,
        months: np.ndarray,
        size: int,
        num_timesteps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        space_time_x: array of shape [H, W, T, D]
        space_x: array of shape [H, W, D]
        time_x: array of shape [T, D]
        static_x: array of shape [D]

        size must be greater or equal to H & W
        """
        assert (space_time_x.shape[0] == space_x.shape[0]) & (
            space_time_x.shape[1] == space_x.shape[1]
        )
        assert space_time_x.shape[2] == time_x.shape[0]
        possible_h = space_time_x.shape[0] - size
        possible_w = space_time_x.shape[1] - size
        assert (possible_h >= 0) & (possible_w >= 0)
        possible_t = space_time_x.shape[2] - num_timesteps
        assert possible_t >= 0

        if possible_h > 0:
            start_h = np.random.choice(possible_h)
        else:
            start_h = possible_h

        if possible_w > 0:
            start_w = np.random.choice(possible_w)
        else:
            start_w = possible_w

        if possible_t > 0:
            start_t = np.random.choice(possible_t)
        else:
            start_t = possible_t

        return (
            space_time_x[
                start_h : start_h + size,
                start_w : start_w + size,
                start_t : start_t + num_timesteps,
            ],
            space_x[start_h : start_h + size, start_w : start_w + size],
            time_x[start_t : start_t + num_timesteps],
            static_x,
            months[start_t : start_t + num_timesteps],
        )

    @staticmethod
    def _fillna(data: np.ndarray, bands_np: np.ndarray):
        """Fill in the missing values in the data array"""
        if data.shape[-1] != len(bands_np):
            raise ValueError(f"Expected data to have {len(bands_np)} bands - got {data.shape[-1]}")
        is_nan_inf = np.isnan(data) | np.isinf(data)
        if not is_nan_inf.any():
            return data

        if len(data.shape) <= 2:
            return np.nan_to_num(data, nan=0)
        if len(data.shape) == 3:
            has_time = False
        elif len(data.shape) == 4:
            has_time = True
        else:
            raise ValueError(
                f"Expected data to be 3D or 4D (x, y, (time), band) - got {data.shape}"
            )

        # treat infinities as NaNs
        data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_per_time_band = np.nanmean(data, axis=(0, 1))  # t, b or b

        mean_per_time_band = np.nan_to_num(mean_per_time_band, nan=0, posinf=0, neginf=0)
        assert not (np.isnan(mean_per_time_band).any() | np.isinf(mean_per_time_band).any())

        if is_nan_inf.any():
            if has_time:
                means_to_fill = (
                    repeat(
                        np.nanmean(mean_per_time_band, axis=0),
                        "b -> h w t b",
                        h=data.shape[0],
                        w=data.shape[1],
                        t=data.shape[2],
                    )
                    * is_nan_inf
                )
            else:
                means_to_fill = (
                    repeat(mean_per_time_band, "b -> h w b", h=data.shape[0], w=data.shape[1])
                    * is_nan_inf
                )
            data = np.nan_to_num(data, nan=0, posinf=0, neginf=0) + means_to_fill
        return data

    def tif_to_h5py_path(self, tif_path: Path) -> Path:
        assert self.h5py_folder is not None
        tif_name = tif_path.stem
        return self.h5py_folder / f"{tif_name}.h5"

    @staticmethod
    def start_month_from_file(tif_path: Path) -> int:
        start_date = tif_path.name.partition("dates=")[2][:10]
        start_month = int(start_date.split("-")[1])
        return start_month

    @classmethod
    def month_array_from_file(cls, tif_path: Path, num_timesteps: int) -> np.ndarray:
        """
        Given a filepath and num_timesteps, extract start_month and return an array of
        months where months[idx] is the month for list(range(num_timesteps))[i]
        """
        # assumes all files are exported with filenames including:
        # *dates=<start_date>*, where the start_date is in a YYYY-MM-dd format
        start_month = cls.start_month_from_file(tif_path)
        # >>> np.fmod(np.array([9., 10, 11, 12, 13, 14]), 12)
        # array([ 9., 10., 11.,  0.,  1.,  2.])
        # - 1 because we want to index from 0
        return np.fmod(np.arange(start_month - 1, start_month - 1 + num_timesteps), 12)

    @classmethod
    def _tif_to_array(cls, tif_path: Path) -> DatasetOutput:
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)
            lon = np.mean(cast(np.ndarray, data.x)).item()
            lat = np.mean(cast(np.ndarray, data.y)).item()

        # this is a bit hackey but is a unique edge case for locations,
        # which are not part of the exported bands but are instead
        # computed here
        static_bands_in_tif = len(EO_STATIC_BANDS) - len(LOCATION_BANDS)

        num_timesteps = (values.shape[0] - len(SPACE_BANDS) - static_bands_in_tif) / len(
            ALL_DYNAMIC_IN_TIME_BANDS
        )
        assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
        dynamic_in_time_x = rearrange(
            values[: -(len(SPACE_BANDS) + static_bands_in_tif)],
            "(t c) h w -> h w t c",
            c=len(ALL_DYNAMIC_IN_TIME_BANDS),
            t=int(num_timesteps),
        )
        dynamic_in_time_x = cls._fillna(dynamic_in_time_x, EO_DYNAMIC_IN_TIME_BANDS_NP)
        space_time_x = dynamic_in_time_x[:, :, :, : -len(TIME_BANDS)]

        # calculate indices, which have shape [h, w, t, 1]
        ndvi = cls.calculate_ndi(space_time_x, band_1="B8", band_2="B4")

        space_time_x = np.concatenate((space_time_x, ndvi), axis=-1)

        time_x = dynamic_in_time_x[:, :, :, -len(TIME_BANDS) :]
        time_x = np.nanmean(time_x, axis=(0, 1))

        space_x = rearrange(
            values[-(len(SPACE_BANDS) + static_bands_in_tif) : -static_bands_in_tif],
            "c h w -> h w c",
        )
        space_x = cls._fillna(space_x, np.array(SPACE_BANDS))

        static_x = values[-static_bands_in_tif:]
        # add DW_STATIC and WC_STATIC
        dw_bands = space_x[:, :, [i for i, v in enumerate(SPACE_BANDS) if v in DW_BANDS]]
        wc_bands = space_x[:, :, [i for i, v in enumerate(SPACE_BANDS) if v in WC_BANDS]]
        static_x = np.concatenate(
            [
                np.nanmean(static_x, axis=(1, 2)),
                to_cartesian(lat, lon),
                np.nanmean(dw_bands, axis=(0, 1)),
                np.nanmean(wc_bands, axis=(0, 1)),
            ]
        )
        static_x = cls._fillna(static_x, np.array(STATIC_BANDS))

        months = cls.month_array_from_file(tif_path, int(num_timesteps))

        try:
            assert not np.isnan(space_time_x).any(), f"NaNs in s_t_x for {tif_path}"
            assert not np.isnan(space_x).any(), f"NaNs in sp_x for {tif_path}"
            assert not np.isnan(time_x).any(), f"NaNs in t_x for {tif_path}"
            assert not np.isnan(static_x).any(), f"NaNs in st_x for {tif_path}"
            assert not np.isinf(space_time_x).any(), f"Infs in s_t_x for {tif_path}"
            assert not np.isinf(space_x).any(), f"Infs in sp_x for {tif_path}"
            assert not np.isinf(time_x).any(), f"Infs in t_x for {tif_path}"
            assert not np.isinf(static_x).any(), f"Infs in st_x for {tif_path}"
            return DatasetOutput(
                space_time_x.astype(np.half),
                space_x.astype(np.half),
                time_x.astype(np.half),
                static_x.astype(np.half),
                months,
            )
        except AssertionError as e:
            raise e

    def _tif_to_array_with_checks(self, idx):
        tif_path = self.tifs[idx]
        try:
            output = self._tif_to_array(tif_path)
            return output
        except Exception as e:
            print(f"Replacing tif {tif_path} due to {e}")
            if idx == 0:
                new_idx = idx + 1
            else:
                new_idx = idx - 1
            self.tifs[idx] = self.tifs[new_idx]
            tif_path = self.tifs[idx]
        output = self._tif_to_array(tif_path)
        return output

    def load_tif(self, idx: int) -> DatasetOutput:
        if self.h5py_folder is None:
            s_t_x, sp_x, t_x, st_x, months = self._tif_to_array_with_checks(idx)
            return DatasetOutput(
                *self.subset_image(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    months,
                    size=self.output_hw,
                    num_timesteps=self.output_timesteps,
                )
            )
        else:
            h5py_path = self.tif_to_h5py_path(self.tifs[idx])
            if h5py_path.exists():
                try:
                    return self.read_and_slice_h5py_file(h5py_path)
                except Exception as e:
                    logger.warn(f"Exception {e} for {self.tifs[idx]}")
                    h5py_path.unlink()
                    s_t_x, sp_x, t_x, st_x, months = self._tif_to_array_with_checks(idx)
                    self.save_h5py(s_t_x, sp_x, t_x, st_x, self.tifs[idx].stem)
                    return DatasetOutput(
                        *self.subset_image(
                            s_t_x, sp_x, t_x, st_x, months, self.output_hw, self.output_timesteps
                        )
                    )
            else:
                s_t_x, sp_x, t_x, st_x, months = self._tif_to_array_with_checks(idx)
                self.save_h5py(s_t_x, sp_x, t_x, st_x, self.tifs[idx].stem)
                return DatasetOutput(
                    *self.subset_image(
                        s_t_x, sp_x, t_x, st_x, months, self.output_hw, self.output_timesteps
                    )
                )

    def save_h5py(self, s_t_x, sp_x, t_x, st_x, tif_stem):
        assert self.h5py_folder is not None
        with h5py.File(self.h5py_folder / f"{tif_stem}.h5", "w") as hf:
            hf.create_dataset("s_t_x", data=s_t_x)
            hf.create_dataset("sp_x", data=sp_x)
            hf.create_dataset("t_x", data=t_x)
            hf.create_dataset("st_x", data=st_x)

    @staticmethod
    def calculate_ndi(input_array: np.ndarray, band_1: str, band_2: str) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDI,
        (band_1 - band_2) / (band_1 + band_2)
        """
        band_1_np = input_array[:, :, :, EO_SPACE_TIME_BANDS.index(band_1)]
        band_2_np = input_array[:, :, :, EO_SPACE_TIME_BANDS.index(band_2)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            return np.expand_dims(
                np.where(
                    (band_1_np + band_2_np) > 0,
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),
                    0,
                ),
                -1,
            )

    def read_and_slice_h5py_file(self, h5py_path: Path):
        with h5py.File(h5py_path, "r") as hf:
            h, w, t, _ = hf["s_t_x"].shape
            start_h, start_w, start_t = self.return_subset_indices(
                h, w, t, self.output_hw, self.output_timesteps
            )
            months = self.month_array_from_file(h5py_path, t)
            output = DatasetOutput(
                hf["s_t_x"][
                    start_h : start_h + self.output_hw,
                    start_w : start_w + self.output_hw,
                    start_t : start_t + self.output_timesteps,
                ],
                hf["sp_x"][
                    start_h : start_h + self.output_hw,
                    start_w : start_w + self.output_hw,
                ],
                hf["t_x"][start_t : start_t + self.output_timesteps],
                hf["st_x"][:],
                months[start_t : start_t + self.output_timesteps],
            )
        return output

    def __getitem__(self, idx):
        if self.h5pys_only:
            return self.read_and_slice_h5py_file(self.h5pys[idx]).normalize(self.normalizer)
        else:
            return self.load_tif(idx).normalize(self.normalizer)

    def process_h5pys(self):
        # iterate through the dataset and save it all as h5pys
        assert self.h5py_folder is not None
        assert not self.h5pys_only
        assert self.cache

        for i in tqdm(range(len(self))):
            # loading the tifs also saves them
            # if they don't exist
            _ = self[i]

    @staticmethod
    def load_normalization_values(path: Path):
        if not path.exists():
            raise ValueError(f"No file found at path {path}")
        with path.open("r") as f:
            norm_dict = json.load(f)
        # we computed the normalizing dict using the same datset
        output_dict = {}
        for key, val in norm_dict.items():
            if "n" not in key:
                output_dict[int(key)] = val
            else:
                output_dict[key] = val
        return output_dict

    def compute_normalization_values(
        self,
        output_hw: int = 96,
        output_timesteps: int = 24,
        estimate_from: Optional[int] = 10000,
    ):
        org_hw = self.output_hw
        self.output_hw = output_hw

        org_t = self.output_timesteps
        self.output_timesteps = output_timesteps

        if estimate_from is not None:
            indices_to_sample = sample(list(range(len(self))), k=estimate_from)
        else:
            indices_to_sample = list(range(len(self)))

        output = ListOfDatasetOutputs([], [], [], [], [])
        for i in tqdm(indices_to_sample):
            s_t_x, sp_x, t_x, st_x, months = self[i]
            output.space_time_x.append(s_t_x.astype(np.float64))
            output.space_x.append(sp_x.astype(np.float64))
            output.time_x.append(t_x.astype(np.float64))
            output.static_x.append(st_x.astype(np.float64))
            output.months.append(months)
        d_o = output.to_datasetoutput()
        norm_dict = {
            "total_n": len(self),
            "sampled_n": len(indices_to_sample),
            len(SPACE_TIME_BANDS): {
                "mean": d_o.space_time_x.mean(axis=(0, 1, 2, 3)).tolist(),
                "std": d_o.space_time_x.std(axis=(0, 1, 2, 3)).tolist(),
            },
            len(SPACE_BANDS): {
                "mean": d_o.space_x.mean(axis=(0, 1, 2)).tolist(),
                "std": d_o.space_x.std(axis=(0, 1, 2)).tolist(),
            },
            len(TIME_BANDS): {
                "mean": d_o.time_x.mean(axis=(0, 1)).tolist(),
                "std": d_o.time_x.std(axis=(0, 1)).tolist(),
            },
            len(STATIC_BANDS): {
                "mean": d_o.static_x.mean(axis=0).tolist(),
                "std": d_o.static_x.std(axis=0).tolist(),
            },
        }

        self.output_hw = org_hw
        self.output_timesteps = org_t

        return norm_dict
