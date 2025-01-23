import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Sequence, Tuple, cast

import h5py
import numpy as np
import torch
from einops import repeat
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset, default_collate
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from ...data.dataset import (
    LOCATION_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    Normalizer,
    to_cartesian,
)
from ...galileo import Encoder
from ...masking import MaskedOutput
from ...utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .bands import BANDS
from .columns import NullableColumns, RequiredColumns
from .datasets import CropHarvest, Task, TestInstance
from .datasets import CropHarvestLabels as OrgCropHarvestLabels
from .utils import NoDataForBoundingBoxError, memoized

logger = logging.getLogger("__main__")


cropharvest_data_dir = data_dir / "cropharvest_data"

CH_BANDS_TO_SPACE_TIME_BANDS = [BANDS.index(s) for s in SPACE_TIME_BANDS if s in BANDS]
SPACE_TIME_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(SPACE_TIME_BANDS) if s in BANDS]

CH_BANDS_TO_SPACE_BANDS = [BANDS.index(s) for s in SPACE_BANDS if s in BANDS]
SPACE_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(SPACE_BANDS) if s in BANDS]

CH_BANDS_TO_TIME_BANDS = [BANDS.index(s) for s in TIME_BANDS if s in BANDS]
TIME_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(TIME_BANDS) if s in BANDS]

LOCATION_BAND_MAPPING = [idx for idx, x in enumerate(STATIC_BANDS) if x in LOCATION_BANDS]


@dataclass
class Hyperparams:
    batch_size: int = 16
    num_workers: int = 0


class CropHarvestLabels(OrgCropHarvestLabels):
    def construct_fao_classification_labels(
        self, task: Task, filter_test: bool = True
    ) -> List[Tuple[Path, int]]:
        gpdf = self.as_geojson()
        if filter_test:
            gpdf = gpdf[gpdf[RequiredColumns.IS_TEST] == False]  # noqa
        if task.bounding_box is not None:
            gpdf = self.filter_geojson(
                gpdf, task.bounding_box, task.include_externally_contributed_labels
            )

        # This should probably be a required column since it has no
        # None values (and shouldn't have any)
        gpdf = gpdf[~gpdf[NullableColumns.CLASSIFICATION_LABEL].isnull()]

        if len(gpdf) == 0:
            raise NoDataForBoundingBoxError

        ys = gpdf[NullableColumns.CLASSIFICATION_LABEL]
        paths = self._dataframe_to_paths(gpdf)

        return [(path, y) for path, y in zip(paths, ys) if path.exists()]


class MultiClassCropHarvest(TorchDataset):
    def __init__(
        self,
        paths_and_y: List[Tuple[Tuple[Path, Path], str]],
        y_string_to_int: Dict[str, int],
    ):
        self.paths_and_y = paths_and_y
        self.y_string_to_int = y_string_to_int

    def __len__(self) -> int:
        return len(self.paths_and_y)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, int]:
        path, y = self.paths_and_y[index]
        satellite_data = h5py.File(path, "r")
        lat = satellite_data.attrs["instance_lat"]
        lon = satellite_data.attrs["instance_lon"]
        return (
            satellite_data.get("array")[:],
            np.array([lat, lon]),
            self.y_string_to_int[y],
        )

    def as_array(
        self, flatten_x: bool = False, num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if num_samples is not None:
            raise NotImplementedError
        indices_to_sample = list(range(len(self)))
        X, latlons, Y = zip(*[self[i] for i in indices_to_sample])
        X_np, latlon_np, y_np = np.stack(X), np.stack(latlons), np.stack(Y)

        if flatten_x:
            X_np = self._flatten_array(X_np)
        return X_np, latlon_np, y_np

    @staticmethod
    def _flatten_array(array: np.ndarray) -> np.ndarray:
        return array.reshape(array.shape[0], -1)


@memoized
def get_eval_datasets():
    return CropHarvest.create_benchmark_datasets(
        root=cropharvest_data_dir, balance_negative_crops=False, normalize=False
    )


def download_cropharvest_data(root_name: str = ""):
    root = Path(root_name) if root_name != "" else cropharvest_data_dir
    if not root.exists():
        root.mkdir()
        CropHarvest(root, download=True)


def model_class_name(model: BaseEstimator) -> str:
    return model.__class__.__name__


class CropHarvestEvalBase:
    """
    Data is automatically downloaded by this class
    """

    start_month = 1
    num_timesteps: Optional[int] = None

    all_classification_sklearn_models = ["LogisticRegression"]

    def __init__(
        self,
        name: str,
        normalizer: Normalizer,
        include_latlons: bool = True,
        seed: int = DEFAULT_SEED,
        ignore_band_groups: Optional[List[str]] = None,
    ):
        self.include_latlons = include_latlons
        self.name = f"{name}{'_latlons' if include_latlons else ''}"
        self.normalizer = normalizer
        self.seed = seed
        self.patch_size = 1

        self.ignore_band_groups = ignore_band_groups
        self.ignore_s_t_band_groups = self.indices_of_ignored(
            SPACE_TIME_BANDS_GROUPS_IDX, ignore_band_groups
        )
        self.ignore_sp_band_groups = self.indices_of_ignored(
            SPACE_BAND_GROUPS_IDX, ignore_band_groups
        )
        self.ignore_t_band_groups = self.indices_of_ignored(
            TIME_BAND_GROUPS_IDX, ignore_band_groups
        )
        self.ignore_st_band_groups = self.indices_of_ignored(
            STATIC_BAND_GROUPS_IDX, ignore_band_groups
        )

    @staticmethod
    def truncate_timesteps(x, num_timesteps: Optional[int]):
        if (num_timesteps is None) or (x is None):
            return x
        else:
            return x[:, :num_timesteps]

    @staticmethod
    def indices_of_ignored(band_groups: OrderedDict, ignore_band_groups: Optional[List]):
        # copied from src.masking with minor modifications. I think thats ok for now
        if ignore_band_groups is None:
            return []
        else:
            ignored_band_indices = []
            for idx, (band, _) in enumerate(band_groups.items()):
                if band in ignore_band_groups:
                    ignored_band_indices.append(idx)
            return ignored_band_indices

    def cropharvest_array_to_normalized_galileo(
        self,
        array: np.ndarray,
        latlons: np.ndarray,
        start_month: int,
        timesteps: Optional[int] = None,
    ):
        array = self.truncate_timesteps(array, timesteps)
        b, t, _ = array.shape

        s_t_x = np.zeros((b, t, len(SPACE_TIME_BANDS)))
        s_t_x[:, :, SPACE_TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_TIME_BANDS]
        s_t_x = repeat(s_t_x, "b t d -> b h w t d", h=1, w=1)
        s_t_m = np.ones((b, 1, 1, t, len(SPACE_TIME_BANDS)))
        s_t_m[:, :, :, :, SPACE_TIME_BANDS_TO_CH_BANDS] = 0
        # turn it from band-space into band-group-space
        s_t_m = s_t_m[:, :, :, :, [g[0] for _, g in SPACE_TIME_BANDS_GROUPS_IDX.items()]]
        # and mask out bands which have been ignored during model training
        s_t_m[:, :, :, :, self.ignore_s_t_band_groups] = 1

        sp_x = np.zeros((b, t, len(SPACE_BANDS)))
        sp_x[:, :, SPACE_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_BANDS]
        sp_x = repeat(sp_x[:, 0], "b d -> b h w d", h=1, w=1)
        sp_m = np.ones((b, 1, 1, len(SPACE_BANDS)))
        sp_m[:, :, :, SPACE_BANDS_TO_CH_BANDS] = 0
        sp_m = sp_m[:, :, :, [g[0] for _, g in SPACE_BAND_GROUPS_IDX.items()]]
        # and mask out bands which have been ignored during model training
        sp_m[:, :, :, self.ignore_sp_band_groups] = 1

        t_x = np.zeros((b, t, len(TIME_BANDS)))
        t_x[:, :, TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_TIME_BANDS]
        t_m = np.ones((b, t, len(TIME_BANDS)))
        t_m[:, :, TIME_BANDS_TO_CH_BANDS] = 0
        t_m = t_m[:, :, [g[0] for _, g in TIME_BAND_GROUPS_IDX.items()]]
        # and mask out bands which have been ignored during model training
        t_m[:, :, self.ignore_t_band_groups] = 1

        st_x = np.zeros((b, len(STATIC_BANDS)))
        st_m = np.ones((b, len(STATIC_BAND_GROUPS_IDX)))
        if self.include_latlons:
            st_m[:, list(STATIC_BAND_GROUPS_IDX).index("location")] = 0
            st_x[:, LOCATION_BAND_MAPPING] = to_cartesian(latlons[:, 0], latlons[:, 1])
        # mask out bands which have been ignored during model training
        st_m[:, self.ignore_st_band_groups] = 1

        months = np.fmod(np.arange(start_month - 1, start_month - 1 + t), 12)
        months = repeat(months, "t -> b t", b=b)

        return masked_output_np_to_tensor(
            self.normalizer(s_t_x),
            self.normalizer(sp_x),
            self.normalizer(t_x),
            self.normalizer(st_x),
            s_t_m,
            sp_m,
            t_m,
            st_m,
            months,
        )

    @staticmethod
    def collate_fn(batch):
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, label = default_collate(batch)
        return MaskedOutput(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months), label

    @torch.no_grad()
    def train_sklearn_model(
        self,
        train_dl: DataLoader,
        pretrained_model: Encoder,
        models: List[str] = ["Random Forest"],
    ) -> Sequence[BaseEstimator]:
        """
        Fit sklearn models on the encodings of the pretrained model.
        For spatial token prediction tasks, encodings and targets are grouped token-wise.
        Either the mode class will be computed or the normalized counts of each class per token.
        This is controlled by the output_mode attribute which can be changed in the subclass.
        """

        for model_mode in models:
            assert model_mode in self.all_classification_sklearn_models
        pretrained_model.eval()

        encodings_list, targets_list = [], []

        for masked_output, label in tqdm(train_dl, desc="Computing encodings for sklearn"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]

            targets_list.append(label.cpu().numpy())

            with torch.no_grad():
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                    s_t_x=s_t_x,
                    sp_x=sp_x,
                    t_x=t_x,
                    st_x=st_x,
                    s_t_m=s_t_m,
                    sp_m=sp_m,
                    t_m=t_m,
                    st_m=st_m,
                    months=months,
                    patch_size=self.patch_size,
                )
                encodings_list.append(
                    pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                    .cpu()
                    .numpy()
                )

        targets_np = np.concatenate(targets_list)
        encodings_np = np.concatenate(encodings_list)

        if len(targets_np.shape) == 2 and targets_np.shape[1] == 1:
            # from [[0], [0], [1]] to [0, 0, 1]
            targets_np = targets_np.ravel()

        fit_models = []
        model_dict = {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=self.seed
            )
        }
        for model in models:
            fit_models.append(clone(model_dict[model]).fit(encodings_np, targets_np))
        return fit_models


class BinaryCropHarvestEval(CropHarvestEvalBase):
    num_outputs = 1

    country_to_sizes: Dict[str, List] = {
        "Kenya": [20, 32, 64, 96, 128, 160, 192, 224, 256, None],
        "Togo": [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, None],
    }

    def __init__(
        self,
        country: str,
        normalizer: Normalizer,
        num_timesteps: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        include_latlons: bool = True,
        eval_mode: str = "test",
        ignore_band_groups: Optional[List[str]] = None,
    ):
        if eval_mode == "val":
            assert country in list(self.country_to_sizes.keys())
        self.eval_mode = eval_mode
        suffix = f"_{sample_size}" if sample_size else ""
        suffix = f"{suffix}_{num_timesteps}" if num_timesteps is not None else suffix
        super().__init__(
            name=f"CropHarvest_{country}{suffix}",
            normalizer=normalizer,
            include_latlons=include_latlons,
            seed=seed,
            ignore_band_groups=ignore_band_groups,
        )

        download_cropharvest_data()

        evaluation_datasets = get_eval_datasets()
        evaluation_datasets = [d for d in evaluation_datasets if country in d.id]
        assert len(evaluation_datasets) == 1
        self.dataset: CropHarvest = evaluation_datasets[0]
        assert self.dataset.task.normalize is False
        self.num_timesteps = num_timesteps
        self.sample_size = sample_size

    @torch.no_grad()
    def _evaluate_model(
        self,
        pretrained_model: Encoder,
        sklearn_model: BaseEstimator,
    ) -> Dict:
        pretrained_model.eval()
        with tempfile.TemporaryDirectory() as results_dir:
            for test_id, test_instance in self.dataset.test_data(max_size=10000):
                savepath = Path(results_dir) / f"{test_id}.nc"

                latlons = np.stack((test_instance.lats, test_instance.lons), axis=-1)
                masked_output = self.cropharvest_array_to_normalized_galileo(
                    cast(np.ndarray, test_instance.x),
                    latlons,
                    self.start_month,
                    self.num_timesteps,
                )
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                    t.to(device) for t in masked_output
                ]
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                    s_t_x=s_t_x,
                    sp_x=sp_x,
                    t_x=t_x,
                    st_x=st_x,
                    s_t_m=s_t_m,
                    sp_m=sp_m,
                    t_m=t_m,
                    st_m=st_m,
                    months=months,
                    patch_size=self.patch_size,
                )
                encodings = (
                    pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                    .cpu()
                    .numpy()
                )
                preds = sklearn_model.predict_proba(encodings)[:, 1]
                ds = test_instance.to_xarray(preds)
                ds.to_netcdf(savepath)

            all_nc_files = list(Path(results_dir).glob("*.nc"))
            combined_instance, combined_preds = TestInstance.load_from_nc(all_nc_files)
            combined_results = combined_instance.evaluate_predictions(combined_preds)

        prefix = sklearn_model.__class__.__name__
        return {f"{self.name}: {prefix}_{key}": val for key, val in combined_results.items()}

    @torch.no_grad()
    def _validate_model(
        self,
        val_dl,
        pretrained_model: Encoder,
        sklearn_model: BaseEstimator,
    ) -> Dict:
        pretrained_model.eval()
        label_list, pred_list = [], []
        for masked_output, label in tqdm(val_dl, desc="Computing encodings for sklearn"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                s_t_x=s_t_x,
                sp_x=sp_x,
                t_x=t_x,
                st_x=st_x,
                s_t_m=s_t_m,
                sp_m=sp_m,
                t_m=t_m,
                st_m=st_m,
                months=months,
                patch_size=self.patch_size,
            )
            encodings = (
                pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                .cpu()
                .numpy()
            )
            preds = sklearn_model.predict_proba(encodings)[:, 1]
            pred_list.append(preds)
            label_list.append(label)

        preds_np = np.concatenate(pred_list)
        labels_np = np.concatenate(label_list)
        combined_results = {
            "auc_roc": roc_auc_score(labels_np, preds_np),
            "f1_score": f1_score(labels_np, preds_np > 0.5),
        }
        prefix = sklearn_model.__class__.__name__
        return {f"{self.name}: {prefix}_{key}": val for key, val in combined_results.items()}

    @staticmethod
    def random_subset(
        array: np.ndarray, latlons: np.ndarray, labels: np.ndarray, fraction: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if fraction is not None:
            num_samples = int(array.shape[0] * fraction)
        else:
            num_samples = array.shape[0]
        return shuffle(array, latlons, labels, random_state=DEFAULT_SEED, n_samples=num_samples)

    def evaluate_model_on_task(
        self,
        pretrained_model: Encoder,
        model_modes: Optional[List[str]] = None,
        fraction: Optional[float] = None,
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert model_mode in self.all_classification_sklearn_models

        array, latlons, labels = self.dataset.as_array()
        if self.eval_mode == "val":
            array, val_array, latlons, val_latlons, labels, val_labels = train_test_split(
                array, latlons, labels, test_size=0.5, random_state=DEFAULT_SEED, stratify=labels
            )
            array, latlons, labels = self.random_subset(array, latlons, labels, fraction=fraction)
            val_array, val_latlons, val_labels = self.random_subset(
                val_array, val_latlons, val_labels, fraction=fraction
            )
            val_dl = DataLoader(
                TensorDataset(
                    *self.cropharvest_array_to_normalized_galileo(
                        val_array,
                        val_latlons,
                        timesteps=self.num_timesteps,
                        start_month=self.start_month,
                    ),
                    torch.from_numpy(val_labels),
                ),
                batch_size=Hyperparams.batch_size,
                shuffle=False,
                num_workers=Hyperparams.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            array, latlons, labels = self.random_subset(array, latlons, labels, fraction=fraction)
            val_dl = None

        train_dl = DataLoader(
            TensorDataset(
                *self.cropharvest_array_to_normalized_galileo(
                    array,
                    latlons,
                    timesteps=self.num_timesteps,
                    start_month=self.start_month,
                ),
                torch.from_numpy(labels),
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
            collate_fn=self.collate_fn,
        )
        trained_sklearn_models = self.train_sklearn_model(train_dl, pretrained_model, model_modes)
        results_dict = {}
        for sklearn_model in trained_sklearn_models:
            if self.eval_mode == "val":
                results_dict.update(self._validate_model(val_dl, pretrained_model, sklearn_model))
            else:
                results_dict.update(self._evaluate_model(pretrained_model, sklearn_model))
        return results_dict


class MultiClassCropHarvestEval(CropHarvestEvalBase):
    num_outputs = 10

    def __init__(
        self,
        normalizer: Normalizer,
        test_ratio: float = 0.2,
        n_per_class: Optional[int] = 100,
        seed: int = DEFAULT_SEED,
        include_latlons: bool = True,
        ignore_band_groups: Optional[List[str]] = None,
    ):
        name_suffix = f"_{n_per_class}" if n_per_class is not None else ""
        super().__init__(
            name=f"CropHarvest_multiclass_global{name_suffix}_{seed}",
            seed=seed,
            normalizer=normalizer,
            include_latlons=include_latlons,
            ignore_band_groups=ignore_band_groups,
        )

        download_cropharvest_data()
        task = Task(normalize=False)
        paths_and_y = CropHarvestLabels(cropharvest_data_dir).construct_fao_classification_labels(
            task, filter_test=True
        )

        y = [x[1] for x in paths_and_y]
        unique_ys = np.unique(y)
        y_string_to_int = {val: idx for idx, val in enumerate(np.unique(y))}

        train_paths_and_y, val_paths_and_y = train_test_split(
            paths_and_y, test_size=test_ratio, stratify=y, random_state=42
        )

        if n_per_class is not None:
            indices_to_keep = []
            y_train = np.array([x[1] for x in train_paths_and_y])
            for y_val in unique_ys:
                y_val_indices = np.where(y_train == y_val)[0]
                indices_to_keep.append(y_val_indices[:n_per_class])
            train_paths_and_y = [train_paths_and_y[i] for i in np.concatenate(indices_to_keep)]
            assert len(train_paths_and_y) <= n_per_class * len(unique_ys)

        array, latlons, labels = MultiClassCropHarvest(
            train_paths_and_y, y_string_to_int
        ).as_array()
        self.train_dataset = TensorDataset(
            *self.cropharvest_array_to_normalized_galileo(
                array, latlons, self.start_month, timesteps=self.num_timesteps
            ),
            torch.from_numpy(labels),
        )
        self.eval_dataset = MultiClassCropHarvest(val_paths_and_y, y_string_to_int)

    @torch.no_grad()
    def _evaluate_models(
        self,
        pretrained_model: Encoder,
        sklearn_models: Sequence[BaseEstimator],
    ) -> Dict:
        dl = DataLoader(
            self.eval_dataset,
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        test_true = []
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }
        for x, latlons, y in dl:
            masked_output = self.cropharvest_array_to_normalized_galileo(
                x, latlons, self.start_month, self.num_timesteps
            )
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                s_t_x=s_t_x,
                sp_x=sp_x,
                t_x=t_x,
                st_x=st_x,
                s_t_m=s_t_m,
                sp_m=sp_m,
                t_m=t_m,
                st_m=st_m,
                months=months,
                patch_size=self.patch_size,
            )
            encodings = (
                pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                .cpu()
                .numpy()
            )
            for model in sklearn_models:
                pred_dict[model_class_name(model)].append(model.predict(encodings))
            test_true.append(y)

        test_true_np = np.concatenate(test_true)
        results_dict = {}

        for model_name_str, pred_list in pred_dict.items():
            test_preds_np = np.concatenate(pred_list, axis=0)
            prefix = f"{model_name_str}"
            results_dict.update(
                {
                    f"{self.name}: {prefix}_num_samples": len(test_true_np),
                    f"{self.name}: {prefix}_f1_score": f1_score(
                        test_true_np, test_preds_np, average="weighted"
                    ),
                }
            )
        return results_dict

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert model_mode in self.all_classification_sklearn_models

        train_dl = DataLoader(
            self.train_dataset,
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
            collate_fn=self.collate_fn,
        )
        trained_sklearn_models = self.train_sklearn_model(train_dl, pretrained_model, model_modes)
        return self._evaluate_models(pretrained_model, trained_sklearn_models)
