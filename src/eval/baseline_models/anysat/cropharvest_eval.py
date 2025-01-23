import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from einops import repeat
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset

from src.eval.cropharvest.columns import NullableColumns, RequiredColumns
from src.eval.cropharvest.datasets import CropHarvest, Task, TestInstance
from src.eval.cropharvest.datasets import CropHarvestLabels as OrgCropHarvestLabels
from src.eval.cropharvest.utils import NoDataForBoundingBoxError, memoized
from src.eval.cropharvest.cropharvest_eval import Hyperparams
from src.utils import DEFAULT_SEED, data_dir, device

from .cropharvest_bands import BANDS
from .wrapper import AnySatWrapper

logger = logging.getLogger("__main__")


path_to_normalizing_dict = Path(__file__).parent / "normalizing_dict.h5"
cropharvest_data_dir = data_dir / "cropharvest_data"


def load_normalizing_dict(path_to_dict: Path = path_to_normalizing_dict) -> Dict[str, np.ndarray]:
    # normalizing dict comes from https://zenodo.org/records/10251170
    # and this function comes from
    # https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/utils.py#L69
    hf = h5py.File(path_to_dict, "r")
    return {"mean": hf.get("mean")[:], "std": hf.get("std")[:]}


class AnySatNormalizer:
    # these are the bands we will replace with the 2*std computation
    # if std = True
    def __init__(self, std_multiplier: float = 1):
        self.std_multiplier = std_multiplier
        norm_dict = load_normalizing_dict()
        # add by -> subtract by
        self.shift_values = norm_dict["mean"]
        self.div_values = norm_dict["std"] * std_multiplier

    @staticmethod
    def _normalize(x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray) -> np.ndarray:
        x = (x - shift_values) / div_values
        return x

    def __call__(self, x: np.ndarray):
        return self._normalize(x, self.shift_values, self.div_values)


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


class BinaryCropHarvestEval:
    start_month = 1
    num_outputs = 1

    country_to_sizes: Dict[str, List] = {
        "Kenya": [20, 32, 64, 96, 128, 160, 192, 224, 256, None],
        "Togo": [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, None],
    }

    all_classification_sklearn_models = [
        "LogisticRegression"
    ]

    def __init__(
        self,
        country: str,
        normalizer: AnySatNormalizer,
        num_timesteps: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        include_latlons: bool = True,
        eval_mode: str = "test",
    ):
        if eval_mode == "val":
            assert country in list(self.country_to_sizes.keys())
        self.eval_mode = eval_mode
        suffix = f"_{sample_size}" if sample_size else ""
        suffix = f"{suffix}_{num_timesteps}" if num_timesteps is not None else suffix
        self.include_latlons = include_latlons
        self.name = f"CropHarvest_{country}{suffix}{'_latlons' if include_latlons else ''}"
        self.seed = seed

        download_cropharvest_data()

        evaluation_datasets = get_eval_datasets()
        evaluation_datasets = [d for d in evaluation_datasets if country in d.id]
        assert len(evaluation_datasets) == 1
        self.dataset: CropHarvest = evaluation_datasets[0]
        assert self.dataset.task.normalize is False
        self.num_timesteps = num_timesteps
        self.sample_size = sample_size
        self.normalize = normalizer

    @staticmethod
    def truncate_timesteps(x, num_timesteps: Optional[int] = None):
        if (num_timesteps is None) or (x is None):
            return x
        else:
            return x[:, :num_timesteps]

    @torch.no_grad()
    def _evaluate_model(
        self,
        pretrained_model: AnySatWrapper,
        sklearn_model: BaseEstimator,
    ) -> Dict:
        pretrained_model.eval()
        with tempfile.TemporaryDirectory() as results_dir:
            for test_id, test_instance in self.dataset.test_data(max_size=10000):
                savepath = Path(results_dir) / f"{test_id}.nc"

                test_x = self.truncate_timesteps(
                    torch.from_numpy(self.normalize(test_instance.x)).to(device).float()  # type: ignore
                )
                s1, s2 = self.s1_and_s2_from_x(test_x)
                num_timesteps = s1.shape[3]  # b h w t d
                encodings = (
                    pretrained_model(
                        s1=s1,
                        s2=s2,
                        months=self.start_month_to_all_months(
                            torch.tensor([self.start_month] * s1.shape[0], device=test_x.device),
                            num_timesteps,
                        ),
                    )
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

    @staticmethod
    def s1_and_s2_from_x(x: Union[np.ndarray, torch.Tensor]):
        s1_band_indices = [BANDS.index(v) for v in AnySatWrapper.INPUT_S1_BAND_ORDERING]
        # this is hacky. The reason is that the wrapper accepts all S2 bands (since some datasets
        # have it) and then discards B1 and B10 since AnySat doesn't use it. CropHarvest doesn't
        # have B1 and B10 so we will it with whatever (B2) since the AnySat wrapper will discard
        # it anyway
        s2_band_indices = [
            BANDS.index(v) if v in BANDS else 2 for v in AnySatWrapper.INPUT_S2_BAND_ORDERING
        ]
        s1, s2 = x[:, :, s1_band_indices], x[:, :, s2_band_indices]
        # add h, w dimensions
        return repeat(s1, "b t d -> b h w t d", h=1, w=1), repeat(
            s2, "b t d -> b h w t d", h=1, w=1
        )

    @staticmethod
    def start_month_to_all_months(month: torch.Tensor, num_timesteps: int):
        return torch.stack(
            [torch.fmod(torch.arange(m, m + num_timesteps, dtype=torch.long), 12) for m in month]
        ).to(month.device)

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        pretrained_model: AnySatWrapper,
        models: List[str] = ["LogisticRegression"],
    ) -> Union[Sequence[BaseEstimator], Dict]:
        for model_mode in models:
            assert model_mode in ["LogisticRegression"]
        pretrained_model.eval()

        encoding_list, target_list = [], []
        for x, y, month in dl:
            x, y, month = [t.to(device) for t in (x, y, month)]
            target_list.append(y.cpu().numpy())
            s1, s2 = self.s1_and_s2_from_x(x)
            num_timesteps = s1.shape[3]  # b h w t d
            with torch.no_grad():
                encodings = (
                    pretrained_model(
                        s1=s1, s2=s2, months=self.start_month_to_all_months(month, num_timesteps)
                    )
                    .cpu()
                    .numpy()
                )
                encoding_list.append(encodings)
        encodings_np = np.concatenate(encoding_list)
        targets = np.concatenate(target_list)
        if len(targets.shape) == 2 and targets.shape[1] == 1:
            targets = targets.ravel()

        fit_models = []
        model_dict = {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=self.seed
            )
        }
        for model in models:
            fit_models.append(clone(model_dict[model]).fit(encodings_np, targets))
        return fit_models

    @staticmethod
    def random_subset(
        array: np.ndarray, labels: np.ndarray, fraction: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if fraction is not None:
            num_samples = int(array.shape[0] * fraction)
        else:
            num_samples = array.shape[0]
        return shuffle(array, labels, random_state=DEFAULT_SEED, n_samples=num_samples)

    def evaluate_model_on_task(
        self,
        pretrained_model: AnySatWrapper,
        model_modes: Optional[List[str]] = None,
        fraction: Optional[float] = None,
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert model_mode in self.all_classification_sklearn_models

        results_dict = {}
        if len(model_modes) > 0:
            array, _, y = self.dataset.as_array(num_samples=self.sample_size)
            array, y = self.random_subset(array, y, fraction=fraction)
            month = np.array([self.start_month] * array.shape[0])
            dl = DataLoader(
                TensorDataset(
                    torch.from_numpy(self.truncate_timesteps(self.normalize(array))).float(),
                    torch.from_numpy(y).long(),
                    torch.from_numpy(month).long(),
                ),
                batch_size=Hyperparams.batch_size,
                shuffle=False,
                num_workers=Hyperparams.num_workers,
            )
            sklearn_models = self.finetune_sklearn_model(
                dl,
                pretrained_model,
                models=model_modes,
            )
            for sklearn_model in sklearn_models:
                results_dict.update(
                    self._evaluate_model(
                        pretrained_model=pretrained_model, sklearn_model=sklearn_model
                    )
                )
        return results_dict
