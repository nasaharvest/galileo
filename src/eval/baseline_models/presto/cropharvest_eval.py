import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import repeat
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset

from src.eval.cropharvest.columns import NullableColumns, RequiredColumns
from src.eval.cropharvest.cropharvest_eval import Hyperparams
from src.eval.cropharvest.datasets import CropHarvest, Task, TestInstance
from src.eval.cropharvest.datasets import CropHarvestLabels as OrgCropHarvestLabels
from src.eval.cropharvest.utils import NoDataForBoundingBoxError, memoized
from src.utils import DEFAULT_SEED, data_dir, device

from .single_file_presto import NUM_DYNAMIC_WORLD_CLASSES, PRESTO_ADD_BY, PRESTO_DIV_BY, Encoder

logger = logging.getLogger("__main__")


cropharvest_data_dir = data_dir / "cropharvest_data"


class PrestoNormalizer:
    # these are the bands we will replace with the 2*std computation
    # if std = True
    def __init__(self, std_multiplier: float = 1):
        self.std_multiplier = std_multiplier
        # add by -> subtract by
        self.shift_values = np.array(PRESTO_ADD_BY) * -1
        self.div_values = np.array(PRESTO_DIV_BY) * std_multiplier

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

    all_classification_sklearn_models = ["LogisticRegression"]

    def __init__(
        self,
        country: str,
        normalizer: PrestoNormalizer,
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

    @staticmethod
    def _mask_to_batch_tensor(
        mask: Optional[np.ndarray], batch_size: int
    ) -> Optional[torch.Tensor]:
        if mask is not None:
            return repeat(torch.from_numpy(mask).to(device), "t c -> b t c", b=batch_size).float()
        return None

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

                test_x = self.truncate_timesteps(
                    torch.from_numpy(self.normalize(test_instance.x)).to(device).float()  # type: ignore
                )
                # mypy fails with these lines uncommented, but this is how we will
                # pass the other values to the model
                test_latlons_np = np.stack([test_instance.lats, test_instance.lons], axis=-1)
                test_latlon = torch.from_numpy(test_latlons_np).to(device).float()
                # mask out DW
                test_dw = self.truncate_timesteps(
                    torch.ones_like(test_x[:, :, 0]).to(device).long() * NUM_DYNAMIC_WORLD_CLASSES
                )
                batch_mask = self.truncate_timesteps(
                    self._mask_to_batch_tensor(None, test_x.shape[0])
                )

                encodings = (
                    pretrained_model(
                        test_x,
                        dynamic_world=test_dw,
                        mask=batch_mask,
                        latlons=test_latlon,
                        month=self.start_month,
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

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        pretrained_model: Encoder,
        models: List[str] = ["LogisticRegression"],
    ) -> Union[Sequence[BaseEstimator], Dict]:
        for model_mode in models:
            assert model_mode in ["LogisticRegression"]
        pretrained_model.eval()

        encoding_list, target_list = [], []
        for x, y, dw, latlons, month in dl:
            x, dw, latlons, y, month = [t.to(device) for t in (x, dw, latlons, y, month)]
            batch_mask = self._mask_to_batch_tensor(None, x.shape[0])
            target_list.append(y.cpu().numpy())
            with torch.no_grad():
                encodings = (
                    pretrained_model(
                        x, dynamic_world=dw, mask=batch_mask, latlons=latlons, month=month
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

        results_dict = {}
        if len(model_modes) > 0:
            array, latlons, y = self.dataset.as_array(num_samples=self.sample_size)
            array, latlons, y = self.random_subset(array, latlons, y, fraction=fraction)
            dw = np.ones_like(array[:, :, 0]) * NUM_DYNAMIC_WORLD_CLASSES
            month = np.array([self.start_month] * array.shape[0])
            dl = DataLoader(
                TensorDataset(
                    torch.from_numpy(self.truncate_timesteps(self.normalize(array))).float(),
                    torch.from_numpy(y).long(),
                    torch.from_numpy(self.truncate_timesteps(dw)).long(),
                    torch.from_numpy(latlons).float(),
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
