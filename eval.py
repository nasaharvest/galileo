import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, cast

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from src import eval as e
from src.data import Dataset, Normalizer
from src.data.config import CONFIG_FILENAME, NORMALIZATION_DICT_FILENAME, OUTPUT_FOLDER
from src.eval.baseline_models import get_model_config
from src.galileo import Encoder, GalileoWrapper
from src.utils import config_dir, device

PARTITIONS = [
    "default",
    "0.20x_train",
    "0.05x_train",
    "0.01x_train",
]

RUNS_PER_SPLIT = {"LP": 5}

EVAL_MODES = ["KNN-5", "KNN-20", "LP"]

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="mmearth_atto")
argparser.add_argument("--benchmark", type=str, default="eurosat")
argparser.add_argument("--eval_mode", type=str, default="FT")
argparser.add_argument("--output_folder", type=str, default="")
argparser.add_argument("--weights_path", type=str, default="/stage/data/RS_baseline_models")
argparser.add_argument("--batch_size", type=int, default=128)
argparser.add_argument("--num_workers", type=int, default=12)
argparser.add_argument("--sweep_norms", dest="sweep_norms", action="store_true")
argparser.add_argument("--patch_size", type=int, default=4)
argparser.add_argument(
    "--pastis_filepath", type=str, default="/stage/data/presto_eval_sets/pastis"
)
argparser.add_argument("--mados_filepath", type=str, default="/stage/data/presto_eval_sets/mados")
argparser.add_argument(
    "--floods_filepath", type=str, default="/stage/data/presto_eval_sets/floods"
)
argparser.add_argument(
    "--breizhcrops_filepath", type=str, default="/stage/data/presto_eval_sets/breizhcrops"
)
argparser.add_argument("--temporal_pooling", type=str, default="mean")

argparser.add_argument("--norm_dataset", type=str, default=None)
argparser.add_argument("--norm_std_multiplier", type=float, default=None)
argparser.add_argument("--lr", type=float, default=None)
argparser.add_argument("--run_id", type=int, default=None)
argparser.add_argument("--partition", type=str, default=None)

argparser.set_defaults(sweep_norms=False)
args = argparser.parse_args().__dict__

if args["output_folder"] == "":
    output_folder = OUTPUT_FOLDER
else:
    output_folder = Path(args["output_folder"])

weights_path = Path(args["weights_path"])
if not weights_path.exists():
    raise ValueError(f"{weights_path} does not exist")

model_name = args["model"]
eval_mode = args["eval_mode"]
benchmark_name = args["benchmark"]
batch_size = args["batch_size"]
num_workers = args["num_workers"]
patch_size = args["patch_size"]
sweep_norms = args["sweep_norms"]
pastis_filepath = Path(args["pastis_filepath"])
mados_filepath = Path(args["mados_filepath"])
floods_filepath = Path(args["floods_filepath"])
breizhcrops_filepath = Path(args["breizhcrops_filepath"])
temporal_pooling = args["temporal_pooling"]
norm_dataset = args["norm_dataset"]
norm_std_multiplier = args["norm_std_multiplier"]
arg_lr = args["lr"]
arg_run_id = args["run_id"]
arg_partition = args["partition"]

if sweep_norms:
    if norm_dataset is not None:
        raise ValueError(f"Can't use norm_dataset {norm_dataset} if sweeping norms")
    if norm_std_multiplier is not None:
        raise ValueError(f"Can't use std_multiplier {norm_std_multiplier} if sweeping norms")
if (norm_dataset is not None) and (norm_dataset != "satlas"):
    if norm_std_multiplier is None:
        raise ValueError("If norm_dataset is not None, norm_std_multiplier must be passed")


BENCHMARKS = {
    "eurosat": {
        "class": e.GeobenchDataset,
        "kwargs": {"dataset_config_file": "m-eurosat.json"},
        "config": "m-eurosat.json",
    },
    "so2sat": {
        "class": e.GeobenchDataset,
        "kwargs": {"dataset_config_file": "m-so2sat.json"},
        "config": "m-so2sat.json",
    },
    "brick-kiln": {
        "class": e.GeobenchDataset,
        "kwargs": {"dataset_config_file": "m-brick-kiln.json"},
        "config": "m-brick-kiln.json",
    },
    "bigearthnet": {
        "class": e.GeobenchDataset,
        "kwargs": {"dataset_config_file": "m-bigearthnet.json"},
        "config": "m-bigearthnet.json",
    },
    "cashew-plant": {
        "class": e.GeobenchDataset,
        "kwargs": {"dataset_config_file": "m-cashew-plant.json"},
        "config": "m-cashew-plant.json",
    },
    "sa-crop-type": {
        "class": e.GeobenchDataset,
        "kwargs": {"dataset_config_file": "m-sa-crop-type.json"},
        "config": "m-sa-crop-type.json",
    },
    "pastis": {
        "class": e.PASTISDataset,
        "kwargs": {"path_to_splits": pastis_filepath},
        "config": "pastis.json",
    },
    "mados": {
        "class": e.MADOSDataset,
        "kwargs": {"path_to_splits": mados_filepath},
        "config": "mados.json",
    },
    "floods": {
        "class": e.Sen1Floods11Dataset,
        "kwargs": {"path_to_splits": floods_filepath},
        "config": "sen1floods11.json",
    },
    "cropharvest_val": {"config": "cropharvest.json"},
    "cropharvest_togo": {"config": "cropharvest.json"},
    "cropharvest_kenya": {"config": "cropharvest.json"},
    "cropharvest_brazil": {"config": "cropharvest.json"},
    "breizhcrops": {
        "class": e.BreizhCropsDataset,
        "kwargs": {"path_to_splits": breizhcrops_filepath},
        "config": "breizhcrops.json",
    },
}

if eval_mode not in EVAL_MODES:
    raise ValueError(f"{eval_mode} not in {EVAL_MODES}")
if benchmark_name not in BENCHMARKS.keys():
    raise ValueError(f"{benchmark_name} not in {BENCHMARKS.keys()}")

model_name_for_savepath = model_name
savepath_prefix = f"{model_name_for_savepath}_{eval_mode}_{benchmark_name}_{patch_size}"
if benchmark_name == "pastis":  # temporal pooling is relevant here
    savepath_prefix = f"{savepath_prefix}_tp{temporal_pooling}"

if arg_partition is not None:
    if arg_partition not in PARTITIONS:
        raise ValueError(f"{arg_partition} not in PARTITIONS")
    print(f"Replacing full partition sweep with {arg_partition}")
    partitions_to_use = [arg_partition]
else:
    partitions_to_use = deepcopy(PARTITIONS)

# breizhcrops does not have other partitions implemented
partitions_to_use = partitions_to_use if benchmark_name != "breizhcrops" else ["default"]

if eval_mode == "LP":
    if arg_run_id is None:
        runs_to_use = list(range(RUNS_PER_SPLIT[eval_mode]))
    else:
        print(f"Replacing full run sweep with {arg_run_id}")
        runs_to_use = [arg_run_id]
        savepath_prefix = f"{savepath_prefix}_runid{arg_run_id}"

    if arg_lr is None:
        if eval_mode == "LP":
            lrs_to_use = e.PROBING_LRs[eval_mode]
        else:
            lrs_to_use = e.FT_LRs
    else:
        print(f"Replacing full lr sweep with {arg_lr}")
        lrs_to_use = [arg_lr]
        savepath_prefix = f"{savepath_prefix}_lr{arg_lr}"

savepath = output_folder / f"{savepath_prefix}.csv"

if savepath.exists():
    results = pd.read_csv(savepath)
else:
    results = None

if "cropharvest" not in benchmark_name:
    benchmark = cast(dict, BENCHMARKS[benchmark_name])
    config_name: str = cast(str, benchmark["config"])
    with (Path("src/eval/configs_v2") / Path(config_name)).open("r") as f:
        config = json.load(f)
    do_pool = True if config["task_type"] == "cls" else False

    # so far we assume only s1 or s2, not both
    s1_or_s2 = "s1" if "s1" in config["band_info"].keys() else "s2"

    try:
        model_dict = get_model_config(model_name, weights_path, s1_or_s2)
        if ("satmae" in model_name) and (config_name == "m-bigearthnet.json"):
            print(f"Updating position embeddings for {model_name}")
            # for satmae on BigEarthNet, we need to adjust position embeddings
            model_dict["args"]["img_size"] = 120
        encoder = model_dict["model"](
            **model_dict["args"], do_pool=do_pool, temporal_pooling=temporal_pooling
        ).to(device)
    except KeyError:
        encoder = GalileoWrapper(
            pretrained_path=weights_path,
            patch_size=patch_size,
            do_pool=do_pool,
            add_layernorm_on_exit=False if eval_mode == "FT" else True,
        ).to(device)

        if benchmark_name == "mados":
            # MADOS is processed differently so it can't use the our norm strategy
            default_norm_strat = {
                "stats": "dataset",
                "type": "norm_no_clip",
                "std_multiplier": 2.0,
            }
        elif s1_or_s2 == "s1":
            # following advice from https://arxiv.org/pdf/2305.13456
            default_norm_strat = {
                "stats": "OURS_S1",
                "type": "norm_no_clip",
                "std_multiplier": 2.0,
            }
        else:
            # following advice from https://arxiv.org/pdf/2305.13456
            default_norm_strat = {"stats": "OURS", "type": "norm_no_clip", "std_multiplier": 2.0}

    norms_for_model = e.get_all_norm_strats(model_name, s1_or_s2)
    if sweep_norms:
        norms_to_use = norms_for_model
    else:
        if norm_dataset is not None:
            if norm_dataset == "satlas":
                norms_to_use = [{"type": "satlas"}]
            else:
                norm_type, _ = e.norm_type_from_model_name(model_name)
                if norm_std_multiplier is not None:
                    norms_to_use = [
                        {
                            "type": norm_type,
                            "stats": norm_dataset,
                            "std_multiplier": norm_std_multiplier,
                        }
                    ]
                else:
                    norms_to_use = [
                        norm for norm in norms_for_model if norm.get("stats", "") == norm_dataset
                    ]

        else:
            # default if its not in the config
            if "models" in config:
                if model_name in config["models"]:
                    norms_to_use = [None]
                else:
                    print(f"No norm strat; using default: {default_norm_strat}")
                    norms_to_use = [default_norm_strat]
            else:
                print(f"No norm strat; using default: {default_norm_strat}")
                norms_to_use = [default_norm_strat]

    for train_partition in partitions_to_use:
        for norm_strat in norms_to_use:
            print(
                f"Running {train_partition} for {model_name}, {benchmark_name} with norm_strat {norm_strat}"
            )
            loaders = e.get_loaders(
                benchmark,
                config,
                model_name,
                args["batch_size"],
                args["num_workers"],
                eval_mode,
                train_partition=train_partition,
                norm_ops=norm_strat,
            )
            print(f"In eval, {len(loaders['train'])}")

            if eval_mode in ["KNN-5", "KNN-20", "K-Means"]:
                if config["task_type"] != "cls":
                    raise ValueError(
                        f"{eval_mode} not supported for {benchmark_name} of task type cls"
                    )
                if (results is not None) and (
                    len(
                        results[
                            (results["partition"] == train_partition)
                            & (results["norm_op"] == str(norm_strat))
                        ]
                    )
                    > 0
                ):
                    print(f"{train_partition}, {norm_strat} in results - skipping")
                    continue
                train_embeddings, train_labels = e.get_embeddings(
                    data_loader=loaders["train"], model=encoder, device=device
                )
                test_embeddings, test_labels = e.get_embeddings(
                    data_loader=loaders["test"], model=encoder, device=device
                )
                test_result = e.run_knn(
                    eval_type=eval_mode,
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    test_embeddings=test_embeddings,
                    test_labels=test_labels,
                    num_classes=config["num_classes"],
                    is_multilabel=config["is_multilabel"],
                    device=device,
                )

                val_embeddings, val_labels = e.get_embeddings(
                    data_loader=loaders["valid"], model=encoder, device=device
                )
                val_result = e.run_knn(
                    eval_type=eval_mode,
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    test_embeddings=val_embeddings,
                    test_labels=val_labels,
                    num_classes=config["num_classes"],
                    is_multilabel=config["is_multilabel"],
                    device=device,
                )
                new_df = pd.DataFrame(
                    {
                        "model_name": [model_name],
                        "benchmark": [benchmark_name],
                        "partition": [train_partition],
                        "test": [test_result],
                        "val": [val_result],
                        "norm_op": [str(norm_strat)],
                    }
                )
                print(new_df)
                if results is not None:
                    results = pd.concat([results, new_df], axis=0)
                else:
                    results = new_df

                results.to_csv(savepath, index=False)

            elif eval_mode == "LP":
                if (model_name == "anysat") and config["task_type"] == "seg":
                    train_subsample: Optional[float] = 1 / 16
                else:
                    train_subsample = None
                t_e, t_l = e.get_embeddings(
                    data_loader=loaders["train"],
                    model=encoder,
                    device=device,
                    subsample_tokens=train_subsample,
                )
                v_e, v_l = e.get_embeddings(
                    data_loader=loaders["valid"], model=encoder, device=device
                )
                te_e, te_l = e.get_embeddings(
                    data_loader=loaders["test"], model=encoder, device=device
                )
                embedding_loaders = {
                    "train": DataLoader(
                        TensorDataset(t_e, t_l), batch_size=batch_size, shuffle=True
                    ),
                    "valid": DataLoader(
                        TensorDataset(v_e, v_l), batch_size=batch_size, shuffle=False
                    ),
                    "test": DataLoader(
                        TensorDataset(te_e, te_l), batch_size=batch_size, shuffle=False
                    ),
                }

                for run_id in runs_to_use:
                    for lr in lrs_to_use:
                        if (results is not None) and (
                            len(
                                results[
                                    (results["partition"] == train_partition)
                                    & (results["lr"] == lr)
                                    & (results["run_id"] == run_id)
                                    & (results["norm_op"] == str(norm_strat))
                                ]
                            )
                            > 0
                        ):
                            print(f"{train_partition}, {run_id}, {lr} in results - skipping")
                            continue
                        if config["task_type"] == "cls":
                            val, test = e.train_and_eval_probe_cls(
                                lr=lr,
                                config=config,
                                loaders=embedding_loaders,
                                in_features=encoder.dim,
                                device=device,
                            )
                        elif config["task_type"] == "seg":
                            val, test = e.train_and_eval_probe_seg(
                                lr=lr,
                                config=config,
                                loaders=embedding_loaders,
                                in_features=encoder.dim,
                                grid_size=encoder.grid_size,
                                device=device,
                            )
                        else:
                            raise ValueError(
                                f"task_type must be cls or seg, not {config['task_type']}"
                            )

                        new_df = pd.DataFrame(
                            {
                                "model_name": [model_name],
                                "benchmark": [benchmark_name],
                                "partition": [train_partition],
                                "val": [val],
                                "test": [test],
                                "lr": [lr],
                                "run_id": [run_id],
                                "norm_op": [str(norm_strat)],
                            }
                        )
                        print(new_df)
                        if results is not None:
                            results = pd.concat([results, new_df], axis=0)
                        else:
                            results = new_df

                        results.to_csv(savepath, index=False)

            elif eval_mode == "FT":
                cache_dir = output_folder / "ft_cache"
                cache_dir.mkdir(exist_ok=True)
                for run_id in runs_to_use:
                    for lr in lrs_to_use:
                        if (results is not None) and (
                            len(
                                results[
                                    (results["partition"] == train_partition)
                                    & (results["lr"] == lr)
                                    & (results["run_id"] == run_id)
                                    & (results["norm_op"] == str(norm_strat))
                                ]
                            )
                            > 0
                        ):
                            print(f"{train_partition}, {run_id}, {lr} in results - skipping")
                            continue
                        if config["task_type"] == "cls":
                            val, test = e.finetune_and_eval_cls(
                                lr=lr,
                                config=config,
                                loaders=loaders,
                                encoder=encoder,
                                device=device,
                                cache_dir=cache_dir,
                            )
                        elif config["task_type"] == "seg":
                            val, test = e.finetune_and_eval_seg(
                                lr=lr,
                                config=config,
                                loaders=loaders,
                                encoder=encoder,
                                device=device,
                            )
                        new_df = pd.DataFrame(
                            {
                                "model_name": [model_name],
                                "benchmark": [benchmark_name],
                                "partition": [train_partition],
                                "val": [val],
                                "test": [test],
                                "lr": [lr],
                                "run_id": [run_id],
                                "norm_op": [str(norm_strat)],
                            }
                        )
                        print(new_df)
                        if results is not None:
                            results = pd.concat([results, new_df], axis=0)
                        else:
                            results = new_df

                        results.to_csv(savepath, index=False)
else:
    input_name_to_args = {
        "cropharvest_val": {"country": "Togo", "eval_mode": "val"},
        "cropharvest_togo": {"country": "Togo", "eval_mode": "test"},
        "cropharvest_kenya": {"country": "Kenya", "eval_mode": "test"},
        "cropharvest_brazil": {"country": "Brazil", "eval_mode": "test"},
    }

    if model_name == "presto":
        model_name_to_save = "presto"
        model_dict = get_model_config(model_name, weights_path, s1_or_s2="cropharvest")
        encoder = model_dict["model"].load_pretrained().encoder.to(device)
        val_task_ts = e.PrestoBinaryCropHarvestEval(
            **input_name_to_args[benchmark_name],  # type: ignore
            normalizer=e.PrestoNormalizer(),
        )
    elif model_name == "anysat":
        model_name_to_save = "anysat"
        # s1_or_s2 does not affect the anysat model
        model_dict = get_model_config(model_name, weights_path, s1_or_s2="s2")
        encoder = model_dict["model"](do_pool=True).to(device)
        val_task_ts = e.AnySatBinaryCropHarvestEval(
            **input_name_to_args[benchmark_name],  # type: ignore
            normalizer=e.AnySatNormalizer(),
        )
    else:
        encoder = Encoder.load_from_folder(weights_path).to(device)
        # isolate the ignore_bands
        config_path = weights_path / CONFIG_FILENAME
        with config_path.open("r") as f:
            config = json.load(f)
        ignore_band_groups = config["training"].get("ignore_band_groups", None)
        print("Running with default normalization (OURS, std=2)")
        val_task_ts = e.BinaryCropHarvestEval(
            normalizer=Normalizer(
                normalizing_dicts=Dataset.load_normalization_values(
                    config_dir / NORMALIZATION_DICT_FILENAME
                ),
                std_multiplier=2,
            ),
            **input_name_to_args[benchmark_name],  # type: ignore
            ignore_band_groups=ignore_band_groups,
        )

    for partition in partitions_to_use:
        partition_to_float = {
            "default": None,  # None means no sampling
        }
        if (results is not None) and (len(results[results["partition"] == partition]) > 0):
            print(f"{partition} in results - skipping")
            continue
        elif partition not in partition_to_float.keys():
            print(f"partition {partition} too small for cropharvest - skipping")
            continue

        output = val_task_ts.evaluate_model_on_task(
            encoder,
            model_modes=["KNNat20Classifier", "LogisticRegression"],
            fraction=partition_to_float[partition],
        )
        # retrieve the appropriate keys
        output_keys = list(output.keys())
        k_key = [k for k in output_keys if "KNNat20" in k and "f1" in k and not k.endswith("_c")][
            0
        ]
        lr_key = [
            k
            for k in output_keys
            if "LogisticRegression" in k and "f1" in k and not k.endswith("_c")
        ][0]
        # save and print
        new_df = pd.DataFrame(
            {
                "model_name": [model_name_to_save],
                "benchmark": [benchmark_name],
                "knn-20": [output[k_key]],
                "lr": [output[lr_key]],
                "partition": [partition],
            }
        )
        print(new_df)
        if results is not None:
            results = pd.concat([results, new_df], axis=0)
        else:
            results = new_df

        results.to_csv(savepath, index=False)
