import argparse
import copy
import json
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Optional, cast

import psutil
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.collate_fns import galileo_collate_fn
from src.config import DEFAULT_SEED
from src.data import Dataset, Normalizer
from src.data.config import (
    CONFIG_FILENAME,
    DATA_FOLDER,
    DECODER_FILENAME,
    EE_PROJECT,
    ENCODER_FILENAME,
    NORMALIZATION_DICT_FILENAME,
    OPTIMIZER_FILENAME,
    OUTPUT_FOLDER,
    TARGET_ENCODER_FILENAME,
    TIFS_FOLDER,
)
from src.galileo import Decoder, Encoder, adjust_learning_rate
from src.loss import construct_target_encoder_masks, do_loss
from src.utils import (
    AverageMeter,
    config_dir,
    device,
    is_bf16_available,
    load_check_config,
    seed_everything,
    timestamp_dirname,
    will_cause_nans,
)

process = psutil.Process()

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

torch.backends.cuda.matmul.allow_tf32 = True
autocast_device = torch.bfloat16 if is_bf16_available() else torch.float32

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="small.json")
argparser.add_argument("--run_name_prefix", type=str, default="")
argparser.add_argument("--h5py_folder", type=str, default="")
argparser.add_argument("--output_folder", type=str, default="")
argparser.add_argument("--download", dest="download", action="store_true")
argparser.add_argument("--h5pys_only", dest="h5pys_only", action="store_true")
argparser.add_argument("--num_workers", dest="num_workers", default=0)
argparser.add_argument("--batch_size", dest="batch_size", default="")
argparser.add_argument("--checkpoint_every_epoch", type=int, default=0)

argparser.set_defaults(download=False)
argparser.set_defaults(cache_in_ram=False)
args = argparser.parse_args().__dict__

if args["h5py_folder"] == "":
    cache_folder = DATA_FOLDER / "h5pys"
else:
    cache_folder = Path(args["h5py_folder"])


restart = False
model_path: Optional[Path] = None
start_epoch = 0
run_id = None
wandb_enabled = True
wandb_org = "nasa-harvest"
wandb_output_dir = Path(__file__).parent

if args["output_folder"] == "":
    output_folder = OUTPUT_FOLDER
else:
    output_folder = Path(args["output_folder"])
    # we expect a single folder in the output_folder.
    # if it is not empty, we are resuming an existing run -
    # retrieve this run's results
    output_dirs = [o for o in output_folder.glob("*") if o.is_dir()]
    if len(output_dirs) > 0:
        assert len(output_dirs) == 1, f"Got more than one output dir: {output_dirs}"
        restart = True
        model_path = output_dirs[0]
        print(f"Restarting run using {model_path}")
        with (model_path / CONFIG_FILENAME).open("r") as f:
            config = json.load(f)
        run_name = config["run_name"]
        start_epoch = config["cur_epoch"]
        run_id = config["wandb_run_id"]


if not restart:
    config = load_check_config(args["config_file"])
    run_name = f"{args['config_file']} config file"
    if args["run_name_prefix"] != "":
        prefix = args["run_name_prefix"]
        run_name = f"{prefix}_{run_name}"
    config["run_name"] = run_name

run = wandb.init(
    name=run_name,
    entity=wandb_org,
    project="galileo",
    dir=wandb_output_dir,
    id=run_id,
    resume="allow",
)
run_id = cast(Run, run).id
config["wandb_run_id"] = run_id

training_config = config["training"]

if args["batch_size"] != "":
    warnings.warn(
        f"Overriding batch size from {training_config['batch_size']} to {args['batch_size']}"
    )
    training_config["batch_size"] = int(args["batch_size"])
    config["training"]["batch_size"] = int(args["batch_size"])

# we seed everything after we call get_random_config(), since
# we want this to differ between runs
seed_everything(DEFAULT_SEED)

print("Loading dataset and dataloader")

dataset = Dataset(
    TIFS_FOLDER,
    download=args["download"],
    h5py_folder=cache_folder,
    h5pys_only=args["h5pys_only"],
)
config["training"]["training_samples"] = len(dataset)

if not restart:
    # we can't reset these values without wandb
    # complaining
    wandb.config.update(config)

if training_config["normalization"] == "std":
    normalizing_dict = dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    print(normalizing_dict, flush=True)
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
    dataset.normalizer = normalizer
else:
    normalizer = Normalizer(std=False)
    dataset.normalizer = normalizer

dataloader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=int(args["num_workers"]),
    collate_fn=partial(
        galileo_collate_fn,
        patch_sizes=training_config["patch_sizes"],
        shape_time_combinations=training_config["shape_time_combinations"],
        st_encode_ratio=training_config["st_encode_ratio"],
        st_decode_ratio=training_config["st_decode_ratio"],
        random_encode_ratio=training_config["random_encode_ratio"],
        random_decode_ratio=training_config["random_decode_ratio"],
        augmentation_strategies=training_config["augmentation"],
        masking_probabilities=training_config["masking_probabilities"],
        max_unmasking_channels=training_config["max_unmasking_channels"],
        random_masking=training_config["random_masking"],
        unmasking_channels_combo=training_config["unmasking_channels_combo"],
        ignore_band_groups=training_config["ignore_band_groups"],
    ),
    pin_memory=True,
)

print("Loading models")
predictor = Decoder(**config["model"]["decoder"]).to(device)
param_groups = [
    {
        "params": predictor.parameters(),
        "name": "decoder",
        "weight_decay": training_config["weight_decay"],
    }
]
encoder = Encoder(**config["model"]["encoder"]).to(device)
param_groups.append(
    {
        "params": encoder.parameters(),
        "name": "encoder",
        "weight_decay": training_config["weight_decay"],
    }
)
second_predictor = None
if training_config["double_loss"] and training_config["double_predictors"]:
    second_predictor = Decoder(**config["model"]["decoder"]).to(device)
    param_groups.append(
        {
            "params": second_predictor.parameters(),
            "name": "second_decoder",
            "weight_decay": training_config["weight_decay"],
        }
    )

if restart:
    assert model_path is not None
    encoder.load_state_dict(torch.load(model_path / ENCODER_FILENAME, map_location=device))
    predictor.load_state_dict(torch.load(model_path / DECODER_FILENAME, map_location=device))
    if second_predictor is not None:
        second_predictor.load_state_dict(
            torch.load(model_path / f"second_{DECODER_FILENAME}", map_location=device)
        )

optimizer = torch.optim.AdamW(
    param_groups,
    lr=0,
    weight_decay=training_config["weight_decay"],
    betas=(training_config["betas"][0], training_config["betas"][1]),
)  # type: ignore
if restart:
    assert model_path is not None
    optimizer.load_state_dict(torch.load(model_path / OPTIMIZER_FILENAME, map_location=device))

assert training_config["effective_batch_size"] % training_config["batch_size"] == 0
iters_to_accumulate = training_config["effective_batch_size"] / training_config["batch_size"]

# setup target encoder and momentum from: https://github.com/facebookresearch/ijepa/blob/main/src/train.py
repeat_aug = 4
steps_per_epoch = len(dataloader) * repeat_aug / iters_to_accumulate
momentum_scheduler = (
    training_config["ema"][0]
    + i
    * (training_config["ema"][1] - training_config["ema"][0])
    / (steps_per_epoch * training_config["num_epochs"])
    for i in range(int(steps_per_epoch * training_config["num_epochs"]) + 1)
)
target_encoder = copy.deepcopy(encoder)
target_encoder.eval()
if restart:
    assert model_path is not None
    target_encoder.load_state_dict(
        torch.load(model_path / TARGET_ENCODER_FILENAME, map_location=device)
    )
    # we also want to step through the momentum scheduler since we are going to fast forward training
    for momentum_epoch in range(start_epoch):
        for i in range(int(steps_per_epoch)):
            _ = next(momentum_scheduler)

for p in target_encoder.parameters():
    p.requires_grad = False

skipped_batches = 0
for e in tqdm(range(start_epoch, training_config["num_epochs"])):
    i = 0
    train_loss = AverageMeter()
    for bs in tqdm(dataloader, total=len(dataloader), leave=False):
        for b_idx, b in enumerate(bs):
            i += 1
            b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
            (
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size,
            ) = b

            if (
                will_cause_nans(s_t_x)
                or will_cause_nans(sp_x)
                or will_cause_nans(t_x)
                or will_cause_nans(st_x)
            ):
                skipped_batches += 1
                warnings.warn(f"Skipping batch with NaNs, {skipped_batches}")
                continue

            with torch.autocast(device_type=device.type, dtype=autocast_device):
                if training_config["double_predictors"] and b_idx > 1:
                    assert second_predictor is not None
                    predictor_to_use = second_predictor
                else:
                    predictor_to_use = predictor
                (p_s_t, p_sp, p_t, p_st) = predictor_to_use(
                    *encoder(
                        s_t_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_m,
                        sp_m,
                        t_m,
                        st_m,
                        months.long(),
                        patch_size=patch_size,
                    ),
                    patch_size=patch_size,
                )
                if ("loss_dict" in config["training"]) and (
                    config["training"]["loss_dict"] == "MAE"
                ):
                    loss = do_loss(
                        training_config["loss_dict"],
                        (
                            p_s_t,
                            p_sp,
                            p_t,
                            p_st,
                            s_t_x,
                            sp_x,
                            t_x,
                            st_x,
                            s_t_m,
                            sp_m,
                            t_m,
                            st_m,
                            patch_size,
                            max(training_config["patch_sizes"]),
                        ),
                    )
                else:
                    with torch.no_grad():
                        if not config["training"]["double_loss"]:
                            t_s_t, t_sp, t_t, t_st, _, _, _, _, _ = target_encoder(
                                s_t_x,
                                sp_x,
                                t_x,
                                st_x,
                                *construct_target_encoder_masks(
                                    s_t_m, sp_m, t_m, st_m, config["training"]["target_masking"]
                                ),
                                months.long(),
                                patch_size=patch_size,
                                exit_after=config["training"]["target_exit_after"],
                                token_exit_cfg=config["training"]["token_exit_cfg"],
                            )
                            loss_dict = training_config["loss_dict"]
                        else:
                            if training_config["random_masking"] != "half":
                                raise ValueError(
                                    "double_loss only possible with random_masking == half"
                                )
                            if b_idx <= 1:
                                # this assumes the collate_fn has a SPACE, TIME, RANDOM, RANDOM
                                # set up.
                                token_exit_cfg = config["training"]["token_exit_cfg"]
                                exit_after = None
                                t_s_t, t_sp, t_t, t_st, _, _, _, _, _ = target_encoder(
                                    s_t_x,
                                    sp_x,
                                    t_x,
                                    st_x,
                                    *construct_target_encoder_masks(
                                        s_t_m,
                                        sp_m,
                                        t_m,
                                        st_m,
                                        config["training"]["target_masking"],
                                    ),
                                    months.long(),
                                    patch_size=patch_size,
                                    exit_after=None,
                                    token_exit_cfg=config["training"]["token_exit_cfg"],
                                )
                                if "loss_dict_st" in training_config:
                                    loss_dict = training_config["loss_dict_st"]
                                else:
                                    loss_dict = training_config["loss_dict"]
                            else:
                                t_s_t, t_sp, t_t, t_st, _, _, _, _, _ = target_encoder(
                                    s_t_x,
                                    sp_x,
                                    t_x,
                                    st_x,
                                    *construct_target_encoder_masks(
                                        s_t_m,
                                        sp_m,
                                        t_m,
                                        st_m,
                                        config["training"]["target_masking"],
                                    ),
                                    months.long(),
                                    patch_size=patch_size,
                                    exit_after=config["training"]["target_exit_after"],
                                    token_exit_cfg=None,
                                )
                                if "loss_dict_random" in training_config:
                                    loss_dict = training_config["loss_dict_random"]
                                else:
                                    loss_dict = training_config["loss_dict"]

                    loss = do_loss(
                        loss_dict,
                        (
                            t_s_t,
                            t_sp,
                            t_t,
                            t_st,
                            p_s_t,
                            p_sp,
                            p_t,
                            p_st,
                            s_t_m[:, 0::patch_size, 0::patch_size],
                            sp_m[:, 0::patch_size, 0::patch_size],
                            t_m,
                            st_m,
                        ),
                    )
                assert not torch.isnan(loss).any(), "NaNs in loss"
            train_loss.update(loss.item(), n=s_t_x.shape[0])

            loss = loss / iters_to_accumulate
            loss.backward()

            if ((i + 1) % iters_to_accumulate == 0) or (i + 1 == len(dataloader)):
                if training_config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                current_lr = adjust_learning_rate(
                    optimizer,
                    epoch=i / (repeat_aug * len(dataloader)) + e,
                    warmup_epochs=training_config["warmup_epochs"],
                    total_epochs=training_config["num_epochs"],
                    max_lr=training_config["max_lr"],
                    min_lr=training_config["final_lr"],
                )

                with torch.no_grad():
                    try:
                        m = next(momentum_scheduler)
                    except StopIteration:
                        m = training_config["ema"][1]
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
    if wandb_enabled:
        to_log = {
            "train_loss": train_loss.average,
            "epoch": e,
            "momentum": m,
            "lr": current_lr,
        }
        wandb.log(to_log, step=e)
    if args["checkpoint_every_epoch"] > 0:
        if e % args["checkpoint_every_epoch"] == 0:
            if model_path is None:
                model_path = output_folder / timestamp_dirname(run_id)
                model_path.mkdir()
            print(f"Checkpointing to {model_path}")
            torch.save(encoder.state_dict(), model_path / ENCODER_FILENAME)
            torch.save(predictor.state_dict(), model_path / DECODER_FILENAME)
            if second_predictor is not None:
                torch.save(
                    second_predictor.state_dict(), model_path / f"second_{DECODER_FILENAME}"
                )
            torch.save(target_encoder.state_dict(), model_path / TARGET_ENCODER_FILENAME)
            torch.save(optimizer.state_dict(), model_path / OPTIMIZER_FILENAME)
            config["cur_epoch"] = e + 1
            with (model_path / CONFIG_FILENAME).open("w") as f:
                json.dump(config, f)

if model_path is None:
    model_path = output_folder / timestamp_dirname(run_id)
    model_path.mkdir()
torch.save(encoder.state_dict(), model_path / ENCODER_FILENAME)
torch.save(predictor.state_dict(), model_path / DECODER_FILENAME)
if second_predictor is not None:
    torch.save(second_predictor.state_dict(), model_path / f"second_{DECODER_FILENAME}")
torch.save(target_encoder.state_dict(), model_path / TARGET_ENCODER_FILENAME)
torch.save(optimizer.state_dict(), model_path / OPTIMIZER_FILENAME)
with (model_path / CONFIG_FILENAME).open("w") as f:
    json.dump(config, f)
