import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, f1_score

from src.galileo import adjust_learning_rate

from .metrics import mean_iou

FT_LRs = [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3]


class EncoderWithHead(nn.Module):
    def __init__(self, encoder, num_classes):
        super(EncoderWithHead, self).__init__()
        self.encoder = deepcopy(encoder)  # just in case
        if encoder.do_pool:
            # for classification
            self.head = nn.Linear(encoder.dim, num_classes)
        else:
            # for segmentation
            logits_per_patch = int(num_classes * encoder.patch_size * encoder.patch_size)
            self.head = nn.Linear(encoder.dim, logits_per_patch)

    def forward(self, **batch):
        features = self.encoder(**batch)
        output = self.head(features)
        return output


def finetune_and_eval_cls(lr, config, loaders, encoder, device, cache_dir):
    finetuned_encoder = finetune_cls(
        data_loader=loaders["train"],
        lr=lr,
        epochs=50,
        encoder=encoder,
        num_classes=config["num_classes"],
        is_multilabel=config["is_multilabel"],
        device=device,
        cache_dir=cache_dir,
    )
    val_acc = evaluate_cls(
        data_loader=loaders["valid"],
        finetuned_encoder=finetuned_encoder,
        is_multilabel=config["is_multilabel"],
        device=device,
    )
    test_acc = evaluate_cls(
        data_loader=loaders["test"],
        finetuned_encoder=finetuned_encoder,
        is_multilabel=config["is_multilabel"],
        device=device,
    )
    print(lr, val_acc, test_acc)
    return val_acc, test_acc


def finetune_and_eval_seg(lr, config, loaders, encoder, device):
    finetuned_encoder = finetune_seg(
        data_loader=loaders["train"],
        lr=lr,
        epochs=50,
        encoder=encoder,
        num_classes=config["num_classes"],
        device=device,
    )
    val_miou = evaluate_seg(
        data_loader=loaders["valid"],
        finetuned_encoder=finetuned_encoder,
        num_classes=config["num_classes"],
        device=device,
    )
    test_miou = evaluate_seg(
        data_loader=loaders["test"],
        finetuned_encoder=finetuned_encoder,
        num_classes=config["num_classes"],
        device=device,
    )
    return val_miou, test_miou


def get_finetune_results(loaders, config, encoder, num_runs, device):
    final_tests = []  # chosen using LR with best val, for each run
    for _ in range(num_runs):
        vals = []
        tests = []
        for lr in FT_LRs:
            if config["task_type"] == "cls":
                val, test = finetune_and_eval_cls(
                    lr=lr, config=config, loaders=loaders, encoder=encoder, device=device
                )
            elif config["task_type"] == "seg":
                val, test = finetune_and_eval_seg(
                    lr=lr, config=config, loaders=loaders, encoder=encoder, device=device
                )
            else:
                raise f"task_type must be cls or seg, not {config['task_type']}"

            vals.append(val)
            tests.append(test)

        final_tests.append(tests[vals.index(max(vals))])

    return final_tests


def finetune_cls(
    data_loader, lr, epochs, encoder, num_classes, is_multilabel, device, cache_dir: Path
):
    epoch_file = cache_dir / "epoch_textfile.json"
    state_dict_file = cache_dir / "state_dict.pt"
    optimizer_params_file = cache_dir / "opt_dict.pt"
    finetuned_encoder = EncoderWithHead(encoder=encoder, num_classes=num_classes).to(device)
    finetuned_encoder = finetuned_encoder.train()
    opt = torch.optim.AdamW(finetuned_encoder.parameters(), lr=lr)

    grad_accum = int(256 / data_loader.batch_size)
    sched_config = {
        "lr": lr,
        "warmup_epochs": int(epochs * 0.1),
        "min_lr": 1.0e-6,
        "epochs": epochs,
    }
    if is_multilabel:
        loss_function: nn.Module = nn.MultiLabelSoftMarginLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    # check the cache, in case we got preempted
    start_epoch = 0
    if epoch_file.exists():
        if (not state_dict_file.exists()) or (not optimizer_params_file.exists()):
            print("Missing a state dict file - removing both")
            epoch_file.unlink(missing_ok=True)
            state_dict_file.unlink(missing_ok=True)
            optimizer_params_file.unlink(missing_ok=True)
        else:
            try:
                with epoch_file.open("r") as f:
                    start_epoch = json.load(f)["last_finished_epoch"] + 1
                finetuned_encoder.load_state_dict(torch.load(state_dict_file))
                opt.load_state_dict(torch.load(optimizer_params_file))
                print(f"Resuming pre-empted job at epoch {start_epoch}")
            except (RuntimeError, EOFError) as e:
                print(f"Got error {e} - restarting run")
                epoch_file.unlink(missing_ok=True)
                state_dict_file.unlink(missing_ok=True)
                optimizer_params_file.unlink(missing_ok=True)
                start_epoch = 0

    print(f"Training on {len(data_loader)} samples from start epoch {start_epoch}")
    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(data_loader):
            batch_labels = batch.pop("target")
            if "s1" in batch:
                batch["s1"] = batch["s1"].to(device).to(torch.bfloat16)
            if "s2" in batch:
                batch["s2"] = batch["s2"].to(device).to(torch.bfloat16)
            if "months" in batch:
                batch["months"] = batch["months"].to(device).long()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = finetuned_encoder(**batch)
                loss = loss_function(logits, batch_labels.to(device))
            (loss / grad_accum).backward()

            if ((i + 1) % grad_accum == 0) or (i + 1 == len(data_loader)):
                epoch_fraction = epoch + (i / len(data_loader))
                lr = adjust_learning_rate(
                    optimizer=opt,
                    epoch=epoch_fraction,
                    total_epochs=sched_config["epochs"],
                    warmup_epochs=sched_config["warmup_epochs"],
                    max_lr=sched_config["lr"],
                    min_lr=sched_config["min_lr"],
                )
                torch.nn.utils.clip_grad_norm_(finetuned_encoder.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        with epoch_file.open("w") as f:
            json.dump({"last_finished_epoch": epoch}, f)
        torch.save(finetuned_encoder.state_dict(), state_dict_file)
        torch.save(opt.state_dict(), optimizer_params_file)

    # delete everything for the new run
    epoch_file.unlink()
    state_dict_file.unlink()
    optimizer_params_file.unlink()
    return finetuned_encoder


def evaluate_cls(data_loader, finetuned_encoder, is_multilabel, device):
    finetuned_encoder = finetuned_encoder.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch_labels = batch.pop("target")
            if "s1" in batch:
                batch["s1"] = batch["s1"].to(device).to(torch.bfloat16)
            if "s2" in batch:
                batch["s2"] = batch["s2"].to(device).to(torch.bfloat16)
            if "months" in batch:
                batch["months"] = batch["months"].to(device).long()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                batch_logits = finetuned_encoder(**batch)  # (bsz, num_classes)

            all_logits.append(batch_logits.float().cpu())
            all_labels.append(batch_labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if is_multilabel:
        all_preds = torch.sigmoid(all_logits) > 0.5
        return f1_score(all_labels, all_preds, average="micro")
    else:
        all_preds = torch.argmax(all_logits, dim=-1)
        return accuracy_score(all_labels, all_preds)


def finetune_seg(data_loader, lr, epochs, encoder, num_classes, device):
    finetuned_encoder = EncoderWithHead(encoder=encoder, num_classes=num_classes).to(device)
    finetuned_encoder = finetuned_encoder.train()
    opt = torch.optim.AdamW(finetuned_encoder.parameters(), lr=lr)
    patch_size = encoder.patch_size

    grad_accum = int(256 / data_loader.batch_size)
    sched_config = {
        "lr": lr,
        "warmup_epochs": int(epochs * 0.1),
        "min_lr": 1.0e-6,
        "epochs": epochs,
    }

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # for MADOS, but ok for others

    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            batch_labels = batch.pop("target")
            if "s1" in batch:
                batch["s1"] = batch["s1"].to(device).to(torch.bfloat16)
            if "s2" in batch:
                batch["s2"] = batch["s2"].to(device).to(torch.bfloat16)
            if "months" in batch:
                batch["months"] = batch["months"].to(device).long()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = finetuned_encoder(**batch)  # (bsz, num_patches, logits_per_patch)
                spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
                logits = rearrange(
                    logits,
                    "b (h w) (c i j) -> b c (h i) (w j)",
                    h=spatial_patches_per_dim,
                    w=spatial_patches_per_dim,
                    c=num_classes,
                    i=patch_size,
                    j=patch_size,
                )
                logits = F.interpolate(
                    logits.float(),
                    size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )  # (bsz, num_classes, H, W)
                loss = loss_function(logits, batch_labels.to(device))

            (loss / grad_accum).backward()

            if ((i + 1) % grad_accum == 0) or (i + 1 == len(data_loader)):
                epoch_fraction = epoch + (i / len(data_loader))
                set_lr = adjust_learning_rate(
                    epoch_fraction, sched_config
                )  # get LR for this epoch
                for g in opt.param_groups:
                    g["lr"] = set_lr  # update

                torch.nn.utils.clip_grad_norm_(finetuned_encoder.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

    return finetuned_encoder


def evaluate_seg(data_loader, finetuned_encoder, num_classes, device):
    finetuned_encoder = finetuned_encoder.eval()
    patch_size = finetuned_encoder.encoder.patch_size

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch_labels = batch.pop("target")
            if "s1" in batch:
                batch["s1"] = batch["s1"].to(device).to(torch.bfloat16)
            if "s2" in batch:
                batch["s2"] = batch["s2"].to(device).to(torch.bfloat16)
            if "months" in batch:
                batch["months"] = batch["months"].to(device).long()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = finetuned_encoder(**batch)  # (bsz, num_patches, logits_per_patch)
                spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
                logits = rearrange(
                    logits,
                    "b (h w) (c i j) -> b c (h i) (w j)",
                    h=spatial_patches_per_dim,
                    w=spatial_patches_per_dim,
                    c=num_classes,
                    i=patch_size,
                    j=patch_size,
                )
                logits = F.interpolate(
                    logits.float(),
                    size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                    mode="bilinear",
                    align_corners=True,
                )  # (bsz, num_classes, H, W)

            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(batch_labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    miou = mean_iou(all_preds, all_labels, num_classes=num_classes, ignore_label=-1)
    return miou
