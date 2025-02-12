import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import accuracy_score, f1_score

from src.galileo import adjust_learning_rate

from .metrics import mean_iou

PROBING_LRs = {
    "LP": [
        1e-4,
        3e-4,
        5e-4,
        8e-4,
        1e-3,
        3e-3,
        5e-3,
        8e-3,
        1e-2,
        3e-2,
        5e-2,
        8e-2,
        1e-1,
        3e-1,
        5e-1,
        8e-1,
    ],
}


def train_and_eval_probe_cls(lr, config, loaders, in_features, device):
    probe = train_probe_cls(
        data_loader=loaders["train"],
        lr=lr,
        epochs=50,
        in_features=in_features,
        num_classes=config["num_classes"],
        is_multilabel=config["is_multilabel"],
        device=device,
    )
    val_acc = evaluate_probe_cls(
        data_loader=loaders["valid"],
        probe=probe,
        is_multilabel=config["is_multilabel"],
        device=device,
    )
    test_acc = evaluate_probe_cls(
        data_loader=loaders["test"],
        probe=probe,
        is_multilabel=config["is_multilabel"],
        device=device,
    )
    return val_acc, test_acc


def train_and_eval_probe_seg(lr, config, loaders, in_features, grid_size, device):
    output_patch_size = math.ceil(config["segmentation_map_height_width"] / grid_size)
    probe = train_probe_seg(
        data_loader=loaders["train"],
        lr=lr,
        epochs=50,
        in_features=in_features,
        num_classes=config["num_classes"],
        patch_size=output_patch_size,
        device=device,
    )
    val_miou = evaluate_probe_seg(
        data_loader=loaders["valid"],
        probe=probe,
        num_classes=config["num_classes"],
        patch_size=output_patch_size,
        device=device,
    )
    test_miou = evaluate_probe_seg(
        data_loader=loaders["test"],
        probe=probe,
        num_classes=config["num_classes"],
        patch_size=output_patch_size,
        device=device,
    )
    return val_miou, test_miou


def train_probe_cls(
    data_loader,
    lr,
    epochs,
    in_features,
    num_classes,
    is_multilabel,
    device,
):
    probe = nn.Sequential(nn.BatchNorm1d(in_features), nn.Linear(in_features, num_classes)).to(
        device
    )
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    sched_config = {
        "lr": lr,
        "warmup_epochs": int(epochs * 0.1),
        "min_lr": 1.0e-5,
        "epochs": epochs,
    }
    probe = probe.train()

    if is_multilabel:
        loss_function = nn.MultiLabelSoftMarginLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            batch_emb, batch_labels = batch  # (bsz, dim), (bsz)
            batch_emb = batch_emb.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = probe(batch_emb)  # (bsz, num_classes)
                loss = loss_function(logits, batch_labels.to(device))

            loss.backward()
            adjust_learning_rate(
                optimizer=opt,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=sched_config["epochs"],
                warmup_epochs=sched_config["warmup_epochs"],
                max_lr=sched_config["lr"],
                min_lr=sched_config["min_lr"],
            )

            opt.step()
            opt.zero_grad()

    return probe


def evaluate_probe_cls(data_loader, probe, is_multilabel, device):
    probe = probe.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch_emb, batch_labels = batch  # (bsz, dim), (bsz)
            batch_emb = batch_emb.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                batch_logits = probe(batch_emb)  # (bsz, num_classes)

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


def train_probe_seg(
    data_loader,
    lr,
    epochs,
    in_features,
    num_classes,
    patch_size,
    probe_type,
    device,
):
    logits_per_patch = int(num_classes * patch_size * patch_size)
    assert probe_type in ["LP", "MLP"]
    if probe_type == "LP":
        probe = nn.Sequential(nn.Linear(in_features, logits_per_patch)).to(device)
    else:
        probe = nn.Sequential(
            nn.Linear(in_features, 2048), nn.GELU(), nn.Linear(2048, logits_per_patch)
        ).to(device)

    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    sched_config = {
        "lr": lr,
        "warmup_epochs": int(epochs * 0.1),
        "min_lr": 1.0e-5,
        "epochs": epochs,
    }
    probe = probe.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # for MADOS, but ok for others

    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            batch_emb, batch_labels = batch  # (bsz, num_patches, dim), (bsz, H, W)
            spatial_patches_per_dim = int(batch_emb.shape[1] ** 0.5)
            batch_emb = batch_emb.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = probe(batch_emb)  # (bsz, num_patches, logits_per_patch)

                # this is a bit hackey
                if batch_labels.shape[1] == batch_labels.shape[2]:
                    logits = rearrange(
                        logits,
                        "b (h w) (c i j) -> b c (h i) (w j)",
                        h=spatial_patches_per_dim,
                        w=spatial_patches_per_dim,
                        c=num_classes,
                        i=patch_size,
                        j=patch_size,
                    )
                    if logits.shape[-2] != batch_labels.shape[-2]:
                        logits = F.interpolate(
                            logits,
                            size=(batch_labels.shape[-2], batch_labels.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        )  # (bsz, num_classes, H, W)
                else:
                    # otherwise, we subsampled in the get_embeddings step
                    logits = rearrange(
                        logits,
                        "b t (c i j) -> b c t (i j)",
                        c=num_classes,
                        i=patch_size,
                        j=patch_size,
                    )
                loss = loss_function(logits, batch_labels.to(device))

            loss.backward()
            adjust_learning_rate(
                optimizer=opt,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=sched_config["epochs"],
                warmup_epochs=sched_config["warmup_epochs"],
                max_lr=sched_config["lr"],
                min_lr=sched_config["min_lr"],
            )

            opt.step()
            opt.zero_grad()

    return probe


def evaluate_probe_seg(
    data_loader,
    probe,
    num_classes,
    patch_size,
    device,
):
    probe = probe.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch_emb, batch_labels = batch  # (bsz, num_patches, dim), (bsz, H, W)
            spatial_patches_per_dim = int(batch_emb.shape[1] ** 0.5)
            batch_emb = batch_emb.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = probe(batch_emb)  # (bsz, num_patches, logits_per_patch)
                logits = rearrange(
                    logits,
                    "b (h w) (c i j) -> b c (h i) (w j)",
                    h=spatial_patches_per_dim,
                    w=spatial_patches_per_dim,
                    c=num_classes,
                    i=patch_size,
                    j=patch_size,
                )
                if logits.shape[-2] != batch_labels.shape[-2]:
                    logits = F.interpolate(
                        logits,
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
