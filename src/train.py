# -*- coding: utf-8 -*-
"""训练脚本: 训练 51 类(50 个目标车型 + unknown)车型分类模型。

用法示例:
    python src/train.py --preset baseline
    python src/train.py --preset improved
    python src/train.py --preset improved --arch resnet34 --epochs 20 --name improved_r34
    python src/train.py --list-presets

每个实验在 runs/<name>/ 下保存配置、逐 epoch 指标和训练日志,
在 models/<name>/best.pt 保存按验证集 Macro-F1 选择的最优模型。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODELS_DIR, RUNS_DIR, TrainConfig, UNKNOWN_INDEX, load_classes  # noqa: E402
from dataset import CarDataset, build_balanced_sampler, build_transform  # noqa: E402
from metrics import accuracy, confusion_matrix, macro_f1  # noqa: E402
from model import NUM_CLASSES, build_model, classifier_parameters, set_backbone_frozen  # noqa: E402

# 实验预设: baseline 为最简基线, 其余用于消融与改进对比
PRESETS: dict[str, dict] = {
    "baseline": dict(
        name="baseline", arch="resnet18", epochs=10, batch_size=32, lr=3e-4,
        label_smoothing=0.0, strong_aug=False, balanced_sampling=False,
    ),
    "ablation_aug_only": dict(
        name="ablation_aug_only", arch="resnet18", epochs=10, batch_size=32, lr=3e-4,
        label_smoothing=0.1, strong_aug=True, balanced_sampling=False,
    ),
    "improved": dict(
        name="improved", arch="resnet18", epochs=14, batch_size=32, lr=3e-4,
        label_smoothing=0.1, strong_aug=True, balanced_sampling=True, warmup_epochs=2,
    ),
    "improved_r34": dict(
        name="improved_r34", arch="resnet34", epochs=12, batch_size=32, lr=2e-4,
        label_smoothing=0.1, strong_aug=True, balanced_sampling=True, warmup_epochs=2,
    ),
    "ablation_no_unknown_cap": dict(
        name="ablation_no_unknown_cap", arch="resnet18", epochs=12, batch_size=32, lr=3e-4,
        label_smoothing=0.1, strong_aug=True, balanced_sampling=True, warmup_epochs=2,
        unknown_weight_cap=100000,
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    head = classifier_parameters(model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [
        {"params": backbone, "lr": cfg.lr * 0.3},
        {"params": head, "lr": cfg.lr},
    ]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"不支持的优化器: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if cfg.scheduler == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    raise ValueError(f"不支持的调度器: {cfg.scheduler}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> dict:
    model.eval()
    all_true, all_pred = [], []
    total_loss, total = 0.0, 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += float(loss) * targets.size(0)
        total += targets.size(0)
        all_pred.append(logits.argmax(1).cpu())
        all_true.append(targets.cpu())
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    cm = confusion_matrix(y_true, y_pred, NUM_CLASSES)
    return {
        "loss": total_loss / max(1, total),
        "macro_f1": macro_f1(y_true, y_pred, NUM_CLASSES),
        "accuracy": accuracy(y_true, y_pred),
        "cm": cm,
    }


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(RUNS_DIR, cfg.name)
    model_dir = os.path.join(MODELS_DIR, cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    class_ids = load_classes()
    train_ds = CarDataset(cfg.data_dir, "train",
                          build_transform(cfg.input_size, True, cfg.strong_aug), class_ids)
    val_ds = CarDataset(cfg.data_dir, "val", build_transform(cfg.input_size, False), class_ids)

    sampler, shuffle = None, True
    if cfg.balanced_sampling:
        sampler = build_balanced_sampler(train_ds, cfg.unknown_weight_cap, cfg.seed)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg.arch, NUM_CLASSES, cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log_path = os.path.join(run_dir, "train.log")
    history_path = os.path.join(run_dir, "history.csv")
    log_file = open(log_path, "a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    config_dump = {k: v for k, v in cfg.__dict__.items() if k != "extra"}
    config_dump.update({
        "device": str(device),
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "unknown_index": UNKNOWN_INDEX,
        "class_ids": class_ids,
    })
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dump, f, ensure_ascii=False, indent=2)

    log(f"实验 {cfg.name} | 结构={cfg.arch} | 设备={device} | 训练集={len(train_ds)} 验证集={len(val_ds)}")
    log(f"增强={'强' if cfg.strong_aug else '弱'} | 均衡采样={cfg.balanced_sampling} | "
        f"标签平滑={cfg.label_smoothing} | epochs={cfg.epochs}")

    best_f1, best_epoch = -1.0, -1
    rows: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_backbone_epochs and epoch == 1:
            set_backbone_frozen(model, True)
            log("冻结骨干网, 仅训练分类头")
        if cfg.freeze_backbone_epochs and epoch == cfg.freeze_backbone_epochs + 1:
            set_backbone_frozen(model, False)
            log("解冻骨干网, 全网络微调")

        model.train()
        running, seen, start = 0.0, 0, time.time()
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss) * targets.size(0)
            seen += targets.size(0)

        metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(running / max(1, seen), 4),
            "val_loss": round(metrics["loss"], 4),
            "val_macro_f1": round(metrics["macro_f1"], 4),
            "val_accuracy": round(metrics["accuracy"], 4),
            "lr": round(optimizer.param_groups[-1]["lr"], 6),
            "seconds": round(time.time() - start, 1),
        }
        rows.append(row)
        log(f"epoch {epoch}/{cfg.epochs}  train_loss={row['train_loss']:.4f}  "
            f"val_loss={row['val_loss']:.4f}  val_macro_f1={row['val_macro_f1']:.4f}  "
            f"val_acc={row['val_accuracy']:.4f}  ({row['seconds']}s)")

        state = {
            "model_state": model.state_dict(),
            "arch": cfg.arch,
            "num_classes": NUM_CLASSES,
            "class_ids": class_ids,
            "unknown_index": UNKNOWN_INDEX,
            "input_size": cfg.input_size,
            "epoch": epoch,
            "val_macro_f1": metrics["macro_f1"],
            "config": config_dump,
        }
        torch.save(state, os.path.join(model_dir, "last.pt"))
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save(state, os.path.join(model_dir, "best.pt"))
            np.save(os.path.join(run_dir, "val_confusion_matrix.npy"), metrics["cm"])

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"name": cfg.name, "arch": cfg.arch,
               "best_val_macro_f1": round(best_f1, 4), "best_epoch": best_epoch,
               "epochs": cfg.epochs, "config": config_dump}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"最优验证集 Macro-F1 = {best_f1:.4f} (epoch {best_epoch}), 模型: {model_dir}/best.pt")
    log_file.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 51 类车型分类模型")
    parser.add_argument("--preset", default="baseline", choices=sorted(PRESETS))
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--arch")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--freeze-backbone-epochs", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--balanced-sampling", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        for key, value in PRESETS.items():
            print(f"{key}: {value}")
        return

    cfg = TrainConfig(**PRESETS[args.preset])
    for key in ("name", "arch", "epochs", "input_size", "num_workers", "freeze_backbone_epochs"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.no_pretrained:
        cfg.pretrained = False
    if args.strong_aug:
        cfg.strong_aug = True
    if args.balanced_sampling:
        cfg.balanced_sampling = True
    train(cfg)


if __name__ == "__main__":
    main()
