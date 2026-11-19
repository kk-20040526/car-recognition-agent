# -*- coding: utf-8 -*-
"""全局路径与配置。

项目根目录 = 本文件的上上级目录, 所有相对路径都相对它解析,
这样无论从哪个工作目录运行脚本, 都能找到数据集与模型。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSES_FILE = os.path.join(PROJECT_ROOT, "classes.txt")
CLASS_INFO_FILE = os.path.join(PROJECT_ROOT, "data", "class_info.json")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
WEB_DIR = os.path.join(PROJECT_ROOT, "src", "web")

UNKNOWN_LABEL = "unknown"
UNKNOWN_INDEX = 50  # 索引 0~49 按 classes.txt 行顺序对应 50 个目标车型, 索引 50 为 unknown


def load_classes(path: str = CLASSES_FILE) -> list[str]:
    """读取 classes.txt (UTF-8 无 BOM, 恰好 50 行四位 ID, 升序)。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != 50:
        raise ValueError(f"classes.txt 必须恰好 50 行, 当前 {len(ids)} 行")
    if len(set(ids)) != 50:
        raise ValueError("classes.txt 存在重复类别 ID")
    for cid in ids:
        if len(cid) != 4 or not cid.isdigit():
            raise ValueError(f"类别 ID 必须为 4 位数字(保留前导零): {cid}")
    if ids != sorted(ids):
        raise ValueError("classes.txt 必须按类别 ID 升序排列")
    return ids


def index_to_label(index: int, class_ids: list[str] | None = None) -> str:
    """模型输出索引 -> 提交用标签(四位车型 ID 或 unknown)。"""
    class_ids = class_ids if class_ids is not None else load_classes()
    return UNKNOWN_LABEL if index == UNKNOWN_INDEX else class_ids[index]


def label_to_index(label: str, class_ids: list[str] | None = None) -> int:
    """数据集目录名 -> 模型输出索引。"""
    class_ids = class_ids if class_ids is not None else load_classes()
    return UNKNOWN_INDEX if label == UNKNOWN_LABEL else class_ids.index(label)


@dataclass
class TrainConfig:
    """一次训练实验的全部超参数。"""

    name: str = "baseline"
    arch: str = "resnet18"
    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    input_size: int = 224
    pretrained: bool = True
    strong_aug: bool = False
    balanced_sampling: bool = False
    unknown_weight_cap: int = 400
    freeze_backbone_epochs: int = 0
    warmup_epochs: int = 1
    num_workers: int = 4
    seed: int = 42
    data_dir: str = DATASET_DIR
    class_info: str = CLASS_INFO_FILE
    extra: dict = field(default_factory=dict)
