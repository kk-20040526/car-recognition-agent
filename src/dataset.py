# -*- coding: utf-8 -*-
"""数据集与数据增强。

目录结构 (dataset/):
    train/<类别 ID>/xxx.jpg ... train/unknown/xxx.jpg
    val/<类别 ID>/xxx.jpg   ... val/unknown/xxx.jpg
类别 ID 取自 classes.txt, 索引 0~49 按 classes.txt 顺序, 索引 50 固定为 unknown。
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from config import UNKNOWN_INDEX, load_classes

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_transform(input_size: int, train: bool, strong: bool = False) -> transforms.Compose:
    """训练/验证两套预处理。训练集增强强度由 strong 控制(用于消融对比)。"""
    if not train:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if not strong:
        ops = [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops = [
            transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomRotation(10),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if strong:
        # 随机擦除: 模拟遮挡, 提升对局部遮挡与背景干扰的鲁棒性
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(ops)


class CarDataset(Dataset):
    """按目录读取图片, 返回 (tensor, 类别索引, 文件名)。"""

    def __init__(self, root: str, split: str, transform, class_ids: list[str] | None = None):
        self.class_ids = class_ids if class_ids is not None else load_classes()
        self.split_dir = os.path.join(root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.split_dir}")
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        # 目录名 -> 索引: 50 个车型 ID 按 classes.txt 顺序, unknown 固定为 50
        for folder in sorted(os.listdir(self.split_dir)):
            folder_path = os.path.join(self.split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder == "unknown":
                index = UNKNOWN_INDEX
            elif folder in self.class_ids:
                index = self.class_ids.index(folder)
            else:
                continue  # 不在本组 50 类内的目录直接忽略
            for name in sorted(os.listdir(folder_path)):
                if name.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append((os.path.join(folder_path, name), index))
        if not self.samples:
            raise RuntimeError(f"{self.split_dir} 下没有找到任何图片")

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(self.class_ids) + 1, dtype=np.int64)
        for _, index in self.samples:
            counts[index] += 1
        return counts

    def __getitem__(self, i: int):
        path, index = self.samples[i]
        image = Image.open(path).convert("RGB")
        return self.transform(image), index, os.path.basename(path)


def build_balanced_sampler(dataset: CarDataset, unknown_cap: int, seed: int = 42) -> WeightedRandomSampler:
    """类别均衡采样: 每个类别权重 ~ 1/有效样本数。

    unknown 类样本来自大量组外车型, 有效样本数按 unknown_cap 截断,
    避免 unknown 因样本量过大而主导每个 batch。
    """
    counts = dataset.class_counts().astype(np.float64)
    effective = counts.copy()
    effective[UNKNOWN_INDEX] = min(counts[UNKNOWN_INDEX], unknown_cap)
    effective = np.maximum(effective, 1.0)
    weights = 1.0 / effective
    sample_weights = torch.as_tensor([weights[index] for _, index in dataset.samples], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True, generator=generator)
