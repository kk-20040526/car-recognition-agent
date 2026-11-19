# -*- coding: utf-8 -*-
"""预下载 ImageNet 预训练权重到本地缓存, 之后训练可以完全离线进行。

用法: python scripts/download_weights.py
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from model import build_model  # noqa: E402

ARCHS = ["resnet18", "resnet34", "efficientnet_b0"]


def main() -> None:
    import torch

    cache = torch.hub.get_dir()
    print("权重缓存目录:", cache, "(可用环境变量 TORCH_HOME 修改)")
    os.makedirs(cache, exist_ok=True)
    for arch in ARCHS:
        print(f"下载/校验 {arch} 预训练权重 …")
        build_model(arch, num_classes=51, pretrained=True)
    print("完成。之后除 --no-pretrained 外的训练命令都可以离线运行。")


if __name__ == "__main__":
    main()
