# -*- coding: utf-8 -*-
"""模型定义: 在 ImageNet 预训练骨干网上替换分类头, 输出 51 类。"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 51  # 50 个目标车型 + unknown


def build_model(arch: str = "resnet18", num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """构建分类模型。

    支持的骨干网: resnet18 / resnet34 / resnet50 / efficientnet_b0 / mobilenet_v3_large。
    预训练权重来自 torchvision 官方 ImageNet 权重, 仅作为特征初始化,
    最终的 51 类分类能力完全由本组数据训练得到。
    """
    weights = None
    if pretrained:
        weights = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
            "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        }.get(arch)
        if weights is None:
            raise ValueError(f"不支持的模型结构: {arch}")

    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型结构: {arch}")
    return model


def classifier_parameters(model: nn.Module) -> list[nn.Parameter]:
    """返回分类头参数(冻结骨干网时用于参数分组)。"""
    for name in ("fc", "classifier"):
        module = getattr(model, name, None)
        if module is not None:
            return list(module.parameters())
    raise ValueError("未找到分类头")


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """冻结/解冻除分类头以外的所有参数。"""
    head_ids = {id(p) for p in classifier_parameters(model)}
    for param in model.parameters():
        if id(param) not in head_ids:
            param.requires_grad = not frozen
