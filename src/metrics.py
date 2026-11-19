# -*- coding: utf-8 -*-
"""评价指标: 51 类 Macro-F1(唯一评分指标)及辅助分析指标。"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 51) -> np.ndarray:
    """行 = 真实类别, 列 = 预测类别。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """逐类 F1: F1 = 2*TP / (2*TP + FP + FN); 分母为 0 时该类别按 0 处理。"""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 51) -> float:
    """51 个类别等权平均的 Macro-F1。"""
    return float(per_class_f1(confusion_matrix(y_true, y_pred, num_classes)).mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def unknown_detection_rate(cm: np.ndarray, unknown_index: int = 50) -> float:
    """未知类检出率: 真实为 unknown 且被预测为 unknown 的比例。"""
    total = cm[unknown_index].sum()
    return float(cm[unknown_index, unknown_index] / total) if total else 0.0


def known_false_reject_rate(cm: np.ndarray, unknown_index: int = 50) -> float:
    """已知类误拒率: 真实为 50 个目标车型但被误判为 unknown 的比例。"""
    known = np.delete(cm, unknown_index, axis=0)
    total = known.sum()
    return float(known[:, unknown_index].sum() / total) if total else 0.0


def plot_confusion(cm: np.ndarray, labels: list[str], path: str, normalize: bool = True) -> None:
    """保存混淆矩阵图(默认按真实类别归一化)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(np.float64)
    if normalize:
        row = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row, out=np.zeros_like(data), where=row > 0)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
