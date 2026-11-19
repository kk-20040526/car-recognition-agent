# -*- coding: utf-8 -*-
"""模型评估与分析: 51 类 Macro-F1、每类指标、混淆矩阵、错误样本与拒识阈值标定。

用法:
    python src/evaluate.py --model models/improved/best.pt --split val
    python src/evaluate.py --model models/improved/best.pt --split test_local --out runs/improved
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR, RUNS_DIR, UNKNOWN_INDEX, load_classes  # noqa: E402
from infer import InferenceModel  # noqa: E402
from metrics import (confusion_matrix, known_false_reject_rate, per_class_f1,  # noqa: E402
                     plot_confusion, unknown_detection_rate)


def collect_split(split_dir: str, class_ids: list[str]) -> tuple[list[str], np.ndarray]:
    """遍历按类别组织的目录, 返回图片路径与真实标签索引。"""
    paths, labels = [], []
    for folder in sorted(os.listdir(split_dir)):
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder == "unknown":
            index = UNKNOWN_INDEX
        elif folder in class_ids:
            index = class_ids.index(folder)
        else:
            continue
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                paths.append(os.path.join(folder_path, name))
                labels.append(index)
    return paths, np.array(labels)


def calibrate_threshold(confidences: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """在验证集上按 Macro-F1 选择"低置信度 -> unknown"的拒识阈值。

    只对预测为非 unknown 的样本生效: 置信度低于阈值的样本改判为 unknown。
    """
    best = {"threshold": 0.0, "macro_f1": -1.0}
    for threshold in np.arange(0.0, 0.96, 0.02):
        adjusted = y_pred.copy()
        adjusted[(confidences < threshold) & (y_pred != UNKNOWN_INDEX)] = UNKNOWN_INDEX
        score = float(per_class_f1(confusion_matrix(y_true, adjusted, 51)).mean())
        if score > best["macro_f1"]:
            best = {"threshold": round(float(threshold), 2), "macro_f1": round(score, 4)}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="评估与错误分析")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test_local", "train"])
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--out", default=None, help="分析结果输出目录, 默认 runs/<模型名>")
    args = parser.parse_args()

    class_ids = load_classes()
    split_dir = os.path.join(args.dataset, args.split)
    out_dir = args.out or os.path.join(RUNS_DIR, os.path.basename(os.path.dirname(args.model)))
    os.makedirs(out_dir, exist_ok=True)

    infer = InferenceModel(args.model)
    paths, y_true = collect_split(split_dir, class_ids)
    print(f"{args.split}: {len(paths)} 张图片, 模型 {args.model} ({infer.arch})")

    rows = infer.predict_many(paths, batch_size=32)
    labels = class_ids + ["unknown"]
    y_pred = np.array([UNKNOWN_INDEX if r["predicted_label"] == "unknown"
                       else class_ids.index(r["predicted_label"]) for r in rows])
    confidences = np.array([r["confidence"] for r in rows])

    cm = confusion_matrix(y_true, y_pred, 51)
    f1 = per_class_f1(cm)
    report = {
        "model": args.model,
        "arch": infer.arch,
        "split": args.split,
        "num_images": len(paths),
        "macro_f1": round(float(f1.mean()), 4),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "unknown_detection_rate": round(unknown_detection_rate(cm), 4),
        "known_false_reject_rate": round(known_false_reject_rate(cm), 4),
        "per_class_f1": {labels[i]: round(float(f1[i]), 4) for i in range(51)},
        "threshold_calibration": calibrate_threshold(confidences, y_true, y_pred),
    }
    report["lowest_f1_classes"] = sorted(report["per_class_f1"].items(), key=lambda kv: kv[1])[:8]
    np.save(os.path.join(out_dir, f"{args.split}_confusion_matrix.npy"), cm)
    plot_confusion(cm, labels, os.path.join(out_dir, f"{args.split}_confusion_matrix.png"))

    # 典型错误样本: 从混淆矩阵里挑出错得最多的类别对, 并列出具体文件
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-errors.ravel()), errors.shape))[0][:8]
    top_confusions = []
    for true_i, pred_i in pairs:
        if errors[true_i, pred_i] == 0:
            continue
        samples = []
        for path, t, r in zip(paths, y_true, rows):
            if t == true_i and r["predicted_label"] == labels[pred_i] and len(samples) < 3:
                samples.append({"image": os.path.basename(path), "confidence": round(r["confidence"], 4)})
        top_confusions.append({
            "true": labels[int(true_i)], "predicted": labels[int(pred_i)],
            "count": int(errors[true_i, pred_i]), "examples": samples,
        })
    report["top_confusions"] = top_confusions

    with open(os.path.join(out_dir, f"{args.split}_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Macro-F1 = {report['macro_f1']:.4f} | Accuracy = {report['accuracy']:.4f}")
    print(f"unknown 检出率 = {report['unknown_detection_rate']:.4f} | "
          f"已知类误拒率 = {report['known_false_reject_rate']:.4f}")
    print(f"拒识阈值标定: {report['threshold_calibration']}")
    print("F1 最低的类别:", report["lowest_f1_classes"])
    print("主要混淆对:")
    for item in top_confusions[:5]:
        print(f"  {item['true']} -> {item['predicted']} 共 {item['count']} 张, 例: {item['examples']}")
    print(f"结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
