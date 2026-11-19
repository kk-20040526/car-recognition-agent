# -*- coding: utf-8 -*-
"""测试集批量测试: 对一个文件夹内的全部图片做推理, 统计结果并写出 CSV。

支持两种输入:
    1) 本机文件夹路径(推荐): 直接扫描目录, 不复制图片;
    2) 浏览器上传的文件夹: 图片先落到 storage/batch/<run_id>/input/ 再测试。

目录里如果按类别分子文件夹存放(目录名等于 classes.txt 的类别 ID 或 unknown),
就自动计算 51 类 Macro-F1、Accuracy、unknown 检出率、已知类误拒率与混淆矩阵;
如果是平铺的一堆图片(没有真实标签), 则只给出预测分布、平均置信度与低置信度数量。
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np

from config import UNKNOWN_INDEX, index_to_label
from metrics import (accuracy, confusion_matrix, known_false_reject_rate, macro_f1,
                     per_class_f1, unknown_detection_rate)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_IMAGES = 200000        # 单次批量测试的图片上限, 避免误选整个磁盘
LARGE_WARNING = 20000      # 超过这个数量时页面提示预计耗时
PREVIEW_ROWS = 60          # 返回给页面的明细行数上限


def collect_folder(root: str, class_ids: list[str]) -> tuple[list[str], list[int | None]]:
    """递归收集图片, 并按所在子目录名推断真实标签(没有则为 None)。"""
    paths: list[str] = []
    labels: list[int | None] = []
    for dirpath, dirnames, filenames in os.walk(root):
        folder = os.path.basename(dirpath)
        if folder == "unknown":
            truth: int | None = UNKNOWN_INDEX
        elif folder in class_ids:
            truth = class_ids.index(folder)
        else:
            truth = None
        for name in sorted(filenames):
            if name.lower().endswith(IMG_EXTENSIONS):
                paths.append(os.path.join(dirpath, name))
                labels.append(truth)
    return paths, labels


def run_batch(infer, paths: list[str], labels: list[int | None], class_ids: list[str],
              batch_size: int = 32, preview_rows: int = PREVIEW_ROWS,
              name_of: dict[str, str] | None = None, threshold: float = 0.0,
              num_workers: int = 0, progress=None) -> dict:
    """执行批量推理并汇总结果。infer 为 InferenceModel 实例。

    name_of 是“类别 ID -> 车型名称”的映射, 用于页面明细与明细表里显示中文名。
    threshold > 0 时按拒识规则处理: 置信度低于阈值的预测改判为 unknown(与提交文件保持一致)。
    num_workers > 0 时并行读图, 适合上万张图片的文件夹; progress(已处理, 总数) 汇报进度。
    """
    name_of = name_of or {}

    def label_text(index: int) -> str:
        label = index_to_label(index, class_ids)
        return name_of.get(label, label)

    started = time.time()
    logits = infer.logits_many(paths, batch_size=batch_size, num_workers=num_workers,
                               progress=progress)
    confidence = np.empty(len(paths), dtype=np.float32)
    for start in range(0, len(paths), 2048):        # 分块算 softmax, 避免为上万张图分配整块概率矩阵
        block = logits[start:start + 2048].astype(np.float32, copy=True)
        block -= block.max(axis=1, keepdims=True)
        np.exp(block, out=block)
        block /= block.sum(axis=1, keepdims=True)
        confidence[start:start + len(block)] = block.max(axis=1)
    raw_predicted = logits.argmax(axis=1)
    predicted = raw_predicted.copy()
    if threshold > 0:
        predicted[(confidence < threshold) & (predicted != UNKNOWN_INDEX)] = UNKNOWN_INDEX

    rows = []
    for position, truth in enumerate(labels):
        index = int(predicted[position])
        name = os.path.basename(paths[position])
        rows.append({
            "image_id": name,
            "image_stem": os.path.splitext(name)[0],
            "predicted_label": index_to_label(index, class_ids),
            "predicted_name": label_text(index),
            "confidence": round(float(confidence[position]), 6),
            "raw_label": index_to_label(int(raw_predicted[position]), class_ids),
            "predicted_index": index,
            "true_index": truth,
            "correct": None if truth is None else int(truth == index),
        })

    labeled = [row for row in rows if row["true_index"] is not None]
    summary: dict = {
        "total": len(rows),
        "labeled": len(labeled),
        "has_ground_truth": len(labeled) == len(rows) and len(rows) > 0,
        "partially_labeled": 0 < len(labeled) < len(rows),
        "elapsed_seconds": round(time.time() - started, 2),
        "avg_confidence": round(float(np.mean([row["confidence"] for row in rows])), 4) if rows else 0.0,
    }

    if labeled:
        y_true = np.array([row["true_index"] for row in labeled])
        y_pred = np.array([row["predicted_index"] for row in labeled])
        matrix = confusion_matrix(y_true, y_pred, 51)
        f1 = per_class_f1(matrix)
        labels = class_ids + ["unknown"]
        summary.update({
            "num_classes": 51,
            "macro_f1": round(macro_f1(y_true, y_pred, 51), 4),
            # 只对测试集中真实出现过的类别求平均, 便于理解“只测了部分类别”的文件夹
            "macro_f1_present": round(float(np.mean(
                [f1[i] for i in sorted({int(v) for v in y_true})])), 4),
            "present_classes": len({int(v) for v in y_true}),
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "unknown_detection_rate": round(unknown_detection_rate(matrix), 4),
            "known_false_reject_rate": round(known_false_reject_rate(matrix), 4),
            "per_class_f1": {labels[i]: round(float(f1[i]), 4) for i in range(51)},
            "confusion_matrix": matrix.tolist(),
            "labels": labels,
        })
        summary["lowest_f1_classes"] = sorted(
            [(labels[i], round(float(f1[i]), 4)) for i in range(51)], key=lambda item: item[1])[:8]

    # 预测分布与低置信度统计(平铺目录也有用)
    distribution: dict[str, int] = {}
    for row in rows:
        distribution[row["predicted_label"]] = distribution.get(row["predicted_label"], 0) + 1
    summary["distribution"] = sorted(distribution.items(), key=lambda item: -item[1])[:15]
    summary["unknown_count"] = distribution.get("unknown", 0)
    summary["low_confidence_count"] = sum(1 for row in rows if row["confidence"] < 0.55)
    summary["misclassified"] = sum(1 for row in labeled if row["correct"] == 0)
    summary["threshold"] = threshold
    summary["raw_unknown_count"] = int((raw_predicted == UNKNOWN_INDEX).sum())
    summary["rejected_by_threshold"] = int((raw_predicted != predicted).sum())

    preview = []
    for row in rows[:preview_rows]:
        preview.append({
            "image_id": row["image_id"],
            "predicted_label": row["predicted_label"],
            "car_name": row["predicted_name"],
            "confidence": round(row["confidence"], 4),
            "true_label": None if row["true_index"] is None
            else label_text(row["true_index"]),
            "correct": row["correct"],
        })
    return {"rows": rows, "summary": summary, "preview": preview, "logits": logits,
            "image_stems": [os.path.splitext(os.path.basename(path))[0] for path in paths]}


def write_predictions(rows: list[dict], path: str) -> None:
    """按任务书规定写出 predictions.csv: image_id,predicted_label,confidence。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_label", "confidence"])
        for row in rows:
            writer.writerow([row["image_id"], row["predicted_label"], f"{row['confidence']:.6f}"])


def write_results(rows: list[dict], path: str, class_ids: list[str]) -> None:
    """写出带真实标签与对错标记的明细表, 便于人工检查错误样本。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_label", "predicted_name", "confidence",
                         "true_label", "correct"])
        for row in rows:
            writer.writerow([
                row["image_id"],
                row["predicted_label"],
                row.get("predicted_name") or index_to_label(row["predicted_index"], class_ids),
                f"{row['confidence']:.6f}",
                "" if row["true_index"] is None else index_to_label(row["true_index"], class_ids),
                "" if row["correct"] is None else row["correct"],
            ])


def write_result_csv(rows: list[dict], path: str) -> None:
    """写出 EduBench 需要的 result.csv: 只有两列, image_id 为文件名去掉扩展名。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_label"])
        for row in rows:
            writer.writerow([row["image_stem"], row["predicted_label"]])


def write_result_npz(stems: list[str], logits: np.ndarray, path: str) -> None:
    """写出 EduBench 需要的 result.npz: image_id(Unicode 字符串) 与 logits(float32)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path,
                        image_id=np.array(stems, dtype=str),
                        logits=np.asarray(logits, dtype=np.float32))


def check_submission(stems: list[str], rows: list[dict], logits: np.ndarray,
                     npz_path: str) -> dict:
    """对批量测试生成的三份提交文件做快速自检, 结果回显到页面。"""
    issues: list[str] = []
    labels = {row["predicted_label"] for row in rows}
    bad = [label for label in labels
           if label != "unknown" and (len(label) != 4 or not label.isdigit())]
    if len(stems) != len(set(stems)):
        issues.append("图片编号存在重复")
    if bad:
        issues.append(f"存在非法标签: {bad[:3]}")
    if logits.shape != (len(stems), 51):
        issues.append(f"logits 形状异常: {logits.shape}")
    if not np.isfinite(logits).all():
        issues.append("logits 含 NaN/Inf")
    size = os.path.getsize(npz_path) if os.path.isfile(npz_path) else 0
    if size > 20 * 1024 * 1024:
        issues.append("result.npz 超过 20 MiB")
    return {
        "ok": not issues,
        "issues": issues,
        "rows": len(rows),
        "unique_ids": len(set(stems)),
        "unknown": sum(1 for row in rows if row["predicted_label"] == "unknown"),
        "npz_mib": round(size / 1024 / 1024, 3),
    }
