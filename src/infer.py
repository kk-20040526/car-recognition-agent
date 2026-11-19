# -*- coding: utf-8 -*-
"""推理脚本: 单张图片识别 + 批量推理生成 predictions.csv。

用法:
    # 单张图片
    python src/infer.py image --image path/to/car.jpg --model models/improved/best.pt

    # 批量推理(教师指定的测试目录)
    python src/infer.py batch --input path/to/test_images --model models/improved/best.pt \
        --output predictions.csv

    # 在本地留出的测试集上评估
    python src/infer.py batch --input dataset/test_local --model models/improved/best.pt \
        --output runs/improved/test_local_predictions.csv --evaluate

输出标签规则: 索引 0~49 -> classes.txt 中对应的四位车型 ID; 索引 50 -> unknown。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UNKNOWN_INDEX, index_to_label, load_classes  # noqa: E402
from dataset import IMG_EXTENSIONS, build_transform  # noqa: E402
from metrics import confusion_matrix, macro_f1, per_class_f1  # noqa: E402
from model import NUM_CLASSES, build_model  # noqa: E402


class _PathDataset(Dataset):
    """只负责读图与预处理的数据集, 用于多进程并行加载(大文件夹时显著提速)。"""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), index


class InferenceModel:
    """封装模型加载与推理, Web 端与命令行共用。"""

    def __init__(self, checkpoint: str, device: str | None = None):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.arch = state.get("arch", "resnet18")
        self.input_size = int(state.get("input_size", 224))
        self.class_ids = state.get("class_ids") or load_classes()
        self.unknown_index = int(state.get("unknown_index", UNKNOWN_INDEX))
        self.model = build_model(self.arch, state.get("num_classes", NUM_CLASSES), pretrained=False)
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()
        self.transform = build_transform(self.input_size, train=False)
        self.metadata = {k: v for k, v in state.items() if k not in ("model_state", "config")}
        self.metadata["trained_config"] = state.get("config", {})

    def label_of(self, index: int) -> str:
        return index_to_label(index, self.class_ids)

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 3) -> dict:
        """单张图片推理, 返回标签、置信度与 Top-K。"""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            logits = self.model(tensor)
        probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
        order = np.argsort(-probs)[:top_k]
        return {
            "image": os.path.basename(image_path),
            "predicted_label": self.label_of(int(order[0])),
            "confidence": float(probs[order[0]]),
            "predicted_index": int(order[0]),
            "is_unknown": bool(order[0] == self.unknown_index),
            "top_k": [
                {"label": self.label_of(int(i)), "index": int(i), "confidence": float(probs[i])}
                for i in order
            ],
        }

    @torch.no_grad()
    def predict_many(self, image_paths: list[str], batch_size: int = 32,
                     threshold: float = 0.0) -> list[dict]:
        """批量推理。threshold > 0 时, 置信度低于该值的预测改判为 unknown(拒识)。"""
        results = []
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start:start + batch_size]
            tensors = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in chunk]).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = self.model(tensors)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            for path, prob in zip(chunk, probs):
                top = int(prob.argmax())
                confidence = float(prob[top])
                index = top
                if threshold > 0 and top != self.unknown_index and confidence < threshold:
                    index = self.unknown_index
                results.append({
                    "image_id": os.path.basename(path),
                    "predicted_label": self.label_of(index),
                    "confidence": confidence,
                })
        return results

    @torch.no_grad()
    def logits_many(self, image_paths: list[str], batch_size: int = 32,
                    num_workers: int = 0, progress=None) -> np.ndarray:
        """返回未经 softmax 的原始分类分数, 形状 (N, 51)。

        EduBench 的 result.npz 要求保存模型对 51 个类别的原始打分, 因此这里不改动数值。
        num_workers > 0 时用多进程并行读图与预处理(适合上万张图片的文件夹),
        progress(已处理, 总数) 用于向页面汇报进度。
        """
        chunks: list[np.ndarray] = []
        done = 0

        def forward(tensors):
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                return self.model(tensors).float().cpu().numpy()

        if num_workers > 0 and len(image_paths) >= 200:
            loader = DataLoader(_PathDataset(image_paths, self.transform), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=False)
            for tensors, _ in loader:
                chunks.append(forward(tensors.to(self.device)))
                done += tensors.size(0)
                if progress:
                    progress(min(done, len(image_paths)), len(image_paths))
        else:
            for start in range(0, len(image_paths), batch_size):
                chunk = image_paths[start:start + batch_size]
                tensors = torch.stack([self.transform(Image.open(p).convert("RGB"))
                                       for p in chunk]).to(self.device)
                chunks.append(forward(tensors))
                done += len(chunk)
                if progress:
                    progress(min(done, len(image_paths)), len(image_paths))
        if not chunks:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32)


def collect_images(path: str) -> list[str]:
    """收集待推理图片。目录存在时递归查找(不含真实标签); 也支持传入单个文件。"""
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for name in sorted(names):
            if name.lower().endswith(IMG_EXTENSIONS):
                files.append(os.path.join(root, name))
    return sorted(files)


def write_predictions(rows: list[dict], output: str) -> None:
    """按任务书规定写出 predictions.csv (UTF-8, 表头固定)。"""
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_label", "confidence"])
        for row in rows:
            writer.writerow([row["image_id"], row["predicted_label"], f"{row['confidence']:.6f}"])


def evaluate_predictions(rows: list[dict], root: str, class_ids: list[str]) -> dict:
    """在本地测试集(按类别目录组织)上计算 51 类 Macro-F1。"""
    y_true, y_pred = [], []
    for row in rows:
        truth = None
        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path) and os.path.isfile(os.path.join(folder_path, row["image_id"])):
                truth = folder
                break
        if truth is None:
            continue
        y_true.append(UNKNOWN_INDEX if truth == "unknown" else class_ids.index(truth))
        y_pred.append(UNKNOWN_INDEX if row["predicted_label"] == "unknown"
                      else class_ids.index(row["predicted_label"]))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, NUM_CLASSES)
    labels = class_ids + ["unknown"]
    f1 = per_class_f1(cm)
    worst = sorted(range(len(labels)), key=lambda i: f1[i])[:5]
    return {
        "num_images": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": macro_f1(y_true, y_pred, NUM_CLASSES),
        "per_class_f1": {labels[i]: round(float(f1[i]), 4) for i in range(len(labels))},
        "worst_classes": [(labels[i], round(float(f1[i]), 4)) for i in worst],
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="车型分类推理")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("image", help="单张图片推理")
    single.add_argument("--image", required=True)
    single.add_argument("--model", required=True)
    single.add_argument("--top-k", type=int, default=3)

    batch = sub.add_parser("batch", help="批量推理并生成 predictions.csv")
    batch.add_argument("--input", required=True, help="图片目录或单个文件")
    batch.add_argument("--model", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--batch-size", type=int, default=32)
    batch.add_argument("--threshold", type=float, default=0.0,
                       help="拒识阈值: 置信度低于该值的预测改判为 unknown, 默认 0(不拒识)")
    batch.add_argument("--evaluate", action="store_true", help="输入目录按类别组织时计算 Macro-F1")
    batch.add_argument("--eval-report")

    args = parser.parse_args()
    infer = InferenceModel(args.model)

    if args.command == "image":
        print(json.dumps(infer.predict(args.image, args.top_k), ensure_ascii=False, indent=2))
        return

    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"未在 {args.input} 找到图片")
    rows = infer.predict_many(images, args.batch_size, threshold=args.threshold)
    if args.threshold > 0:
        rejected = sum(1 for row in rows if row["predicted_label"] == "unknown")
        print(f"拒识阈值 {args.threshold:.2f}: 最终判为 unknown 的图片 {rejected} 张")
    write_predictions(rows, args.output)
    print(f"已写入 {len(rows)} 条预测 -> {args.output}")

    if args.evaluate:
        report = evaluate_predictions(rows, args.input, infer.class_ids)
        print(f"本地测试集: {report['num_images']} 张, Accuracy={report['accuracy']:.4f}, "
              f"Macro-F1={report['macro_f1']:.4f}")
        print("F1 最低的 5 个类别:", report["worst_classes"])
        path = args.eval_report or os.path.join(os.path.dirname(args.output), "eval_report.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"评估报告 -> {path}")


if __name__ == "__main__":
    main()
