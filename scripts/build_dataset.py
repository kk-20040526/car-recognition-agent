# -*- coding: utf-8 -*-
"""数据准备脚本: 生成 classes.txt 与 51 类数据集(50 个目标车型 + unknown)。

处理流程:
    1. 从原始训练集目录中按类别 ID 升序取前 50 类作为本组目标车型, 写 classes.txt;
    2. 其余类别作为 unknown 来源, 按品牌分散抽样(每类 30 张)组成 unknown 训练集;
    3. 官方验证集(val)按 6:4 划分: 60% 作为调参用的验证集, 40% 作为本地测试集;
    4. 目标车型与 unknown 的样本分别落到 dataset/{train,val,test_local}/ 下,
       Windows 上使用硬链接, 不额外占用磁盘空间。

用法: python scripts/build_dataset.py
运行前请修改下面的原始数据路径。
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# ---------------------------------------------------------------- 原始数据位置
# 2026-11-10 改为本机路径：数据集已从 数据集.zip 解压到 01_车型识别Agent\数据集_解压\
# 原值（队友机器）留作注释，换机器时改回或改成自己的路径即可：
#   RAW_TRAIN_DIR = r"C:\Users\陈双清\Desktop\train"
#   RAW_ARCHIVE   = r"C:\Users\陈双清\Desktop\数据集.zip"
RAW_TRAIN_DIR = r"C:\Users\20937\Desktop\01_车型识别Agent\数据\数据集_解压\train"   # 827 类训练图片目录
RAW_ARCHIVE = r"C:\Users\20937\Desktop\01_车型识别Agent\数据\数据集.zip"                              # 含 val/ 与 cls_info/class_info.json

CLASS_INFO_OUT = os.path.join(PROJECT_ROOT, "data", "class_info.json")
CLASSES_TXT = os.path.join(PROJECT_ROOT, "classes.txt")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

TARGET_COUNT = 50               # 目标车型类别数
UNKNOWN_SOURCE_COUNT = 50       # unknown 样本来自多少个组外类别
UNKNOWN_PER_CLASS_TRAIN = 30    # 每个来源类别的训练样本数
UNKNOWN_PER_CLASS_VAL = 8       # 每个来源类别的验证样本数
UNKNOWN_PER_CLASS_TEST = 6      # 每个来源类别的本地测试样本数
VAL_RATIO = 0.6                 # 官方验证集中用于调参的比例
SEED = 20260916


def link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_type(type_info: str) -> tuple[str, str]:
    parts = (type_info.split("==") + [""])[0].split("####")
    return (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else "")


def main() -> None:
    if not os.path.isdir(RAW_TRAIN_DIR):
        raise SystemExit(f"未找到原始训练集目录: {RAW_TRAIN_DIR}")

    # 1) 提取类别说明文件
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    with zipfile.ZipFile(RAW_ARCHIVE) as zf:
        if not os.path.isfile(CLASS_INFO_OUT):
            with zf.open("cls_info/class_info.json") as src, open(CLASS_INFO_OUT, "wb") as dst:
                shutil.copyfileobj(src, dst)
    with open(CLASS_INFO_OUT, "r", encoding="utf-8") as f:
        info = {record["id"]: record for record in json.load(f)}

    # 2) 选定 50 个目标类别并写 classes.txt
    folders = sorted(d for d in os.listdir(RAW_TRAIN_DIR)
                     if os.path.isdir(os.path.join(RAW_TRAIN_DIR, d)))
    if len(folders) < TARGET_COUNT:
        raise SystemExit(f"训练集只有 {len(folders)} 个类别目录, 少于 {TARGET_COUNT}")
    target_ids = folders[:TARGET_COUNT]
    with open(CLASSES_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(target_ids)) + "\n")

    # 3) 选择 unknown 来源类别: 优先样本多、品牌不重复的组外类别
    counts = {cid: len(os.listdir(os.path.join(RAW_TRAIN_DIR, cid)))
              for cid in folders if cid not in set(target_ids)}
    rest = sorted(counts, key=lambda cid: (-counts[cid], cid))
    source_ids, brands = [], set()
    for cid in rest:
        brand = info[cid]["name_from_new"].split("_")[0]
        if brand in brands:
            continue
        source_ids.append(cid)
        brands.add(brand)
        if len(source_ids) >= UNKNOWN_SOURCE_COUNT:
            break
    if len(source_ids) < UNKNOWN_SOURCE_COUNT:
        for cid in rest:
            if cid not in source_ids:
                source_ids.append(cid)
            if len(source_ids) >= UNKNOWN_SOURCE_COUNT:
                break

    rng = random.Random(SEED)
    for split in ("train", "val", "test_local"):
        for cid in target_ids + ["unknown"]:
            os.makedirs(os.path.join(DATASET_DIR, split, cid), exist_ok=True)

    # 4) 目标车型: 训练集全量, 官方验证集按 6:4 切分
    for cid in target_ids:
        for name in sorted(os.listdir(os.path.join(RAW_TRAIN_DIR, cid))):
            link_or_copy(os.path.join(RAW_TRAIN_DIR, cid, name),
                         os.path.join(DATASET_DIR, "train", cid, name))

    with zipfile.ZipFile(RAW_ARCHIVE) as zf:
        wanted: dict[str, list[tuple[str, str]]] = {}
        for name in zf.namelist():
            if name.startswith("__MACOSX/"):
                continue
            parts = name.split("/")
            cid = parts[1] if len(parts) == 3 else ""
            if parts[0] == "val" and cid in set(target_ids) | set(source_ids) and parts[2]:
                wanted.setdefault(cid, []).append((name, parts[2]))

        val_split = {}
        for cid, entries in wanted.items():
            entries.sort(key=lambda item: item[1])
            cut = max(1, int(round(len(entries) * VAL_RATIO)))
            val_split[cid] = (entries[:cut], entries[cut:])

        for cid in target_ids:
            for bucket, subset in zip(("val", "test_local"), val_split[cid]):
                for name, filename in subset:
                    dst = os.path.join(DATASET_DIR, bucket, cid, filename)
                    with zf.open(name) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)

        # 5) unknown: 从组外类别抽样, 训练与验证/测试样本互不重复
        unknown_stats = []
        for cid in source_ids:
            files = sorted(os.listdir(os.path.join(RAW_TRAIN_DIR, cid)))
            picked = rng.sample(files, min(UNKNOWN_PER_CLASS_TRAIN, len(files)))
            for name in picked:
                link_or_copy(os.path.join(RAW_TRAIN_DIR, cid, name),
                             os.path.join(DATASET_DIR, "train", "unknown", f"{cid}_{name}"))

            keep, hold = val_split[cid]
            take_val = rng.sample(keep, min(UNKNOWN_PER_CLASS_VAL, len(keep)))
            take_test = rng.sample(hold, min(UNKNOWN_PER_CLASS_TEST, len(hold)))
            for bucket, subset in (("val", take_val), ("test_local", take_test)):
                for name, filename in subset:
                    dst = os.path.join(DATASET_DIR, bucket, "unknown", f"{cid}_{filename}")
                    with zf.open(name) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
            unknown_stats.append({"id": cid, "name": info[cid]["name_from_new"],
                                  "type": "####".join(split_type(info[cid]["type_info"])),
                                  "train": len(picked), "val": len(take_val),
                                  "test": len(take_test)})

    # 6) 统计信息, 供实训报告引用
    summary = {
        "seed": SEED,
        "target_classes": [
            {"car_id": cid, "car_name": info[cid]["name_from_new"],
             "body_type": split_type(info[cid]["type_info"])[0],
             "level": split_type(info[cid]["type_info"])[1],
             "train": len(os.listdir(os.path.join(DATASET_DIR, "train", cid))),
             "val": len(os.listdir(os.path.join(DATASET_DIR, "val", cid))),
             "test": len(os.listdir(os.path.join(DATASET_DIR, "test_local", cid)))}
            for cid in target_ids
        ],
        "unknown_sources": unknown_stats,
        "split_sizes": {
            split: sum(len(os.listdir(os.path.join(DATASET_DIR, split, d)))
                       for d in os.listdir(os.path.join(DATASET_DIR, split)))
            for split in ("train", "val", "test_local")
        },
        "val_ratio_for_tuning": VAL_RATIO,
    }
    with open(os.path.join(DATASET_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("classes.txt:", len(target_ids), "个类别")
    print("样本数:", summary["split_sizes"])
    print("unknown 来源类别数:", len(unknown_stats))


if __name__ == "__main__":
    main()
