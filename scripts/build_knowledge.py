# -*- coding: utf-8 -*-
"""由教师提供的 class_info.json 生成车型资料库 data/car_knowledge.json。

资料卡里的字段全部来自类别说明文件(品牌、车系、车身形式、级别),
描述文案也是由这些结构化字段拼装而成, 不引入任何外部或凭空编造的参数。

用法: python scripts/build_knowledge.py
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import CLASS_INFO_FILE, DATASET_DIR, load_classes  # noqa: E402

OUT_FILE = os.path.join(PROJECT_ROOT, "data", "car_knowledge.json")


def split_type(type_info: str) -> tuple[str, str, str, str]:
    """type_info 形如 'SUV####紧凑型SUV==SUV####Compact SUV'。"""
    parts = (type_info.split("==") + [""])[:2]
    zh_parts = [p for p in parts[0].split("####") if p]
    en_parts = [p for p in parts[1].split("####") if p]
    body = zh_parts[0] if zh_parts else ""
    level = zh_parts[1] if len(zh_parts) > 1 else ""
    body_en = en_parts[0] if en_parts else ""
    level_en = en_parts[1] if len(en_parts) > 1 else ""
    return body, level, body_en, level_en


def describe(cid: str, name: str, brand: str, model: str, body: str, level: str) -> str:
    """用结构化字段生成一段可读的车型简介。"""
    return (f"{name} 是{brand}旗下的{model}车系, 车身形式为{body}, 级别为{level}。"
            f"在本实训的 827 类车型体系中, 它的类别 ID 是 {cid}, 属于{body}-{level}这一细分市场。")


def main() -> None:
    class_ids = load_classes()
    with open(CLASS_INFO_FILE, "r", encoding="utf-8") as f:
        info = json.load(f)

    cards = {}
    for record in info:
        cid = record["id"]
        name = record["name_from_new"]
        brand, _, model = name.partition("_")
        model = model or name
        body, level, body_en, level_en = split_type(record["type_info"])
        cards[cid] = {
            "id": cid,
            "name": name,
            "brand": brand,
            "model": model,
            "body_type": body,
            "level": level,
            "body_type_en": body_en,
            "level_en": level_en,
            "supported": cid in class_ids,
            "description": describe(cid, name, brand, model, body, level),
        }

    # 同级别/同品牌的可选车型(只在支持的 50 类内推荐, 保证推荐结果可被本组模型识别)
    supported = [cards[cid] for cid in class_ids]
    for card in supported:
        same_level = [c["id"] for c in supported
                      if c["id"] != card["id"] and c["level"] == card["level"]]
        same_brand = [c["id"] for c in supported
                      if c["id"] != card["id"] and c["brand"] == card["brand"]]
        same_body = [c["id"] for c in supported
                     if c["id"] != card["id"] and c["body_type"] == card["body_type"]]
        merged: list[str] = []
        for cid in same_level + same_brand + same_body:
            if cid not in merged:
                merged.append(cid)
        card["similar_ids"] = merged[:5]
        card["same_level_ids"] = same_level[:5]
        card["same_brand_ids"] = same_brand[:5]

    summary = {
        "total_cards": len(cards),
        "supported_cards": len(supported),
        "body_types": sorted({c["body_type"] for c in supported}),
        "levels": sorted({c["level"] for c in supported}),
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cards": cards}, f, ensure_ascii=False, indent=2)
    print(f"已生成 {OUT_FILE}: {len(cards)} 张车型资料卡, 其中 {len(supported)} 张属于本组支持的 50 类")


if __name__ == "__main__":
    main()
