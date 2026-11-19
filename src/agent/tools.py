# -*- coding: utf-8 -*-
"""智能体工具集。

工具清单(共 8 个, 其中视觉工具 1 个, 业务工具 7 个):
    classify_car               视觉分类工具: 真实调用本组训练模型
    query_car_info             查询车型资料卡
    search_cars                按品牌/级别/车身/关键词检索车型
    add_car_favorite           把车型加入收藏夹
    remove_car_favorite        从收藏夹删除车型
    list_car_favorites         查看收藏夹
    search_recognition_history 检索历史识别记录
    compare_cars               两款车型的属性对比
    recommend_similar_cars     推荐同级别/同品牌车型

每个工具都返回结构化结果, 失败时返回 {"ok": False, "error": ...}, 由智能体向用户解释。
"""
from __future__ import annotations

import json
import os
import time

from database import (add_favorite, add_recognition, last_recognition, list_favorites,
                      remove_favorite, search_recognitions)

MAX_TOOL_ROUNDS = 6


def load_knowledge(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "classify_car",
            "description": "识别用户上传图片中的车型, 返回类别 ID、车型名称与置信度。"
                           "只支持本组登记的 50 个车型, 不属于这些车型时返回 unknown。"
                           "本轮有新图片时必须调用本工具; 若本轮没有新图片可省略参数, "
                           "系统会自动使用本会话最近一次上传的图片。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "图片文件名, 不填则使用本会话最近上传的图片"},
                    "top_k": {"type": "integer", "description": "返回置信度最高的前几个候选, 默认 3"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_car_info",
            "description": "按车型 ID 或中文车型名查询车型资料卡(品牌、车系、车身形式、级别、描述等)。"
                           "名称不精确时先调用 search_cars 找候选, 不要把名称猜成别的车型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {"type": "string", "description": "四位车型 ID 或中文车型名, 如 0000 / 大众朗逸; 不确定时不要自行编造 ID"},
                },
                "required": ["car_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cars",
            "description": "按品牌、级别、车身形式或关键词检索车型。用于回答“某类车有哪些”“举几个例子”, "
                           "也用于对比前查找候选。返回车型 ID、名称、品牌、级别、车身形式和是否支持图片识别。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "车型名、品牌或车系关键词, 可为空"},
                    "brand": {"type": "string", "description": "品牌, 如 江铃、大众, 可为空"},
                    "level": {"type": "string", "description": "级别或细分级别, 如 轻卡、微面、MPV、紧凑型车、中大型SUV, 可为空"},
                    "body_type": {"type": "string", "description": "车身形式, 如 卡车、SUV、轿车、MPV; 轻卡/微面这类细分级别请填 level, 可为空"},
                    "limit": {"type": "integer", "description": "返回数量上限, 默认 5"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_car_favorite",
            "description": "把车型加入收藏夹。用户说“记录下来”“收藏一下”时调用。"
                           "不传 car_id 时默认收藏本会话最近一次识别出的车型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {"type": "string", "description": "四位车型 ID 或中文车型名, 不填则使用最近一次识别结果; 不确定时不要自行编造 ID"},
                    "note": {"type": "string", "description": "用户补充的备注, 可为空"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_car_favorite",
            "description": "从收藏夹中删除一个车型。用户说“取消收藏”“删掉这个收藏”时调用。"
                           "不传 car_id 时默认删除本会话最近一次识别出的车型。"
                           "删除收藏必须调用本工具; 不得在不调用工具的情况下声称已删除。",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {"type": "string", "description": "四位车型 ID 或车型名称, 不填则使用最近一次识别结果; 不确定时不要自行编造 ID"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_car_favorites",
            "description": "查看当前收藏夹中的所有车型。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_recognition_history",
            "description": "检索历史识别记录, 可按车型 ID、车型名称或图片名模糊匹配。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "关键词, 留空表示返回最近若干条"},
                    "limit": {"type": "integer", "description": "返回条数上限, 默认 10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cars",
            "description": "对比两款车型的品牌、级别、车身形式、描述等属性。"
                           "用户提到“收藏里的某车”时, 先调用 list_car_favorites 拿到 car_id。"
                           "名称不精确时先调用 search_cars 找候选并向用户确认, 不要自己编 ID。",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id_a": {"type": "string", "description": "第一个车型 ID 或中文车型名, 不确定时不要编造 ID"},
                    "car_id_b": {"type": "string", "description": "第二个车型 ID 或中文车型名, 不确定时不要编造 ID"},
                },
                "required": ["car_id_a", "car_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_similar_cars",
            "description": "根据某个车型推荐本组支持的相似车型(同级别/同品牌)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "car_id": {"type": "string", "description": "车型 ID 或中文车型名, 不填则使用最近一次识别结果; 不确定时不要自行编造 ID"},
                    "limit": {"type": "integer", "description": "推荐数量, 默认 3"},
                },
            },
        },
    },
]


class ToolContext:
    """一次对话轮次内的工具上下文。"""

    def __init__(self, session_id: str, image_path: str = "", image_id: str = ""):
        self.session_id = session_id
        self.image_path = image_path
        self.image_id = image_id


class CarTools:
    """工具实现。视觉模型与知识库都在初始化时真实加载。"""

    def __init__(self, classifier, knowledge: dict, confidence_threshold: float = 0.55,
                 unknown_threshold: float = 0.40, db_path: str | None = None):
        self.classifier = classifier          # InferenceModel 实例(真实加载模型)
        self.knowledge = knowledge["cards"] if "cards" in knowledge else knowledge
        self.confidence_threshold = confidence_threshold
        self.unknown_threshold = unknown_threshold
        self.db_path = db_path
        self.name_to_id = {card["name"]: cid for cid, card in self.knowledge.items()}

    # ------------------------------------------------------------- helpers
    def _db(self) -> dict:
        return {"db_path": self.db_path} if self.db_path else {}

    @staticmethod
    def _normalize(text: str) -> str:
        """把车型名归一化, 便于匹配“大众朗逸”和“大众_朗逸”这类写法。"""
        return str(text or "").replace("_", "").replace(" ", "").replace("-", "").strip().lower()

    def _resolve_id(self, car_id: str | None, ctx: ToolContext) -> tuple[str | None, str | None]:
        """把车型 ID 或口语化名称解析成车型 ID; 未指定时回退到最近一次识别结果。

        名称匹配到多个车型时返回候选列表, 由智能体向用户确认, 不静默选第一个。
        """
        if car_id:
            raw = str(car_id).strip()
            if raw in self.knowledge:
                return raw, None
            target = self._normalize(raw)
            exact = [cid for cid, card in self.knowledge.items()
                     if self._normalize(card["name"]) == target]
            if len(exact) == 1:
                return exact[0], None
            matches = []
            for cid, card in self.knowledge.items():
                name = card["name"]
                norm = self._normalize(name)
                if target and (target in norm or norm in target):
                    matches.append((cid, name, bool(card.get("supported"))))
            matches = sorted(set(matches), key=lambda item: (not item[2], len(item[1])))
            if len(matches) == 1:
                return matches[0][0], None
            if len(matches) > 1:
                preview = "、".join(f"{cid} {name}" for cid, name, _ in matches[:6])
                return None, (f"“{raw}”匹配到多个车型: {preview}。"
                              f"请确认是哪一款(传四位 ID 或更完整的名称)。")
            return None, f"未找到车型“{raw}”。可以先调用 search_cars 查候选, 或换一个更完整的名称。"
        record = last_recognition(ctx.session_id, **self._db())
        if record and record["is_unknown"]:
            return None, "最近一次识别结果是 unknown(不属于本组 50 类), 没有对应车型"
        if record:
            return record["predicted_label"], None
        return None, "本会话还没有可用的识别结果, 请先上传一张车辆图片"

    def _card(self, car_id: str) -> dict | None:
        return self.knowledge.get(car_id)

    def _confidence_level(self, confidence: float, is_unknown: bool) -> str:
        """把置信度映射成四档, 供展示、收藏留痕与后续规则使用。"""
        if is_unknown:
            return "unknown"
        if confidence >= self.confidence_threshold:
            return "high"
        if confidence >= self.unknown_threshold:
            return "medium"
        return "low"

    # -------------------------------------------------------------- vision
    def classify_car(self, ctx: ToolContext, image_id: str | None = None, top_k: int = 3) -> dict:
        """视觉工具: 真实推理本组训练模型, 并把结果写入识别记录。"""
        image_path = ""
        if image_id and ctx.image_path:
            candidate = os.path.join(os.path.dirname(ctx.image_path), image_id)
            if os.path.isfile(candidate):
                image_path = candidate
            elif os.path.basename(ctx.image_path) == image_id:
                image_path = ctx.image_path
        if not image_path:
            image_path = ctx.image_path
        if not image_path or not os.path.isfile(image_path):
            return {"ok": False, "error": "没有找到可识别的图片, 请先上传一张车辆图片"}
        if self.classifier is None:
            return {"ok": False, "error": "视觉模型未加载, 请先训练模型或检查 config.yaml 中的 model_path"}

        started = time.time()
        try:
            result = self.classifier.predict(image_path, top_k=max(1, min(int(top_k or 3), 5)))
        except Exception as exc:  # 图片损坏、格式异常等
            return {"ok": False, "error": f"图片识别失败: {exc}"}

        label = result["predicted_label"]
        confidence = result["confidence"]
        card = self._card(label) if label != "unknown" else None
        is_unknown = label == "unknown"
        level = self._confidence_level(confidence, is_unknown)
        low_confidence = level in ("medium", "low")

        add_recognition(ctx.session_id, image_path, os.path.basename(image_path), label,
                        card["name"] if card else "unknown", confidence,
                        result["top_k"], is_unknown, **self._db())

        return {
            "ok": True,
            "image_id": os.path.basename(image_path),
            "car_id": None if is_unknown else label,
            "car_name": card["name"] if card else "unknown",
            "confidence": round(confidence, 4),
            "is_unknown": is_unknown,
            "low_confidence": low_confidence,
            "confidence_level": level,
            "confidence_threshold": self.confidence_threshold,
            "top_k": [
                {"car_id": item["label"],
                 "car_name": self._card(item["label"])["name"] if self._card(item["label"]) else "unknown",
                 "confidence": round(item["confidence"], 4)}
                for item in result["top_k"]
            ],
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    # ------------------------------------------------------------ business
    def query_car_info(self, ctx: ToolContext, car_id: str) -> dict:
        resolved, error = self._resolve_id(car_id, ctx)
        if error:
            return {"ok": False, "error": error}
        card = self._card(resolved)
        if card is None:
            return {"ok": False, "error": f"车型 {resolved} 不在本组资料库中"}
        similar = [self._card(cid) for cid in card.get("similar_ids", [])]
        return {
            "ok": True,
            "car_id": card["id"],
            "car_name": card["name"],
            "brand": card["brand"],
            "model": card["model"],
            "body_type": card["body_type"],
            "level": card["level"],
            "supported_by_model": card.get("supported", False),
            "description": card["description"],
            "similar_car_ids": card.get("similar_ids", []),
            "similar_car_names": [item["name"] for item in similar if item],
        }

    def search_cars(self, ctx: ToolContext, keyword: str = "", brand: str = "",
                    level: str = "", body_type: str = "", limit: int = 5) -> dict:
        """按品牌/级别/车身/关键词检索车型, 供“某类车有哪些”和对比前找候选使用。"""
        limit = max(1, min(int(limit or 5), 20))
        kw, br = self._normalize(keyword), self._normalize(brand)
        lv, bt = self._normalize(level), self._normalize(body_type)
        hits: list[dict] = []
        for card in self.knowledge.values():
            name = self._normalize(card["name"])
            if kw and kw not in name and kw not in self._normalize(card.get("description", "")):
                continue
            if br and br != self._normalize(card.get("brand", "")):
                continue
            if lv and lv not in self._normalize(card.get("level", "")) and lv not in self._normalize(card.get("body_type", "")):
                continue
            if bt and bt not in self._normalize(card.get("body_type", "")) and bt not in self._normalize(card.get("level", "")):
                continue
            hits.append(card)
        hits.sort(key=lambda card: (not card.get("supported"), len(card["name"])))
        return {
            "ok": True,
            "count": len(hits),
            "cars": [{"car_id": card["id"], "car_name": card["name"],
                      "brand": card.get("brand", ""), "model": card.get("model", ""),
                      "level": card.get("level", ""), "body_type": card.get("body_type", ""),
                      "supported": bool(card.get("supported"))}
                     for card in hits[:limit]],
        }

    def add_car_favorite(self, ctx: ToolContext, car_id: str | None = None, note: str = "") -> dict:
        resolved, error = self._resolve_id(car_id, ctx)
        if error:
            return {"ok": False, "error": error}
        card = self._card(resolved)
        if card is None or not card.get("supported"):
            return {"ok": False, "error": f"车型 {resolved} 不在本组支持的 50 类中, 未加入收藏"}
        # 收藏留痕: 若收藏的正是本会话最近识别出的车型, 把图片与置信度一并记录
        record = last_recognition(ctx.session_id, **self._db())
        image_path, confidence, level = "", 0.0, ""
        if record and record["predicted_label"] == resolved:
            image_path = record["image_path"]
            confidence = float(record["confidence"])
            level = self._confidence_level(confidence, record["is_unknown"])
        result = add_favorite(card["id"], card["name"], note or "", source="agent",
                              image_path=image_path, confidence=confidence,
                              confidence_level=level, **self._db())
        result.update({"ok": True, "car_name": card["name"],
                       "confidence": round(confidence, 4), "confidence_level": level})
        return result

    def list_car_favorites(self, ctx: ToolContext) -> dict:
        items = list_favorites(**self._db())
        result = []
        for item in items:
            card = self._card(item["car_id"]) or {}
            result.append({"car_id": item["car_id"], "car_name": item["car_name"],
                           "note": item["note"],
                           "brand": card.get("brand", ""), "level": card.get("level", ""),
                           "body_type": card.get("body_type", ""),
                           "confidence": round(item["confidence"], 4),
                           "confidence_level": item["confidence_level"]})
        return {"ok": True, "count": len(result), "favorites": result}

    def remove_car_favorite(self, ctx: ToolContext, car_id: str | None = None) -> dict:
        resolved, error = self._resolve_id(car_id, ctx)
        if error:
            return {"ok": False, "error": error}
        result = remove_favorite(resolved, **self._db())
        result["ok"] = result["status"] == "removed"
        return result

    def search_recognition_history(self, ctx: ToolContext, keyword: str = "", limit: int = 10) -> dict:
        items = search_recognitions(keyword or "", int(limit or 10), **self._db())
        return {"ok": True, "count": len(items),
                "records": [{"image_id": item["image_id"], "car_id": item["predicted_label"],
                             "car_name": item["car_name"],
                             "confidence": round(item["confidence"], 4),
                             "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["created_at"]))}
                            for item in items]}

    def compare_cars(self, ctx: ToolContext, car_id_a: str, car_id_b: str) -> dict:
        resolved_a, error_a = self._resolve_id(car_id_a, ctx)
        if error_a:
            return {"ok": False, "error": error_a}
        resolved_b, error_b = self._resolve_id(car_id_b, ctx)
        if error_b:
            return {"ok": False, "error": error_b}
        card_a, card_b = self._card(resolved_a), self._card(resolved_b)
        if card_a is None or card_b is None:
            return {"ok": False, "error": "存在不在本组资料库中的车型"}
        same_level = card_a["level"] == card_b["level"]
        same_body = card_a["body_type"] == card_b["body_type"]
        if same_level and same_body:
            advice = "两款车属于同一级别、同一车身形式, 可以直接比较, 重点看品牌和车系偏好。"
        elif same_body:
            advice = "两款车车身形式相同但级别不同, 可以比较尺寸和定位, 先确认客户用途。"
        else:
            advice = "两款车车身形式不同, 通常面向不同使用场景, 不建议直接比较; 先按用途筛选。"
        return {
            "ok": True,
            "car_a": {"car_id": card_a["id"], "car_name": card_a["name"], "brand": card_a["brand"],
                      "model": card_a.get("model", ""), "body_type": card_a["body_type"],
                      "level": card_a["level"], "description": card_a.get("description", "")},
            "car_b": {"car_id": card_b["id"], "car_name": card_b["name"], "brand": card_b["brand"],
                      "model": card_b.get("model", ""), "body_type": card_b["body_type"],
                      "level": card_b["level"], "description": card_b.get("description", "")},
            "same_level": same_level,
            "same_body_type": same_body,
            "summary": ("两款车属于同一级别" if same_level else "两款车级别不同")
                       + ("、同一车身形式。" if same_body else "、车身形式不同。"),
            "advice": advice,
        }

    def recommend_similar_cars(self, ctx: ToolContext, car_id: str | None = None,
                               limit: int = 3) -> dict:
        resolved, error = self._resolve_id(car_id, ctx)
        if error:
            return {"ok": False, "error": error}
        card = self._card(resolved)
        if card is None:
            return {"ok": False, "error": f"车型 {resolved} 不在本组支持的车型中"}
        picks = []
        for cid in card.get("similar_ids", []):
            other = self._card(cid)
            if other and len(picks) < int(limit or 3):
                picks.append({"car_id": other["id"], "car_name": other["name"],
                              "level": other["level"], "brand": other["brand"]})
        return {"ok": True, "base_car": {"car_id": card["id"], "car_name": card["name"],
                                         "level": card["level"]},
                "recommendations": picks}

    # ------------------------------------------------------------ dispatch
    def registry(self) -> dict:
        return {
            "classify_car": lambda ctx, **kw: self.classify_car(ctx, **kw),
            "query_car_info": lambda ctx, **kw: self.query_car_info(ctx, **kw),
            "search_cars": lambda ctx, **kw: self.search_cars(ctx, **kw),
            "add_car_favorite": lambda ctx, **kw: self.add_car_favorite(ctx, **kw),
            "remove_car_favorite": lambda ctx, **kw: self.remove_car_favorite(ctx, **kw),
            "list_car_favorites": lambda ctx, **kw: self.list_car_favorites(ctx, **kw),
            "search_recognition_history": lambda ctx, **kw: self.search_recognition_history(ctx, **kw),
            "compare_cars": lambda ctx, **kw: self.compare_cars(ctx, **kw),
            "recommend_similar_cars": lambda ctx, **kw: self.recommend_similar_cars(ctx, **kw),
        }

    def execute(self, name: str, arguments: str | dict, ctx: ToolContext) -> dict:
        """执行工具调用, 统一处理参数解析错误与异常。"""
        registry = self.registry()
        if name not in registry:
            return {"ok": False, "error": f"未知工具: {name}"}
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"工具参数不是合法 JSON: {exc}"}
        else:
            args = arguments or {}
        try:
            return registry[name](ctx, **args)
        except TypeError as exc:
            return {"ok": False, "error": f"工具参数不匹配: {exc}"}
        except Exception as exc:
            return {"ok": False, "error": f"工具执行失败: {exc}"}
