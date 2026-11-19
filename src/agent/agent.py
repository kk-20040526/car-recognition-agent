# -*- coding: utf-8 -*-
"""智能体主流程: qwen-flash + 工具调用循环 + 会话管理。

一轮对话的执行流程:
    1. 读取本会话历史消息(含上一轮的识别结果), 拼装成 DeepSeek 的 messages;
    2. 把工具定义一起发给模型;
    3. 模型返回 tool_calls 时, 由本程序真实执行工具(视觉工具会加载本组模型推理);
    4. 把工具结果回传给模型, 直到模型给出最终回答或达到最大轮数;
    5. 全过程写入数据库, 便于 Web 端展示工具调用记录。
"""
from __future__ import annotations

import json
import os
import time

from database import add_message, get_messages, last_recognition, session_exists, touch_session
from llm import DeepSeekClient, LLMError
from tools import MAX_TOOL_ROUNDS, CarTools, ToolContext

SYSTEM_PROMPT = """你是“识车助手”, 帮助二手车商和用户识别、查找、了解、收藏和对比汽车。

【能力范围】
- 图片识别只覆盖 50 类常见车型(类别 ID 为四位数字), 不属于这 50 类的图片会返回 unknown(未知车型)。
- 资料库里有 827 款车型的资料, 可以按名称或条件查询; 不在 50 类里的车型只能查资料, 不能通过图片识别。
- 图片识别必须通过 classify_car 工具完成, 不允许凭文字描述猜测具体车型。
- 车型资料必须来自 query_car_info / search_cars 返回的资料卡, 不要编造价格、动力、销量、保值率等资料库里没有的参数,
  也不要额外添加资料卡以外的评价或卖点。

【能力边界】
- 一张图里有多辆车时, 本工具只识别画面里最主要的一辆车; 多车场景需要先做车辆检测再逐车分类, 不在当前范围, 要如实说明。
- 用户问价格、保值率、车况、里程、维修记录时, 明确说资料库里没有这类数据, 不要估算或编造。

【工具使用规则】
1. 用户上传了新图片时, 先调用 classify_car。
2. 用户问“某类车有哪些 / 举几个例子 / 有没有某某品牌的车”时, 调用 search_cars 检索候选,
   不要凭记忆列车型。用户说“介绍一下某车”时调用 query_car_info; 即使用户没有上传图片、
   直接说了车型名, 也可以直接查, 不要要求用户先上传图片。
3. 需要对比时调用 compare_cars。两款车的名称必须准确:
   - 不确定用户指哪一款时, 先调用 search_cars 找候选, 或根据工具返回的候选列表向用户确认,
     不要自己编四位 ID, 也不要把名称猜成别的车型。
   - 用户说“收藏里的某车”时, 先调用 list_car_favorites 拿到 car_id, 再对比、查资料或删除。
   - 两款车级别或车身形式不同时, 要主动说明“不同级别 / 不同用途”, 不要只报相同或不同。
4. 需要推荐时调用 recommend_similar_cars。
5. 用户说“记录下来 / 收藏一下 / 加入收藏”时调用 add_car_favorite; 问“我的收藏”时调用 list_car_favorites。
   add_car_favorite 只能新增或更新收藏, 不能删除。
   任何收藏 / 删除操作都必须先调用工具, 只有工具返回 ok=true 才能声称已完成;
   不得只凭文字声称已写入或已删除。
6. 用户说“取消收藏 / 删掉这个收藏”时调用 remove_car_favorite; 删除收藏必须调用本工具,
   不得在不调用工具的情况下声称已删除。
7. 用户追问历史识别记录时调用 search_recognition_history。
8. 一个任务可以连续调用多个工具。例如“这是什么车?介绍一下,然后加入收藏”需要依次调用
   classify_car -> query_car_info -> add_car_favorite。

【识别结果处理】
- 若 classify_car 返回 is_unknown=true, 必须告诉用户“这张图片不在能识别的 50 类车型里”,
  不得编造车型, 也不得继续按某个确定车型执行收藏等业务操作。
- 若返回 low_confidence=true, 要说明置信度较低, 列出 top_k 候选, 请用户确认或换一张更清晰的图片;
  用户明确确认后必须调用 add_car_favorite 完成收藏, 并在回复中说明这是一次低置信度收藏;
  不得只在回复里声称已收藏而不调用工具。
  如果用户在同一句话里已经明确确认(如“就按这个收藏”“确认收藏”“是它”), 直接调用 add_car_favorite,
  不要再重复询问确认。
- 回答中要给出车型名称、类别 ID 和置信度(百分比), 用简洁的中文回答。

【多轮对话】
- 同一会话里保留上下文。用户说“帮我记录下来”“那它呢”这类话时, 默认指本会话最近一次识别出的车型,
  此时 add_car_favorite、query_car_info 等工具可以不传 car_id。
- 用户说“收藏里的那辆 / 我收藏的车”时, 先调用 list_car_favorites, 不要用最近识别的车顶替。

【异常处理】
- 工具返回 ok=false 时, 用通俗的话向用户说明原因(例如没找到图片、车型不在支持范围), 并给出下一步建议。

【表达要求】
- 对用户不要提工具名、函数名、参数名或“调用工具”这类技术词, 用“我帮你查一下”“我帮你对比”来表达。
- 车型名称里的下划线要换成空格再展示, 例如“大众_朗逸”写成“大众 朗逸”。
- 回答保持简洁, 先给结论, 再给必要细节。
"""


class CarAgent:
    def __init__(self, llm: DeepSeekClient, tools: CarTools, db_path: str | None = None,
                 history_messages: int = 12):
        self.llm = llm
        self.tools = tools
        self.db_path = db_path
        self.history_messages = history_messages

    def _db(self) -> dict:
        return {"db_path": self.db_path} if self.db_path else {}

    def _build_messages(self, session_id: str, user_text: str, image_path: str) -> list[dict]:
        """系统提示 + 最近若干轮历史 + 本轮用户输入。"""
        history = get_messages(session_id, limit=200, **self._db())
        if history and history[-1]["role"] == "user":
            history = history[:-1]  # 本轮用户消息在末尾单独补上(带图片说明), 避免重复
        window = history[-self.history_messages:]
        # 保证窗口从一条 user 消息开始, 避免把 tool 调用链截断导致接口报错
        while window and window[0]["role"] != "user":
            window = window[1:]
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in window:
            role = item["role"]
            if role == "tool":
                messages.append({"role": "tool", "tool_call_id": item["tool_call_id"],
                                 "content": item["content"]})
            elif role == "assistant":
                entry: dict = {"role": "assistant", "content": item["content"] or ""}
                if item["tool_calls"]:
                    entry["tool_calls"] = item["tool_calls"]
                messages.append(entry)
            else:
                messages.append({"role": "user", "content": item["content"]})

        suffix = ""
        if image_path:
            suffix = f"\n[本轮用户上传了图片: {os.path.basename(image_path)}]"
        messages.append({"role": "user", "content": (user_text or "请看看这张图片") + suffix})
        return messages

    def run_turn(self, session_id: str, user_text: str, image_path: str = "") -> dict:
        """执行一轮对话, 返回最终回答与工具调用记录。"""
        if not session_exists(session_id, **self._db()):
            return {"reply": "会话不存在, 请刷新页面重新开始。", "tool_calls": [], "error": True}

        add_message(session_id, "user", user_text or "", image_path=image_path, **self._db())
        messages = self._build_messages(session_id, user_text, image_path)
        ctx = ToolContext(session_id, image_path=image_path,
                          image_id=os.path.basename(image_path) if image_path else "")
        tool_log: list[dict] = []
        reply = ""

        if image_path:
            started = time.time()
            args = json.dumps({"image_id": os.path.basename(image_path)}, ensure_ascii=False)
            result = self.tools.execute("classify_car", args, ctx)
            tool_log.append({"name": "classify_car", "arguments": args,
                             "result": result, "elapsed_ms": int((time.time() - started) * 1000)})
            messages.insert(1, {
                "role": "system",
                "content": "本轮用户上传的图片已经由 classify_car 完成识别, 结果如下: "
                           + json.dumps(result, ensure_ascii=False)
                           + "。不要重复调用 classify_car; 直接基于这个结果回答, 需要资料/收藏/对比时再调用其他工具。",
            })

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                message = self.llm.chat(messages, tools=self.tools_schemas())
                payload = DeepSeekClient.to_message_dict(message)
                messages.append(payload)
                if not payload.get("tool_calls"):
                    reply = payload.get("content") or ""
                    add_message(session_id, "assistant", reply, **self._db())
                    break

                add_message(session_id, "assistant", payload.get("content") or "",
                            tool_calls=payload["tool_calls"], **self._db())
                for call in payload["tool_calls"]:
                    name = call["function"]["name"]
                    started = time.time()
                    result = self.tools.execute(name, call["function"]["arguments"], ctx)
                    elapsed = int((time.time() - started) * 1000)
                    tool_log.append({
                        "name": name,
                        "arguments": call["function"]["arguments"],
                        "result": result,
                        "elapsed_ms": elapsed,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    add_message(session_id, "tool",
                                json.dumps(result, ensure_ascii=False),
                                tool_call_id=call["id"], **self._db())
            else:
                reply = "已经连续调用了多轮工具仍未完成, 请把需求拆成更小的步骤再试一次。"
                add_message(session_id, "assistant", reply, **self._db())
        except LLMError as exc:
            reply = f"抱歉, 智能体调用失败: {exc}"
            add_message(session_id, "assistant", reply, **self._db())
            touch_session(session_id, **self._db())
            return {"reply": reply, "tool_calls": tool_log, "error": True}

        reply = self._fallback_favorite(session_id, user_text, ctx, tool_log, reply)
        touch_session(session_id, **self._db())
        return {
            "reply": reply,
            "tool_calls": tool_log,
            "low_confidence": any(
                isinstance(item["result"], dict) and item["result"].get("low_confidence")
                for item in tool_log
            ),
            "error": False,
        }

    @staticmethod
    def _wants_favorite(text: str) -> bool:
        text = text or ""
        excludes = ("取消收藏", "删除收藏", "不要收藏", "别收藏", "不用收藏",
                    "收藏里", "我的收藏", "查看收藏", "有哪些收藏")
        hints = ("加入收藏", "加到收藏", "收藏一下", "收藏这辆", "收藏它",
                 "存起来", "记下来", "加入车源库")
        if any(item in text for item in excludes):
            return False
        return any(item in text for item in hints)

    def _fallback_favorite(self, session_id, user_text, ctx, tool_log, reply):
        """模型只说不做时, 后端补一次真实收藏, 避免“嘴上已收藏、库里没有”。"""
        calls = [item for item in tool_log if item.get("name") == "add_car_favorite"]
        if calls:
            result = calls[-1].get("result") or {}
            if not result.get("ok") and "收藏" in reply and ("已" in reply or "成功" in reply):
                return f"收藏没有成功：{result.get('error', '未知原因')}"
            return reply
        if not self._wants_favorite(user_text):
            if "已自动" in reply and "收藏" in reply:
                return "识别完成。如需加入收藏，告诉我一声即可。"
            return reply
        record = last_recognition(session_id, **self._db())
        if not record:
            return "还没有识别到车型，请先发一张车图，再说加入收藏。"
        if record.get("is_unknown"):
            return "这辆车不在支持的 50 类里，不能加入收藏。"
        args = json.dumps({"car_id": record["predicted_label"],
                           "note": "用户要求加入收藏（后端兜底）"}, ensure_ascii=False)
        result = self.tools.execute("add_car_favorite", args, ctx)
        tool_log.append({"name": "add_car_favorite", "arguments": args,
                         "result": result, "elapsed_ms": 0})
        if not result.get("ok"):
            return f"收藏没有成功：{result.get('error', '未知原因')}"
        confidence = float(result.get("confidence") or 0)
        suffix = f"，置信度 {confidence:.2%}。" if confidence else "。"
        return f"已将 {result['car_name']}（ID: {result['car_id']}）加入收藏{suffix}"

    @staticmethod
    def tools_schemas() -> list[dict]:
        from tools import TOOL_SCHEMAS
        return TOOL_SCHEMAS
