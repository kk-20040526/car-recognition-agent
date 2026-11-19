# -*- coding: utf-8 -*-
"""DeepSeek-V4-Flash 客户端封装(OpenAI 兼容接口 + Function Calling)。"""
from __future__ import annotations


class LLMError(RuntimeError):
    """调用大模型失败时抛出, 由上层转成用户可读的提示。"""


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-v4-flash", temperature: float = 0.3,
                 timeout: float = 60.0):
        if not api_key:
            raise LLMError("未配置 DEEPSEEK_API_KEY, 请在 .env 或环境变量中填写后再启动系统")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(f"缺少 openai 依赖: {exc}") from exc

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """发起一次对话请求, 返回 message 对象(可能包含 tool_calls)。"""
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"DeepSeek 接口调用失败: {exc}") from exc
        if not response.choices:
            raise LLMError("DeepSeek 返回了空结果")
        return response.choices[0].message

    @staticmethod
    def to_message_dict(message) -> dict:
        """把 SDK 返回的 message 转成可持久化、可回传的字典。"""
        data: dict = {"role": "assistant", "content": message.content or ""}
        if getattr(message, "tool_calls", None):
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]
        return data
