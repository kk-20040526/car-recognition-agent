# -*- coding: utf-8 -*-
"""验证 DeepSeek 配置是否可用: 读取 .env 里的 Key, 真实调用一次接口。

用法: python scripts/check_deepseek.py

输出会告诉你:
    1) Key 有没有读到;
    2) 接口是否连得通、Key 是否有效;
    3) config.yaml 里的模型名是否可用, 不可用时列出账号下可用的模型名。
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "agent"))

from settings import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    deepseek = settings["deepseek"]
    key = deepseek.get("api_key", "")

    if not key:
        print("[1/3] 未读到 API Key")
        print("      请确认项目根目录下有 .env 文件, 且其中一行是: DEEPSEEK_API_KEY=sk-xxxx")
        print("      快捷方式: powershell -ExecutionPolicy Bypass -File scripts\\set_api_key.ps1")
        return 1
    print(f"[1/3] 已读取 API Key: {key[:6]}...{key[-4:]}（长度 {len(key)}）")

    try:
        from openai import OpenAI
    except ImportError:
        print("      缺少 openai 依赖: python -m pip install -r requirements.txt")
        return 1

    client = OpenAI(api_key=key, base_url=deepseek["base_url"], timeout=30)
    try:
        models = [item.id for item in client.models.list().data]
        print(f"[2/3] 接口连通, Key 有效。账号下可用模型: {', '.join(models[:12])}")
    except Exception as exc:
        print(f"[2/3] 接口调用失败: {exc}")
        if "Connection error" in str(exc) or "timed out" in str(exc).lower():
            print("      看起来是网络不通: 请在你自己的终端(本机 PowerShell)里运行本脚本,"
                  " 并确认能访问 https://api.deepseek.com")
        else:
            print("      常见原因: Key 填错或已失效、账号没有余额")
        return 1

    model = deepseek["model"]
    if model not in models:
        print(f"[3/3] config.yaml 里的模型名 {model} 不在可用列表中")
        print("      请把 config.yaml 的 deepseek.model 改成上面列出的某个名字(例如 deepseek-chat)")
        return 2

    try:
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字: 就绪"}],
            temperature=0,
        )
        print(f"[3/3] 模型 {model} 对话成功, 返回: {reply.choices[0].message.content!r}")
    except Exception as exc:
        print(f"[3/3] 模型 {model} 调用失败: {exc}")
        return 1

    print("\n全部通过: 启动系统后即可真实对话。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
