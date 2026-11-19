# -*- coding: utf-8 -*-
"""运行配置: 读取 config.yaml 与环境变量(.env)。

API Key 只从环境变量 / .env 读取, 不写在代码、报告或截图里。
"""
from __future__ import annotations

import os

import yaml

from config import PROJECT_ROOT

CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.yaml")

DEFAULT_SETTINGS = {
    "model_path": "models/improved/best.pt",
    "knowledge_path": "data/car_knowledge.json",
    "db_path": "storage/app.db",
    "upload_dir": "storage/uploads",
    "confidence_threshold": 0.55,
    "unknown_threshold": 0.40,
    "history_messages": 12,
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
        "temperature": 0.3,
        "timeout": 60,
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}


def _load_dotenv() -> None:
    """极简 .env 读取, 避免额外依赖。

    .env 中的配置优先于系统环境变量: 本项目的 Key 以 .env 为准,
    避免在别的终端里运行时被无关的环境变量覆盖。
    """
    path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def load_settings() -> dict:
    _load_dotenv()
    settings = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_SETTINGS.items()}
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        for key, value in user_config.items():
            if isinstance(value, dict) and isinstance(settings.get(key), dict):
                settings[key].update(value)
            else:
                settings[key] = value
    deepseek = settings["deepseek"]
    deepseek["api_key"] = os.environ.get(deepseek.get("api_key_env", "DEEPSEEK_API_KEY"), "").strip()
    # DeepSeek 的 Key 以 sk- 开头; 其他内容(空值或 .env.example 里的中文占位符)一律按未配置处理
    if not deepseek["api_key"].startswith("sk-"):
        deepseek["api_key"] = ""
    return settings


def resolve(path: str) -> str:
    """把配置里的相对路径解析为项目内的绝对路径。"""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
