# -*- coding: utf-8 -*-
"""演示前自检：一条命令判断这台机器现在能不能跑起识车助手。

用法（在项目根目录）：
    .\.venv\Scripts\python.exe scripts\check_ready.py

它检查 6 件事：
    1) Python 与 torch 是否可用
    2) classes.txt 是否为恰好 50 行合法 ID
    3) models/improved/best.pt 是否存在、能否被加载成 51 类模型
    4) 车型资料库 data/car_knowledge.json 是否存在
    5) .env 里的 DEEPSEEK_API_KEY 是否配置
    6) 结论：能不能开演示

注意：本脚本只读不写，不会修改任何文件。
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

OK, BAD, WARN = "[OK]", "[BLOCK]", "[WARN]"
blockers = []
warns = []

print("project:", PROJECT_ROOT)
print("-" * 60)

# 1) Python + torch
print("1) Python / torch")
print("   python:", sys.version.split()[0])
try:
    import torch
    print("   torch : %s (cuda=%s)" % (torch.__version__, torch.cuda.is_available()))
except Exception as exc:
    print("   torch : 不可用 -> %s" % exc)
    blockers.append("未安装 torch：先跑 scripts\\setup_env.ps1")

# 2) classes.txt
print("2) classes.txt")
try:
    from config import load_classes
    ids = load_classes()
    print("   %s 50 类，首 %s 末 %s" % (OK, ids[0], ids[-1]))
except Exception as exc:
    print("   %s %s" % (BAD, exc))
    blockers.append("classes.txt 格式不对（必须恰好 50 行四位 ID、无注释、升序、UTF-8 无 BOM）")

# 3) 权重
print("3) models/improved/best.pt")
weights = os.path.join(PROJECT_ROOT, "models", "improved", "best.pt")
if not os.path.exists(weights):
    print("   %s 不存在：%s" % (BAD, weights))
    blockers.append("缺模型权重 best.pt（找队友要他训练产出的 checkpoint，放到 models\\improved\\best.pt）")
else:
    size = os.path.getsize(weights) / 1024 / 1024
    print("   文件存在 %.1f MB" % size)
    try:
        from model import build_model, NUM_CLASSES
        model = build_model(arch="resnet18", num_classes=NUM_CLASSES, pretrained=False)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        # 本项目 train.py 存的是 {"model_state":..., "arch":..., "class_ids":...}
        # 早期写成 state["state_dict"]，取不到就整包喂给 load_state_dict，
        # 结果 missing=102/unexpected=9 全是误报。这里按真实格式取。
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
            meta_arch = ckpt.get("arch")
            meta_nc = ckpt.get("num_classes")
            meta_f1 = ckpt.get("val_macro_f1")
            meta_ep = ckpt.get("epoch")
            print("   checkpoint: arch=%s num_classes=%s best_epoch=%s val_macro_f1=%s"
                  % (meta_arch, meta_nc,
                     meta_ep, ("%.4f" % meta_f1) if isinstance(meta_f1, float) else meta_f1))
            if meta_arch and meta_arch != "resnet18":
                warns.append("权重训练用的是 %s，但自检按 resnet18 建结构" % meta_arch)
            if meta_nc and int(meta_nc) != NUM_CLASSES:
                warns.append("权重 num_classes=%s 与代码 NUM_CLASSES=%d 不一致"
                             % (meta_nc, NUM_CLASSES))
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 2:
            print("   %s 能加载但不完整：missing=%d unexpected=%d" % (WARN, len(missing), len(unexpected)))
            warns.append("权重结构与 resnet18+51 类不完全一致，先跑一次 infer 确认")
        else:
            print("   %s 权重可加载（missing=%d unexpected=%d）" % (OK, len(missing), len(unexpected)))
    except Exception as exc:
        print("   %s 加载失败：%s" % (BAD, exc))
        blockers.append("best.pt 无法加载（可能不是 state_dict 或类别数不是 51）")

# 4) 知识库
print("4) data/car_knowledge.json")
kb = os.path.join(PROJECT_ROOT, "data", "car_knowledge.json")
print("   %s %.0f KB" % (OK, os.path.getsize(kb) / 1024) if os.path.exists(kb)
      else "   %s 不存在" % BAD)
if not os.path.exists(kb):
    blockers.append("资料库缺失：跑 scripts\\build_knowledge.py")

# 5) API Key（按 config.yaml 里 deepseek.api_key_env 指定的变量名去找，不再写死 DEEPSEEK_API_KEY）
env_path = os.path.join(PROJECT_ROOT, ".env")
KEY_ENV = "DEEPSEEK_API_KEY"
try:
    import yaml as _yaml
    _cfg = os.path.join(PROJECT_ROOT, "config.yaml")
    if os.path.exists(_cfg):
        with open(_cfg, "r", encoding="utf-8") as _f:
            _y = _yaml.safe_load(_f) or {}
        KEY_ENV = str((_y.get("deepseek") or {}).get("api_key_env") or KEY_ENV).strip()
        _MODEL = str((_y.get("deepseek") or {}).get("model") or "").strip()
        _BASE = str((_y.get("deepseek") or {}).get("base_url") or "").strip()
except Exception:
    _MODEL = _BASE = ""
print("5) %s（模型 %s @ %s）" % (KEY_ENV, _MODEL or "?", _BASE or "?"))
key = ""
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith(KEY_ENV + "="):
                key = line.split("=", 1)[-1].strip().strip('"').strip("'")
if key and not key.startswith("sk-你的"):
    print("   %s 已配置（%s...%s，长度 %d）" % (OK, key[:6], key[-4:], len(key)))
else:
    print("   %s 未配置：复制 .env.example 为 .env 并填入 %s" % (BAD, KEY_ENV))
    blockers.append("没有 %s：没有 Key 时 Agent 无法调用大模型，只能做纯视觉识别" % KEY_ENV)

# 6) 结论
print("-" * 60)
if blockers:
    print("结论：还不能完整演示，有 %d 个阻断项：" % len(blockers))
    for i, b in enumerate(blockers, 1):
        print("  %d. %s" % (i, b))
else:
    print("结论：可以演示。建议先用一张图预热一次，再正式开讲。")
for w in warns:
    print("  提示：%s" % w)

sys.exit(0 if not blockers else 1)
