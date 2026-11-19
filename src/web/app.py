# -*- coding: utf-8 -*-
"""Web 系统后端(FastAPI): 图片上传、多轮对话、工具调用记录、收藏与历史查询。

启动:
    python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
或:
    python src/web/app.py
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(WEB_DIR)
AGENT_DIR = os.path.join(SRC_DIR, "agent")
for path in (SRC_DIR, AGENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent import CarAgent  # noqa: E402
from batch import (MAX_IMAGES, check_submission, collect_folder, run_batch,  # noqa: E402
                   write_predictions, write_result_csv, write_result_npz, write_results)
from config import PROJECT_ROOT, load_classes  # noqa: E402
from database import (add_message, create_session, get_messages, list_favorites,  # noqa: E402
                      search_recognitions, session_exists)
from llm import DeepSeekClient, LLMError  # noqa: E402
from settings import load_settings, resolve  # noqa: E402
from tools import TOOL_SCHEMAS, CarTools, ToolContext, load_knowledge  # noqa: E402

SETTINGS = load_settings()
DB_PATH = resolve(SETTINGS["db_path"])
UPLOAD_DIR = resolve(SETTINGS["upload_dir"])
BATCH_DIR = resolve("storage/batch")
SUBMISSION_DIR = resolve("submission/student_submit")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

app = FastAPI(title="识车助手 · 车型识别与收藏智能体", version="1.0")
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")

_state: dict = {"classifier": None, "tools": None, "agent": None, "load_error": ""}


def bootstrap() -> None:
    """启动时真实加载视觉模型、知识库、工具与智能体。"""
    class_ids = load_classes()
    model_path = resolve(SETTINGS["model_path"])
    if os.path.isfile(model_path):
        try:
            from infer import InferenceModel
            _state["classifier"] = InferenceModel(model_path)
            print(f"[启动] 已加载视觉模型: {model_path} ({_state['classifier'].arch})")
        except Exception as exc:
            _state["load_error"] = f"视觉模型加载失败: {exc}"
            print(f"[启动] {_state['load_error']}")
    else:
        _state["load_error"] = f"未找到模型文件 {model_path}, 请先训练或修改 config.yaml 中的 model_path"
        print(f"[启动] {_state['load_error']}")

    knowledge_path = resolve(SETTINGS["knowledge_path"])
    knowledge = load_knowledge(knowledge_path) if os.path.isfile(knowledge_path) else {"cards": {}}
    _state["tools"] = CarTools(_state["classifier"], knowledge,
                               confidence_threshold=SETTINGS["confidence_threshold"],
                               unknown_threshold=SETTINGS["unknown_threshold"],
                               db_path=DB_PATH)
    deepseek = SETTINGS["deepseek"]
    try:
        llm = DeepSeekClient(deepseek["api_key"], deepseek["base_url"], deepseek["model"],
                             deepseek["temperature"], deepseek["timeout"])
        _state["agent"] = CarAgent(llm, _state["tools"], db_path=DB_PATH,
                                   history_messages=SETTINGS["history_messages"])
        print(f"[启动] 智能体就绪: {deepseek['model']} @ {deepseek['base_url']}")
    except LLMError as exc:
        _state["agent"] = None
        _state["load_error"] = str(exc)
        print(f"[启动] {exc}")
    _state["class_ids"] = class_ids


bootstrap()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "static", "index.html"))


@app.get("/api/status")
def status() -> dict:
    model_meta = _state["classifier"].metadata if _state["classifier"] else {}
    return {
        "model_loaded": _state["classifier"] is not None,
        "agent_ready": _state["agent"] is not None,
        "message": _state["load_error"],
        "model_path": SETTINGS["model_path"],
        "model_arch": model_meta.get("arch"),
        "model_val_macro_f1": model_meta.get("val_macro_f1"),
        "confidence_threshold": SETTINGS["confidence_threshold"],
        "unknown_threshold": SETTINGS["unknown_threshold"],
        "deepseek_model": SETTINGS["deepseek"]["model"],
        "tools": [item["function"]["name"] for item in TOOL_SCHEMAS],
    }


@app.get("/api/classes")
def classes() -> dict:
    knowledge_path = resolve(SETTINGS["knowledge_path"])
    knowledge = load_knowledge(knowledge_path) if os.path.isfile(knowledge_path) else {"cards": {}}
    cards = knowledge.get("cards", {})
    items = []
    for cid in load_classes():
        card = cards.get(cid, {})
        items.append({"car_id": cid, "car_name": card.get("name", cid),
                      "brand": card.get("brand", ""), "level": card.get("level", ""),
                      "body_type": card.get("body_type", "")})
    return {"count": len(items), "classes": items}


@app.post("/api/sessions")
def new_session() -> dict:
    session_id = create_session(DB_PATH)
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}/messages")
def session_messages(session_id: str) -> dict:
    if not session_exists(session_id, DB_PATH):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"messages": get_messages(session_id, db_path=DB_PATH)}


@app.post("/api/chat")
async def chat(session_id: str = Form(...), message: str = Form(""),
               image: UploadFile | None = File(None)) -> JSONResponse:
    if not session_exists(session_id, DB_PATH):
        raise HTTPException(status_code=404, detail="会话不存在")
    if _state["agent"] is None:
        return JSONResponse({
            "reply": f"智能体未就绪: {_state['load_error']}。请检查 DEEPSEEK_API_KEY 与模型文件配置。",
            "tool_calls": [], "error": True,
        })

    image_path = ""
    if image is not None and image.filename:
        folder = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(folder, exist_ok=True)
        safe_name = os.path.basename(image.filename)
        image_path = os.path.join(folder, f"{int(time.time() * 1000)}_{safe_name}")
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    try:
        result = _state["agent"].run_turn(session_id, message, image_path)
    except Exception as exc:  # 兜底: 任何未预期异常都给出可读提示
        return JSONResponse({"reply": f"系统处理失败: {exc}", "tool_calls": [], "error": True})
    result["image_url"] = (f"/uploads/{session_id}/{os.path.basename(image_path)}"
                           if image_path else "")
    result["session_id"] = session_id
    return JSONResponse(result)


@app.post("/api/favorites")
def add_favorite_api(car_id: str = Form(...), note: str = Form("")) -> dict:
    result = _state["tools"].add_car_favorite(type("Ctx", (), {"session_id": "web"})(), car_id, note)
    return result


@app.get("/api/favorites")
def favorites() -> dict:
    items = list_favorites(DB_PATH)
    tools = _state.get("tools")
    knowledge = tools.knowledge if tools else {}
    for item in items:
        card = knowledge.get(item["car_id"]) or {}
        item["brand"] = card.get("brand", "")
        item["level"] = card.get("level", "")
        item["body_type"] = card.get("body_type", "")
        path = item.get("image_path") or ""
        if path:
            rel = os.path.relpath(path, UPLOAD_DIR).replace("\\", "/")
            if not rel.startswith(".."):
                item["image_url"] = f"/uploads/{rel}"
    return {"count": len(items), "favorites": items}


@app.delete("/api/favorites/{car_id}")
def delete_favorite(car_id: str) -> dict:
    """收藏项删除按钮使用: 真正从数据库移除。"""
    tools = _state.get("tools")
    if tools is None:
        raise HTTPException(status_code=503, detail="工具尚未就绪")
    result = tools.remove_car_favorite(ToolContext("web"), car_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "删除失败"))
    return result


@app.get("/api/cars/{car_id}")
def car_detail(car_id: str) -> dict:
    """收藏详情弹窗使用: 返回完整车型资料卡。"""
    tools = _state.get("tools")
    card = (tools.knowledge.get(car_id) if tools else None) or None
    if card is None:
        raise HTTPException(status_code=404, detail=f"车型 {car_id} 不存在")
    similar = [tools.knowledge.get(cid) for cid in card.get("similar_ids", [])]
    return {
        "car_id": card["id"], "car_name": card["name"], "brand": card.get("brand", ""),
        "model": card.get("model", ""), "body_type": card.get("body_type", ""),
        "level": card.get("level", ""), "description": card.get("description", ""),
        "supported": bool(card.get("supported")),
        "similar_cars": [{"car_id": item["id"], "car_name": item["name"]}
                         for item in similar if item],
    }


class CompareRequest(BaseModel):
    car_id_a: str
    car_id_b: str


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    """收藏夹“对比”按钮使用: 直接比较两款车, 不经过大模型。"""
    tools = _state.get("tools")
    if tools is None:
        raise HTTPException(status_code=503, detail="工具尚未就绪")
    result = tools.compare_cars(ToolContext("web-compare"), req.car_id_a, req.car_id_b)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "对比失败"))
    return result


@app.get("/api/recognitions")
def recognitions(keyword: str = "", limit: int = 20) -> dict:
    items = search_recognitions(keyword, limit, DB_PATH)
    return {"count": len(items), "records": items}


@app.post("/api/batch/pick_folder")
def pick_folder() -> dict:
    """在本机弹出文件夹选择窗口, 返回所选路径。

    优先使用系统自带的 Windows 文件夹选择框(不依赖 Tcl/Tk, 中文路径也正常);
    失败时退回 tkinter; 都不可用时提示用户直接在输入框填写路径。
    """
    import subprocess
    import tempfile

    def read_result(temp_path: str) -> str:
        if os.path.isfile(temp_path):
            with open(temp_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                return f.read().strip()
        return ""

    # 方式一: Windows 自带文件夹选择框
    temp_file = os.path.join(tempfile.gettempdir(), f"pickfolder_{uuid.uuid4().hex}.txt")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms | Out-Null\n"
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog\n"
        "$dialog.Description = 'Please choose the test image folder'\n"
        "$dialog.ShowNewFolderButton = $false\n"
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {\n"
        f"  Set-Content -LiteralPath '{temp_file}' -Value $dialog.SelectedPath -Encoding UTF8\n"
        "}\n")
    try:
        subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script],
                       capture_output=True, timeout=90)
        path = read_result(temp_file)
        if path and os.path.isdir(path):
            os.remove(temp_file)
            return {"ok": True, "path": path}
    except Exception:
        pass
    finally:
        if os.path.isfile(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass

    # 方式二: tkinter(部分精简版 Python 没有 Tcl/Tk, 会失败)
    code = ("import tkinter as tk\n"
            "from tkinter import filedialog\n"
            "root = tk.Tk()\n"
            "root.withdraw()\n"
            "root.attributes('-topmost', True)\n"
            "picked = filedialog.askdirectory(title='请选择测试集文件夹')\n"
            "root.destroy()\n"
            "print(picked or '')\n")
    try:
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                   text=True, timeout=90)
        output = (completed.stdout or "").strip()
        path = output.splitlines()[-1].strip() if output else ""
        if path and os.path.isdir(path):
            return {"ok": True, "path": path}
    except Exception:
        pass

    return {"ok": False,
            "error": "没能打开文件夹选择窗口(当前 Python 环境缺少图形界面组件)。"
                     "请改用「从浏览器选择文件夹」按钮, 或直接在输入框里粘贴文件夹路径"}


_jobs: dict = {}                 # job_id -> 进度与结果
_job_gate = threading.Lock()     # 同一时间只运行一个批量任务, 其余排队


def _run_batch_job(job_id: str, scan_root: str, run_id: str, paths: list, labels: list,
                   batch_size: int, threshold: float, num_workers: int) -> None:
    """后台线程: 执行批量推理, 写文件, 更新任务状态。"""
    job = _jobs[job_id]
    class_ids = _state["class_ids"]
    with _job_gate:
        job.update(state="running", started_at=time.time())

        def progress(done: int, total: int) -> None:
            job["processed"] = done
            job["total"] = total

        try:
            cards = getattr(_state["tools"], "knowledge", {}) or {}
            name_of = {cid: card.get("name", cid) for cid, card in cards.items()}
            name_of["unknown"] = "unknown（不属于本组支持的车型）"
            result = run_batch(_state["classifier"], paths, labels, class_ids,
                               batch_size=batch_size, name_of=name_of, threshold=threshold,
                               num_workers=num_workers, progress=progress)
            _write_all_outputs(run_id, scan_root, result, class_ids)
            job.update(state="done", summary=job["summary"], finished_at=time.time())
        except Exception as exc:
            job.update(state="error", error=f"批量测试失败: {exc}", finished_at=time.time())


def _write_all_outputs(run_id: str, scan_root: str, result: dict, class_ids: list) -> None:
    """写出运行目录、项目根目录与 EduBench 提交目录的全部产物, 并更新任务摘要。"""
    run_dir = os.path.join(BATCH_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    write_predictions(result["rows"], os.path.join(run_dir, "predictions.csv"))
    write_results(result["rows"], os.path.join(run_dir, "batch_results.csv"), class_ids)
    write_result_csv(result["rows"], os.path.join(run_dir, "result.csv"))
    write_result_npz(result["image_stems"], result["logits"], os.path.join(run_dir, "result.npz"))
    # 同时刷新项目根目录的 predictions.csv, 直接作为提交材料
    write_predictions(result["rows"], os.path.join(PROJECT_ROOT, "predictions.csv"))
    # 写出 EduBench 平台需要的三个文件, 并重新打包
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    shutil.copyfile(os.path.join(PROJECT_ROOT, "classes.txt"),
                    os.path.join(SUBMISSION_DIR, "classes.txt"))
    write_result_csv(result["rows"], os.path.join(SUBMISSION_DIR, "result.csv"))
    write_result_npz(result["image_stems"], result["logits"],
                     os.path.join(SUBMISSION_DIR, "result.npz"))
    package_dir = os.path.dirname(SUBMISSION_DIR)
    zip_path = os.path.join(package_dir, "student_submit.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(os.listdir(SUBMISSION_DIR)):
            archive.write(os.path.join(SUBMISSION_DIR, name), f"student_submit/{name}")

    summary = result["summary"]
    check = check_submission(result["image_stems"], result["rows"], result["logits"],
                             os.path.join(SUBMISSION_DIR, "result.npz"))
    summary.update({
        "folder": scan_root,
        "run_id": run_id,
        "predictions_url": f"/api/batch/download/{run_id}/predictions.csv",
        "results_url": f"/api/batch/download/{run_id}/batch_results.csv",
        "result_csv_url": f"/api/batch/download/{run_id}/result.csv",
        "result_npz_url": f"/api/batch/download/{run_id}/result.npz",
        "submission_dir": os.path.normpath(SUBMISSION_DIR),
        "submission_zip": os.path.normpath(zip_path),
        "check": check,
        "root_csv": "predictions.csv",
    })
    for key in _jobs:
        if _jobs[key].get("run_id") == run_id:
            _jobs[key]["summary"] = summary
            _jobs[key]["preview"] = result["preview"]


@app.post("/api/batch/run")
async def batch_run(session_id: str = Form(""), mode: str = Form("path"),
                    folder_path: str = Form(""), batch_size: int = Form(32),
                    threshold: float = Form(-1.0),
                    files: list[UploadFile] = File(default=[])) -> JSONResponse:
    """启动一次批量测试(后台执行), 立即返回任务号, 前端轮询进度。"""
    if _state["classifier"] is None:
        return JSONResponse({"ok": False,
                             "error": _state["load_error"] or "视觉模型未加载, 请先训练模型并检查 config.yaml"},
                            status_code=200)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    class_ids = _state["class_ids"]

    if mode == "upload":
        if not files:
            return JSONResponse({"ok": False, "error": "没有收到图片, 请重新选择文件夹"}, status_code=200)
        input_dir = os.path.join(BATCH_DIR, run_id, "input")
        for item in files:
            relative = (item.filename or "").replace("\\", "/").lstrip("/")
            parts = [part for part in relative.split("/") if part not in ("", ".", "..")]
            if not parts:
                continue
            target = os.path.join(input_dir, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                shutil.copyfileobj(item.file, f)
        scan_root = input_dir
    else:
        raw_path = (folder_path or "").strip().strip('"')
        if not raw_path:
            return JSONResponse({"ok": False, "error": "请先选择或填写一个文件夹路径"}, status_code=200)
        scan_root = resolve(raw_path)
        if not os.path.isdir(scan_root):
            return JSONResponse({"ok": False, "error": f"文件夹不存在: {scan_root}"}, status_code=200)

    paths, labels = collect_folder(scan_root, class_ids)
    if not paths:
        return JSONResponse({"ok": False,
                             "error": f"该文件夹里没有找到图片(支持 jpg/jpeg/png/bmp/webp): {scan_root}"},
                            status_code=200)
    if len(paths) > MAX_IMAGES:
        return JSONResponse({"ok": False,
                             "error": f"图片数量 {len(paths)} 超过单次上限 {MAX_IMAGES} 张"}, status_code=200)

    if threshold is None or threshold < 0:
        threshold = float((SETTINGS.get("submission") or {}).get("rejection_threshold", 0.6))
    batch_size = max(1, min(int(batch_size or 32), 128))
    # 图片多时用多进程读图(CPU 上大约能快 30%-60%), 图片少时没必要付进程启动开销
    num_workers = 4 if len(paths) >= 2000 else 0

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"state": "queued", "total": len(paths), "processed": 0, "run_id": run_id,
                     "folder": scan_root, "created_at": time.time(), "threshold": threshold,
                     "num_workers": num_workers}
    worker = threading.Thread(target=_run_batch_job,
                              args=(job_id, scan_root, run_id, paths, labels,
                                    batch_size, float(threshold), num_workers),
                              daemon=True)
    worker.start()
    return JSONResponse({"ok": True, "job_id": job_id, "state": "queued", "total": len(paths),
                         "run_id": run_id, "threshold": float(threshold)})


@app.get("/api/batch/job/{job_id}")
def batch_job(job_id: str) -> dict:
    """查询批量任务进度; 完成后返回统计结果与明细预览。"""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    payload = {key: value for key, value in job.items() if key not in ("summary", "preview")}
    payload["ok"] = job.get("state") != "error"
    payload["elapsed"] = round(time.time() - job.get("created_at", time.time()), 1)
    if job.get("state") == "done":
        payload["summary"] = job.get("summary")
        payload["preview"] = job.get("preview")
    return payload


@app.get("/api/batch/download/{run_id}/{filename}")
def batch_download(run_id: str, filename: str) -> FileResponse:
    path = os.path.join(BATCH_DIR, os.path.basename(run_id), os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path, filename=os.path.basename(filename))


@app.get("/uploads/{session_id}/{filename}")
def uploaded_file(session_id: str, filename: str) -> FileResponse:
    path = os.path.join(UPLOAD_DIR, os.path.basename(session_id), os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
