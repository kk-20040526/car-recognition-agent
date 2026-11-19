# -*- coding: utf-8 -*-
"""数据存储层(SQLite): 会话、消息、识别记录、收藏夹。

所有业务工具都通过本模块真实读写数据库, 不使用固定的假数据。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "storage", "app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_calls  TEXT DEFAULT '[]',
    tool_call_id TEXT DEFAULT '',
    image_path  TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS recognitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    image_path      TEXT NOT NULL,
    image_id        TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    car_name        TEXT DEFAULT '',
    confidence      REAL NOT NULL,
    top_k           TEXT DEFAULT '[]',
    is_unknown      INTEGER DEFAULT 0,
    created_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS favorites (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    car_id      TEXT NOT NULL,
    car_name    TEXT NOT NULL,
    note        TEXT DEFAULT '',
    source      TEXT DEFAULT 'manual',
    image_path  TEXT DEFAULT '',
    confidence  REAL DEFAULT 0,
    confidence_level TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    UNIQUE (car_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_recognitions_session ON recognitions(session_id, id);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """给旧版 favorites 表补列(幂等), 避免已有数据库因缺列报错。"""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(favorites)")}
    additions = {
        "image_path": "TEXT DEFAULT ''",
        "confidence": "REAL DEFAULT 0",
        "confidence_level": "TEXT DEFAULT ''",
    }
    for name, ddl in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE favorites ADD COLUMN {name} {ddl}")
    conn.commit()


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# ---------------------------------------------------------------- sessions
def create_session(db_path: str = DB_PATH, title: str = "") -> str:
    session_id = uuid.uuid4().hex[:12]
    now = time.time()
    with connect(db_path) as conn:
        conn.execute("INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                     (session_id, title, now, now))
    return session_id


def touch_session(session_id: str, db_path: str = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time(), session_id))


def session_exists(session_id: str, db_path: str = DB_PATH) -> bool:
    with connect(db_path) as conn:
        return conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None


# ---------------------------------------------------------------- messages
def add_message(session_id: str, role: str, content: str, tool_calls: list | None = None,
                image_path: str = "", tool_call_id: str = "", db_path: str = DB_PATH) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, image_path, tool_call_id,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(tool_calls or [], ensure_ascii=False),
             image_path, tool_call_id, time.time()))
        return int(cursor.lastrowid)


def get_messages(session_id: str, limit: int = 200, db_path: str = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id LIMIT ?",
            (session_id, limit)).fetchall()
    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "tool_calls": json.loads(row["tool_calls"] or "[]"),
            "image_path": row["image_path"],
            "tool_call_id": row["tool_call_id"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


# ------------------------------------------------------------ recognitions
def add_recognition(session_id: str, image_path: str, image_id: str, predicted_label: str,
                    car_name: str, confidence: float, top_k: list, is_unknown: bool,
                    db_path: str = DB_PATH) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO recognitions (session_id, image_path, image_id, predicted_label, car_name,"
            " confidence, top_k, is_unknown, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, image_path, image_id, predicted_label, car_name, confidence,
             json.dumps(top_k, ensure_ascii=False), int(is_unknown), time.time()))
        return int(cursor.lastrowid)


def last_recognition(session_id: str, db_path: str = DB_PATH) -> dict | None:
    """取同一会话中最近一次识别结果, 供多轮对话直接引用。"""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM recognitions WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "image_path": row["image_path"],
        "image_id": row["image_id"],
        "predicted_label": row["predicted_label"],
        "car_name": row["car_name"],
        "confidence": row["confidence"],
        "top_k": json.loads(row["top_k"] or "[]"),
        "is_unknown": bool(row["is_unknown"]),
        "created_at": row["created_at"],
    }


def search_recognitions(keyword: str = "", limit: int = 10, db_path: str = DB_PATH) -> list[dict]:
    """按车型 ID / 车型名称检索历史识别记录。"""
    sql = "SELECT * FROM recognitions"
    params: list = []
    if keyword:
        sql += " WHERE predicted_label LIKE ? OR car_name LIKE ? OR image_id LIKE ?"
        like = f"%{keyword}%"
        params = [like, like, like]
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": row["id"],
            "image_id": row["image_id"],
            "predicted_label": row["predicted_label"],
            "car_name": row["car_name"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


# --------------------------------------------------------------- favorites
def add_favorite(car_id: str, car_name: str, note: str = "", source: str = "manual",
                 image_path: str = "", confidence: float = 0.0, confidence_level: str = "",
                 db_path: str = DB_PATH) -> dict:
    with connect(db_path) as conn:
        existing = conn.execute("SELECT id FROM favorites WHERE car_id = ?", (car_id,)).fetchone()
        if existing:
            conn.execute("UPDATE favorites SET note = ?, car_name = ?, image_path = ?, "
                         "confidence = ?, confidence_level = ? WHERE car_id = ?",
                         (note, car_name, image_path, confidence, confidence_level, car_id))
            return {"status": "updated", "car_id": car_id, "car_name": car_name}
        conn.execute("INSERT INTO favorites (car_id, car_name, note, source, image_path, confidence,"
                     " confidence_level, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (car_id, car_name, note, source, image_path, confidence, confidence_level, time.time()))
    return {"status": "added", "car_id": car_id, "car_name": car_name}


def list_favorites(db_path: str = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM favorites ORDER BY id DESC").fetchall()
    return [
        {"car_id": row["car_id"], "car_name": row["car_name"], "note": row["note"],
         "source": row["source"], "created_at": row["created_at"],
         "image_path": row["image_path"] or "", "confidence": float(row["confidence"] or 0),
         "confidence_level": row["confidence_level"] or ""}
        for row in rows
    ]


def remove_favorite(car_id: str, db_path: str = DB_PATH) -> dict:
    with connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM favorites WHERE car_id = ?", (car_id,))
        return {"status": "removed" if cursor.rowcount else "not_found", "car_id": car_id}
