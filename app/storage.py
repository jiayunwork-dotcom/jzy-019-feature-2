"""具名光路档的进程内 SQLite 存取。

每次操作使用短连接：WAL 模式下读写可并发，短连接也避免读事务
长期挂起而阻塞写操作。写操作另用进程内锁串行化。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator

from .elements import Element
from .errors import OpticsError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paths (
    name      TEXT PRIMARY KEY,
    elements  TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def _elements_to_json(elements: Iterable[Element]) -> str:
    return json.dumps(
        [{"type": e.kind, "params": e.params} for e in elements],
        ensure_ascii=False,
    )


def _elements_from_json(payload: str) -> list[Element]:
    data = json.loads(payload)
    return [Element(kind=item["type"], params=item["params"]) for item in data]


class PathStore:
    """单文件 SQLite，默认落盘在进程工作目录（可用 OPTICS_DB 覆盖）。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.environ.get("OPTICS_DB", "optics_paths.db")
        self._write_lock = threading.Lock()
        # 建表与 WAL 切换都是写操作，只在初始化时做一次。
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            yield conn
        finally:
            conn.close()

    def register(self, name: str, elements: list[Element], *, overwrite: bool = False) -> None:
        payload = _elements_to_json(elements)
        with self._write_lock, self._connect() as conn:
            if overwrite:
                conn.execute(
                    "INSERT INTO paths(name, elements) VALUES (?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET elements = excluded.elements",
                    (name, payload),
                )
            else:
                try:
                    conn.execute(
                        "INSERT INTO paths(name, elements) VALUES (?, ?)",
                        (name, payload),
                    )
                except sqlite3.IntegrityError:
                    raise OpticsError(
                        "duplicate_name",
                        f"光路档 {name!r} 已登记",
                        name=name,
                        status_code=409,
                    )
            conn.commit()

    def get(self, name: str) -> list[Element]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT elements FROM paths WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            raise OpticsError(
                "unknown_path",
                f"光路档 {name!r} 未登记，不猜测；请先登记或用 /paths 登记或用 /trace 直接追迹",
                name=name,
                status_code=404,
            )
        return _elements_from_json(row["elements"])

    def list_paths(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, elements, created_at FROM paths ORDER BY rowid"
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            elements = _elements_from_json(row["elements"])
            result.append({
                "name": row["name"],
                "created_at": row["created_at"],
                "elements": [e.describe() for e in elements],
            })
        return result

    def seed(self, name: str, elements: list[Element]) -> None:
        """登记示范档；已存在同名档则保留不动。"""
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO paths(name, elements) VALUES (?, ?)",
                (name, _elements_to_json(elements)),
            )
            conn.commit()

    def exists(self, name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM paths WHERE name = ?", (name,)
            ).fetchone()
        return row is not None
