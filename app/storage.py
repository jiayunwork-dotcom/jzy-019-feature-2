"""具名光路档与材料档的进程内 SQLite 存取。

两类档各一张表，同库文件。每次操作使用短连接：WAL 模式下读写可并发，
短连接也避免读事务长期挂起而阻塞写操作。写操作另用进程内锁串行化。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from .dispersion import build_dispersion
from .elements import Element
from .errors import OpticsError
from .materials import Material

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paths (
    name      TEXT PRIMARY KEY,
    elements  TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS materials (
    name      TEXT PRIMARY KEY,
    spec      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _elements_to_json(elements: Iterable[Element]) -> str:
    return json.dumps(
        [{"type": e.kind, "params": e.params} for e in elements],
        ensure_ascii=False,
    )


def _elements_from_json(payload: str) -> list[Element]:
    data = json.loads(payload)
    return [Element(kind=item["type"], params=item["params"]) for item in data]


class _SQLiteStore:
    """单文件 SQLite，默认落盘在进程工作目录（可用 OPTICS_DB 覆盖）。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.environ.get("OPTICS_DB", "optics_paths.db")
        self._write_lock = threading.Lock()
        # 建表与 WAL 切换都是写操作，只在初始化时做一次。
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
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


class PathStore(_SQLiteStore):
    """光路档存取。"""

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
                "SELECT name, elements, created_at FROM paths ORDER BY name"
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


class MaterialStore(_SQLiteStore):
    """材料档存取：原样保留登记的色散描述，读取时重建色散模型。"""

    def register(self, name: str, spec: dict[str, Any], *, overwrite: bool = False) -> None:
        # 先在写库前构造一次，拒绝非法色散描述
        build_dispersion(spec)
        payload = json.dumps(spec, ensure_ascii=False)
        with self._write_lock, self._connect() as conn:
            if overwrite:
                conn.execute(
                    "INSERT INTO materials(name, spec) VALUES (?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET spec = excluded.spec",
                    (name, payload),
                )
            else:
                try:
                    conn.execute(
                        "INSERT INTO materials(name, spec) VALUES (?, ?)",
                        (name, payload),
                    )
                except sqlite3.IntegrityError:
                    raise OpticsError(
                        "duplicate_name",
                        f"材料档 {name!r} 已登记",
                        name=name,
                        status_code=409,
                    )
            conn.commit()

    def get(self, name: str) -> Material:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT spec FROM materials WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            raise OpticsError(
                "unknown_material",
                f"材料档 {name!r} 未登记；请先用 /materials 登记",
                material=name,
                status_code=404,
            )
        return Material(name=name, model=build_dispersion(json.loads(row["spec"])))

    def all_materials(self) -> list[Material]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, spec FROM materials ORDER BY name"
            ).fetchall()
        return [
            Material(name=row["name"], model=build_dispersion(json.loads(row["spec"])))
            for row in rows
        ]

    def materials_map(self) -> dict[str, Material]:
        return {material.name: material for material in self.all_materials()}

    def list_materials(self) -> list[dict[str, object]]:
        return [material.describe() for material in self.all_materials()]

    def seed(self, material: Material) -> None:
        """登记内置材料档；同名保留不动。"""
        spec = material.model.describe()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO materials(name, spec) VALUES (?, ?)",
                (material.name, json.dumps(spec, ensure_ascii=False)),
            )
            conn.commit()
