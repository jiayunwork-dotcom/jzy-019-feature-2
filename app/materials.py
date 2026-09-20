"""具名材料档：色散模型的进程内 SQLite 存取。

与光路档共用一个数据库文件（``OPTICS_DB``），单建 ``materials`` 表。
每次操作使用短连接（WAL 下读写可并发），写操作用进程内锁串行化。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from .dispersion import (
    CauchyModel,
    ConstantIndex,
    DispersionModel,
    SellmeierModel,
    fit_cauchy,
)
from .errors import OpticsError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


@dataclass(frozen=True)
class Material:
    """一份具名材料档：色散模型 + 登记时的原始描述。"""

    name: str
    model: DispersionModel
    description: dict[str, object]

    def index(self, wavelength_nm: float) -> float:
        try:
            return self.model.index(wavelength_nm)
        except OpticsError as exc:
            # 补上材料名，方便调用方定位是哪块玻璃算不出来
            exc.extra.setdefault("material", self.name)
            raise

    def describe(self) -> dict[str, object]:
        return {"name": self.name, **self.description}


def model_from_description(description: dict[str, object]) -> DispersionModel:
    """按登记时的描述重建色散模型（入参检查保证描述合法）。"""
    description_type = description["description_type"]
    if description_type == "samples":
        samples = [
            (float(p["wavelength"]), float(p["n"]))
            for p in description["samples"]  # type: ignore[index]
        ]
        return fit_cauchy(samples)
    coefficients = description["coefficients"]
    assert isinstance(coefficients, dict)
    model_name = description["model"]
    if model_name == "constant":
        return ConstantIndex(n=float(coefficients["n"]))
    if model_name == "cauchy":
        return CauchyModel(
            A=float(coefficients["A"]),
            B=float(coefficients["B"]),
            C=float(coefficients.get("C", 0.0)),
        )
    if model_name == "sellmeier":
        return SellmeierModel(
            B=tuple(float(v) for v in coefficients["B"]),  # type: ignore[arg-type]
            C=tuple(float(v) for v in coefficients["C"]),  # type: ignore[arg-type]
        )
    raise OpticsError(  # pragma: no cover - 入参检查已拦住
        "invalid_material", f"未知色散模型 {model_name!r}"
    )


def _material_from_row(name: str, payload: str) -> Material:
    description = json.loads(payload)
    return Material(
        name=name,
        model=model_from_description(description),
        description=description,
    )


class MaterialStore:
    """具名材料档的存取；与 :class:`PathStore` 同库不同表。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.environ.get("OPTICS_DB", "optics_paths.db")
        self._write_lock = threading.Lock()
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

    def register(self, name: str, description: dict[str, object]) -> Material:
        """登记一份材料档；``description`` 必须已过入参检查。"""
        material = Material(
            name=name,
            model=model_from_description(description),
            description=description,
        )
        payload = json.dumps(description, ensure_ascii=False)
        with self._write_lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO materials(name, description) VALUES (?, ?)",
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
        return material

    def get(self, name: str) -> Material:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT description FROM materials WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            raise OpticsError(
                "unknown_material",
                f"材料档 {name!r} 未登记，不猜测；请先登记或用 /materials 查看已登记材料",
                name=name,
                status_code=404,
            )
        return _material_from_row(name, row["description"])

    def index_of(self, name: str, wavelength_nm: float) -> float:
        """按名取材料并在给定波长求折射率（追迹时的现算入口）。"""
        return self.get(name).index(wavelength_nm)

    def list_materials(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, created_at FROM materials ORDER BY rowid"
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            material = _material_from_row(row["name"], row["description"])
            entry: dict[str, object] = {
                **material.describe(),
                "created_at": row["created_at"],
            }
            # 采样点档同时给出拟合所得的 Cauchy 系数，便于核对
            if entry.get("description_type") == "samples":
                entry["fitted"] = material.model.describe()
            result.append(entry)
        return result

    def seed(self, name: str, description: dict[str, object]) -> None:
        """登记内置示范材料；已存在同名档则保留不动。"""
        payload = json.dumps(description, ensure_ascii=False)
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO materials(name, description) VALUES (?, ?)",
                (name, payload),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# 内置示范玻璃（Sellmeier 系数，λ 以 µm 计，数据为公开牌号值）
# ---------------------------------------------------------------------------

#: N-BK7 冕玻璃
BK7_DESCRIPTION: dict[str, object] = {
    "description_type": "coefficients",
    "model": "sellmeier",
    "coefficients": {
        "B": [1.03961212, 0.231792344, 1.01046945],
        "C": [0.00600069867, 0.0200179144, 103.560653],
    },
}

#: SF10 火石玻璃
SF10_DESCRIPTION: dict[str, object] = {
    "description_type": "coefficients",
    "model": "sellmeier",
    "coefficients": {
        "B": [1.62153902, 0.256287842, 1.64447552],
        "C": [0.0122241457, 0.0595736775, 147.468793],
    },
}

DEMO_MATERIALS: dict[str, dict[str, object]] = {
    "N-BK7": BK7_DESCRIPTION,
    "SF10": SF10_DESCRIPTION,
}
