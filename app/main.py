"""FastAPI 对外接口：材料档登记、近轴光路登记与多波长追迹、物像求解、色差归约。

启动：uvicorn app.main:app
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .chromatic import chromatic_aberration
from .demo import TELESCOPE_DEMO_NAME, telescope_demo_elements
from .elements import Element
from .errors import OpticsError
from .materials import DEMO_MATERIALS, MaterialStore
from .spectral import single_trace_payload, trace_spectrum
from .storage import PathStore
from .validation import (
    normalize_chromatic_basis,
    normalize_material_body,
    normalize_object_distance,
    normalize_ray,
    normalize_register_body,
    normalize_trace_body,
    normalize_wavelength_value,
    normalize_wavelengths,
    require_wavelengths_if_materials,
)


store = PathStore()
materials = MaterialStore()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 内置共焦望远镜示范档与两块示范玻璃（N-BK7 / SF10）
    store.seed(TELESCOPE_DEMO_NAME, telescope_demo_elements())
    for glass_name, description in DEMO_MATERIALS.items():
        materials.seed(glass_name, description)
    yield


app = FastAPI(
    title="近轴光路追迹服务",
    version="2.0.0",
    description="薄透镜、空气间隔、球面折射面的近轴矩阵追迹与成像求解；"
                "折射面可点名具名玻璃，按波长现算折射率，支持多波长追迹与色差归约（仅 HTTP）。",
    lifespan=lifespan,
)


@app.exception_handler(OpticsError)
async def optics_error_handler(_request: Request, exc: OpticsError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.to_dict()})


async def _read_json(request: Request) -> Any:
    raw = await request.body()
    if not raw:
        raise OpticsError("invalid_body", "请求体为空，需要 JSON 对象", status_code=400)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpticsError(
            "invalid_body",
            f"请求体不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
            status_code=400,
        )


def _spectrum_response(elements: list[Element],
                       ray: tuple[float, float],
                       s: float | None,
                       wavelengths: list[float],
                       basis: tuple[float, float] | None) -> dict[str, object]:
    """多波长追迹响应：逐波长结果 + 跨波长色差归约。"""
    results = trace_spectrum(elements, ray, s, wavelengths, materials.index_of)
    return {
        "wavelengths": list(wavelengths),
        "wavelength_unit": "nm",
        "request": {
            "ray": {"y": ray[0], "u": ray[1]},
            "s": s,
            "s_meaning": "null 表示平行光入射（物在无穷远）",
        },
        "elements": [e.describe() for e in elements],
        "results": results,
        "chromatic": chromatic_aberration(results, wavelengths, basis, s),
    }


def _dispatch_trace(elements: list[Element],
                    ray: tuple[float, float],
                    s: float | None,
                    wavelengths: list[float] | None,
                    basis: tuple[float, float] | None) -> dict[str, object]:
    """老式单波长请求走原路；给了 wavelengths 就走多波长编排。"""
    if wavelengths is None:
        return single_trace_payload(elements, ray, s)
    return _spectrum_response(elements, ray, s, wavelengths, basis)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 材料档
# ---------------------------------------------------------------------------

@app.post("/materials", status_code=200)
async def register_material(request: Request) -> dict[str, object]:
    """登记一份具名材料档：采样点描述或解析系数描述（二选一）。"""
    name, description = normalize_material_body(await _read_json(request))
    material = materials.register(name, description)
    return {"registered": name, "material": material.describe()}


@app.get("/materials")
def list_materials() -> dict[str, object]:
    """列出全部材料档：描述方式与全部系数或采样点。"""
    return {"materials": materials.list_materials()}


@app.get("/materials/{name}/index")
def material_index(name: str, wavelength: str | None = None) -> dict[str, object]:
    """点名材料在给定波长（nm）现算折射率。"""
    if wavelength is None:
        raise OpticsError(
            "missing_field",
            "缺少查询参数 wavelength（单位 nm）",
            field="wavelength",
            status_code=400,
        )
    try:
        raw: Any = float(wavelength)
    except ValueError:
        raise OpticsError(
            "invalid_wavelength",
            f"查询参数 wavelength 必须是数值（单位 nm），收到 {wavelength!r}",
            field="wavelength",
        )
    wl = normalize_wavelength_value(raw)
    material = materials.get(name)
    return {
        "material": name,
        "wavelength": wl,
        "wavelength_unit": "nm",
        "n": material.index(wl),
        "model": material.model.describe(),
    }


# ---------------------------------------------------------------------------
# 光路档
# ---------------------------------------------------------------------------

@app.get("/paths")
def list_paths() -> dict[str, object]:
    """列出全部具名档，给出元件种类与参数全文。"""
    return {"paths": store.list_paths()}


@app.post("/paths")
async def register_path(request: Request) -> dict[str, object]:
    """登记一条具名光路档（进程内 SQLite）。"""
    name, elements = normalize_register_body(await _read_json(request))
    store.register(name, elements)
    return {"registered": name, "elements": [e.describe() for e in elements]}


@app.post("/paths/{name}/trace")
async def trace_named(name: str, request: Request) -> dict[str, object]:
    """点名一档追迹；未登记的档名拒绝，不猜测。可带 wavelengths 走多波长。"""
    body = await _read_json(request)
    elements = store.get(name)
    if not isinstance(body, dict):
        raise OpticsError("invalid_body", "请求体必须是 JSON 对象", status_code=400)
    ray = normalize_ray(body.get("ray"))
    s = normalize_object_distance(body.get("s"))
    wavelengths = normalize_wavelengths(body.get("wavelengths"))
    basis = normalize_chromatic_basis(body.get("chromatic"))
    require_wavelengths_if_materials(elements, wavelengths)
    result = _dispatch_trace(elements, ray, s, wavelengths, basis)
    result["path_name"] = name
    return result


@app.post("/trace")
async def trace_once(request: Request) -> dict[str, object]:
    """当次请求内联元件清单，只用一次，不登记。可带 wavelengths 走多波长。"""
    elements, ray, s, wavelengths, basis = normalize_trace_body(await _read_json(request))
    return _dispatch_trace(elements, ray, s, wavelengths, basis)
