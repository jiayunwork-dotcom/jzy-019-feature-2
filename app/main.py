"""FastAPI 对外接口：材料档、光路档登记，单波长/多波长近轴追迹。

启动：uvicorn app.main:app

波长维度说明：``/trace`` 与 ``/paths/{name}/trace`` 的请求体可选带
``wavelengths``（μm 数组）。不带波长时走旧的单波长路径，响应结构与
旧版完全一致（顶层 system_abcd / imaging / outgoing_ray ...）；带波长
时返回 ``results``（每波长一份，矩阵与光线严格隔离）与跨波长色差
``chromatic_aberration``。
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import demo
from .chromatic import chromatic_aberration
from .dispersion import DEFAULT_WAVELENGTH_UM
from .elements import Element
from .errors import OpticsError
from .materials import normalize_material_body
from .multispectral import trace_multiwavelength
from .storage import MaterialStore, PathStore
from .validation import (
    normalize_elements,
    normalize_object_distance,
    normalize_ray,
    normalize_reference_pair,
    normalize_register_body,
    normalize_trace_body,
    normalize_wavelengths,
)


store = PathStore()
material_store = MaterialStore()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 内置材料档 + 示范光路档（均幂等，同名保留）
    for material in demo.seed_materials():
        material_store.seed(material)
    for name, elements in demo.seed_paths():
        store.seed(name, elements)
    yield


app = FastAPI(
    title="近轴光路追迹服务",
    version="2.0.0",
    description=(
        "薄透镜、空气间隔、球面折射面的近轴矩阵追迹与成像求解；"
        "支持具名色散材料（constant/cauchy/sellmeier/sampled）与多波长色差计算。仅 HTTP。"
    ),
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


def _request_block(ray: tuple[float, float],
                   s: float | None,
                   wavelengths: list[float]) -> dict[str, object]:
    block: dict[str, object] = {
        "ray": {"y": ray[0], "u": ray[1]},
        "s": s,
        "s_meaning": "null 表示平行光入射（物在无穷远）",
        "wavelengths": wavelengths,
        "wavelength_unit": "um",
    }
    return block


def _run_trace(elements: list[Element],
               ray: tuple[float, float],
               s: float | None,
               wavelengths: list[float] | None,
               pair: tuple[float, float] | None) -> dict[str, object]:
    if wavelengths is None:
        # 旧单波长路径：d 线波长下追迹一次，响应保持旧结构
        wavelength_list = [DEFAULT_WAVELENGTH_UM]
        results = trace_multiwavelength(
            elements, ray, s, wavelength_list, material_store.materials_map()
        )
        result = results[0]
        result["request"] = _request_block(ray, s, wavelength_list)
        return result

    results = trace_multiwavelength(
        elements, ray, s, wavelengths, material_store.materials_map()
    )
    assert pair is not None
    short, long_ = pair
    response: dict[str, object] = {
        "wavelengths": wavelengths,
        "wavelength_unit": "um",
        "count": len(wavelengths),
        "results": results,
        "chromatic_aberration": chromatic_aberration(results, short, long_),
        "request": _request_block(ray, s, wavelengths),
    }
    return response


def _trace_from_body(body: Any) -> dict[str, object]:
    known = material_store.all_materials()
    known_names = {material.name for material in known}
    elements, ray, s, wavelengths, pair = normalize_trace_body(
        body, known_materials=known_names
    )
    return _run_trace(elements, ray, s, wavelengths, pair)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 材料档
# ---------------------------------------------------------------------------

@app.get("/materials")
def list_materials() -> dict[str, object]:
    """列出全部具名材料档：描述方式与全部系数或采样点。"""
    return {"materials": material_store.list_materials()}


@app.post("/materials")
async def register_material(request: Request) -> dict[str, object]:
    """登记一种具名材料档（进程内 SQLite）。"""
    material = normalize_material_body(await _read_json(request))
    material_store.register(material.name, material.model.describe())
    return {"registered": material.name, "material": material.describe()}


@app.get("/materials/{name}")
def get_material(name: str) -> dict[str, object]:
    material = material_store.get(name)
    return material.describe()


@app.get("/materials/{name}/index")
def material_index(name: str, wavelength: float) -> dict[str, object]:
    """点名材料在给定波长（μm）处现算折射率。"""
    material = material_store.get(name)
    n = material.model.index(wavelength)
    return {
        "material": name,
        "wavelength": wavelength,
        "wavelength_unit": "um",
        "index": n,
        "description_kind": material.model.type_name,
    }


# ---------------------------------------------------------------------------
# 光路档
# ---------------------------------------------------------------------------

@app.get("/paths")
def list_paths() -> dict[str, object]:
    """列出全部具名光路档，给出元件种类与参数全文。"""
    return {"paths": store.list_paths()}


@app.post("/paths")
async def register_path(request: Request) -> dict[str, object]:
    """登记一条具名光路档（进程内 SQLite）。"""
    known_names = {material.name for material in material_store.all_materials()}
    name, elements = normalize_register_body(
        await _read_json(request), known_materials=known_names
    )
    store.register(name, elements)
    return {"registered": name, "elements": [e.describe() for e in elements]}


@app.post("/paths/{name}/trace")
async def trace_named(name: str, request: Request) -> dict[str, object]:
    """点名一档追迹；未登记的档名/材料档拒绝，不猜测。"""
    body = await _read_json(request)
    if not isinstance(body, dict):
        raise OpticsError("invalid_body", "请求体必须是 JSON 对象", status_code=400)
    # 先做请求体校验（与 /trace 一致的退回语义），再取档
    ray = normalize_ray(body.get("ray"))
    s = normalize_object_distance(body.get("s"))
    wavelengths = normalize_wavelengths(body.get("wavelengths"))
    pair = None
    if wavelengths is not None:
        pair = normalize_reference_pair(body.get("reference_wavelengths"), wavelengths)
    elif body.get("reference_wavelengths") is not None:
        raise OpticsError(
            "invalid_reference_wavelength",
            "给出 reference_wavelengths 时必须同时给出 wavelengths 数组",
        )
    elements = store.get(name)
    # 光路档内引用的材料档也在追迹前点名核验（直接规范化存储的原始参数，
    # 不经过 describe() 往返，避免具名介质被误当成字符串折射率）
    known_names = {material.name for material in material_store.all_materials()}
    normalize_elements(
        [{"type": element.kind, "params": dict(element.params)} for element in elements],
        known_materials=known_names,
    )
    result = _run_trace(elements, ray, s, wavelengths, pair)
    result["path_name"] = name
    return result


@app.post("/trace")
async def trace_once(request: Request) -> dict[str, object]:
    """当次请求内联元件清单，只用一次，不登记。"""
    return _trace_from_body(await _read_json(request))
