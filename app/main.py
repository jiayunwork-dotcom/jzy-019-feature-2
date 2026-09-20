"""FastAPI 对外接口：近轴光路登记与追迹、物像求解。

启动：uvicorn app.main:app
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .demo import TELESCOPE_DEMO_NAME, telescope_demo_elements
from .elements import Element
from .errors import OpticsError
from .storage import PathStore
from .systems import (
    ROUNDTRIP_TOL,
    solve_imaging,
    reverse_trace_residual,
    trace,
)
from .validation import (
    normalize_object_distance,
    normalize_ray,
    normalize_register_body,
    normalize_trace_body,
)


store = PathStore()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 内置共焦望远镜示范档
    store.seed(TELESCOPE_DEMO_NAME, telescope_demo_elements())
    yield


app = FastAPI(
    title="近轴光路追迹服务",
    version="1.0.0",
    description="薄透镜、空气间隔、球面折射面的近轴矩阵追迹与成像求解（仅 HTTP）。",
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


def _trace_response(elements: list[Element],
                    ray: tuple[float, float],
                    s: float | None) -> dict[str, object]:
    result = trace(elements, ray)
    imaging = solve_imaging(elements, s)
    residual, restored = reverse_trace_residual(elements, ray)
    if residual > ROUNDTRIP_TOL:  # 数学上不应发生；守住往返关系
        raise OpticsError(
            "roundtrip_error",
            f"正向后逆向未能还原入射光线，残差 {residual:g} 超过容差 {ROUNDTRIP_TOL:g}",
            status_code=500,
            residual=residual,
        )

    system = result["system_matrix"]
    assert isinstance(system, dict)
    result["imaging"] = {
        **imaging,
        "object_distance_convention": "物在第一面左侧，s 取正；null 表示物在无穷远",
    }
    result["roundtrip"] = {
        "restored_ray": {"y": restored[0], "u": restored[1]},
        "residual": residual,
        "tolerance": ROUNDTRIP_TOL,
    }
    result["request"] = {
        "ray": {"y": ray[0], "u": ray[1]},
        "s": s,
        "s_meaning": "null 表示平行光入射（物在无穷远）",
    }
    result["system_abcd"] = {
        "A": system["A"], "B": system["B"], "C": system["C"], "D": system["D"],
    }
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    """点名一档追迹；未登记的档名拒绝，不猜测。"""
    body = await _read_json(request)
    elements = store.get(name)
    if not isinstance(body, dict):
        raise OpticsError("invalid_body", "请求体必须是 JSON 对象", status_code=400)
    ray = normalize_ray(body.get("ray"))
    s = normalize_object_distance(body.get("s"))
    result = _trace_response(elements, ray, s)
    result["path_name"] = name
    return result


@app.post("/trace")
async def trace_once(request: Request) -> dict[str, object]:
    """当次请求内联元件清单，只用一次，不登记。"""
    elements, ray, s = normalize_trace_body(await _read_json(request))
    return _trace_response(elements, ray, s)
