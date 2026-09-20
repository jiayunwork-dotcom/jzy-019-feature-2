"""入参检查：把原始 JSON 体转成规范化对象，不合规即抛出带类型的错误。"""

from __future__ import annotations

import math
from typing import Any

from .elements import Element
from .errors import OpticsError

#: 接受的元件名 -> 规范名
TYPE_ALIASES: dict[str, str] = {
    "space": "space",
    "air_space": "space",
    "airspace": "space",
    "propagate": "space",
    "lens": "lens",
    "thin_lens": "lens",
    "thinlens": "lens",
    "refract": "refract",
    "refraction": "refract",
    "spherical_refraction": "refract",
    "surface": "refract",
}

#: 每种规范元件需要的数值参数
REQUIRED_PARAMS: dict[str, dict[str, tuple[str, ...]]] = {
    "space": {"L": ("L", "length", "d", "distance")},
    "lens": {"f": ("f", "focal_length", "focal", "focalLength")},
    "refract": {
        "R": ("R", "radius", "radiusOfCurvature"),
        "n1": ("n1", "n_in", "nBefore"),
        "n2": ("n2", "n_out", "nAfter"),
    },
}


def _require_object(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OpticsError(
            "invalid_body",
            f"{what}必须是 JSON 对象，收到 {type(value).__name__}",
            status_code=400,
        )
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpticsError(
            "non_finite_number",
            f"参数 {field!r} 必须是有限数值，收到 {value!r}",
            field=field,
            value=value,
        )
    number = float(value)
    if not math.isfinite(number):
        raise OpticsError(
            "non_finite_number",
            f"参数 {field!r} 必须是有限数值，收到 {value!r}",
            field=field,
            value=value,
        )
    return number


def normalize_element(raw: Any, index: int) -> Element:
    """检查并规范化第 ``index``（1 起）个元件。"""
    where = f"第 {index} 个元件"
    if not isinstance(raw, dict):
        raise OpticsError(
            "invalid_element",
            f"{where}必须是对象，收到 {type(raw).__name__}",
            index=index,
        )

    if "type" not in raw:
        raise OpticsError("missing_field", f"{where}缺少 type 字段", index=index, field="type")

    raw_type = raw["type"]
    if not isinstance(raw_type, str):
        raise OpticsError(
            "unknown_element",
            f"{where}的 type 必须是字符串，收到 {raw_type!r}",
            index=index,
        )
    kind = TYPE_ALIASES.get(raw_type.strip())
    if kind is None:
        raise OpticsError(
            "unknown_element",
            f"{where}的元件种类 {raw_type!r} 未知，"
            f"支持：space（空气间隔）、lens（薄透镜）、refract（球面折射）",
            index=index,
            type=raw_type,
        )

    if "params" not in raw or not isinstance(raw["params"], dict):
        raise OpticsError(
            "missing_field",
            f"{where}缺少 params 对象",
            index=index,
            field="params",
        )
    raw_params = raw["params"]

    params: dict[str, float] = {}
    for canonical, aliases in REQUIRED_PARAMS[kind].items():
        present = [name for name in aliases if name in raw_params]
        if not present:
            raise OpticsError(
                "missing_field",
                f"{where}（{kind}）缺少参数 {canonical!r}",
                index=index,
                field=canonical,
            )
        params[canonical] = _finite_number(raw_params[present[0]], f"{kind}.{canonical}")

    if kind == "space":
        if params["L"] < 0.0:
            raise OpticsError(
                "negative_spacing",
                f"{where}空气间隔 L={params['L']:g} 为负，"
                "负的间隔不能说成正向传播",
                index=index,
                L=params["L"],
            )
    elif kind == "lens":
        if params["f"] == 0.0:
            raise OpticsError(
                "zero_focal_length",
                f"{where}薄透镜焦距为零，焦距不得为零",
                index=index,
            )
    else:  # refract
        if params["R"] == 0.0:
            raise OpticsError(
                "zero_radius",
                f"{where}球面曲率半径 R 为零，平面折射请显式建模或使用非零 R",
                index=index,
            )
        if params["n1"] <= 0.0 or params["n2"] <= 0.0:
            raise OpticsError(
                "invalid_refractive_index",
                f"{where}折射率必须为正，收到 n1={params['n1']:g}, n2={params['n2']:g}",
                index=index,
                n1=params["n1"],
                n2=params["n2"],
            )

    return Element(kind=kind, params=params)


def normalize_elements(raw: Any) -> list[Element]:
    if not isinstance(raw, list) or not raw:
        raise OpticsError(
            "missing_field",
            "elements 必须是非空元件清单",
            field="elements",
            status_code=400,
        )
    return [normalize_element(item, i + 1) for i, item in enumerate(raw)]


def normalize_ray(raw: Any, *, required: bool = True) -> tuple[float, float] | None:
    if raw is None:
        if required:
            raise OpticsError("missing_field", "缺少 ray 对象", field="ray")
        return None
    ray = _require_object(raw, "ray")
    for name in ("y", "u"):
        if name not in ray:
            raise OpticsError("missing_field", f"ray 缺少 {name!r} 分量", field=name)
    return _finite_number(ray["y"], "ray.y"), _finite_number(ray["u"], "ray.u")


def normalize_object_distance(raw: Any) -> float | None:
    """物距：正数，或 null 表示物在无穷远。"""
    if raw is None:
        return None
    value = _finite_number(raw, "s")
    if value <= 0.0:
        raise OpticsError(
            "invalid_object_distance",
            f"物距 s 必须为正（物在第一面左侧），收到 {value:g}",
            s=value,
        )
    return value


def normalize_trace_body(body: Any) -> tuple[list[Element], tuple[float, float], float | None]:
    data = _require_object(body, "请求体")
    if "elements" not in data:
        raise OpticsError("missing_field", "缺少 elements 元件清单", field="elements")
    elements = normalize_elements(data["elements"])
    ray = normalize_ray(data.get("ray"))
    s = normalize_object_distance(data.get("s"))
    return elements, ray, s


def normalize_register_body(body: Any) -> tuple[str, list[Element]]:
    data = _require_object(body, "请求体")
    for name in ("name", "elements"):
        if name not in data:
            raise OpticsError("missing_field", f"缺少 {name!r} 字段", field=name)
    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise OpticsError(
            "invalid_name",
            "光路档名必须是非空字符串",
            field="name",
            status_code=400,
        )
    return name.strip(), normalize_elements(data["elements"])
