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
        "n1": ("n1", "n_in", "nBefore", "material1", "material_in"),
        "n2": ("n2", "n_out", "nAfter", "material2", "material_out"),
    },
}

#: 色散模型名 -> 需要的系数
MODEL_COEFFICIENTS: dict[str, tuple[str, ...]] = {
    "constant": ("n",),
    "cauchy": ("A", "B"),
    "sellmeier": ("B", "C"),
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


def _medium_spec(value: Any, field: str, index: int) -> float | str:
    """折射面一侧的介质：正有限数（恒定折射率）或非空字符串（材料档名）。"""
    if isinstance(value, str):
        name = value.strip()
        if not name:
            raise OpticsError(
                "invalid_refractive_index",
                f"第 {index} 个元件的材料档名是空字符串",
                index=index,
                field=field,
            )
        return name
    return _positive_index(value, field, index)


def _positive_index(value: Any, field: str, index: int) -> float:
    number = _finite_number(value, field)
    if number <= 0.0:
        raise OpticsError(
            "invalid_refractive_index",
            f"第 {index} 个元件折射率必须为正，收到 {number:g}",
            index=index,
            value=number,
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

    params: dict[str, float | str] = {}
    for canonical, aliases in REQUIRED_PARAMS[kind].items():
        present = [name for name in aliases if name in raw_params]
        if not present:
            raise OpticsError(
                "missing_field",
                f"{where}（{kind}）缺少参数 {canonical!r}",
                index=index,
                field=canonical,
            )
        raw_value = raw_params[present[0]]
        if kind == "refract" and canonical in ("n1", "n2"):
            # 介质可以是写死的折射率，也可以是具名材料档
            params[canonical] = _medium_spec(raw_value, f"{kind}.{canonical}", index)
        else:
            params[canonical] = _finite_number(raw_value, f"{kind}.{canonical}")

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
        # n1/n2 已在 _medium_spec 里分别按数值或材料名检查过

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


def has_material_reference(elements: list[Element]) -> bool:
    """元件清单里是否有折射面点名了材料档（而非写死折射率）。"""
    return any(
        e.kind == "refract"
        and (isinstance(e.params["n1"], str) or isinstance(e.params["n2"], str))
        for e in elements
    )


def normalize_wavelength_value(raw: Any, field: str = "wavelength") -> float:
    """单个波长：正有限数，单位 nm。"""
    value = _finite_number(raw, field)
    if value <= 0.0:
        raise OpticsError(
            "invalid_wavelength",
            f"波长必须为正（单位 nm），收到 {value:g}",
            field=field,
            wavelength=value,
        )
    return value


def normalize_wavelengths(raw: Any) -> list[float] | None:
    """波长清单：缺省为 None（老式单波长请求），否则非空正数清单。"""
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise OpticsError(
            "invalid_wavelength",
            "wavelengths 必须是非空波长清单（单位 nm）",
            field="wavelengths",
        )
    return [
        normalize_wavelength_value(item, f"wavelengths[{i}]")
        for i, item in enumerate(raw)
    ]


def normalize_chromatic_basis(raw: Any) -> tuple[float, float] | None:
    """色差基准：{"short_wavelength": λ短, "long_wavelength": λ长}，缺省 None。"""
    if raw is None:
        return None
    data = _require_object(raw, "chromatic")
    for name in ("short_wavelength", "long_wavelength"):
        if name not in data:
            raise OpticsError(
                "missing_field",
                f"chromatic 缺少 {name!r} 字段",
                field=name,
            )
    short = normalize_wavelength_value(data["short_wavelength"], "short_wavelength")
    long = normalize_wavelength_value(data["long_wavelength"], "long_wavelength")
    if short >= long:
        raise OpticsError(
            "invalid_chromatic_basis",
            f"色差基准要求短波 < 长波，收到 short={short:g}, long={long:g}",
            short_wavelength=short,
            long_wavelength=long,
        )
    return short, long


def require_wavelengths_if_materials(elements: list[Element],
                                     wavelengths: list[float] | None) -> None:
    """折射面点名材料档时必须显式给出波长清单，不静默取默认谱线。"""
    if wavelengths is None and has_material_reference(elements):
        raise OpticsError(
            "missing_field",
            "折射面点名了材料档，必须给出 wavelengths 才能按波长现算折射率",
            field="wavelengths",
        )


def normalize_trace_body(
    body: Any,
) -> tuple[list[Element], tuple[float, float], float | None,
           list[float] | None, tuple[float, float] | None]:
    data = _require_object(body, "请求体")
    if "elements" not in data:
        raise OpticsError("missing_field", "缺少 elements 元件清单", field="elements")
    elements = normalize_elements(data["elements"])
    ray = normalize_ray(data.get("ray"))
    s = normalize_object_distance(data.get("s"))
    wavelengths = normalize_wavelengths(data.get("wavelengths"))
    basis = normalize_chromatic_basis(data.get("chromatic"))
    require_wavelengths_if_materials(elements, wavelengths)
    return elements, ray, s, wavelengths, basis


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


# ---------------------------------------------------------------------------
# 材料档登记体
# ---------------------------------------------------------------------------

def _normalize_samples(raw: Any) -> list[dict[str, float]]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise OpticsError(
            "invalid_material",
            "samples 至少需要 2 个采样点才能拟合色散曲线",
            field="samples",
        )
    points: list[dict[str, float]] = []
    seen: set[float] = set()
    for i, item in enumerate(raw):
        point = _require_object(item, f"采样点 samples[{i}]")
        for name in ("wavelength", "n"):
            if name not in point:
                raise OpticsError(
                    "missing_field",
                    f"采样点 samples[{i}] 缺少 {name!r} 字段",
                    field=name,
                )
        wavelength = normalize_wavelength_value(point["wavelength"],
                                                f"samples[{i}].wavelength")
        n = _positive_index(point["n"], f"samples[{i}].n", i + 1)
        if wavelength in seen:
            raise OpticsError(
                "invalid_material",
                f"采样点波长 {wavelength:g} nm 重复，同一波长只能给一个折射率",
                field="samples",
                wavelength=wavelength,
            )
        seen.add(wavelength)
        points.append({"wavelength": wavelength, "n": n})
    points.sort(key=lambda p: p["wavelength"])
    return points


def _normalize_coefficients(raw: Any) -> dict[str, object]:
    """解析系数描述：{"model": ..., "coefficients": {...}}。"""
    data = _require_object(raw, "coefficients")
    model = data.get("model")
    if not isinstance(model, str) or model not in MODEL_COEFFICIENTS:
        raise OpticsError(
            "invalid_material",
            f"coefficients.model 必须是 {sorted(MODEL_COEFFICIENTS)} 之一，"
            f"收到 {model!r}",
            field="model",
        )
    if model == "constant":
        if "n" not in data:
            raise OpticsError("missing_field", "constant 模型缺少系数 'n'", field="n")
        return {"model": model,
                "coefficients": {"n": _positive_index(data["n"], "constant.n", 1)}}
    if model == "cauchy":
        for name in ("A", "B"):
            if name not in data:
                raise OpticsError(
                    "missing_field", f"cauchy 模型缺少系数 {name!r}", field=name)
        return {
            "model": model,
            "coefficients": {
                "A": _finite_number(data["A"], "cauchy.A"),
                "B": _finite_number(data["B"], "cauchy.B"),
                "C": _finite_number(data.get("C", 0.0), "cauchy.C"),
            },
        }
    # sellmeier
    for name in ("B", "C"):
        if name not in data:
            raise OpticsError(
                "missing_field", f"sellmeier 模型缺少系数数组 {name!r}", field=name)
    b_raw, c_raw = data["B"], data["C"]
    if (not isinstance(b_raw, list) or not isinstance(c_raw, list)
            or not b_raw or len(b_raw) != len(c_raw) or len(b_raw) > 6):
        raise OpticsError(
            "invalid_material",
            "sellmeier 的 B、C 必须是等长数组，1 到 6 对系数",
            field="coefficients",
        )
    b = [_finite_number(v, f"sellmeier.B[{i}]") for i, v in enumerate(b_raw)]
    c = [_finite_number(v, f"sellmeier.C[{i}]") for i, v in enumerate(c_raw)]
    for i, ci in enumerate(c):
        if ci <= 0.0:
            raise OpticsError(
                "invalid_material",
                f"sellmeier.C[{i}]={ci:g} 必须为正（共振波长平方，单位 µm²）",
                field="C",
            )
    return {"model": model, "coefficients": {"B": b, "C": c}}


def normalize_material_body(body: Any) -> tuple[str, dict[str, object]]:
    """材料档登记体：具名 + 采样点描述或解析系数描述（二选一）。"""
    data = _require_object(body, "请求体")
    if "name" not in data:
        raise OpticsError("missing_field", "缺少 name 字段", field="name")
    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise OpticsError(
            "invalid_name",
            "材料档名必须是非空字符串",
            field="name",
            status_code=400,
        )
    has_samples = "samples" in data
    has_coefficients = "coefficients" in data
    if has_samples == has_coefficients:
        raise OpticsError(
            "invalid_material",
            "材料档必须且只能给一种描述：samples（采样点）或 coefficients（解析系数）",
            status_code=400,
        )
    if has_samples:
        description: dict[str, object] = {
            "description_type": "samples",
            "samples": _normalize_samples(data["samples"]),
        }
    else:
        description = {
            "description_type": "coefficients",
            **_normalize_coefficients(data["coefficients"]),
        }
    return name.strip(), description
