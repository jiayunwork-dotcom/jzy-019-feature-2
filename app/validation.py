"""入参检查：把原始 JSON 体转成规范化对象，不合规即抛出带类型的错误。

折射面的 n1/n2 既可以是正数（旧写法，等价于全波段恒定的无色散材料），
也可以是具名材料档名；薄透镜既可以直接给焦距 f，也可以给
(r1, r2, glass)，由色散模型按波长现算折射率再走造镜者公式。
"""

from __future__ import annotations

import math
from typing import Any, Iterable

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

_SPACE_KEYS = ("L", "length", "d", "distance")
_LENS_F_KEYS = ("f", "focal_length", "focal", "focalLength")
_R1_KEYS = ("r1", "R1", "radius1", "front_radius")
_R2_KEYS = ("r2", "R2", "radius2", "back_radius")
_GLASS_KEYS = ("glass", "material", "glass_name", "glassName")
_RADIUS_KEYS = ("R", "radius", "radiusOfCurvature")
_N1_KEYS = ("n1", "n_in", "nBefore")
_N2_KEYS = ("n2", "n_out", "nAfter")
_N1_MATERIAL_KEYS = ("n1_material", "n1_glass", "n1Material")
_N2_MATERIAL_KEYS = ("n2_material", "n2_glass", "n2Material")


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


def _take_any(raw_params: dict[str, Any], keys: tuple[str, ...]) -> Any:
    present = [key for key in keys if key in raw_params]
    return raw_params[present[0]] if present else None


def _normalize_radius(raw: Any, field: str) -> float | None:
    """曲率半径：非零有限数，或 null 表示平面。"""
    if raw is None:
        return None
    radius = _finite_number(raw, "refract.R")
    if radius == 0.0:
        raise OpticsError(
            "zero_radius",
            "参数 'refract.R' 曲率半径为零，平面折射请显式用 null 建模或使用非零 R",
            field="refract.R",
        )
    return radius


def _medium_spec(raw_params: dict[str, Any],
                 index: int,
                 side: str,
                 number_keys: tuple[str, ...],
                 material_keys: tuple[str, ...],
                 known_materials: Iterable[str] | None) -> float | str:
    """折射面一侧的介质：正折射率（旧写法）或具名材料档名。"""
    raw_number = _take_any(raw_params, number_keys)
    raw_material = _take_any(raw_params, material_keys)
    where = f"第 {index} 个元件"

    if raw_material is not None:
        if raw_number is not None:
            raise OpticsError(
                "invalid_element",
                f"{where}折射面 {side} 侧同时给了折射率与材料档，二者只能取其一",
                index=index,
                side=side,
            )
        if not isinstance(raw_material, str) or not raw_material.strip():
            raise OpticsError(
                "invalid_material_reference",
                f"{where}折射面 {side} 侧材料档名必须是非空字符串，收到 {raw_material!r}",
                index=index,
                side=side,
            )
        name = raw_material.strip()
        if known_materials is not None and name not in known_materials:
            raise OpticsError(
                "unknown_material",
                f"{where}折射面 {side} 侧引用的材料档 {name!r} 未登记",
                index=index,
                material=name,
                side=side,
                status_code=404,
            )
        return name

    if raw_number is None:
        raise OpticsError(
            "missing_field",
            f"{where}折射面 {side} 侧缺少折射率（{number_keys[0]}）或材料档（{material_keys[0]}）",
            index=index,
            field=number_keys[0],
        )
    number = _finite_number(raw_number, f"refract.{side}")
    if number <= 0.0:
        raise OpticsError(
            "invalid_refractive_index",
            f"{where}折射面 {side} 侧折射率必须为正，收到 {number:g}",
            index=index,
            side=side,
            value=number,
        )
    return number


def normalize_element(raw: Any,
                      index: int,
                      *,
                      known_materials: Iterable[str] | None = None) -> Element:
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

    if kind == "space":
        raw_l = _take_any(raw_params, _SPACE_KEYS)
        if raw_l is None:
            raise OpticsError(
                "missing_field",
                f"{where}（space）缺少参数 'L'",
                index=index,
                field="L",
            )
        length = _finite_number(raw_l, "space.L")
        if length < 0.0:
            raise OpticsError(
                "negative_spacing",
                f"{where}空气间隔 L={length:g} 为负，"
                "负的间隔不能说成正向传播",
                index=index,
                L=length,
            )
        return Element(kind="space", params={"L": length})

    if kind == "lens":
        raw_f = _take_any(raw_params, _LENS_F_KEYS)
        raw_glass = _take_any(raw_params, _GLASS_KEYS)
        if raw_f is not None and raw_glass is not None:
            raise OpticsError(
                "invalid_element",
                f"{where}薄透镜不能同时给焦距 f 与玻璃材料：直接给 f 表示"
                "无色散薄透镜；给 r1/r2/glass 才随波长色散",
                index=index,
            )
        if raw_glass is not None:
            if not isinstance(raw_glass, str) or not raw_glass.strip():
                raise OpticsError(
                    "invalid_material_reference",
                    f"{where}薄透镜材料档名必须是非空字符串，收到 {raw_glass!r}",
                    index=index,
                )
            glass = raw_glass.strip()
            if known_materials is not None and glass not in known_materials:
                raise OpticsError(
                    "unknown_material",
                    f"{where}薄透镜引用的材料档 {glass!r} 未登记",
                    index=index,
                    material=glass,
                    status_code=404,
                )
            r1 = _normalize_radius(_take_any(raw_params, _R1_KEYS), "lens.r1")
            r2 = _normalize_radius(_take_any(raw_params, _R2_KEYS), "lens.r2")
            if r1 is None and r2 is None:
                raise OpticsError(
                    "zero_focal_length",
                    f"{where}挂玻璃 {glass!r} 的薄透镜两面均为平面，"
                    "造镜者公式给出零焦度（平板玻璃不聚焦）",
                    index=index,
                    material=glass,
                )
            return Element(kind="lens", params={"r1": r1, "r2": r2, "glass": glass})

        if raw_f is None:
            raise OpticsError(
                "missing_field",
                f"{where}（lens）缺少焦距 f，或缺少 r1/r2/glass 色散薄透镜描述",
                index=index,
                field="f",
            )
        focal_length = _finite_number(raw_f, "lens.f")
        if focal_length == 0.0:
            raise OpticsError(
                "zero_focal_length",
                f"{where}薄透镜焦距为零，焦距不得为零",
                index=index,
            )
        return Element(kind="lens", params={"f": focal_length})

    # refract
    # refract：R 可显式给 null 表示平面；完全缺字段才算缺参
    if not any(key in raw_params for key in _RADIUS_KEYS):
        raise OpticsError(
            "missing_field",
            f"{where}（refract）缺少曲率半径 R（平面请给 null）",
            index=index,
            field="R",
        )
    radius = _normalize_radius(_take_any(raw_params, _RADIUS_KEYS), "refract.R")
    n1 = _medium_spec(raw_params, index, "n1(入射)", _N1_KEYS,
                      _N1_MATERIAL_KEYS, known_materials)
    n2 = _medium_spec(raw_params, index, "n2(出射)", _N2_KEYS,
                      _N2_MATERIAL_KEYS, known_materials)
    params: dict[str, Any] = {"n1": n1, "n2": n2}
    if radius is not None:
        params["R"] = radius
    return Element(kind="refract", params=params)


def normalize_elements(raw: Any,
                       *,
                       known_materials: Iterable[str] | None = None) -> list[Element]:
    if not isinstance(raw, list) or not raw:
        raise OpticsError(
            "missing_field",
            "elements 必须是非空元件清单",
            field="elements",
            status_code=400,
        )
    return [
        normalize_element(item, i + 1, known_materials=known_materials)
        for i, item in enumerate(raw)
    ]


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


def normalize_wavelengths(raw: Any) -> list[float] | None:
    """波长组（μm）：非空、逐项正且有限、去重后升序。

    返回 ``None`` 表示调用方走旧的单波长（不带波长）路径。
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise OpticsError(
            "invalid_wavelengths",
            "wavelengths 必须是非空数组（波长以 μm 计）",
            status_code=400,
        )
    wavelengths: list[float] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise OpticsError(
                "invalid_wavelengths",
                f"wavelengths[{i}] 必须是数值，收到 {item!r}",
                index=i,
            )
        value = float(item)
        if not math.isfinite(value) or value <= 0.0:
            raise OpticsError(
                "invalid_wavelengths",
                f"wavelengths[{i}]={value:g} 必须是正的有限值（μm）",
                index=i,
            )
        wavelengths.append(value)
    unique = sorted(set(wavelengths))
    if len(unique) != len(wavelengths):
        raise OpticsError(
            "invalid_wavelengths",
            f"wavelengths 不得重复，收到 {wavelengths}",
        )
    return unique


def normalize_reference_pair(raw: Any,
                             wavelengths: list[float]) -> tuple[float, float]:
    """色差基准的长、短两条谱线；缺省取波长组的两端。"""
    if raw is None:
        return wavelengths[0], wavelengths[-1]
    pair = _require_object(raw, "reference_wavelengths")
    if "short" not in pair or "long" not in pair:
        raise OpticsError(
            "invalid_reference_wavelength",
            "reference_wavelengths 需要同时给出 short 与 long（μm）",
        )
    short = _finite_number(pair["short"], "reference.short")
    long_ = _finite_number(pair["long"], "reference.long")
    if short <= 0.0 or long_ <= 0.0:
        raise OpticsError(
            "invalid_reference_wavelength",
            f"基准波长必须为正，收到 short={short:g}, long={long_:g}",
        )
    if short >= long_:
        raise OpticsError(
            "invalid_reference_wavelength",
            f"要求 short < long，收到 short={short:g}, long={long_:g}",
            short=short,
            long=long_,
        )
    allowed = set(wavelengths)
    for label, value in (("short", short), ("long", long_)):
        if not any(abs(value - w) <= 1e-12 for w in allowed):
            raise OpticsError(
                "invalid_reference_wavelength",
                f"基准 {label} 波长 {value:g} μm 不在本次追迹波长清单 {wavelengths} 内",
                label=label,
                wavelength=value,
            )
    return short, long_


def normalize_trace_body(
    body: Any,
    *,
    known_materials: Iterable[str] | None = None,
) -> tuple[list[Element], tuple[float, float], float | None,
          list[float] | None, tuple[float, float] | None]:
    """一次性追迹请求体 -> (元件, 光线, 物距, 波长组, 色差基准)。"""
    data = _require_object(body, "请求体")
    if "elements" not in data:
        raise OpticsError("missing_field", "缺少 elements 元件清单", field="elements")
    elements = normalize_elements(data["elements"], known_materials=known_materials)
    ray = normalize_ray(data.get("ray"))
    s = normalize_object_distance(data.get("s"))
    wavelengths = normalize_wavelengths(data.get("wavelengths"))
    pair = None
    if wavelengths is not None:
        pair = normalize_reference_pair(data.get("reference_wavelengths"), wavelengths)
    elif data.get("reference_wavelengths") is not None:
        raise OpticsError(
            "invalid_reference_wavelength",
            "给出 reference_wavelengths 时必须同时给出 wavelengths 数组",
        )
    return elements, ray, s, wavelengths, pair


def normalize_register_body(
    body: Any,
    *,
    known_materials: Iterable[str] | None = None,
) -> tuple[str, list[Element]]:
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
    return name.strip(), normalize_elements(
        data["elements"], known_materials=known_materials
    )
