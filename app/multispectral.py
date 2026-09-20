"""多波长系统：每个波长独立解析材料、连乘矩阵、解共轭像面。

波长之间没有任何共享的中间量：每个波长用自己的
:class:`~app.materials.MaterialResolver` 把元件档解析成数值元件，
再交给 :mod:`app.systems` 里的单波长连乘/求解逻辑。
"""

from __future__ import annotations

from typing import Any, Sequence

from .elements import (
    Element,
    lensmaker_focal_length,
)
from .errors import OpticsError
from .materials import MaterialResolver
from .systems import (
    ROUNDTRIP_TOL,
    solve_imaging,
    reverse_trace_residual,
    trace,
)

#: 空气中薄透镜的围介质折射率（造镜者公式在空气中）
SURROUND_INDEX = 1.0


def resolve_element(element: Element,
                    index: int,
                    resolver: MaterialResolver) -> Element:
    """把一个元件档在当前波长下解析成只含数值参数的元件。

    未登记材料、非正折射率、零焦度都在拼矩阵之前在此退回，
    错误带元件序号与波长。
    """
    wavelength = resolver.wavelength
    params = dict(element.params)

    if element.kind == "space":
        return Element("space", {"L": float(params["L"])})

    if element.kind == "lens":
        if params.get("glass") is not None:
            glass = params["glass"]
            n = resolver.index_of(glass)
            try:
                focal_length = lensmaker_focal_length(
                    params.get("r1"), params.get("r2"), n
                )
            except ZeroDivisionError:
                raise OpticsError(
                    "zero_focal_length",
                    f"第 {index} 个元件（薄透镜/玻璃 {glass!r}）在波长 "
                    f"{wavelength:g} μm 处由造镜者公式算出的合成焦度为零，"
                    "焦距不存在",
                    index=index,
                    material=glass,
                    wavelength=wavelength,
                )
            resolved_params: dict[str, Any] = {
                "f": focal_length,
                "glass": glass,
                "glass_index": n,
                "r1": params.get("r1"),
                "r2": params.get("r2"),
            }
            return Element("lens", resolved_params)
        return Element("lens", {"f": float(params["f"])})

    if element.kind == "refract":
        resolved: dict[str, Any] = {"R": params.get("R")}
        for side in ("n1", "n2"):
            value = params[side]
            if isinstance(value, str):
                resolved[side] = resolver.index_of(value)
                resolved[f"{side}_material"] = value
            else:
                number = float(value)
                if number <= 0.0:  # 入参检查通常已拦截，守住最后一道
                    raise OpticsError(
                        "invalid_refractive_index",
                        f"第 {index} 个元件（折射面）{side}={number:g} 非正",
                        index=index,
                        side=side,
                    )
                resolved[side] = number
        return Element("refract", resolved)

    # 入参检查保证不会到达这里
    raise OpticsError(  # pragma: no cover
        "unknown_element",
        f"第 {index} 个元件种类 {element.kind!r} 未知",
        index=index,
    )


def resolved_indices(element: Element) -> dict[str, float]:
    """给出该（已解析）元件在当前波长用到的折射率，便于结果核对。"""
    if element.kind == "lens" and "glass_index" in element.params:
        return {
            "lens_index": element.params["glass_index"],
            "surround_index": SURROUND_INDEX,
        }
    if element.kind == "refract":
        return {"n1": element.params["n1"], "n2": element.params["n2"]}
    return {}


def trace_at_wavelength(elements: Sequence[Element],
                        ray: tuple[float, float],
                        object_distance: float | None,
                        resolver: MaterialResolver,
                        *,
                        check_roundtrip: bool = True) -> dict[str, Any]:
    """单波长追迹：解析 -> 连乘 -> 解像面 -> 往返校验。结果自包含。"""
    resolved = [
        resolve_element(element, i + 1, resolver)
        for i, element in enumerate(elements)
    ]

    result = trace(resolved, ray)
    imaging = solve_imaging(resolved, object_distance)
    residual, restored = reverse_trace_residual(resolved, ray)
    if check_roundtrip and residual > ROUNDTRIP_TOL:  # 数学上不应发生
        raise OpticsError(
            "roundtrip_error",
            f"波长 {resolver.wavelength:g} μm 正向后逆向未能还原入射光线，"
            f"残差 {residual:g} 超过容差 {ROUNDTRIP_TOL:g}",
            status_code=500,
            wavelength=resolver.wavelength,
            residual=residual,
        )

    # 在每段里补上该波长实际使用的折射率（折射面两侧 / 薄透镜玻璃）
    for segment, resolved_element in zip(result["segments"], resolved):
        segment["indices"] = resolved_indices(resolved_element)

    result["wavelength"] = resolver.wavelength
    result["wavelength_unit"] = "um"
    result["imaging"] = {
        **imaging,
        "object_distance_convention": "物在第一面左侧，s 取正；null 表示物在无穷远",
    }
    result["roundtrip"] = {
        "restored_ray": {"y": restored[0], "u": restored[1]},
        "residual": residual,
        "tolerance": ROUNDTRIP_TOL,
    }
    system = result["system_matrix"]
    assert isinstance(system, dict)
    result["system_abcd"] = {
        "A": system["A"], "B": system["B"], "C": system["C"], "D": system["D"],
    }
    return result


def trace_multiwavelength(elements: Sequence[Element],
                          ray: tuple[float, float],
                          object_distance: float | None,
                          wavelengths: Sequence[float],
                          materials: dict[str, Any],
                          *,
                          check_roundtrip: bool = True) -> list[dict[str, Any]]:
    """对一组波长各自独立追迹。

    ``materials`` 是只读的名称 -> :class:`~app.materials.Material` 快照；
    每个波长新建解析器，矩阵、光线、缓存互不共享。
    """
    per_wavelength: list[dict[str, Any]] = []
    for wavelength in wavelengths:
        resolver = MaterialResolver(materials, wavelength)
        per_wavelength.append(
            trace_at_wavelength(
                elements, ray, object_distance, resolver,
                check_roundtrip=check_roundtrip,
            )
        )
    return per_wavelength
