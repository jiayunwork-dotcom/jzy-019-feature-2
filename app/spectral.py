"""多波长追迹编排：按波长解析材料、逐波长独立连乘与像面求解。

每个波长各自解析出一份数值元件清单，各自走一遍单波长追迹，
波长之间不共享任何中间量——矩阵、光线、像面严格隔离。
"""

from __future__ import annotations

from typing import Callable, Sequence

from .elements import Element
from .errors import OpticsError
from .systems import (
    ROUNDTRIP_TOL,
    reverse_trace_residual,
    solve_imaging,
    trace,
)

#: 材料档查询入口：(材料名, 波长 nm) -> 折射率
IndexResolver = Callable[[str, float], float]


def resolve_elements(elements: Sequence[Element],
                     index_of: IndexResolver,
                     wavelength_nm: float) -> list[Element]:
    """把折射面上的材料名按当前波长现算成数值折射率。

    空气间隔与薄透镜不吃色散，原样保留；写死折射率的老式折射面也原样
    保留（等价于无色散材料）。未登记的材料档在这里、即追迹之前退回，
    并点明错在第几个元件。
    """
    resolved: list[Element] = []
    for i, element in enumerate(elements):
        if element.kind != "refract":
            resolved.append(element)
            continue
        params: dict[str, float | str] = dict(element.params)
        for key in ("n1", "n2"):
            value = params[key]
            if isinstance(value, str):
                try:
                    params[key] = index_of(value, wavelength_nm)
                except OpticsError as exc:
                    exc.extra.setdefault("index", i + 1)
                    exc.extra.setdefault("wavelength", wavelength_nm)
                    raise
        resolved.append(Element("refract", params))
    return resolved


def single_trace_payload(elements: Sequence[Element],
                         ray: tuple[float, float],
                         s: float | None,
                         *,
                         include_request: bool = True) -> dict[str, object]:
    """一条（已解析的）元件清单在单一波长下的完整追迹结果。"""
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
    if include_request:
        result["request"] = {
            "ray": {"y": ray[0], "u": ray[1]},
            "s": s,
            "s_meaning": "null 表示平行光入射（物在无穷远）",
        }
    result["system_abcd"] = {
        "A": system["A"], "B": system["B"], "C": system["C"], "D": system["D"],
    }
    return result


def trace_spectrum(elements: Sequence[Element],
                   ray: tuple[float, float],
                   s: float | None,
                   wavelengths: Sequence[float],
                   index_of: IndexResolver) -> list[dict[str, object]]:
    """对一组波长逐个独立追迹，返回按请求顺序排列的逐波长结果。

    每个波长的元件清单是独立解析出来的副本，系统矩阵、出射光线、
    像面各算各的，一个波长的中间量不会落到另一个波长头上。
    """
    results: list[dict[str, object]] = []
    for wavelength in wavelengths:
        resolved = resolve_elements(elements, index_of, wavelength)
        payload = single_trace_payload(resolved, ray, s, include_request=False)
        payload["wavelength"] = wavelength
        results.append(payload)
    return results
