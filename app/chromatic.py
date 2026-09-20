"""跨波长色差归约：在逐波长追迹结果之上算轴向色差与倍率色差。

轴向色差：指定长短两条谱线的像距之差（短波像距 − 长波像距）。
倍率色差：这两条谱线的横向放大率之差（短波放大率 − 长波放大率）。
物在无穷远时，像距即后焦距，轴向色差就是后焦距在轴上拉开的距离。
"""

from __future__ import annotations

from typing import Sequence

from .errors import OpticsError


def _find_result(results: Sequence[dict[str, object]],
                 wavelength: float) -> dict[str, object]:
    for result in results:
        if result["wavelength"] == wavelength:
            return result
    available = [r["wavelength"] for r in results]
    raise OpticsError(
        "invalid_chromatic_basis",
        f"色差基准波长 {wavelength:g} nm 不在本次追迹的波长清单 {available} 里",
        wavelength=wavelength,
        wavelengths=available,
    )


def select_basis(wavelengths: Sequence[float],
                 basis: tuple[float, float] | None
                 ) -> tuple[float, float, str]:
    """确定色差基准谱线：请求指定优先，否则取波长清单的最短与最长。"""
    if basis is not None:
        return basis[0], basis[1], "request"
    return min(wavelengths), max(wavelengths), "default_min_max"


def chromatic_aberration(results: Sequence[dict[str, object]],
                         wavelengths: Sequence[float],
                         basis: tuple[float, float] | None,
                         object_distance: float | None) -> dict[str, object]:
    """在逐波长结果上归约跨波长色差量。"""
    if len(set(wavelengths)) < 2:
        return {
            "basis": None,
            "axial": None,
            "lateral": None,
            "note": "只给了一个波长，无跨波长色差可算",
        }

    short, long, source = select_basis(wavelengths, basis)
    short_result = _find_result(results, short)
    long_result = _find_result(results, long)
    short_imaging = short_result["imaging"]
    long_imaging = long_result["imaging"]
    assert isinstance(short_imaging, dict) and isinstance(long_imaging, dict)

    report: dict[str, object] = {
        "basis": {
            "short_wavelength": short,
            "long_wavelength": long,
            "source": source,
        },
    }

    # ---- 轴向色差：短波像距 − 长波像距 ----
    s_short = short_imaging["image_distance"]
    s_long = long_imaging["image_distance"]
    axial: dict[str, object] = {
        "image_distance_short": s_short,
        "image_distance_long": s_long,
        "definition": "轴向色差 = 短波像距 − 长波像距；物在无穷远时即后焦距之差",
    }
    if s_short is None or s_long is None:
        axial["difference"] = None
        axial["absolute"] = None
        axial["note"] = "至少一条谱线无有限像距（无焦或像在无穷远），轴向色差无定义"
    else:
        difference = float(s_short) - float(s_long)
        axial["difference"] = difference
        axial["absolute"] = abs(difference)
    report["axial"] = axial

    # ---- 倍率色差：短波放大率 − 长波放大率 ----
    m_short = short_imaging["magnification"]
    m_long = long_imaging["magnification"]
    lateral: dict[str, object] = {
        "magnification_short": m_short,
        "magnification_long": m_long,
        "definition": "倍率色差 = 短波放大率 − 长波放大率",
    }
    if object_distance is None:
        lateral["difference"] = None
        lateral["absolute"] = None
        lateral["note"] = "物在无穷远时放大率恒为 0，倍率色差不适用"
    elif m_short is None or m_long is None:
        lateral["difference"] = None
        lateral["absolute"] = None
        lateral["note"] = "至少一条谱线无有限放大率（无焦系统），倍率色差无定义"
    else:
        difference = float(m_short) - float(m_long)
        lateral["difference"] = difference
        lateral["absolute"] = abs(difference)
    report["lateral"] = lateral

    return report
