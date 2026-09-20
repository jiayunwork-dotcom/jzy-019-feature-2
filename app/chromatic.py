"""跨波长色差归约：在已按波长隔离的追迹结果上做差分统计。

本模块不做任何追迹、不解析材料，只消费每波长的自包含结果。

- 轴向（纵向）色差：长短两条基准谱线像距之差
  ``LCA = s'(λ_long) - s'(λ_short)``。
  正常色散（n 蓝大、f 蓝短）下有限物距像距蓝短红长，LCA > 0；
  物在无穷远时对后焦距 ``BFL = -A/C`` 做同样的差分。
- 倍率色差：两谱线横向放大率之差 ``TCA = m(λ_long) - m(λ_short)``。
"""

from __future__ import annotations

from typing import Any, Sequence

from .errors import OpticsError


def _result_for(wavelength_results: Sequence[dict[str, Any]],
                wavelength: float) -> dict[str, Any]:
    for result in wavelength_results:
        if result["wavelength"] == wavelength:
            return result
    raise OpticsError(  # pragma: no cover（入参检查已保证基准波长在清单内）
        "invalid_reference_wavelength",
        f"基准波长 {wavelength:g} μm 不在本次追迹波长清单内",
        wavelength=wavelength,
    )


def _image_distance(result: dict[str, Any]) -> float | None:
    imaging = result["imaging"]
    distance = imaging["image_distance"]
    if distance is None:
        return None
    return float(distance)


def chromatic_aberration(wavelength_results: Sequence[dict[str, Any]],
                         short_wavelength: float,
                         long_wavelength: float
                         ) -> dict[str, Any]:
    """计算两基准谱线之间的轴向色差与倍率色差。

    返回字段全部点名基准波长，方向定义写在响应里，杜绝符号歧义。
    """
    short_result = _result_for(wavelength_results, short_wavelength)
    long_result = _result_for(wavelength_results, long_wavelength)

    s_short = _image_distance(short_result)
    s_long = _image_distance(long_result)

    if s_short is None or s_long is None:
        axial: float | None = None
        axial_kind = "unavailable"
    else:
        axial = s_long - s_short
        # 物在无穷远时每波长的 image_distance 即后焦距
        infinity_mode = all(
            result["imaging"]["object_distance"] is None
            for result in wavelength_results
        )
        axial_kind = "back_focal_length_difference" if infinity_mode else "image_distance_difference"

    m_short = short_result["imaging"]["magnification"]
    m_long = long_result["imaging"]["magnification"]
    if m_short is None or m_long is None:
        transverse: float | None = None
    else:
        transverse = float(m_long) - float(m_short)

    return {
        "reference_wavelengths": {
            "short": short_wavelength,
            "long": long_wavelength,
            "unit": "um",
        },
        "axial_chromatic_aberration": axial,
        "axial_kind": axial_kind,
        "axial_sign_convention": (
            "s'(long) - s'(short)；正常色散有限物距像距蓝短红长时为正；"
            "无穷远物时为后焦距之差 BFL(long) - BFL(short)"
        ),
        "transverse_chromatic_aberration": transverse,
        "transverse_sign_convention": "m(long) - m(short)",
        "short_image_distance": s_short,
        "long_image_distance": s_long,
        "short_magnification": m_short,
        "long_magnification": m_long,
    }
