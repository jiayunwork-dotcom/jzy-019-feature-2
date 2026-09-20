"""系统连乘、光线追迹与物像求解。"""

from __future__ import annotations

import math
from typing import Sequence

from .elements import (
    Element,
    IDENTITY,
    det,
    element_matrix,
    inverse,
    mat_apply,
    matmul,
)
from .errors import OpticsError

Matrix4 = tuple[float, float, float, float]

#: 空气系统行列式容差（钉死）
DET_TOL = 1e-12
#: 往返追迹还原容差
ROUNDTRIP_TOL = 1e-10
#: |C| 不超过该值即判定接近无焦
AFOCAL_C_TOL = 1e-12

AIR_KINDS = {"space", "lens"}


def _as_abcd(m: Matrix4) -> dict[str, float]:
    a, b, c, d = m
    return {"A": a, "B": b, "C": c, "D": d}


def segment_matrices(elements: Sequence[Element]) -> list[Matrix4]:
    return [element_matrix(e) for e in elements]


def system_matrix(elements: Sequence[Element]) -> Matrix4:
    """按光线前进方向依次相乘：M = M_n ... M_2 M_1。"""
    result: Matrix4 = IDENTITY
    for m in segment_matrices(elements):
        result = matmul(m, result)
    return result


def is_air_system(elements: Sequence[Element]) -> bool:
    """只含空气间隔与薄透镜（不含球面折射面）即视为空气系统。"""
    return all(e.kind in AIR_KINDS for e in elements)


def check_air_determinants(elements: Sequence[Element],
                           matrices: Sequence[Matrix4]) -> None:
    """空气系统：每一段及整段乘积行列式都必须在容差内等于 1。

    含球面折射面的系统不做此检查（单块折射面行列式为 n1/n2，
    回到同一介质时整段乘积行列式才回到 1，由往返测试兜底）。
    """
    if not is_air_system(elements):
        return
    cumulative: Matrix4 = IDENTITY
    for index, m in enumerate(matrices):
        segment_det = det(m)
        if not math.isfinite(segment_det) or abs(segment_det - 1.0) > DET_TOL:
            raise OpticsError(
                "determinant_error",
                f"第 {index + 1} 个元件矩阵行列式 {segment_det!r} 偏离 1 "
                f"超过容差 {DET_TOL:g}",
                status_code=500,
            )
        cumulative = matmul(m, cumulative)
        product_det = det(cumulative)
        if not math.isfinite(product_det) or abs(product_det - 1.0) > DET_TOL:
            raise OpticsError(
                "determinant_error",
                f"前 {index + 1} 段连乘行列式 {product_det!r} 偏离 1 "
                f"超过容差 {DET_TOL:g}",
                status_code=500,
            )


def trace(elements: Sequence[Element],
          ray: tuple[float, float]) -> dict[str, object]:
    """正向追迹，返回每段矩阵、系统矩阵、出入射光线。"""
    matrices = segment_matrices(elements)
    check_air_determinants(elements, matrices)

    system = system_matrix(elements)
    outgoing = mat_apply(system, ray)

    segments: list[dict[str, object]] = []
    for index, (element, m) in enumerate(zip(elements, matrices)):
        segments.append({
            "index": index + 1,
            "type": element.kind,
            "params": dict(element.params),
            "matrix": _as_abcd(m),
            "determinant": det(m),
        })

    return {
        "elements": [e.describe() for e in elements],
        "segments": segments,
        "system_matrix": _as_abcd(system),
        "determinant": det(system),
        "air_system": is_air_system(elements),
        "incoming_ray": {"y": ray[0], "u": ray[1]},
        "outgoing_ray": {"y": outgoing[0], "u": outgoing[1]},
    }


def reverse_trace_residual(elements: Sequence[Element],
                           ray: tuple[float, float]) -> tuple[float, tuple[float, float]]:
    """正向追迹后，按相反次序用各段矩阵的逆作用回去，返还原残差与还原光线。"""
    matrices = segment_matrices(elements)
    system = system_matrix(elements)
    outgoing = mat_apply(system, ray)

    back = outgoing
    for m in reversed(matrices):
        back = mat_apply(inverse(m), back)

    residual = max(abs(back[0] - ray[0]), abs(back[1] - ray[1]))
    return residual, back


def solve_imaging(elements: Sequence[Element],
                  object_distance: float | None) -> dict[str, object]:
    """由整段矩阵的共轭条件求像距与横向放大率。

    物在第一面左侧，物距 ``s`` 取正；``None`` 表示物在无穷远（平行光入射）。
    像距从最后一面（出射参考面）向右量起，虚像为负。

    轴外物点高 h 发出的光线在第一面处 y1 = h + s u0，经过
    系统矩阵 M=(A,B;C,D) 后再传播 s' 到达像面，像高与 u0 无关
    （系数必须为零）::

        A s + B + s'(C s + D) = 0
        s' = -(A s + B) / (C s + D)

    横向放大率（det M = 1）::

        m = A + s' C = 1 / (C s + D)

    单透镜代回即还原高斯公式 ``1/s + 1/s' = 1/f``。
    """
    a, b, c, d = system_matrix(elements)
    afocal = abs(c) <= AFOCAL_C_TOL
    effective_focal_length = None if afocal else -1.0 / c

    result: dict[str, object] = {
        "afocal": afocal,
        "effective_focal_length": effective_focal_length,
        "object_distance": object_distance,
    }

    if object_distance is None:
        if afocal:
            # 无焦系统把平行光仍映射为平行光：像在无穷远，无有限像距；
            # 角放大率 = D（共焦望远镜为 -f1/f2）。
            result.update({
                "image_distance": None,
                "magnification": None,
                "note": "无焦系统，平行光出射仍平行，像在无穷远",
            })
        else:
            # 平行光入射，像落在后焦面：后焦距 BFL = -A/C，空气中 EFL = -1/C
            result.update({
                "image_distance": -a / c,
                "magnification": 0.0,
            })
        return result

    s = float(object_distance)
    denominator = c * s + d

    if afocal:
        # C=0：无有限焦点，但有限物距仍共轭到有限（虚/实）像面
        result.update({
            "image_distance": -(a * s + b) / d,
            "magnification": 1.0 / d,
            "note": "系统接近无焦（|C| <= {:g}），无有限焦距".format(AFOCAL_C_TOL),
        })
        return result

    if abs(denominator) <= AFOCAL_C_TOL:
        # 物在前焦面附近，出射光平行，像在无穷远
        result.update({
            "image_distance": None,
            "magnification": 0.0,
            "note": "出射光平行，像在无穷远",
        })
        return result

    image_distance = -(a * s + b) / denominator
    magnification = 1.0 / denominator
    result.update({
        "image_distance": image_distance,
        "magnification": magnification,
    })
    return result
