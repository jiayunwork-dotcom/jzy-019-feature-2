"""元件矩阵。

光线统一用列向量 ``(y, u)`` 表示：``y`` 为高度，``u`` 为近轴角。
全程只使用这套角度表示，不引入光学方向余弦。

矩阵以 (A, B, C, D) 四元组表示，作用于列向量::

    | y' |   | A B | | y |
    | u' | = | C D | | u | |

元件分两类形态：

- 登记/请求里的元件是 *元件档*（:class:`Element`）：折射面的 n1/n2
  可以是数值（无色散），也可以是具名材料档名；薄透镜可以直接给焦距 f，
  也可以给 (r1, r2, glass) 由造镜者公式随波长定焦距。
- 追迹每个波长前先把元件档解析成只含数值参数的 *解析元件*，再拼矩阵。
  ``element_matrix`` 只接受解析元件（数值），波长之间彼此隔离。
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple


class Element(NamedTuple):
    """登记、追迹时使用的元件描述（可能含具名材料引用，待按波长解析）。"""

    kind: str  # "space" | "lens" | "refract"
    params: dict[str, Any]

    def describe(self) -> dict[str, object]:
        return {"type": self.kind, "params": dict(self.params)}


def matmul(m1: tuple[float, float, float, float],
           m2: tuple[float, float, float, float]
           ) -> tuple[float, float, float, float]:
    """按光线前进方向连乘：结果作用于光线时先 ``m2`` 后 ``m1``。"""
    a1, b1, c1, d1 = m1
    a2, b2, c2, d2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
    )


def mat_apply(m: tuple[float, float, float, float],
              ray: tuple[float, float]) -> tuple[float, float]:
    a, b, c, d = m
    y, u = ray
    return a * y + b * u, c * y + d * u


def det(m: tuple[float, float, float, float]) -> float:
    a, b, c, d = m
    return a * d - b * c


def inverse(m: tuple[float, float, float, float]
            ) -> tuple[float, float, float, float]:
    """2x2 矩阵的逆。近轴元件矩阵行列式非零。"""
    a, b, c, d = m
    delta = a * d - b * c
    return d / delta, -b / delta, -c / delta, a / delta


IDENTITY = (1.0, 0.0, 0.0, 1.0)


def space_matrix(length: float) -> tuple[float, float, float, float]:
    """自由空间（空气间隔）传播 L。

    第一行 (1, L)，第二行 (0, 1)，行列式恒为 1。
    """
    return (1.0, float(length), 0.0, 1.0)


def lens_matrix(focal_length: float) -> tuple[float, float, float, float]:
    """焦距为 f 的薄透镜。

    第一行 (1, 0)，第二行 (-1/f, 1)，行列式恒为 1。
    ``f`` 不得为零（零焦距在入参检查阶段即退回）。
    """
    f = float(focal_length)
    return (1.0, 0.0, -1.0 / f, 1.0)


def refract_matrix(radius: float | None,
                   n1: float,
                   n2: float) -> tuple[float, float, float, float]:
    """球面折射面，光线由折射率 ``n1`` 的介质进入 ``n2``。

    高度在面上连续，近轴角按近轴折射定律折入新介质::

        n2 u2 = n1 u1 - (n2 - n1) y / R

    R 符号与光路前进方向一致：曲率中心在面的右侧（凸向入射一侧）取正；
    ``R=None`` 表示平面（曲率为 0），折光项消失。
    两侧折射率相等时 ``- (n2-n1)/R`` 项为 0，矩阵退化为单位阵。
    行列式 n1/n2，整段系统若回到同一介质，乘积行列式仍回到 1。
    """
    power = 0.0 if radius is None else (float(n1) - float(n2)) / (float(n2) * float(radius))
    return (1.0, 0.0, power, float(n1) / float(n2))


def lensmaker_focal_length(r1: float | None,
                           r2: float | None,
                           n: float) -> float:
    """空气中薄透镜造镜者公式：``1/f = (n-1)(1/r1 - 1/r2)``。

    r 为 ``None`` 表示该面为平面（曲率为 0）。合成焦度为零时焦距不存在，
    由调用方在入参检查阶段退回。
    """
    curvature1 = 0.0 if r1 is None else 1.0 / float(r1)
    curvature2 = 0.0 if r2 is None else 1.0 / float(r2)
    power = (float(n) - 1.0) * (curvature1 - curvature2)
    if not math.isfinite(power) or power == 0.0:
        raise ZeroDivisionError("薄透镜造镜者公式给出的合成焦度为零")
    return 1.0 / power


def _numeric(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # 具名材料引用必须先经 MaterialResolver 解析后才能拼矩阵
        raise TypeError(f"{field} 尚未解析为数值折射率：{value!r}")
    return float(value)


def element_matrix(element: Element) -> tuple[float, float, float, float]:
    """由 *已解析*（参数全部数值化）元件拼矩阵。"""
    if element.kind == "space":
        return space_matrix(element.params["L"])
    if element.kind == "lens":
        if "f" not in element.params:
            # 挂玻璃的薄透镜必须先按当前波长解析出 f
            raise TypeError("挂玻璃的薄透镜必须先按波长解析为数值焦距")
        return lens_matrix(element.params["f"])
    if element.kind == "refract":
        return refract_matrix(
            element.params["R"],
            _numeric(element.params["n1"], "n1"),
            _numeric(element.params["n2"], "n2"),
        )
    # 入参检查保证不会到达这里
    raise ValueError(f"未知元件种类: {element.kind}")  # pragma: no cover
