"""元件矩阵。

光线统一用列向量 ``(y, u)`` 表示：``y`` 为高度，``u`` 为近轴角。
全程只使用这套角度表示，不引入光学方向余弦。

矩阵以 (A, B, C, D) 四元组表示，作用于列向量::

    | y' |   | A B | | y |
    | u' | = | C D | | u |
"""

from __future__ import annotations

from typing import NamedTuple


class Element(NamedTuple):
    """登记、追迹时使用的规范化元件描述。

    ``params`` 的折射率分量可以是数值（恒定折射率），也可以是字符串
    （具名材料档）；拼矩阵前必须先把材料名按当前波长解析成数值。
    """

    kind: str          # "space" | "lens" | "refract"
    params: dict[str, float | str]

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


def refract_matrix(radius: float,
                   n1: float,
                   n2: float) -> tuple[float, float, float, float]:
    """球面折射面，光线由折射率 ``n1`` 的介质进入 ``n2``。

    高度在面上连续，近轴角按近轴折射定律折入新介质::

        n2 u2 = n1 u1 - (n2 - n1) y / R

    R 符号与光路前进方向一致：曲率中心在面的右侧（凸向入射一侧）取正。
    两侧折射率相等时 ``- (n2-n1)/R`` 项为 0，矩阵退化为单位阵。
    行列式 n1/n2，整段系统若回到同一介质，乘积行列式仍回到 1。
    """
    return (1.0, 0.0, (float(n1) - float(n2)) / (float(n2) * float(radius)),
            float(n1) / float(n2))


def element_matrix(element: Element) -> tuple[float, float, float, float]:
    if element.kind == "space":
        return space_matrix(element.params["L"])
    if element.kind == "lens":
        return lens_matrix(element.params["f"])
    if element.kind == "refract":
        n1, n2 = element.params["n1"], element.params["n2"]
        if isinstance(n1, str) or isinstance(n2, str):
            # 材料名必须先经 spectral.resolve_elements 按波长解析成数值
            raise ValueError("折射面仍挂着未解析的材料名，不能拼矩阵")
        return refract_matrix(element.params["R"], n1, n2)
    # 入参检查保证不会到达这里
    raise ValueError(f"未知元件种类: {element.kind}")  # pragma: no cover
