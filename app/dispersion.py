"""波长 → 折射率的色散模型。

本模块只管"给波长、算折射率"这一件事，不碰追迹主流程。

约定：
- 对外波长一律用纳米（nm）。
- Cauchy 模型 ``n(λ) = A + B/λ² + C/λ⁴`` 直接以 nm 计（B 单位 nm²，C 单位 nm⁴）。
- Sellmeier 模型沿用文献惯例，λ 以微米（µm）计、C 系数单位 µm²，
  模型内部把 nm 换算成 µm 再求值。
- 采样点描述先按 Cauchy 形式做最小二乘拟合（2 点用两项，≥3 点用三项），
  内插与外推都沿拟合出的色散曲线走，不做线性硬连。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .errors import OpticsError

#: 折射率合理性容差下限（折射率必须为正且有限）
_MIN_INDEX = 1e-6


class DispersionModel(Protocol):
    """色散模型协议：给定波长（nm）返回折射率。"""

    def index(self, wavelength_nm: float) -> float:
        ...

    def describe(self) -> dict[str, object]:
        """列出该模型的描述方式与全部系数。"""
        ...


def _check_index(n: float, wavelength_nm: float, what: str) -> float:
    if not math.isfinite(n) or n < _MIN_INDEX:
        raise OpticsError(
            "dispersion_evaluation_failed",
            f"{what}在波长 {wavelength_nm:g} nm 处算出的折射率 {n!r} 非正或非有限",
            wavelength=wavelength_nm,
        )
    return n


@dataclass(frozen=True)
class ConstantIndex:
    """无色散材料：任意波长返回同一折射率（老式写死 n1/n2 的等价物）。"""

    n: float

    def index(self, wavelength_nm: float) -> float:
        return _check_index(self.n, wavelength_nm, "恒定折射率材料")

    def describe(self) -> dict[str, object]:
        return {"model": "constant", "coefficients": {"n": self.n}}


@dataclass(frozen=True)
class CauchyModel:
    """Cauchy 色散：``n(λ) = A + B/λ² + C/λ⁴``，λ 以 nm 计。"""

    A: float
    B: float
    C: float = 0.0

    def index(self, wavelength_nm: float) -> float:
        lam2 = wavelength_nm * wavelength_nm
        n = self.A + self.B / lam2 + self.C / (lam2 * lam2)
        return _check_index(n, wavelength_nm, "Cauchy 材料")

    def describe(self) -> dict[str, object]:
        return {
            "model": "cauchy",
            "coefficients": {"A": self.A, "B": self.B, "C": self.C},
            "wavelength_unit": "nm",
        }


@dataclass(frozen=True)
class SellmeierModel:
    """Sellmeier 色散：``n² = 1 + Σ Bᵢ λ² / (λ² − Cᵢ)``，λ 以 µm 计。"""

    B: tuple[float, ...]
    C: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.B) != len(self.C) or not self.B:
            raise OpticsError(
                "invalid_material",
                "Sellmeier 模型的 B、C 系数个数必须相同且至少一对",
            )

    def index(self, wavelength_nm: float) -> float:
        lam_um = wavelength_nm / 1000.0
        l2 = lam_um * lam_um
        n2 = 1.0
        for b, c in zip(self.B, self.C):
            denominator = l2 - c
            if denominator == 0.0:
                raise OpticsError(
                    "dispersion_evaluation_failed",
                    f"Sellmeier 材料在波长 {wavelength_nm:g} nm 处撞上共振极点，无法求值",
                    wavelength=wavelength_nm,
                )
            n2 += b * l2 / denominator
        if not math.isfinite(n2) or n2 <= 0.0:
            raise OpticsError(
                "dispersion_evaluation_failed",
                f"Sellmeier 材料在波长 {wavelength_nm:g} nm 处 n²={n2!r} 非正或非有限",
                wavelength=wavelength_nm,
            )
        return _check_index(math.sqrt(n2), wavelength_nm, "Sellmeier 材料")

    def describe(self) -> dict[str, object]:
        return {
            "model": "sellmeier",
            "coefficients": {"B": list(self.B), "C": list(self.C)},
            "wavelength_unit": "um",
        }


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """部分主元高斯消元解小规模稠密线性方程组（拟合最多 3 阶）。"""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-300:
            raise OpticsError(
                "invalid_material",
                "采样点拟合色散曲线时法方程奇异：采样波长可能过于接近或重复",
            )
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        residual = sum(aug[row][k] * solution[k] for k in range(row + 1, n))
        solution[row] = (aug[row][n] - residual) / aug[row][row]
    return solution


def fit_cauchy(samples: Sequence[tuple[float, float]]) -> CauchyModel:
    """把 (波长 nm, 折射率) 采样点最小二乘拟合成 Cauchy 模型。

    2 个点拟合 ``A + B/λ²`` 两项，3 个及以上拟合 ``A + B/λ² + C/λ⁴`` 三项；
    点数恰好等于项数时为精确内插，更多点时按最小二乘。外推同样沿该
    有理曲线走，不在相邻采样点之间线性硬连。
    """
    if len(samples) < 2:
        raise OpticsError(
            "invalid_material",
            f"采样点描述至少需要 2 个 (波长, 折射率) 点，收到 {len(samples)} 个",
        )
    terms = 2 if len(samples) == 2 else 3

    def basis(wavelength_nm: float) -> list[float]:
        lam2 = wavelength_nm * wavelength_nm
        row = [1.0, 1.0 / lam2]
        if terms == 3:
            row.append(1.0 / (lam2 * lam2))
        return row

    normal = [[0.0] * terms for _ in range(terms)]
    rhs = [0.0] * terms
    for wavelength_nm, index in samples:
        row = basis(wavelength_nm)
        for i in range(terms):
            rhs[i] += row[i] * index
            for j in range(terms):
                normal[i][j] += row[i] * row[j]
    coeffs = _solve_linear(normal, rhs)
    model = CauchyModel(A=coeffs[0], B=coeffs[1], C=coeffs[2] if terms == 3 else 0.0)
    # 拟合结果至少在采样点自身上必须给出正的折射率
    for wavelength_nm, _ in samples:
        model.index(wavelength_nm)
    return model
