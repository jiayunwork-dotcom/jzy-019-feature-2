"""波长 -> 折射率的色散模型（与追迹主流程完全独立的纯数学模块）。

波长一律以微米（μm）计。支持四类描述：

- ``constant``：全波段折射率恒定（旧的裸 n1/n2 写法等价于此类）；
- ``cauchy``：柯西解析系数
  ``n(λ) = c0 + c1/λ² + c2/λ⁴ + c3/λ⁶``；
- ``sellmeier``：Sellmeier 解析系数（1~3 项）
  ``n² = 1 + Σ B_i λ²/(λ² - C_i)``；
- ``sampled``：若干条参考谱线 ``(λ, n)`` 采样点，按柯西色散关系做
  最小二乘拟合（点数恰等于系数数时为精确插值），内插、外推都走拟合
  出来的解析柯西曲线，绝不线性硬连。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .errors import OpticsError

#: 缺省参考波长（氦 d 线），旧单波长请求不带波长时使用
DEFAULT_WAVELENGTH_UM = 0.5876
#: 建档时用 d 线做一次折射率正定性自检
GUARD_WAVELENGTH_UM = 0.5876
MAX_CAUCHY_TERMS = 4


class DispersionModel(Protocol):
    """折射率随波长变化的模型。"""

    type_name: str

    def index(self, wavelength: float) -> float:
        """给定波长（μm）处的折射率。"""

    def describe(self) -> dict[str, Any]:
        """可登记、可往返序列化的完整描述（含全部系数/采样点）。"""


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpticsError(
            "invalid_dispersion",
            f"{what} 必须是数值，收到 {value!r}",
            field=what,
            value=value,
        )
    number = float(value)
    if not math.isfinite(number):
        raise OpticsError(
            "invalid_dispersion",
            f"{what} 必须是有限数值，收到 {value!r}",
            field=what,
            value=value,
        )
    return number


@dataclass(frozen=True)
class ConstantIndex:
    """无色散材料：折射率与波长无关。"""

    n: float
    type_name: str = "constant"

    def index(self, wavelength: float) -> float:
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            raise OpticsError(
                "invalid_wavelength",
                f"波长必须为正的有限值（μm），收到 {wavelength!r}",
                wavelength=wavelength,
            )
        return self.n

    def describe(self) -> dict[str, Any]:
        return {"type": "constant", "n": self.n}


@dataclass(frozen=True)
class CauchyDispersion:
    """柯西级数 n(λ) = c0 + c1/λ² + c2/λ⁴ + ...（1~4 项）。"""

    coefficients: tuple[float, ...]
    type_name: str = "cauchy"

    def index(self, wavelength: float) -> float:
        lam = _check_wavelength(wavelength)
        inv_lambda2 = 1.0 / (lam * lam)
        power = 1.0
        total = 0.0
        for coefficient in self.coefficients:
            total += coefficient * power
            power *= inv_lambda2
        return total

    def describe(self) -> dict[str, Any]:
        return {"type": "cauchy", "coefficients": list(self.coefficients)}


@dataclass(frozen=True)
class SellmeierDispersion:
    """Sellmeier 方程 n² = 1 + Σ B_i λ²/(λ² - C_i)。"""

    b: tuple[float, ...]
    c: tuple[float, ...]
    type_name: str = "sellmeier"

    def index(self, wavelength: float) -> float:
        lam = _check_wavelength(wavelength)
        lam2 = lam * lam
        total = 1.0
        for b_i, c_i in zip(self.b, self.c):
            denominator = lam2 - c_i
            if denominator == 0.0:
                # λ=sqrt(C_i) 恰好落在谐振极点上，级数发散
                raise OpticsError(
                    "invalid_wavelength",
                    f"波长 {wavelength:g} μm 恰在 Sellmeier 极点 "
                    f"{math.sqrt(c_i):g} μm 上，折射率发散",
                    wavelength=wavelength,
                )
            # 注意：分母在远紫外谐振项处可以为负（如 BK7 的 C3≈103.56），
            # 这是标准 Sellmeier 系数在可见光区的正常经验用法，只看最终 n²。
            total += b_i * lam2 / denominator
        if total <= 0.0:
            raise OpticsError(
                "invalid_refractive_index",
                f"Sellmeier 模型在 {wavelength:g} μm 处 n²={total:g} 非正",
                wavelength=wavelength,
            )
        return math.sqrt(total)

    def describe(self) -> dict[str, Any]:
        return {"type": "sellmeier", "B": list(self.b), "C": list(self.c)}


@dataclass(frozen=True)
class SampledDispersion:
    """参考谱线采样点 + 柯西拟合曲线。

    内插与外推一律使用 ``fitted`` 这条解析柯西曲线；采样点原样保留，
    登记/列出时连同拟合系数与残差一并报出。
    """

    wavelengths: tuple[float, ...]
    indices: tuple[float, ...]
    fitted: CauchyDispersion
    fit_rms: float
    type_name: str = "sampled"

    def index(self, wavelength: float) -> float:
        lam = _check_wavelength(wavelength)
        return self.fitted.index(lam)

    def within_range(self, wavelength: float) -> bool:
        return self.wavelengths[0] <= wavelength <= self.wavelengths[-1]

    def describe(self) -> dict[str, Any]:
        return {
            "type": "sampled",
            "terms": len(self.fitted.coefficients),
            "samples": [list(pair) for pair in zip(self.wavelengths, self.indices)],
            "wavelength_range": [self.wavelengths[0], self.wavelengths[-1]],
            "wavelength_unit": "um",
            "fitted_cauchy_coefficients": list(self.fitted.coefficients),
            "fit_rms_residual": self.fit_rms,
            "extrapolation": "采样范围之外按拟合的柯西解析曲线外推（非线性硬连）",
        }


def _check_wavelength(wavelength: float) -> float:
    if isinstance(wavelength, bool) or not isinstance(wavelength, (int, float)):
        raise OpticsError(
            "invalid_wavelength",
            f"波长必须是数值（μm），收到 {wavelength!r}",
            wavelength=wavelength,
        )
    lam = float(wavelength)
    if not math.isfinite(lam) or lam <= 0.0:
        raise OpticsError(
            "invalid_wavelength",
            f"波长必须为正的有限值（μm），收到 {wavelength!r}",
            wavelength=wavelength,
        )
    return lam


# ---------------------------------------------------------------------------
# 线性最小二乘（列主元高斯消元 + 列尺度改善条件数），不引第三方依赖
# ---------------------------------------------------------------------------

def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    # 列尺度：每列除以该列最大绝对值
    scales: list[float] = []
    for col in range(n):
        scale = max(abs(matrix[row][col]) for row in range(n))
        scales.append(scale if scale > 0.0 else 1.0)
    a = [
        [matrix[row][col] / scales[col] for col in range(n)] + [rhs[row]]
        for row in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise OpticsError(
                "invalid_dispersion",
                "采样点的柯西设计矩阵奇异，无法拟合色散曲线；"
                "请拉开采样波长间隔或减少系数项数",
            )
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        pivot_value = a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] / pivot_value
            if factor:
                for k in range(col, n + 1):
                    a[row][k] -= factor * a[col][k]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        residual = a[row][n] - sum(a[row][col] * solution[col] for col in range(row + 1, n))
        solution[row] = residual / a[row][row]
    return [solution[col] / scales[col] for col in range(n)]


def _fit_cauchy(wavelengths: Sequence[float],
                indices: Sequence[float],
                terms: int) -> tuple[CauchyDispersion, float]:
    """在基函数 (1, λ⁻², λ⁻⁴, ...) 上做（精确/最小二乘）拟合。"""
    x_values = [1.0 / (lam * lam) for lam in wavelengths]
    rows = [[x ** power for power in range(terms)] for x in x_values]

    if len(wavelengths) == terms:
        coefficients = _solve_linear(rows, list(indices))
    else:
        # 法方程 (XᵀX) c = Xᵀy
        normal = [[0.0] * terms for _ in range(terms)]
        target = [0.0] * terms
        for row, y in zip(rows, indices):
            for i in range(terms):
                target[i] += row[i] * y
                for j in range(terms):
                    normal[i][j] += row[i] * row[j]
        coefficients = _solve_linear(normal, target)

    model = CauchyDispersion(tuple(coefficients))
    residual_squared = 0.0
    for lam, n in zip(wavelengths, indices):
        residual_squared += (model.index(lam) - n) ** 2
    rms = math.sqrt(residual_squared / len(wavelengths))
    for coefficient in coefficients:
        if not math.isfinite(coefficient):
            raise OpticsError("invalid_dispersion", "柯西拟合系数非有限数")
    return model, rms


# ---------------------------------------------------------------------------
# 由登记的 JSON 描述构造模型
# ---------------------------------------------------------------------------

def _build_constant(spec: dict[str, Any]) -> ConstantIndex:
    if "n" not in spec:
        raise OpticsError("invalid_dispersion", "constant 模型缺少 n", field="n")
    n = _finite(spec["n"], "constant.n")
    if n <= 0.0:
        raise OpticsError(
            "invalid_refractive_index",
            f"恒定折射率必须为正，收到 n={n:g}",
            n=n,
        )
    return ConstantIndex(n=n)


def _build_cauchy(spec: dict[str, Any]) -> CauchyDispersion:
    raw = spec.get("coefficients")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_CAUCHY_TERMS:
        raise OpticsError(
            "invalid_dispersion",
            "cauchy 模型需要 coefficients 数组（1~4 个系数，按 λ^0,λ^-2,... 排序）",
            field="coefficients",
        )
    coefficients = tuple(_finite(v, f"cauchy.coefficients[{i}]") for i, v in enumerate(raw))
    model = CauchyDispersion(coefficients)
    _assert_positive_index(model, "cauchy")
    return model


def _build_sellmeier(spec: dict[str, Any]) -> SellmeierDispersion:
    b_raw, c_raw = spec.get("B"), spec.get("C")
    if not isinstance(b_raw, list) or not isinstance(c_raw, list):
        raise OpticsError(
            "invalid_dispersion",
            "sellmeier 模型需要 B、C 两个等长数组（1~3 项，波长以 μm 计）",
        )
    if not 1 <= len(b_raw) <= 3 or len(b_raw) != len(c_raw):
        raise OpticsError(
            "invalid_dispersion",
            "sellmeier 的 B、C 必须等长且为 1~3 项",
        )
    b = tuple(_finite(v, f"sellmeier.B[{i}]") for i, v in enumerate(b_raw))
    c = tuple(_finite(v, f"sellmeier.C[{i}]") for i, v in enumerate(c_raw))
    model = SellmeierDispersion(b, c)
    _assert_positive_index(model, "sellmeier")
    return model


def _build_sampled(spec: dict[str, Any]) -> SampledDispersion:
    raw_samples = spec.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < 3:
        raise OpticsError(
            "invalid_dispersion",
            "sampled 模型至少需要 3 条 [波长μm, 折射率] 参考谱线，"
            "才能按柯西色散关系拟合（不允许线性硬连）",
            field="samples",
        )

    points: list[tuple[float, float]] = []
    for i, sample in enumerate(raw_samples):
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise OpticsError(
                "invalid_dispersion",
                f"samples[{i}] 必须是 [波长, 折射率] 二元组",
                index=i,
            )
        lam = _check_wavelength(_finite(sample[0], f"samples[{i}].wavelength"))
        n = _finite(sample[1], f"samples[{i}].index")
        if n <= 0.0:
            raise OpticsError(
                "invalid_refractive_index",
                f"samples[{i}] 折射率必须为正，收到 n={n:g}",
                index=i,
            )
        points.append((lam, n))

    points.sort(key=lambda pair: pair[0])
    wavelengths = tuple(p[0] for p in points)
    indices = tuple(p[1] for p in points)
    if len(set(wavelengths)) != len(wavelengths):
        raise OpticsError("invalid_dispersion", "sampled 参考谱线波长不得重复")

    terms = spec.get("terms", min(3, len(points)))
    if not isinstance(terms, int) or isinstance(terms, bool) or not 1 <= terms <= len(points):
        raise OpticsError(
            "invalid_dispersion",
            f"terms 必须是 1~{len(points)} 之间的整数，收到 {terms!r}",
        )
    if terms > MAX_CAUCHY_TERMS:
        raise OpticsError(
            "invalid_dispersion",
            f"柯西拟合最多 {MAX_CAUCHY_TERMS} 项，收到 terms={terms}",
        )

    # 正常色散下 n 应随 λ 单调下降；不强制卡死（有些档故意非标准），
    # 但点数/项数足够时拟合必须能正定性自检。
    fitted, rms = _fit_cauchy(wavelengths, indices, terms)
    model = SampledDispersion(
        wavelengths=wavelengths,
        indices=indices,
        fitted=fitted,
        fit_rms=rms,
    )
    _assert_positive_index(model, "sampled")
    return model


def _assert_positive_index(model: DispersionModel, label: str) -> None:
    try:
        n = model.index(GUARD_WAVELENGTH_UM)
    except OpticsError:
        # 模型自身已说明原因（如极点），不在建档处再包一层
        raise
    if not math.isfinite(n) or n <= 0.0:
        raise OpticsError(
            "invalid_refractive_index",
            f"{label} 模型在 {GUARD_WAVELENGTH_UM:g} μm 处折射率为 {n!r}，必须为正",
        )


_BUILDERS = {
    "constant": _build_constant,
    "cauchy": _build_cauchy,
    "sellmeier": _build_sellmeier,
    "sampled": _build_sampled,
}


def build_dispersion(spec: Any) -> DispersionModel:
    """按登记描述构造色散模型；描述非法即抛带类型的错误。"""
    if not isinstance(spec, dict):
        raise OpticsError(
            "invalid_dispersion",
            f"色散描述必须是对象，收到 {type(spec).__name__}",
        )
    type_name = spec.get("type")
    builder = _BUILDERS.get(type_name)
    if builder is None:
        raise OpticsError(
            "invalid_dispersion",
            f"色散类型 {type_name!r} 未知，支持：constant / cauchy / sellmeier / sampled",
            type=type_name,
        )
    return builder(spec)
