"""色散模型的纯计算测试：Cauchy / Sellmeier / 恒定、采样点拟合、求值失败。"""

from __future__ import annotations

import math

import pytest

from app.dispersion import (
    CauchyModel,
    ConstantIndex,
    SellmeierModel,
    fit_cauchy,
)
from app.errors import OpticsError
from app.materials import Material, model_from_description

# 常用夫琅禾费谱线（nm）
F_LINE = 486.1327
D_LINE = 587.5618
C_LINE = 656.2725

# N-BK7 公开 Sellmeier 系数（λ 以 µm 计）
BK7_B = (1.03961212, 0.231792344, 1.01046945)
BK7_C = (0.00600069867, 0.0200179144, 103.560653)


# ---------------------------------------------------------------------------
# 恒定折射率：全波段一致，色差为零的根基
# ---------------------------------------------------------------------------

def test_constant_index_is_wavelength_independent():
    model = ConstantIndex(1.5)
    for wavelength in (400.0, F_LINE, D_LINE, C_LINE, 800.0, 2000.0):
        assert model.index(wavelength) == 1.5


# ---------------------------------------------------------------------------
# Cauchy 模型：按定义公式求值
# ---------------------------------------------------------------------------

def test_cauchy_model_matches_hand_formula():
    model = CauchyModel(A=1.5046, B=4200.0, C=1.5e8)
    for wavelength in (F_LINE, D_LINE, C_LINE):
        expected = 1.5046 + 4200.0 / wavelength**2 + 1.5e8 / wavelength**4
        assert model.index(wavelength) == pytest.approx(expected, rel=1e-15)
    # 短波折射率高、长波折射率低（正常色散）
    assert model.index(F_LINE) > model.index(D_LINE) > model.index(C_LINE)


# ---------------------------------------------------------------------------
# Sellmeier 模型：N-BK7 在 d 线的公认折射率 1.51680
# ---------------------------------------------------------------------------

def test_sellmeier_bk7_matches_catalogue_values():
    model = SellmeierModel(B=BK7_B, C=BK7_C)
    assert model.index(D_LINE) == pytest.approx(1.51680, abs=1e-5)
    assert model.index(F_LINE) == pytest.approx(1.52238, abs=1e-5)
    assert model.index(C_LINE) == pytest.approx(1.51432, abs=1e-5)


def test_sellmeier_mismatched_coefficient_counts_rejected():
    with pytest.raises(OpticsError) as exc:
        SellmeierModel(B=(1.0, 0.2), C=(0.01,))
    assert exc.value.code == "invalid_material"


# ---------------------------------------------------------------------------
# 采样点拟合：精确还原生成它的 Cauchy 曲线；外推沿曲线而非线性硬连
# ---------------------------------------------------------------------------

def test_fit_cauchy_recovers_generating_curve_exactly():
    true_model = CauchyModel(A=1.5046, B=4200.0, C=1.5e8)
    sample_wavelengths = [404.6561, 486.1327, 546.074, 587.5618, 656.2725, 768.2]
    samples = [(w, true_model.index(w)) for w in sample_wavelengths]

    fitted = fit_cauchy(samples)
    assert fitted.A == pytest.approx(true_model.A, rel=1e-9)
    assert fitted.B == pytest.approx(true_model.B, rel=1e-9)
    assert fitted.C == pytest.approx(true_model.C, rel=1e-9)

    # 采样点之间的内插与采样范围之外的外推都落在同一条 Cauchy 曲线上
    for wavelength in (450.0, 500.0, 600.0, 700.0, 900.0, 350.0):
        assert fitted.index(wavelength) == pytest.approx(
            true_model.index(wavelength), abs=1e-9)


def test_fit_cauchy_extrapolation_is_not_linear():
    # 用两个采样点拟合 A + B/λ²，外推值必须等于 Cauchy 外推而非线性外推
    true_model = CauchyModel(A=1.50, B=5000.0)
    w1, w2 = 500.0, 600.0
    samples = [(w1, true_model.index(w1)), (w2, true_model.index(w2))]
    fitted = fit_cauchy(samples)

    w_far = 900.0
    n1, n2 = true_model.index(w1), true_model.index(w2)
    linear_extrapolation = n2 + (n2 - n1) / (w2 - w1) * (w_far - w2)
    assert fitted.index(w_far) == pytest.approx(true_model.index(w_far), abs=1e-12)
    assert abs(fitted.index(w_far) - linear_extrapolation) > 1e-4


def test_fit_cauchy_two_points_uses_two_terms():
    fitted = fit_cauchy([(500.0, 1.52), (600.0, 1.51)])
    assert fitted.C == 0.0
    assert fitted.index(500.0) == pytest.approx(1.52, abs=1e-12)
    assert fitted.index(600.0) == pytest.approx(1.51, abs=1e-12)


def test_fit_cauchy_requires_at_least_two_points():
    with pytest.raises(OpticsError) as exc:
        fit_cauchy([(550.0, 1.5)])
    assert exc.value.code == "invalid_material"


# ---------------------------------------------------------------------------
# 求值失败：折射率非正或非有限时带类型退回
# ---------------------------------------------------------------------------

def test_cauchy_non_positive_index_rejected_at_evaluation():
    model = CauchyModel(A=-10.0, B=0.0)
    with pytest.raises(OpticsError) as exc:
        model.index(D_LINE)
    assert exc.value.code == "dispersion_evaluation_failed"


def test_sellmeier_negative_n_squared_rejected_at_evaluation():
    # 构造一个在可见光波段 n² < 0 的病态模型
    model = SellmeierModel(B=(-5.0,), C=(0.01,))
    with pytest.raises(OpticsError) as exc:
        model.index(D_LINE)
    assert exc.value.code == "dispersion_evaluation_failed"


def test_material_evaluation_error_carries_material_name():
    material = Material(
        name="bad_glass",
        model=CauchyModel(A=-10.0, B=0.0),
        description={"description_type": "coefficients",
                     "model": "cauchy", "A": -10.0, "B": 0.0, "C": 0.0},
    )
    with pytest.raises(OpticsError) as exc:
        material.index(D_LINE)
    assert exc.value.extra["material"] == "bad_glass"


# ---------------------------------------------------------------------------
# 同一玻璃、两种描述：采样点档与解析系数档给出一致折射率
# ---------------------------------------------------------------------------

def test_samples_and_coefficients_describing_same_glass_agree():
    coefficients_description = {
        "description_type": "coefficients",
        "model": "cauchy",
        "coefficients": {"A": 1.5046, "B": 4200.0, "C": 1.5e8},
    }
    coefficient_model = model_from_description(coefficients_description)
    sample_wavelengths = [404.6561, 486.1327, 546.074, 587.5618, 656.2725, 768.2]
    samples_description = {
        "description_type": "samples",
        "samples": [
            {"wavelength": w, "n": coefficient_model.index(w)}
            for w in sample_wavelengths
        ],
    }
    sample_model = model_from_description(samples_description)

    for wavelength in (F_LINE, 500.0, D_LINE, 620.0, C_LINE, 700.0):
        assert sample_model.index(wavelength) == pytest.approx(
            coefficient_model.index(wavelength), abs=1e-9)


def test_constant_description_roundtrip():
    model = model_from_description(
        {"description_type": "coefficients", "model": "constant",
         "coefficients": {"n": 1.5}})
    assert isinstance(model, ConstantIndex)
    assert model.index(D_LINE) == 1.5
