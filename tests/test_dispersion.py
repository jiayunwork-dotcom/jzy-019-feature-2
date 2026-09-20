"""色散模型与材料档：多谱线折射率、柯西拟合、两种描述等价。"""

from __future__ import annotations

import pytest

from app.dispersion import (
    CauchyDispersion,
    SampledDispersion,
    SellmeierDispersion,
    build_dispersion,
)
from app.errors import OpticsError

# BK7 Sellmeier（Schott N-BK7，波长 μm）
BK7_B = (1.03961217, 0.231792344, 1.01046945)
BK7_C = (0.00600069867, 0.0200179144, 103.560653)

# 参考值（Schott 数据表，六位小数）
BK7_TABLE = {
    0.48613: 1.52238,  # F
    0.58756: 1.51680,  # d
    0.65627: 1.51432,  # C
}


def test_sellmeier_matches_glass_table_at_reference_lines():
    model = SellmeierDispersion(BK7_B, BK7_C)
    for wavelength, expected in BK7_TABLE.items():
        assert model.index(wavelength) == pytest.approx(expected, abs=2e-5)


def test_normal_dispersion_index_drops_with_wavelength():
    model = SellmeierDispersion(BK7_B, BK7_C)
    blue = model.index(0.4861)
    green = model.index(0.5876)
    red = model.index(0.6563)
    assert blue > green > red


def test_constant_index_is_achromatic():
    model = build_dispersion({"type": "constant", "n": 1.5})
    values = {model.index(w) for w in (0.4, 0.55, 0.7, 1.2)}
    assert values == {1.5}


def test_cauchy_analytic_formula():
    coefficients = (1.5046, 0.00420, 0.00003)
    model = CauchyDispersion(coefficients)
    for wavelength in (0.45, 0.55, 0.7):
        expected = (
            coefficients[0]
            + coefficients[1] / wavelength**2
            + coefficients[2] / wavelength**4
        )
        assert model.index(wavelength) == pytest.approx(expected, abs=1e-14)


def test_sampled_fit_is_analytic_not_piecewise_linear():
    # 在一条真实柯西曲线上取 5 个点：拟合后曲线上的折射率与解析式一致，
    # 而两点线性内插会明显不同（尤其在带 c2 项时）。
    coefficients = (1.5046, 0.00420, 0.00002)
    analytic = CauchyDispersion(coefficients)
    sample_wavelengths = [0.45, 0.5, 0.5876, 0.63, 0.7]
    samples = [[w, analytic.index(w)] for w in sample_wavelengths]
    model = build_dispersion({"type": "sampled", "samples": samples})

    assert isinstance(model, SampledDispersion)
    assert model.fit_rms < 1e-9

    # 内插点走拟合曲线，与解析式一致（线性硬连不可能做到）
    for wavelength in (0.47, 0.54, 0.61, 0.68):
        assert model.index(wavelength) == pytest.approx(
            analytic.index(wavelength), abs=1e-9
        )

    # 外推也有解析依据：范围外仍按柯西曲线走，不拒绝、不线性延长
    assert model.within_range(0.6)
    assert not model.within_range(0.75)
    assert model.index(0.75) == pytest.approx(analytic.index(0.75), abs=1e-9)


def test_sampled_and_analytic_describe_same_glass_agree():
    """同一块玻璃：采样点档与解析系数档算出的折射率一致。"""
    coefficients = (1.5046, 0.0042, 0.0)
    sample_wavelengths = [0.4861, 0.5461, 0.5876, 0.6563, 0.7065]
    samples = [
        [w, coefficients[0] + coefficients[1] / w**2]
        for w in sample_wavelengths
    ]
    analytic = build_dispersion({"type": "cauchy", "coefficients": list(coefficients)})
    sampled = build_dispersion({"type": "sampled", "samples": samples, "terms": 2})

    for wavelength in (0.46, 0.5, 0.55, 0.6, 0.68, 0.72):
        assert sampled.index(wavelength) == pytest.approx(
            analytic.index(wavelength), abs=1e-12
        )
    # 采样点上自然严格落在原值
    for wavelength, _ in samples:
        assert sampled.index(wavelength) == pytest.approx(
            coefficients[0] + coefficients[1] / wavelength**2, abs=1e-12
        )


def test_sampled_describe_lists_every_sample_and_coefficients():
    samples = [[0.4861, 1.522], [0.5876, 1.517], [0.6563, 1.514]]
    described = build_dispersion({"type": "sampled", "samples": samples}).describe()
    assert described["type"] == "sampled"
    assert len(described["samples"]) == 3
    assert len(described["fitted_cauchy_coefficients"]) == described["terms"]
    assert described["wavelength_range"] == [0.4861, 0.6563]
    assert described["fit_rms_residual"] < 1e-9


def test_dispersion_validation_rejects_bad_specs():
    bad_specs = [
        {"type": "constant", "n": -1.0},
        {"type": "constant", "n": float("nan")},
        {"type": "cauchy", "coefficients": []},
        {"type": "cauchy", "coefficients": [1, 2, 3, 4, 5]},
        {"type": "sellmeier", "B": [1.0], "C": [1.0, 2.0]},
        {"type": "sampled", "samples": [[0.5, 1.5], [0.6, 1.49]]},  # 少于3点
        {"type": "sampled", "samples": [[0.5, 1.5], [0.5, 1.5], [0.6, 1.49]]},  # 重复波长
        {"type": "sampled", "samples": [[0.5, 1.5], [0.6, 0.0], [0.7, 1.49]]},  # 非正折射率
        {"type": "mystery"},
    ]
    for spec in bad_specs:
        with pytest.raises(OpticsError) as exc:
            build_dispersion(spec)
        assert exc.value.code in {"invalid_dispersion", "invalid_refractive_index"}


def test_nonpositive_wavelength_rejected_by_models():
    for model in (
        SellmeierDispersion(BK7_B, BK7_C),
        CauchyDispersion((1.5, 0.004)),
        build_dispersion({"type": "constant", "n": 1.5}),
    ):
        for bad in (0.0, -0.5, float("inf"), float("nan")):
            with pytest.raises(OpticsError) as exc:
                model.index(bad)
            assert exc.value.code == "invalid_wavelength"
