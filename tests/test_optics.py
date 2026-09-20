"""纯光学计算测试，不依赖 Web 框架。"""

from __future__ import annotations

import math

import pytest

from app.elements import Element, element_matrix
from app.errors import OpticsError
from app.systems import (
    AFOCAL_C_TOL,
    DET_TOL,
    ROUNDTRIP_TOL,
    check_air_determinants,
    reverse_trace_residual,
    segment_matrices,
    solve_imaging,
    system_matrix,
    trace,
)
from app.validation import normalize_element, normalize_elements


# ---------------------------------------------------------------------------
# 空气系统行列式为 1：空气间隔接空气间隔，中间只夹薄透镜
# ---------------------------------------------------------------------------

def test_air_system_determinant_is_one_including_consecutive_spaces():
    elements = normalize_elements([
        {"type": "space", "params": {"L": 30.0}},
        {"type": "lens", "params": {"f": 100.0}},
        {"type": "space", "params": {"L": 20.0}},
        {"type": "space", "params": {"L": 15.0}},   # 空气间隔直接接空气间隔
        {"type": "lens", "params": {"f": -50.0}},   # 发散透镜
        {"type": "space", "params": {"L": 5.0}},
    ])
    matrices = segment_matrices(elements)
    check_air_determinants(elements, matrices)  # 内部逐段+连乘断言，不抛即通过

    for m in matrices:
        assert abs((m[0] * m[3] - m[1] * m[2]) - 1.0) <= DET_TOL
    assert abs((lambda m: m[0] * m[3] - m[1] * m[2])(system_matrix(elements)) - 1.0) <= DET_TOL


def test_matrix_shapes_match_spec():
    space = element_matrix(Element("space", {"L": 7.0}))
    assert space == pytest.approx((1.0, 7.0, 0.0, 1.0))

    lens = element_matrix(Element("lens", {"f": 100.0}))
    assert lens[0] == 1.0 and lens[1] == 0.0
    assert lens[2] == pytest.approx(-0.01) and lens[3] == 1.0


# ---------------------------------------------------------------------------
# 单透镜还原高斯公式：1/s + 1/s' = 1/f
# ---------------------------------------------------------------------------

def test_single_thin_lens_recovers_gaussian_formula():
    f = 120.0
    elements = normalize_elements([{"type": "lens", "params": {"f": f}}])
    a, b, c, d = system_matrix(elements)
    assert (a, b, d) == (1.0, 0.0, 1.0)
    assert c == pytest.approx(-1.0 / f)

    for s in (60.0, 150.0, 200.0, 1000.0):
        imaging = solve_imaging(elements, s)
        s_prime = imaging["image_distance"]
        magnification = imaging["magnification"]
        assert s_prime is not None
        # 倒数和等于焦距倒数
        assert 1.0 / s + 1.0 / s_prime == pytest.approx(1.0 / f, abs=1e-12)
        # 放大率 m = -s'/s
        assert magnification == pytest.approx(-s_prime / s, abs=1e-12)
        assert imaging["effective_focal_length"] == pytest.approx(f)

    # 物在前焦面：出射平行，像在无穷远
    assert solve_imaging(elements, f)["image_distance"] is None

    # 平行光入射：像落在后焦面，EFL = -1/C = f
    infinity_imaging = solve_imaging(elements, None)
    assert infinity_imaging["image_distance"] == pytest.approx(f)
    assert infinity_imaging["effective_focal_length"] == pytest.approx(-1.0 / c)
    assert infinity_imaging["magnification"] == 0.0


# ---------------------------------------------------------------------------
# 共焦望远镜：C 接近 0、放大率为负 = -f2/f1；错开间距焦度必须重新出现
# ---------------------------------------------------------------------------

def test_confocal_telescope_is_afocal_with_negative_magnification():
    f1, f2 = 100.0, 40.0
    elements = normalize_elements([
        {"type": "lens", "params": {"f": f1}},
        {"type": "space", "params": {"L": f1 + f2}},
        {"type": "lens", "params": {"f": f2}},
    ])
    a, b, c, d = system_matrix(elements)
    assert abs(c) < AFOCAL_C_TOL

    # 平行光进、平行光出：像在无穷远
    infinity_imaging = solve_imaging(elements, None)
    assert infinity_imaging["afocal"] is True
    assert infinity_imaging["image_distance"] is None
    assert infinity_imaging["effective_focal_length"] is None

    # 有限物距仍共轭到有限像面（虚像），横向放大率 = -f2/f1
    s = 1000.0
    imaging = solve_imaging(elements, s)
    assert imaging["afocal"] is True
    assert imaging["image_distance"] is not None
    assert imaging["magnification"] == pytest.approx(-f2 / f1, abs=1e-12)
    assert imaging["magnification"] < 0.0
    # s' = -(A s + B)/D
    assert imaging["image_distance"] == pytest.approx(-(a * s + b) / d, abs=1e-10)
    # det = 1 时 A = 1/D
    assert a == pytest.approx(1.0 / d, abs=1e-12)


def test_defocused_telescope_regains_optical_power():
    f1, f2 = 100.0, 40.0
    confocal = f1 + f2

    for delta in (-1.0, -0.1, +0.1, +1.0):
        elements = normalize_elements([
            {"type": "lens", "params": {"f": f1}},
            {"type": "space", "params": {"L": confocal + delta}},
            {"type": "lens", "params": {"f": f2}},
        ])
        _, _, c, _ = system_matrix(elements)
        # C = delta / (f1 f2)，不再是无焦
        assert abs(c) > AFOCAL_C_TOL
        assert c == pytest.approx(delta / (f1 * f2), abs=1e-12)
        imaging = solve_imaging(elements, None)
        assert imaging["afocal"] is False
        assert imaging["effective_focal_length"] is not None
        assert math.isfinite(imaging["effective_focal_length"])


# ---------------------------------------------------------------------------
# 正向再逆向还原入射光线（含进出玻璃）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("elements,ray", [
    (
        [
            {"type": "lens", "params": {"f": 80.0}},
            {"type": "space", "params": {"L": 12.0}},
            {"type": "lens", "params": {"f": -30.0}},
        ],
        (5.0, 0.01),
    ),
    (
        # 空气进玻璃再出到空气
        [
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": 1.5}},
            {"type": "space", "params": {"L": 10.0}},
            {"type": "refract", "params": {"R": -50.0, "n1": 1.5, "n2": 1.0}},
        ],
        (3.0, -0.02),
    ),
])
def test_forward_then_inverse_restores_ray(elements, ray):
    elements = normalize_elements(elements)
    residual, restored = reverse_trace_residual(elements, ray)
    assert residual <= ROUNDTRIP_TOL
    assert restored[0] == pytest.approx(ray[0], abs=ROUNDTRIP_TOL)
    assert restored[1] == pytest.approx(ray[1], abs=ROUNDTRIP_TOL)


def test_glass_roundtrip_system_determinant_returns_to_one():
    elements = normalize_elements([
        {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": 1.5}},
        {"type": "space", "params": {"L": 10.0}},
        {"type": "refract", "params": {"R": -50.0, "n1": 1.5, "n2": 1.0}},
    ])
    a, b, c, d = system_matrix(elements)
    assert a * d - b * c == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 球面折射：等折射率退化为恒等；近轴折射定律方向正确
# ---------------------------------------------------------------------------

def test_refraction_equal_indices_is_identity():
    element = normalize_element(
        {"type": "refract", "params": {"R": 50.0, "n1": 1.5, "n2": 1.5}}, 1)
    m = element_matrix(element)
    assert m == pytest.approx((1.0, 0.0, 0.0, 1.0))

    ray = (2.0, 0.03)
    result = trace([element], ray)
    assert result["outgoing_ray"]["y"] == pytest.approx(ray[0])
    assert result["outgoing_ray"]["u"] == pytest.approx(ray[1])


def test_refraction_air_to_glass_bends_toward_axis_for_convex_surface():
    # 凸面（R>0），轴上平行光从空气进玻璃应向光轴弯折：u2 < 0
    element = normalize_element(
        {"type": "refract", "params": {"R": 100.0, "n1": 1.0, "n2": 1.5}}, 1)
    result = trace([element], (1.0, 0.0))
    # u2 = n1/n2 u1 - (n2-n1)/(n2 R) y = -0.5/150
    assert result["outgoing_ray"]["u"] == pytest.approx(-0.5 / 150.0, abs=1e-12)
    assert result["outgoing_ray"]["y"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 退回项：焦距为零、未知元件、负间隔、非正折射率、非有限数
# ---------------------------------------------------------------------------

def test_zero_focal_length_is_rejected():
    with pytest.raises(OpticsError) as exc:
        normalize_element({"type": "lens", "params": {"f": 0.0}}, 2)
    assert exc.value.code == "zero_focal_length"
    assert exc.value.extra["index"] == 2


def test_unknown_element_is_rejected():
    with pytest.raises(OpticsError) as exc:
        normalize_element({"type": "wormhole", "params": {}}, 1)
    assert exc.value.code == "unknown_element"


def test_negative_spacing_is_rejected():
    with pytest.raises(OpticsError) as exc:
        normalize_element({"type": "air_space", "params": {"L": -5.0}}, 1)
    assert exc.value.code == "negative_spacing"


def test_non_positive_refractive_index_is_rejected():
    with pytest.raises(OpticsError) as exc:
        normalize_element(
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": 0.0}}, 1)
    assert exc.value.code == "invalid_refractive_index"


def test_non_finite_number_is_rejected():
    with pytest.raises(OpticsError) as exc:
        normalize_element({"type": "lens", "params": {"f": float("nan")}}, 1)
    assert exc.value.code == "non_finite_number"


def test_missing_field_is_rejected():
    with pytest.raises(OpticsError) as exc:
        normalize_element({"type": "lens"}, 1)
    assert exc.value.code == "missing_field"


# ---------------------------------------------------------------------------
# 改焦距必须真正改变 EFL 与放大率
# ---------------------------------------------------------------------------

def test_changing_focal_length_changes_results():
    def run(f):
        elements = normalize_elements([{"type": "lens", "params": {"f": f}}])
        imaging = solve_imaging(elements, 300.0)
        return imaging["effective_focal_length"], imaging["magnification"]

    efl1, mag1 = run(100.0)
    efl2, mag2 = run(150.0)
    assert efl1 != efl2
    assert mag1 != mag2
    assert efl1 == pytest.approx(100.0)
    assert efl2 == pytest.approx(150.0)
