"""多波长追迹 HTTP 行为：色差场景、描述等价、回归、并行隔离、非法输入。"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.demo import ACHROMAT_NAME, SINGLE_BK7_NAME
from app.main import app

client_cm = TestClient(app)
client = client_cm.__enter__()

F_LINE = 0.4861
D_LINE = 0.5876
C_LINE = 0.6563
VISIBLE_THREE = [F_LINE, D_LINE, C_LINE]


@pytest.fixture(scope="module", autouse=True)
def registered_equivalence_materials():
    """同一块柯西玻璃的两种描述：解析系数档 vs 采样点档。"""
    coefficients = (1.5046, 0.0042)
    client.post("/materials", json={
        "name": "equiv_cauchy_analytic",
        "dispersion": {"type": "cauchy", "coefficients": list(coefficients)},
    })
    sample_wavelengths = [0.4861, 0.5461, 0.5876, 0.6563, 0.7065]
    samples = [
        [w, coefficients[0] + coefficients[1] / w**2]
        for w in sample_wavelengths
    ]
    client.post("/materials", json={
        "name": "equiv_cauchy_sampled",
        "dispersion": {"type": "sampled", "samples": samples, "terms": 2},
    })
    yield


def _trace(body):
    response = client.post("/trace", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _by_wavelength(body):
    return {item["wavelength"]: item for item in body["results"]}


# ---------------------------------------------------------------------------
# 无色散材料：各波长像距完全一致、色差为零（回归钉死）
# ---------------------------------------------------------------------------

def test_constant_index_request_is_fully_achromatic():
    body = _trace({
        "elements": [
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": 1.5}},
            {"type": "space", "params": {"L": 10.0}},
            {"type": "refract", "params": {"R": -50.0, "n1": 1.5, "n2": 1.0}},
        ],
        "ray": {"y": 1.5, "u": 0.02},
        "s": 300.0,
        "wavelengths": [0.4, 0.5, 0.5876, 0.7],
    })
    matrices = [tuple(result["system_abcd"][k] for k in "ABCD")
                for result in body["results"]]
    assert matrices[1:] == matrices[:-1]
    image_distances = {result["imaging"]["image_distance"] for result in body["results"]}
    assert len(image_distances) == 1
    magnifications = {result["imaging"]["magnification"] for result in body["results"]}
    assert len(magnifications) == 1

    ca = body["chromatic_aberration"]
    assert ca["axial_chromatic_aberration"] == 0.0
    assert ca["transverse_chromatic_aberration"] == 0.0
    assert ca["reference_wavelengths"]["short"] == 0.4
    assert ca["reference_wavelengths"]["long"] == 0.7


def test_bare_focal_length_lens_and_spaces_are_wavelength_invariant():
    body = _trace({
        "elements": [
            {"type": "lens", "params": {"f": 80.0}},
            {"type": "space", "params": {"L": 12.0}},
            {"type": "lens", "params": {"f": -40.0}},
        ],
        "ray": {"y": 1.0, "u": 0.0},
        "s": 250.0,
        "wavelengths": [0.45, 0.55, 0.65],
    })
    matrices = [result["system_abcd"] for result in body["results"]]
    assert matrices[1] == matrices[0] and matrices[2] == matrices[0]
    for result in body["results"]:
        assert result["determinant"] == pytest.approx(1.0, abs=1e-12)
    assert body["chromatic_aberration"]["axial_chromatic_aberration"] == 0.0


# ---------------------------------------------------------------------------
# 单薄透镜 + 真实色散玻璃：蓝短红长，轴向色差符号/量值可手算核对
# ---------------------------------------------------------------------------

def test_single_dispersive_lens_blue_short_red_long():
    body = _trace({
        "elements": [{
            "type": "lens",
            "params": {"r1": 103.3597, "r2": -103.3597, "glass": "BK7"},
        }],
        "ray": {"y": 1.0, "u": 0.0},
        "s": 300.0,
        "wavelengths": VISIBLE_THREE,
    })
    results = _by_wavelength(body)
    s_blue = results[F_LINE]["imaging"]["image_distance"]
    s_green = results[D_LINE]["imaging"]["image_distance"]
    s_red = results[C_LINE]["imaging"]["image_distance"]

    # 蓝端焦距短 -> 同一物距下像距短；像距依次拉开
    assert s_blue < s_green < s_red

    # 手算（造镜者公式 + Sellmeier）：f_F≈98.932, f_d=100.000, f_C≈100.482
    assert results[F_LINE]["imaging"]["effective_focal_length"] == pytest.approx(98.93177, abs=1e-4)
    assert results[D_LINE]["imaging"]["effective_focal_length"] == pytest.approx(100.0, abs=1e-4)
    assert results[C_LINE]["imaging"]["effective_focal_length"] == pytest.approx(100.4816, abs=1e-4)

    # 像距：1/s' = 1/f - 1/300
    for wavelength, f_expected in (
        (F_LINE, 98.93177), (D_LINE, 100.0), (C_LINE, 100.4816),
    ):
        expected_sp = 1.0 / (1.0 / f_expected - 1.0 / 300.0)
        assert results[wavelength]["imaging"]["image_distance"] == pytest.approx(
            expected_sp, abs=1e-4
        )

    ca = body["chromatic_aberration"]
    # 轴向色差 = s'(C) - s'(F) ≈ 151.0862 - 147.6093 = 3.4770，方向为正
    assert ca["axial_chromatic_aberration"] == pytest.approx(3.47696, abs=1e-3)
    assert ca["axial_chromatic_aberration"] > 0.0
    assert ca["axial_kind"] == "image_distance_difference"
    assert ca["reference_wavelengths"] == {
        "short": F_LINE, "long": C_LINE, "unit": "um",
    }

    # 倍率色差 = m(C) - m(F)（正常色散 m 随像距增大而绝对值增大，故为负）
    m_blue = results[F_LINE]["imaging"]["magnification"]
    m_red = results[C_LINE]["imaging"]["magnification"]
    assert ca["transverse_chromatic_aberration"] == pytest.approx(m_red - m_blue, abs=1e-12)


def test_dispersive_lens_infinity_reports_bfl_per_wavelength():
    body = _trace({
        "elements": [{
            "type": "lens",
            "params": {"r1": 103.3597, "r2": -103.3597, "glass": "BK7"},
        }],
        "ray": {"y": 2.0, "u": 0.0},
        "s": None,
        "wavelengths": VISIBLE_THREE,
    })
    results = _by_wavelength(body)
    bfl_blue = results[F_LINE]["imaging"]["image_distance"]
    bfl_red = results[C_LINE]["imaging"]["image_distance"]
    # 平行光：后焦距即焦距，蓝短红长
    assert bfl_blue == pytest.approx(98.93177, abs=1e-4)
    assert bfl_red == pytest.approx(100.4816, abs=1e-4)
    assert bfl_blue < results[D_LINE]["imaging"]["image_distance"] < bfl_red

    ca = body["chromatic_aberration"]
    assert ca["axial_kind"] == "back_focal_length_difference"
    assert ca["axial_chromatic_aberration"] == pytest.approx(bfl_red - bfl_blue, abs=1e-12)


def test_named_demo_single_bk7_path_matches_inline():
    response = client.post(f"/paths/{SINGLE_BK7_NAME}/trace", json={
        "ray": {"y": 1.0, "u": 0.0}, "s": 300.0,
        "wavelengths": VISIBLE_THREE,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["path_name"] == SINGLE_BK7_NAME
    results = _by_wavelength(body)
    assert results[F_LINE]["imaging"]["image_distance"] < \
        results[C_LINE]["imaging"]["image_distance"]


# ---------------------------------------------------------------------------
# 消色差双胶合：F/C 轴向色差归零，d 线留二级光谱残余
# ---------------------------------------------------------------------------

def test_achromat_doublet_long_short_coincide_middle_residual():
    body = _trace({
        "elements": [
            {"type": "lens", "params": {"r1": 31.9120605, "r2": -75.0, "glass": "BK7"}},
            {"type": "lens", "params": {"r1": -75.0, "r2": 128.695564, "glass": "F2"}},
        ],
        "ray": {"y": 1.0, "u": 0.0},
        "s": 500.0,
        "wavelengths": VISIBLE_THREE,
    })
    results = _by_wavelength(body)
    s_f = results[F_LINE]["imaging"]["image_distance"]
    s_d = results[D_LINE]["imaging"]["image_distance"]
    s_c = results[C_LINE]["imaging"]["image_distance"]

    # 长短两谱线像距落到同一处（容差内），中间谱线残留二级光谱
    assert s_f == pytest.approx(s_c, abs=1e-6)
    assert abs(s_d - s_f) > 1e-4
    # d 线 f=100，s=500 -> s'=125；F/C 为 f≈100.050 -> s'≈125.078
    assert s_d == pytest.approx(125.0, abs=1e-6)
    assert s_f == pytest.approx(125.07823, abs=1e-4)

    ca = body["chromatic_aberration"]
    assert abs(ca["axial_chromatic_aberration"]) <= 1e-6
    assert ca["short_image_distance"] == pytest.approx(s_f, abs=1e-12)
    assert ca["long_image_distance"] == pytest.approx(s_c, abs=1e-12)


def test_achromat_demo_path_is_achromatic():
    response = client.post(f"/paths/{ACHROMAT_NAME}/trace", json={
        "ray": {"y": 1.0, "u": 0.0}, "s": 500.0,
        "wavelengths": VISIBLE_THREE,
    })
    assert response.status_code == 200, response.text
    ca = response.json()["chromatic_aberration"]
    assert abs(ca["axial_chromatic_aberration"]) <= 1e-6


def test_achromat_infinity_lca_is_zero_but_secondary_spectrum_remains():
    body = _trace({
        "elements": [
            {"type": "lens", "params": {"r1": 31.9120605, "r2": -75.0, "glass": "BK7"}},
            {"type": "lens", "params": {"r1": -75.0, "r2": 128.695564, "glass": "F2"}},
        ],
        "ray": {"y": 1.0, "u": 0.0},
        "s": None,
        "wavelengths": VISIBLE_THREE,
    })
    results = _by_wavelength(body)
    assert results[F_LINE]["imaging"]["image_distance"] == pytest.approx(
        results[C_LINE]["imaging"]["image_distance"], abs=1e-6
    )
    assert abs(results[D_LINE]["imaging"]["image_distance"]
               - results[F_LINE]["imaging"]["image_distance"]) > 1e-4
    assert abs(body["chromatic_aberration"]["axial_chromatic_aberration"]) <= 1e-6


# ---------------------------------------------------------------------------
# 采样点档与解析系数档描述同一块玻璃：折射率一致、追迹结果不漂移
# ---------------------------------------------------------------------------

def test_sampled_and_analytic_material_agree_over_http():
    for wavelength in (0.5, 0.56, 0.62, 0.69):
        a = client.get(
            f"/materials/equiv_cauchy_analytic/index?wavelength={wavelength}"
        ).json()
        s = client.get(
            f"/materials/equiv_cauchy_sampled/index?wavelength={wavelength}"
        ).json()
        assert a["index"] == pytest.approx(s["index"], abs=1e-12)


def test_trace_does_not_drift_with_description_kind():
    wavelengths = [0.5, 0.6, 0.7]
    bodies = []
    for glass in ("equiv_cauchy_analytic", "equiv_cauchy_sampled"):
        bodies.append(_trace({
            "elements": [{
                "type": "lens",
                "params": {"r1": 60.0, "r2": -60.0, "glass": glass},
            }],
            "ray": {"y": 1.0, "u": 0.0},
            "s": 250.0,
            "wavelengths": wavelengths,
        }))
    for wavelength in wavelengths:
        a = _by_wavelength(bodies[0])[wavelength]
        s = _by_wavelength(bodies[1])[wavelength]
        for key in "ABCD":
            assert a["system_abcd"][key] == pytest.approx(
                s["system_abcd"][key], abs=1e-12
            )
        assert a["imaging"]["image_distance"] == pytest.approx(
            s["imaging"]["image_distance"], abs=1e-9
        )


def test_materials_listing_shows_kind_and_full_coefficients():
    listing = {item["name"]: item for item in client.get("/materials").json()["materials"]}
    assert listing["BK7"]["description_kind"] == "sellmeier"
    assert len(listing["BK7"]["model"]["B"]) == 3
    assert listing["equiv_cauchy_sampled"]["description_kind"] == "sampled"
    assert len(listing["equiv_cauchy_sampled"]["model"]["samples"]) == 5


def test_refractive_surface_named_glass_changes_with_wavelength():
    body = _trace({
        "elements": [
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2_material": "BK7"}},
        ],
        "ray": {"y": 1.0, "u": 0.0},
        "s": None,
        "wavelengths": VISIBLE_THREE,
    })
    results = _by_wavelength(body)
    n_blue = results[F_LINE]["segments"][0]["indices"]["n2"]
    n_red = results[C_LINE]["segments"][0]["indices"]["n2"]
    assert n_blue == pytest.approx(1.52238, abs=2e-5)
    assert n_red == pytest.approx(1.51432, abs=2e-5)
    assert n_blue > n_red


# ---------------------------------------------------------------------------
# 旧单波长写死折射率的老请求行为不变
# ---------------------------------------------------------------------------

def test_legacy_single_wavelength_request_shape_unchanged():
    response = client.post("/trace", json={
        "elements": [{"type": "lens", "params": {"f": 100.0}}],
        "ray": {"y": 5.0, "u": 0.01},
        "s": 200.0,
    })
    assert response.status_code == 200
    body = response.json()
    # 旧响应的顶层键全部保留
    for key in ("system_abcd", "imaging", "outgoing_ray", "incoming_ray",
                "segments", "roundtrip", "determinant", "elements"):
        assert key in body
    assert "results" not in body
    assert (body["system_abcd"]["A"], body["system_abcd"]["B"],
            body["system_abcd"]["C"], body["system_abcd"]["D"]) == (1.0, 0.0, -0.01, 1.0)
    assert body["imaging"]["image_distance"] == pytest.approx(200.0)
    assert body["imaging"]["magnification"] == pytest.approx(-1.0)
    # 数值与单波长时完全一致（不因加了波长维而漂移）
    legacy = client.post("/trace", json={
        "elements": [{"type": "lens", "params": {"f": 100.0}}],
        "ray": {"y": 5.0, "u": 0.01}, "s": 200.0,
        "wavelengths": [0.5876],
    }).json()["results"][0]
    assert legacy["system_abcd"] == body["system_abcd"]
    assert legacy["imaging"]["image_distance"] == body["imaging"]["image_distance"]


def test_legacy_refractive_request_value_unchanged():
    response = client.post("/trace", json={
        "elements": [
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": 1.5}},
            {"type": "space", "params": {"L": 10.0}},
            {"type": "refract", "params": {"R": -50.0, "n1": 1.5, "n2": 1.0}},
        ],
        "ray": {"y": 1.5, "u": 0.02},
    })
    body = response.json()
    assert body["determinant"] == pytest.approx(1.0, abs=1e-12)
    assert body["roundtrip"]["residual"] <= 1e-10
    # n1==n2 退化成什么都不做
    identity = client.post("/trace", json={
        "elements": [{"type": "refract",
                      "params": {"R": 50.0, "n1": 1.5, "n2": 1.5}}],
        "ray": {"y": 2.0, "u": 0.03},
    }).json()
    assert identity["outgoing_ray"]["y"] == pytest.approx(2.0)
    assert identity["outgoing_ray"]["u"] == pytest.approx(0.03)
    assert tuple(identity["segments"][0]["matrix"][k] for k in "ABCD") == (
        pytest.approx(1.0), pytest.approx(0.0),
        pytest.approx(0.0), pytest.approx(1.0),
    )


# ---------------------------------------------------------------------------
# 波长间严格隔离：各结果互不共享矩阵/光线
# ---------------------------------------------------------------------------

def test_per_wavelength_results_are_isolated():
    body = _trace({
        "elements": [{
            "type": "lens",
            "params": {"r1": 103.3597, "r2": -103.3597, "glass": "BK7"},
        }],
        "ray": {"y": 1.0, "u": 0.0},
        "s": 300.0,
        "wavelengths": [0.45, 0.5, 0.5876, 0.65, 0.7],
    })
    assert body["count"] == 5
    assert [r["wavelength"] for r in body["results"]] == [0.45, 0.5, 0.5876, 0.65, 0.7]
    matrices = [r["system_abcd"] for r in body["results"]]
    assert all(matrices[i] is not matrices[j] for i in range(5) for j in range(5) if i != j)
    outgoing = [r["outgoing_ray"]["u"] for r in body["results"]]
    assert len(set(outgoing)) == 5  # 各波长折光不同，出射角彼此不同


def test_parallel_multiwavelength_traces_do_not_cross_contaminate():
    async def call(wavelengths, y):
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.post("/trace", json={
                "elements": [{
                    "type": "lens",
                    "params": {"r1": 103.3597, "r2": -103.3597, "glass": "BK7"},
                }],
                "ray": {"y": y, "u": 0.0},
                "s": 300.0,
                "wavelengths": wavelengths,
            })

    async def scenario():
        return await asyncio.gather(
            call([0.4861, 0.5876, 0.6563], 1.0),
            call([0.45, 0.7], 2.0),
            call([0.4861, 0.5876, 0.6563], 3.0),
            call([0.55], 4.0),
        )

    responses = asyncio.run(scenario())
    for response in responses:
        assert response.status_code == 200, response.text
    a, b, c, d = (r.json() for r in responses)

    # 同两组波长的作业结果彼此一致（与入射高度无关），且不混入 0.45/0.7/0.55
    def matrix_map(body):
        return {r["wavelength"]: r["system_abcd"] for r in body["results"]}

    assert matrix_map(a) == matrix_map(c)
    assert sorted(matrix_map(a)) == [0.4861, 0.5876, 0.6563]
    assert sorted(matrix_map(b)) == [0.45, 0.7]
    assert sorted(matrix_map(d)) == [0.55]
    # 出射高度只随本作业入射高度线性变化，不串作业
    assert a["results"][0]["outgoing_ray"]["y"] * 3.0 == pytest.approx(
        c["results"][0]["outgoing_ray"]["y"]
    )
    assert b["results"][0]["outgoing_ray"]["y"] == pytest.approx(2.0)
    assert d["results"][0]["outgoing_ray"]["y"] == pytest.approx(4.0)
    # 色差基准也各自独立
    assert a["chromatic_aberration"]["reference_wavelengths"]["short"] == 0.4861
    assert b["chromatic_aberration"]["reference_wavelengths"]["short"] == 0.45


# ---------------------------------------------------------------------------
# 各类非法输入在追迹前退回，错误带类型与元件序号
# ---------------------------------------------------------------------------

def _expect_error(body, code, status_code=422):
    response = client.post("/trace", json=body)
    assert response.status_code == status_code, response.text
    assert response.json()["error"]["code"] == code, response.text
    return response.json()["error"]


def test_multiwavelength_invalid_inputs_rejected():
    base_elements = [{"type": "lens", "params": {"f": 50.0}}]
    base_ray = {"y": 1.0, "u": 0.0}

    _expect_error(
        {"elements": base_elements, "ray": base_ray, "wavelengths": []},
        "invalid_wavelengths", 400,
    )
    _expect_error(
        {"elements": base_elements, "ray": base_ray, "wavelengths": [0.5, 0.5]},
        "invalid_wavelengths",
    )
    _expect_error(
        {"elements": base_elements, "ray": base_ray, "wavelengths": [-0.5]},
        "invalid_wavelengths",
    )
    response = client.post(
        "/trace",
        content=('{"elements":[{"type":"lens","params":{"f":50}}],'
                 '"ray":{"y":1,"u":0},"wavelengths":[0.5,1e999]}'),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_wavelengths"
    _expect_error(
        {"elements": base_elements, "ray": base_ray,
         "wavelengths": [0.45, 0.65],
         "reference_wavelengths": {"short": 0.45, "long": 0.55}},
        "invalid_reference_wavelength",
    )
    _expect_error(
        {"elements": base_elements, "ray": base_ray,
         "wavelengths": [0.45, 0.65],
         "reference_wavelengths": {"short": 0.65, "long": 0.45}},
        "invalid_reference_wavelength",
    )
    _expect_error(
        {"elements": base_elements, "ray": base_ray,
         "reference_wavelengths": {"short": 0.45, "long": 0.65}},
        "invalid_reference_wavelength",
    )


def test_unknown_material_rejected_before_trace():
    error = _expect_error(
        {
            "elements": [
                {"type": "lens", "params": {"f": 100.0}},
                {"type": "lens",
                 "params": {"r1": 30.0, "r2": -30.0, "glass": "UNOBTANIUM"}},
            ],
            "ray": {"y": 1.0, "u": 0.0},
            "wavelengths": [0.5],
        },
        "unknown_material", 404,
    )
    assert error["index"] == 2
    assert error["material"] == "UNOBTANIUM"

    error = _expect_error(
        {
            "elements": [{"type": "refract",
                          "params": {"R": 30.0, "n1": 1.0, "n2_material": "MISSING"}}],
            "ray": {"y": 1.0, "u": 0.0},
            "wavelengths": [0.5],
        },
        "unknown_material", 404,
    )
    assert error["index"] == 1
    assert error["material"] == "MISSING"
    assert "n2" in error["side"]


def test_legacy_element_errors_still_typed_and_indexed():
    error = _expect_error(
        {"elements": [
            {"type": "lens", "params": {"f": 50.0}},
            {"type": "space", "params": {"L": -2.0}},
        ], "ray": {"y": 1.0, "u": 0.0}, "wavelengths": [0.5]},
        "negative_spacing",
    )
    assert error["index"] == 2

    _expect_error(
        {"elements": [{"type": "lens", "params": {"f": 0.0}}],
         "ray": {"y": 1.0, "u": 0.0}, "wavelengths": [0.5]},
        "zero_focal_length",
    )
    _expect_error(
        {"elements": [{"type": "wormhole", "params": {}}],
         "ray": {"y": 1.0, "u": 0.0}, "wavelengths": [0.5]},
        "unknown_element",
    )
    _expect_error(
        {"elements": [{"type": "refract",
                       "params": {"R": 30.0, "n1": 0.0, "n2": 1.5}}],
         "ray": {"y": 1.0, "u": 0.0}, "wavelengths": [0.5]},
        "invalid_refractive_index",
    )


def test_plano_plano_glass_lens_rejected_as_zero_power():
    _expect_error(
        {"elements": [{"type": "lens",
                       "params": {"r1": None, "r2": None, "glass": "BK7"}}],
         "ray": {"y": 1.0, "u": 0.0}, "wavelengths": [0.5]},
        "zero_focal_length",
    )


def test_invalid_material_registration_rejected():
    response = client.post("/materials", json={"name": "bad"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "missing_field"

    response = client.post("/materials", json={
        "name": "bad2",
        "dispersion": {"type": "sampled", "samples": [[0.5, 1.5]]},
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_dispersion"

    response = client.post("/materials", json={"dispersion": {"type": "constant", "n": 1.5}})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_name"


def test_duplicate_material_rejected():
    response = client.post("/materials", json={
        "name": "BK7", "dispersion": {"type": "constant", "n": 1.5},
    })
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "duplicate_name"
    # 原档未被覆盖
    assert client.get("/materials/BK7").json()["description_kind"] == "sellmeier"


def test_roundtrip_holds_at_every_wavelength_with_glass():
    body = _trace({
        "elements": [
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2_material": "BK7"}},
            {"type": "space", "params": {"L": 12.0}},
            {"type": "refract", "params": {"R": -50.0, "n1_material": "BK7", "n2": 1.0}},
        ],
        "ray": {"y": 1.5, "u": 0.02},
        "s": None,
        "wavelengths": [0.4861, 0.5876, 0.6563],
    })
    for result in body["results"]:
        assert result["roundtrip"]["residual"] <= 1e-10
        assert result["roundtrip"]["restored_ray"]["y"] == pytest.approx(1.5, abs=1e-10)
        assert result["roundtrip"]["restored_ray"]["u"] == pytest.approx(0.02, abs=1e-10)
