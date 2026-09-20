"""多波长追迹与色差的 HTTP 接口测试。

谱线（nm）：F=486.1327（蓝）、d=587.5618（黄）、C=656.2725（红）。
手算基准值由同一组 Sellmeier 系数独立算出（见各测试注释）。
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import app

client_cm = TestClient(app)
client = client_cm.__enter__()

F_LINE = 486.1327
D_LINE = 587.5618
C_LINE = 656.2725

# N-BK7 / SF10 公开 Sellmeier 系数（λ 以 µm 计）
BK7_COEFF = {
    "B": [1.03961212, 0.231792344, 1.01046945],
    "C": [0.00600069867, 0.0200179144, 103.560653],
}
SF10_COEFF = {
    "B": [1.62153902, 0.256287842, 1.64447552],
    "C": [0.0122241457, 0.0595736775, 147.468793],
}

# 手算基准：BK7 折射率
N_F, N_D, N_C = 1.52237629, 1.51680003, 1.51432235

# 消色差双胶合（BK7+SF10，f_d=100mm）：由 Φ_d=0.01、Φ_F=Φ_C 解出
ACHROMAT_R1 = 50.0
ACHROMAT_R2 = -67.37074459703513
ACHROMAT_R3 = -259.8202730859087

# 同一块 Cauchy 玻璃的两种描述
CAUCHY_A, CAUCHY_B, CAUCHY_C = 1.5046, 4200.0, 1.5e8


def _cauchy_n(wavelength: float) -> float:
    return CAUCHY_A + CAUCHY_B / wavelength**2 + CAUCHY_C / wavelength**4


def _singlet(material: str, r1: float = 50.0, r2: float = -50.0) -> list[dict]:
    return [
        {"type": "refract", "params": {"R": r1, "n1": 1.0, "n2": material}},
        {"type": "refract", "params": {"R": r2, "n1": material, "n2": 1.0}},
    ]


def _achromat() -> list[dict]:
    return [
        {"type": "refract",
         "params": {"R": ACHROMAT_R1, "n1": 1.0, "n2": "TEST_CROWN"}},
        {"type": "refract",
         "params": {"R": ACHROMAT_R2, "n1": "TEST_CROWN", "n2": "TEST_FLINT"}},
        {"type": "refract",
         "params": {"R": ACHROMAT_R3, "n1": "TEST_FLINT", "n2": 1.0}},
    ]


@pytest.fixture(scope="module", autouse=True)
def registered_materials():
    client.post("/materials", json={
        "name": "TEST_CROWN",
        "coefficients": {"model": "sellmeier", **BK7_COEFF},
    })
    client.post("/materials", json={
        "name": "TEST_FLINT",
        "coefficients": {"model": "sellmeier", **SF10_COEFF},
    })
    client.post("/materials", json={
        "name": "TEST_CAUCHY",
        "coefficients": {"model": "cauchy",
                         "A": CAUCHY_A, "B": CAUCHY_B, "C": CAUCHY_C},
    })
    client.post("/materials", json={
        "name": "TEST_SAMPLES",
        "samples": [
            {"wavelength": w, "n": _cauchy_n(w)}
            for w in (404.6561, 486.1327, 546.074, 587.5618, 656.2725, 768.2)
        ],
    })
    client.post("/paths", json={
        "name": "crown_singlet",
        "elements": _singlet("TEST_CROWN"),
    })
    yield


def _trace(elements, wavelengths, s=300.0, ray=None, chromatic=None):
    body = {
        "elements": elements,
        "ray": ray or {"y": 1.0, "u": 0.0},
        "s": s,
        "wavelengths": wavelengths,
    }
    if chromatic is not None:
        body["chromatic"] = chromatic
    resp = client.post("/trace", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _index(material: str, wavelength: float) -> float:
    resp = client.get(f"/materials/{material}/index", params={"wavelength": wavelength})
    assert resp.status_code == 200, resp.text
    return resp.json()["n"]


# ---------------------------------------------------------------------------
# 材料档：登记、列出、点名取折射率
# ---------------------------------------------------------------------------

def test_material_registration_and_listing_shows_full_description():
    listing = client.get("/materials").json()["materials"]
    by_name = {m["name"]: m for m in listing}

    crown = by_name["TEST_CROWN"]
    assert crown["description_type"] == "coefficients"
    assert crown["model"] == "sellmeier"
    assert crown["coefficients"]["B"] == BK7_COEFF["B"]
    assert crown["coefficients"]["C"] == BK7_COEFF["C"]

    sampled = by_name["TEST_SAMPLES"]
    assert sampled["description_type"] == "samples"
    assert len(sampled["samples"]) == 6
    assert sampled["samples"][0]["wavelength"] == 404.6561
    # 采样点档同时给出拟合所得的 Cauchy 系数
    assert sampled["fitted"]["model"] == "cauchy"
    assert sampled["fitted"]["coefficients"]["A"] == pytest.approx(CAUCHY_A, rel=1e-9)

    # 内置示范玻璃已就位
    assert "N-BK7" in by_name and "SF10" in by_name


def test_material_index_endpoint_matches_hand_computed_bk7():
    assert _index("TEST_CROWN", F_LINE) == pytest.approx(N_F, abs=1e-6)
    assert _index("TEST_CROWN", D_LINE) == pytest.approx(N_D, abs=1e-6)
    assert _index("TEST_CROWN", C_LINE) == pytest.approx(N_C, abs=1e-6)
    # 正常色散：蓝 > 黄 > 红
    assert _index("TEST_CROWN", F_LINE) > _index("TEST_CROWN", D_LINE) \
        > _index("TEST_CROWN", C_LINE)


def test_same_glass_two_descriptions_give_same_index():
    for wavelength in (F_LINE, 500.0, D_LINE, 620.0, C_LINE, 700.0):
        n_coeff = _index("TEST_CAUCHY", wavelength)
        n_samples = _index("TEST_SAMPLES", wavelength)
        assert n_samples == pytest.approx(n_coeff, abs=1e-9)


def test_duplicate_material_rejected():
    resp = client.post("/materials", json={
        "name": "TEST_CROWN",
        "coefficients": {"model": "constant", "n": 1.5},
    })
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "duplicate_name"


@pytest.mark.parametrize("body,code", [
    ({"name": "x"}, "invalid_material"),                                  # 两种描述都不给
    ({"name": "x", "samples": [], "coefficients": {"model": "constant", "n": 1.5}},
     "invalid_material"),                                                 # 两种描述都给
    ({"name": "x", "samples": [{"wavelength": 550.0, "n": 1.5}]},
     "invalid_material"),                                                 # 采样点不足两个
    ({"name": "x", "samples": [{"wavelength": 550.0, "n": 1.5},
                               {"wavelength": 550.0, "n": 1.6}]},
     "invalid_material"),                                                 # 采样波长重复
    ({"name": "x", "samples": [{"wavelength": -550.0, "n": 1.5},
                               {"wavelength": 600.0, "n": 1.6}]},
     "invalid_wavelength"),                                               # 采样波长非正
    ({"name": "x", "samples": [{"wavelength": 550.0, "n": -1.5},
                               {"wavelength": 600.0, "n": 1.6}]},
     "invalid_refractive_index"),                                         # 采样折射率非正
    ({"name": "x", "coefficients": {"model": "unknown", "A": 1.5}},
     "invalid_material"),                                                 # 未知模型
    ({"name": "x", "coefficients": {"model": "sellmeier",
                                    "B": [1.0], "C": [-0.01]}},
     "invalid_material"),                                                 # Sellmeier C 非正
    ({"name": "x", "coefficients": {"model": "sellmeier", "B": [1.0]}},
     "missing_field"),                                                    # Sellmeier 缺 C
    ({"name": "x", "coefficients": {"model": "cauchy", "A": 1.5}},
     "missing_field"),                                                    # Cauchy 缺 B
    ({"coefficients": {"model": "constant", "n": 1.5}}, "missing_field"),  # 缺名字
])
def test_invalid_material_registration_rejected(body, code):
    resp = client.post("/materials", json=body)
    assert resp.status_code in (400, 422)
    assert resp.json()["error"]["code"] == code


def test_unknown_material_index_rejected():
    resp = client.get("/materials/no_such_glass/index",
                      params={"wavelength": 550.0})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_material"


def test_material_index_bad_wavelength_rejected():
    resp = client.get("/materials/TEST_CROWN/index")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "missing_field"

    resp = client.get("/materials/TEST_CROWN/index", params={"wavelength": "abc"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_wavelength"

    resp = client.get("/materials/TEST_CROWN/index", params={"wavelength": -5})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_wavelength"


# ---------------------------------------------------------------------------
# 回归：老式写死折射率的请求，加不加波长行为都必须不变
# ---------------------------------------------------------------------------

def test_legacy_single_wavelength_request_shape_unchanged():
    resp = client.post("/trace", json={
        "elements": [{"type": "lens", "params": {"f": 100.0}}],
        "ray": {"y": 5.0, "u": 0.01},
        "s": 200.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    # 老响应的键一个不少，也不多出色差字段
    for key in ("system_abcd", "imaging", "outgoing_ray", "roundtrip",
                "segments", "request"):
        assert key in body
    assert "results" not in body and "chromatic" not in body
    assert body["imaging"]["image_distance"] == pytest.approx(200.0)
    assert body["imaging"]["magnification"] == pytest.approx(-1.0)


def test_constant_index_elements_are_achromatic_across_wavelengths():
    # 写死 n1/n2 的折射面 = 无色散材料：各波长结果必须完全一致
    elements = [
        {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": 1.5}},
        {"type": "refract", "params": {"R": -50.0, "n1": 1.5, "n2": 1.0}},
    ]
    body = _trace(elements, [F_LINE, D_LINE, C_LINE], s=300.0)
    results = body["results"]
    assert [r["wavelength"] for r in results] == [F_LINE, D_LINE, C_LINE]

    first = results[0]
    for other in results[1:]:
        assert other["system_abcd"] == first["system_abcd"]
        assert other["imaging"]["image_distance"] == first["imaging"]["image_distance"]
        assert other["imaging"]["magnification"] == first["imaging"]["magnification"]
        assert other["outgoing_ray"] == first["outgoing_ray"]

    chromatic = body["chromatic"]
    assert chromatic["axial"]["difference"] == 0.0
    assert chromatic["lateral"]["difference"] == 0.0


def test_thin_lens_and_space_do_not_disperse():
    elements = [
        {"type": "lens", "params": {"f": 100.0}},
        {"type": "space", "params": {"L": 30.0}},
        {"type": "lens", "params": {"f": -80.0}},
    ]
    body = _trace(elements, [F_LINE, D_LINE, C_LINE, 700.0], s=500.0)
    results = body["results"]
    for other in results[1:]:
        assert other["system_abcd"] == results[0]["system_abcd"]
        assert other["imaging"] == results[0]["imaging"]
    assert body["chromatic"]["axial"]["difference"] == 0.0
    assert body["chromatic"]["lateral"]["difference"] == 0.0


# ---------------------------------------------------------------------------
# 单薄透镜配真实色散玻璃：蓝短红长，轴向色差可手算核对
# ---------------------------------------------------------------------------

def test_dispersive_singlet_spreads_image_distances_blue_short_red_long():
    body = _trace(_singlet("TEST_CROWN"), [F_LINE, D_LINE, C_LINE], s=300.0)
    results = {r["wavelength"]: r for r in body["results"]}

    # 每个波长的折射面都按该波长现算了折射率再拼矩阵
    assert results[F_LINE]["elements"][0]["params"]["n2"] == pytest.approx(N_F, abs=1e-6)
    assert results[C_LINE]["elements"][0]["params"]["n2"] == pytest.approx(N_C, abs=1e-6)

    # 薄透镜造镜者公式：f(λ) = 1 / ((n(λ)-1)(1/R1 - 1/R2)) = 25/(n-1)
    for wavelength, n in ((F_LINE, N_F), (D_LINE, N_D), (C_LINE, N_C)):
        f_expected = 25.0 / (n - 1.0)
        assert results[wavelength]["imaging"]["effective_focal_length"] == \
            pytest.approx(f_expected, abs=1e-5)
        # 同一物距下的高斯像距 1/s + 1/s' = 1/f
        s_prime_expected = 1.0 / (1.0 / f_expected - 1.0 / 300.0)
        assert results[wavelength]["imaging"]["image_distance"] == \
            pytest.approx(s_prime_expected, abs=1e-5)

    s_f = results[F_LINE]["imaging"]["image_distance"]
    s_d = results[D_LINE]["imaging"]["image_distance"]
    s_c = results[C_LINE]["imaging"]["image_distance"]
    # 蓝端焦距短、红端焦距长：三个像距依次拉开
    assert s_f < s_d < s_c

    # 轴向色差 = 短波像距 − 长波像距，符号为负，量值与手算一致
    chromatic = body["chromatic"]
    assert chromatic["basis"]["short_wavelength"] == F_LINE
    assert chromatic["basis"]["long_wavelength"] == C_LINE
    axial = chromatic["axial"]["difference"]
    assert axial == pytest.approx(s_f - s_c, abs=1e-12)
    assert axial == pytest.approx(-1.06408254, abs=1e-6)
    assert axial < 0.0

    # 倍率色差 = 短波放大率 − 长波放大率
    m_f = results[F_LINE]["imaging"]["magnification"]
    m_c = results[C_LINE]["imaging"]["magnification"]
    lateral = chromatic["lateral"]["difference"]
    assert lateral == pytest.approx(m_f - m_c, abs=1e-12)
    assert lateral == pytest.approx(0.00354694, abs=1e-6)


def test_infinity_object_reports_per_wavelength_back_focal_distance():
    body = _trace(_singlet("TEST_CROWN"), [F_LINE, D_LINE, C_LINE], s=None)
    results = {r["wavelength"]: r for r in body["results"]}

    # 每个波长各自给出后焦距，蓝短红长
    bfl = {w: results[w]["imaging"]["image_distance"] for w in (F_LINE, D_LINE, C_LINE)}
    assert bfl[F_LINE] == pytest.approx(25.0 / (N_F - 1.0), abs=1e-5)
    assert bfl[C_LINE] == pytest.approx(25.0 / (N_C - 1.0), abs=1e-5)
    assert bfl[F_LINE] < bfl[D_LINE] < bfl[C_LINE]

    # 轴向色差即后焦距在轴上拉开的距离
    axial = body["chromatic"]["axial"]
    assert axial["difference"] == pytest.approx(bfl[F_LINE] - bfl[C_LINE], abs=1e-12)
    assert axial["difference"] < 0.0
    # 物在无穷远时倍率色差不适用，须明说
    assert body["chromatic"]["lateral"]["difference"] is None
    assert "note" in body["chromatic"]["lateral"]


# ---------------------------------------------------------------------------
# 消色差双胶合：长短两谱线焦度合一，中间谱线留二级光谱
# ---------------------------------------------------------------------------

def test_achromatic_doublet_nulls_axial_color_but_keeps_secondary_spectrum():
    body = _trace(_achromat(), [F_LINE, D_LINE, C_LINE], s=None)
    results = {r["wavelength"]: r for r in body["results"]}
    bfl = {w: results[w]["imaging"]["image_distance"] for w in (F_LINE, D_LINE, C_LINE)}

    # 设计目标：f_d = 100mm，F 与 C 焦度合一
    assert bfl[D_LINE] == pytest.approx(100.0, abs=1e-6)
    assert bfl[F_LINE] == pytest.approx(bfl[C_LINE], abs=1e-9)

    # 轴向色差（F−C）在容差内归零
    axial = body["chromatic"]["axial"]
    assert abs(axial["difference"]) < 1e-6
    assert axial["difference"] == pytest.approx(bfl[F_LINE] - bfl[C_LINE], abs=1e-15)

    # 中间谱线留残余（二级光谱）：d 线后焦距偏离 F/C 约 -0.0529mm
    secondary = bfl[D_LINE] - bfl[F_LINE]
    assert secondary == pytest.approx(-0.0529465, abs=1e-4)
    assert abs(secondary) > 1e-3  # 明显不为零：残余确实在


def test_achromat_finite_object_distance_also_achromatic():
    body = _trace(_achromat(), [F_LINE, D_LINE, C_LINE], s=10000.0)
    results = {r["wavelength"]: r for r in body["results"]}
    s_f = results[F_LINE]["imaging"]["image_distance"]
    s_c = results[C_LINE]["imaging"]["image_distance"]
    s_d = results[D_LINE]["imaging"]["image_distance"]
    assert abs(body["chromatic"]["axial"]["difference"]) < 1e-6
    assert abs(s_d - s_f) > 1e-3  # 二级光谱仍在
    assert s_f == pytest.approx(s_c, abs=1e-9)


# ---------------------------------------------------------------------------
# 同一玻璃两种描述：追迹结果不随描述方式漂移
# ---------------------------------------------------------------------------

def test_trace_results_do_not_depend_on_material_description():
    wavelengths = [F_LINE, D_LINE, C_LINE]
    by_coeff = _trace(_singlet("TEST_CAUCHY"), wavelengths, s=300.0)
    by_samples = _trace(_singlet("TEST_SAMPLES"), wavelengths, s=300.0)
    for r_coeff, r_samples in zip(by_coeff["results"], by_samples["results"]):
        assert r_samples["system_abcd"] == pytest.approx(r_coeff["system_abcd"], abs=1e-9)
        assert r_samples["imaging"]["image_distance"] == pytest.approx(
            r_coeff["imaging"]["image_distance"], abs=1e-7)


# ---------------------------------------------------------------------------
# 波长之间严格隔离：每个波长的矩阵等于该波长折射率独立算出的矩阵
# ---------------------------------------------------------------------------

def test_wavelength_results_are_isolated_and_independent():
    wavelengths = [F_LINE, D_LINE, C_LINE]
    body = _trace(_singlet("TEST_CROWN"), wavelengths, s=300.0)
    for result in body["results"]:
        w = result["wavelength"]
        n_here = _index("TEST_CROWN", w)
        # 用写死折射率的老式元件独立重算同一波长，必须逐位一致
        reference = _trace([
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": n_here}},
            {"type": "refract", "params": {"R": -50.0, "n1": n_here, "n2": 1.0}},
        ], [w], s=300.0)["results"][0]
        assert result["system_abcd"] == reference["system_abcd"]
        assert result["outgoing_ray"] == reference["outgoing_ray"]
        assert result["imaging"]["image_distance"] == \
            reference["imaging"]["image_distance"]
    # 三个波长的系统矩阵两两不同（色散确实进了矩阵）
    abcds = [tuple(r["system_abcd"][k] for k in "ABCD") for r in body["results"]]
    assert len(set(abcds)) == 3


# ---------------------------------------------------------------------------
# 点名光路的多波长追迹
# ---------------------------------------------------------------------------

def test_named_path_multi_wavelength_trace():
    resp = client.post("/paths/crown_singlet/trace", json={
        "ray": {"y": 1.0, "u": 0.0},
        "s": 300.0,
        "wavelengths": [F_LINE, D_LINE, C_LINE],
        "chromatic": {"short_wavelength": F_LINE, "long_wavelength": C_LINE},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path_name"] == "crown_singlet"
    assert len(body["results"]) == 3
    assert body["chromatic"]["basis"]["source"] == "request"
    assert body["chromatic"]["axial"]["difference"] < 0.0

    # 与内联追迹同一光路结果一致
    inline = _trace(_singlet("TEST_CROWN"), [F_LINE, D_LINE, C_LINE], s=300.0)
    for named, inl in zip(body["results"], inline["results"]):
        assert named["system_abcd"] == inl["system_abcd"]


def test_chromatic_basis_defaults_to_min_max_wavelengths():
    body = _trace(_singlet("TEST_CROWN"), [C_LINE, F_LINE, D_LINE], s=300.0)
    basis = body["chromatic"]["basis"]
    assert basis["short_wavelength"] == F_LINE
    assert basis["long_wavelength"] == C_LINE
    assert basis["source"] == "default_min_max"


# ---------------------------------------------------------------------------
# 错误：未登记材料、缺波长、坏波长、坏色差基准
# ---------------------------------------------------------------------------

def test_unknown_material_in_element_rejected_with_index():
    resp = client.post("/trace", json={
        "elements": [
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": "no_such"}},
            {"type": "refract", "params": {"R": -50.0, "n1": "no_such", "n2": 1.0}},
        ],
        "ray": {"y": 1.0, "u": 0.0},
        "s": 300.0,
        "wavelengths": [D_LINE],
    })
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "unknown_material"
    assert err["index"] == 1  # 错在第一个元件


def test_material_reference_without_wavelengths_rejected():
    resp = client.post("/trace", json={
        "elements": _singlet("TEST_CROWN"),
        "ray": {"y": 1.0, "u": 0.0},
        "s": 300.0,
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "missing_field"

    # 点名光路同样守这条规矩
    resp = client.post("/paths/crown_singlet/trace",
                       json={"ray": {"y": 1.0, "u": 0.0}, "s": 300.0})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "missing_field"


@pytest.mark.parametrize("wavelengths", [[], [-550.0], [0.0], ["green"]])
def test_invalid_wavelengths_rejected(wavelengths):
    resp = client.post("/trace", json={
        "elements": [{"type": "lens", "params": {"f": 100.0}}],
        "ray": {"y": 1.0, "u": 0.0},
        "wavelengths": wavelengths,
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] in ("invalid_wavelength", "non_finite_number")


def test_non_finite_wavelength_rejected():
    # 原始 JSON 体里带 NaN 记号，服务端解析为非有限数后退回
    resp = client.post(
        "/trace",
        content='{"elements":[{"type":"lens","params":{"f":100}}],'
                '"ray":{"y":1,"u":0},"wavelengths":[550.0, NaN]}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "non_finite_number"


def test_chromatic_basis_must_be_traced_wavelengths():
    resp = client.post("/trace", json={
        "elements": _singlet("TEST_CROWN"),
        "ray": {"y": 1.0, "u": 0.0},
        "s": 300.0,
        "wavelengths": [F_LINE, C_LINE],
        "chromatic": {"short_wavelength": F_LINE, "long_wavelength": 600.0},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_chromatic_basis"


def test_chromatic_basis_short_must_be_below_long():
    resp = client.post("/trace", json={
        "elements": _singlet("TEST_CROWN"),
        "ray": {"y": 1.0, "u": 0.0},
        "s": 300.0,
        "wavelengths": [F_LINE, C_LINE],
        "chromatic": {"short_wavelength": C_LINE, "long_wavelength": F_LINE},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_chromatic_basis"


def test_single_wavelength_reports_no_chromatic():
    body = _trace(_singlet("TEST_CROWN"), [D_LINE], s=300.0)
    assert len(body["results"]) == 1
    assert body["chromatic"]["axial"] is None
    assert body["chromatic"]["lateral"] is None
    assert "note" in body["chromatic"]


# ---------------------------------------------------------------------------
# 并行两组波长追迹互不污染
# ---------------------------------------------------------------------------

def test_parallel_wavelength_traces_do_not_cross_contaminate():
    async def call(elements, wavelengths, s):
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.post("/trace", json={
                "elements": elements,
                "ray": {"y": 1.0, "u": 0.0},
                "s": s,
                "wavelengths": wavelengths,
            })

    set_a = [F_LINE, D_LINE, C_LINE]
    set_b = [F_LINE, C_LINE]  # 双胶合的消色差基准对

    async def scenario():
        return await asyncio.gather(
            call(_singlet("TEST_CROWN"), set_a, 300.0),
            call(_achromat(), set_b, None),
            call(_singlet("TEST_CROWN"), set_a, 300.0),
            call(_achromat(), set_b, None),
        )

    responses = asyncio.run(scenario())
    for r in responses:
        assert r.status_code == 200, r.text
    a1, b1, a2, b2 = (r.json() for r in responses)

    # 各回各的波长清单
    assert a1["wavelengths"] == set_a and a2["wavelengths"] == set_a
    assert b1["wavelengths"] == set_b and b2["wavelengths"] == set_b
    # 同请求两次结果一致
    assert a1["results"] == a2["results"]
    assert b1["results"] == b2["results"]
    # 单透镜有色散、双胶合消色差，互不串扰
    assert a1["chromatic"]["axial"]["difference"] < 0.0
    assert abs(b1["chromatic"]["axial"]["difference"]) < 1e-6
    # 并行结果与串行重算一致
    serial = _trace(_singlet("TEST_CROWN"), set_a, s=300.0)
    assert a1["results"] == serial["results"]
