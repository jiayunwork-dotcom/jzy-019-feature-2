"""HTTP 接口测试：登记、列出、点名追迹、一次性追迹、并行隔离。"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.demo import TELESCOPE_DEMO_NAME
from app.main import app

# 作为上下文管理器进入时触发 lifespan startup（登记内置示范档）
client_cm = TestClient(app)
client = client_cm.__enter__()


def _lens_space_lens(f1, spacing, f2):
    return [
        {"type": "lens", "params": {"f": f1}},
        {"type": "space", "params": {"L": spacing}},
        {"type": "air_space", "params": {"length": 0.0}},
        {"type": "lens", "params": {"f": f2}},
    ]


@pytest.fixture(scope="module", autouse=True)
def registered_paths():
    client.post("/paths", json={
        "name": "single_f100",
        "elements": [{"type": "lens", "params": {"f": 100.0}}],
    })
    client.post("/paths", json={
        "name": "double_a",
        "elements": _lens_space_lens(100.0, 30.0, 50.0),
    })
    client.post("/paths", json={
        "name": "double_b",
        "elements": _lens_space_lens(200.0, 80.0, 75.0),
    })
    yield


# ---------------------------------------------------------------------------
# 登记 / 列出 / 示范档
# ---------------------------------------------------------------------------

def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_path_seeded_and_lists_full_params():
    listing = client.get("/paths").json()["paths"]
    names = [item["name"] for item in listing]
    assert TELESCOPE_DEMO_NAME in names

    demo = next(item for item in listing if item["name"] == TELESCOPE_DEMO_NAME)
    kinds = [e["type"] for e in demo["elements"]]
    assert kinds == ["lens", "space", "lens"]
    # 元件种类与参数全文
    assert demo["elements"][0]["params"]["f"] == 100.0
    assert demo["elements"][1]["params"]["L"] == 150.0
    assert demo["elements"][2]["params"]["f"] == 50.0


def test_demo_trace_is_afocal_negative_magnification():
    resp = client.post(f"/paths/{TELESCOPE_DEMO_NAME}/trace",
                       json={"ray": {"y": 2.0, "u": 0.0}, "s": 1000.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert abs(body["system_abcd"]["C"]) < 1e-12
    imaging = body["imaging"]
    assert imaging["afocal"] is True
    assert imaging["magnification"] < 0.0
    assert imaging["magnification"] == pytest.approx(-0.5)
    # 有限物距共轭：s' = -(A s + B)/D = -((-0.5)*1000 + 150)/(-2) = -175（虚像）
    assert imaging["image_distance"] == pytest.approx(-175.0, abs=1e-9)
    # 出射光线：轴上光线 y=0 检验角放大率 u_out = D u_in = -f1/f2 * u_in
    resp_ang = client.post(f"/paths/{TELESCOPE_DEMO_NAME}/trace",
                           json={"ray": {"y": 0.0, "u": 0.01}, "s": 1000.0})
    assert resp_ang.json()["outgoing_ray"]["u"] == pytest.approx(-0.02, abs=1e-12)

    # 物在无穷远：像距为空，仍为无焦
    resp_inf = client.post(f"/paths/{TELESCOPE_DEMO_NAME}/trace",
                           json={"ray": {"y": 2.0, "u": 0.0}, "s": None})
    imaging_inf = resp_inf.json()["imaging"]
    assert imaging_inf["image_distance"] is None
    assert imaging_inf["magnification"] is None


def test_duplicate_name_rejected():
    resp = client.post("/paths", json={
        "name": "single_f100",
        "elements": [{"type": "lens", "params": {"f": 100.0}}],
    })
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "duplicate_name"


# ---------------------------------------------------------------------------
# 点名追迹：结果含 A/B/C/D、EFL、像距、放大率、出射 y/u
# ---------------------------------------------------------------------------

def test_named_trace_returns_full_result_and_gaussian_image():
    resp = client.post("/paths/single_f100/trace",
                       json={"ray": {"y": 5.0, "u": 0.01}, "s": 200.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    abcd = body["system_abcd"]
    assert (abcd["A"], abcd["B"], abcd["C"], abcd["D"]) == (1.0, 0.0, -0.01, 1.0)
    assert body["imaging"]["effective_focal_length"] == pytest.approx(100.0)
    # 1/200 + 1/s' = 1/100 -> s' = 200
    assert body["imaging"]["image_distance"] == pytest.approx(200.0)
    assert body["imaging"]["magnification"] == pytest.approx(-1.0)
    # 出射 y/u
    assert body["outgoing_ray"]["y"] == pytest.approx(5.0)
    assert body["outgoing_ray"]["u"] == pytest.approx(0.01 - 5.0 / 100.0)
    # 每段矩阵都在
    assert body["segments"][0]["matrix"]["C"] == -0.01
    # 往返还原
    assert body["roundtrip"]["residual"] <= 1e-10


def test_infinity_object_hits_back_focal_plane():
    resp = client.post("/paths/single_f100/trace",
                       json={"ray": {"y": 5.0, "u": 0.0}, "s": None})
    body = resp.json()
    assert body["imaging"]["image_distance"] == pytest.approx(100.0)
    assert body["imaging"]["magnification"] == 0.0


def test_inline_trace_once_is_not_registered():
    resp = client.post("/trace", json={
        "elements": [{"type": "lens", "params": {"f": 100.0 / 3.0}}],
        "ray": {"y": 1.0, "u": 0.0},
        "s": 100.0,
    })
    assert resp.status_code == 200
    expected = 1.0 / (3.0 / 100.0 - 1.0 / 100.0)  # 1/s' = 1/f - 1/s
    assert resp.json()["imaging"]["image_distance"] == pytest.approx(expected, abs=1e-9)

    names = [p["name"] for p in client.get("/paths").json()["paths"]]
    assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# 未知档名、零焦距、未知元件：HTTP 层带类型的错误
# ---------------------------------------------------------------------------

def test_unknown_path_name_rejected():
    resp = client.post("/paths/does_not_exist/trace",
                       json={"ray": {"y": 1.0, "u": 0.0}})
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "unknown_path"
    assert "does_not_exist" in err["message"]


def test_zero_focal_length_rejected_over_http():
    resp = client.post("/trace", json={
        "elements": [{"type": "lens", "params": {"f": 0.0}}],
        "ray": {"y": 1.0, "u": 0.0},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "zero_focal_length"


def test_unknown_element_rejected_over_http():
    resp = client.post("/trace", json={
        "elements": [{"type": "grating", "params": {}}],
        "ray": {"y": 1.0, "u": 0.0},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unknown_element"


def test_negative_spacing_and_bad_index_rejected():
    resp = client.post("/trace", json={
        "elements": [
            {"type": "lens", "params": {"f": 50.0}},
            {"type": "space", "params": {"L": -1.0}},
        ],
        "ray": {"y": 1.0, "u": 0.0},
    })
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "negative_spacing"
    assert err["index"] == 2


def test_non_positive_index_and_non_finite_rejected():
    resp = client.post("/trace", json={
        "elements": [{"type": "refract",
                      "params": {"R": 30.0, "n1": 1.0, "n2": -1.5}}],
        "ray": {"y": 1.0, "u": 0.0},
    })
    assert resp.json()["error"]["code"] == "invalid_refractive_index"

    # 原始 JSON 体里带 Infinity 记号，服务端解析为非有限数后退回
    resp = client.post(
        "/trace",
        content='{"elements":[{"type":"lens","params":{"f":1e999}}],"ray":{"y":1,"u":0}}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "non_finite_number"


def test_missing_ray_field_rejected():
    resp = client.post("/paths/single_f100/trace", json={"s": 200.0})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "missing_field"


# ---------------------------------------------------------------------------
# 并行两档追迹结果互不污染
# ---------------------------------------------------------------------------

def test_parallel_traces_do_not_cross_contaminate():
    async def call(name, y):
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.post(
                f"/paths/{name}/trace",
                json={"ray": {"y": y, "u": 0.0}, "s": 500.0},
            )

    async def scenario():
        results = await asyncio.gather(
            call("double_a", 1.0),
            call("double_b", 2.0),
            call("double_a", 3.0),
            call("double_b", 4.0),
        )
        return results

    responses = asyncio.run(scenario())
    for r in responses:
        assert r.status_code == 200, r.text

    a1, b1, a2, b2 = (r.json() for r in responses)

    # 两次同档结果彼此一致（除入参与出射高度外）
    assert a1["system_abcd"] == a2["system_abcd"]
    assert b1["system_abcd"] == b2["system_abcd"]
    # 两档的系统矩阵不同，且各归各档
    assert a1["system_abcd"] != b1["system_abcd"]
    assert a1["path_name"] == "double_a"
    assert b1["path_name"] == "double_b"
    # 出射高度只由本档入射高度决定
    assert a1["outgoing_ray"]["y"] * 3.0 == pytest.approx(a2["outgoing_ray"]["y"])
    assert b1["outgoing_ray"]["y"] * 2.0 == pytest.approx(b2["outgoing_ray"]["y"])


def test_refractive_path_roundtrip_over_http():
    resp = client.post("/trace", json={
        "elements": [
            {"type": "refract", "params": {"R": 50.0, "n1": 1.0, "n2": 1.5}},
            {"type": "space", "params": {"L": 10.0}},
            {"type": "refract", "params": {"R": -50.0, "n1": 1.5, "n2": 1.0}},
        ],
        "ray": {"y": 1.5, "u": 0.02},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["determinant"] == pytest.approx(1.0, abs=1e-12)
    assert body["roundtrip"]["residual"] <= 1e-10
