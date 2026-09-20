# 近轴光路追迹服务

把薄透镜、空气间隔、球面折射面串成一条近轴光路的 HTTP 服务。
给定入射高度 `y` 与近轴角 `u`，返回出射光线、每段矩阵、整段系统矩阵，
以及给定物距对应的像距与横向放大率。

- Python 3.12 + FastAPI，仅提供 HTTP 接口，无前端、无账户
- 具名光路档写入进程内 SQLite（WAL），可登记 / 列出 / 按名追迹
- 内置共焦望远镜示范档 `telescope_demo`（f1=100、f2=50、间距 150，放大率 -0.5，C≈0）
- 也可在当次请求里内联元件清单，只追一次不落库

## 约定

光线列向量全程为 `(y, u)`（高度、近轴角），不中途切换方向余弦。

| 元件 | 矩阵 |
|------|------|
| 空气间隔传播 L | `[[1, L], [0, 1]]` |
| 薄透镜 f（f≠0） | `[[1, 0], [-1/f, 1]]` |
| 球面折射 (R, n1, n2) | `[[1, 0], [(n1-n2)/(n2 R)], n1/n2]` |

- 系统矩阵按光线前进方向连乘：`M = M_n … M_2 M_1`
- 空气系统（只有间隔和薄透镜）：每段及整段乘积行列式在 `1e-12` 容差内为 1
- 物在第一面左侧、物距 `s` 取正；`s = null` 表示物在无穷远
- 共轭条件：`s' = -(A s + B)/(C s + D)`，横向放大率 `m = 1/(C s + D)`
  （单透镜代入即还原 `1/s + 1/s' = 1/f`）
- 平行光入射像落在后焦面，空气中有效焦距 `EFL = -1/C`
- 共焦望远镜（间距 = f1+f2）`C≈0` 判为无焦，角放大率 `-f1/f2`；
  间距一旦拉开或缩短，C 重新非零，焦度重新出现
- 折射面 `n1==n2` 退化为单位阵；空气进玻璃再回空气，整段行列式回到 1
- 任何合法光路正向追迹后按相反次序乘各段逆矩阵，入射 `(y,u)` 在 `1e-10` 内还原

## 构建与运行

```bash
docker build -t paraxial-optics .
docker run --rm -p 8000:8000 paraxial-optics
```

数据库默认在容器卷 `/data/optics_paths.db`（可用环境变量 `OPTICS_DB` 覆盖）。

本地直跑：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口

### `GET /health`
健康检查。

### `POST /paths` — 登记具名档
```json
{
  "name": "doublet",
  "elements": [
    {"type": "lens",  "params": {"f": 100}},
    {"type": "space", "params": {"L": 30}},
    {"type": "lens",  "params": {"f": 50}}
  ]
}
```
元件类型：`space`（别名 `air_space`/`propagate`，参数 `L`）、
`lens`（参数 `f`）、`refract`（参数 `R`、`n1`、`n2`）。
重名返回 `409 duplicate_name`。

### `GET /paths` — 列出全部档（元件种类与参数全文）

### `POST /paths/{name}/trace` — 点名追迹
```json
{"ray": {"y": 5.0, "u": 0.01}, "s": 200}
```
未登记档名返回 `404 unknown_path`，不猜测。

### `POST /trace` — 内联清单，只用一次
```json
{
  "elements": [{"type": "lens", "params": {"f": 100}}],
  "ray": {"y": 5.0, "u": 0.01},
  "s": null
}
```

追迹响应含：每段矩阵与行列式、系统 A/B/C/D、整段行列式、有效焦距、
给定物距的像距与横向放大率、出射 `y/u`、往返还原残差。

错误统一形如 `{"error": {"code": "...", "message": "...", ...}}`，
带类型的 code 包括：`unknown_element`、`negative_spacing`、
`zero_focal_length`、`invalid_refractive_index`、`non_finite_number`、
`missing_field`、`unknown_path`、`duplicate_name`、`invalid_body` 等。

## 模块划分

| 文件 | 职责 |
|------|------|
| `app/elements.py` | 元件矩阵与 2×2 矩阵运算 |
| `app/systems.py` | 系统连乘、正向/反向追迹、物像求解 |
| `app/storage.py` | 具名档的进程内 SQLite 存取 |
| `app/validation.py` | 入参检查与规范化 |
| `app/errors.py` | 带类型的错误 |
| `app/demo.py` | 内置共焦望远镜示范档 |
| `app/main.py` | FastAPI 路由 |

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

钉死的场景：空气系统行列式为 1、单透镜还原高斯公式、共焦望远镜 C≈0 且放大率为负、
错开间距焦度重现、正向再逆向还原入射光线、进出玻璃行列式回到 1、等折射率退化、
焦距为零/未知元件/未知档名被拒、并行两档追迹互不污染。
