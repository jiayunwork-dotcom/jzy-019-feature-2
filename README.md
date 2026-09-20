# 近轴光路追迹服务（多波长 / 色差）

把薄透镜、空气间隔、球面折射面串成一条近轴光路的 HTTP 服务。
给定入射高度 `y` 与近轴角 `u`，可在一组波长上分别追迹，逐波长返回
出射光线、每段矩阵、整段系统矩阵、像距与放大率，并归约出轴向/倍率色差。

- Python 3.12 + FastAPI，仅提供 HTTP 接口，无前端、无账户
- **具名色散材料档**写入进程内 SQLite（WAL），支持四类描述：
  `constant` / `cauchy` / `sellmeier` / `sampled`（参考谱线采样点按柯西色散关系拟合，
  内插外推均走解析曲线，不线性硬连）
- **具名光路档**同样落 SQLite，可登记 / 列出 / 按名追迹
- 一次请求吃一串波长：逐波长独立构造系统矩阵、独立解共轭像面，矩阵与光线严格隔离
- 跨波长色差：轴向（像距/后焦距之差）与倍率（放大率之差），并点名基准的长、短谱线
- 旧写法完全保留：折射面直接写死 `n1`/`n2`、薄透镜直接写 `f` 等价于全波段恒定材料，
  不带 `wavelengths` 的旧请求响应结构与数值不变
- 内置示范档：`telescope_demo`（共焦望远镜）、`single_bk7_lens`（BK7 单透镜，
  蓝短红长）、`achromat_doublet`（BK7+F2 消色差双胶合，F/C 消色差、d 线留二级光谱）

## 约定

光线列向量全程为 `(y, u)`（高度、近轴角），不中途切换方向余弦。
**长度单位为毫米，波长单位为微米（μm）。**

| 元件 | 矩阵 |
|------|------|
| 空气间隔传播 L | `[[1, L], [0, 1]]` |
| 薄透镜 f（f≠0） | `[[1, 0], [-1/f, 1]]` |
| 球面折射 (R, n1, n2) | `[[1, 0], [(n1-n2)/(n2 R)], n1/n2]`（`R=null` 为平面） |

- 系统矩阵按光线前进方向连乘：`M = M_n … M_2 M_1`
- 空气系统（只有间隔和薄透镜）：每段及整段乘积行列式在 `1e-12` 容差内为 1
- 物在第一面左侧、物距 `s` 取正；`s = null` 表示物在无穷远
- 共轭条件：`s' = -(A s + B)/(C s + D)`，横向放大率 `m = 1/(C s + D)`
  （单透镜代入即还原 `1/s + 1/s' = 1/f`）
- 平行光入射像落在后焦面：`BFL = -A/C`，空气中 `EFL = -1/C`
- 挂玻璃薄透镜在空气中按造镜者公式定焦：`1/f = (n(λ)-1)(1/r1 - 1/r2)`
- 共焦望远镜（间距 = f1+f2）`C≈0` 判为无焦，角放大率 `-f1/f2`
- 折射面 `n1==n2` 退化为单位阵；正向再反向乘各段逆矩阵，入射 `(y,u)` 在 `1e-10` 内还原

## 构建与运行

```bash
docker build -t paraxial-optics .
docker run --rm -p 8000:8000 paraxial-optics
```

数据库默认在容器卷 `/data/optics_paths.db`（可用环境变量 `OPTICS_DB` 覆盖），
材料档与光路档同库存放。容器启动即对外开放全部接口并幂等写入内置材料与示范档。

本地直跑：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口

### 材料档

#### `POST /materials` — 登记具名材料

```json
{ "name": "my_glass", "dispersion": { "type": "sellmeier",
  "B": [1.03961217, 0.231792344, 1.01046945],
  "C": [0.00600069867, 0.0200179144, 103.560653] } }
```

四种描述方式（波长一律 μm）：

```json
{ "type": "constant", "n": 1.5 }
{ "type": "cauchy", "coefficients": [c0, c1, c2] }
// n(λ) = c0 + c1/λ² + c2/λ⁴（1~4 项）

{ "type": "sellmeier", "B": [B1,B2,B3], "C": [C1,C2,C3] }
// n² = 1 + Σ B_i λ²/(λ²-C_i)

{ "type": "sampled",
  "samples": [[0.4861, 1.52238], [0.5876, 1.51680], [0.6563, 1.51432]],
  "terms": 3 }
// ≥3 条 [波长, 折射率]；按柯西关系最小二乘拟合（点数=项数时为精确插值），
// 列出时回报全部采样点、拟合系数与残差；内外插均走该解析曲线
```

- `GET /materials`：列出全部材料（描述方式 + 全部系数/采样点）
- `GET /materials/{name}`：点名取一档
- `GET /materials/{name}/index?wavelength=0.55`：现算某波长折射率
- 重名 `409 duplicate_name`；非法色散 `422 invalid_dispersion` /
  `invalid_refractive_index`

内置材料：`air`（constant 1.0）、`BK7`、`F2`（Sellmeier）。

### 光路档与追迹

#### `POST /paths` — 登记具名光路

```json
{
  "name": "lens_group",
  "elements": [
    {"type": "lens",  "params": {"r1": 31.912, "r2": -75.0, "glass": "BK7"}},
    {"type": "lens",  "params": {"r1": -75.0, "r2": 128.696, "glass": "F2"}}
  ]
}
```

元件：

- `space`（别名 `air_space`/`propagate`）：`params: {"L": ...}`，`L>=0`
- `lens`：要么 `{"f": ...}`（无色散，旧写法），要么
  `{"r1": ..., "r2": ..., "glass": "材料名"}`；曲率可给 `null` 表示平面
- `refract`：`{"R": ..., "n1": ..., "n2": ...}`（旧写法，正折射率），
  或把任一侧换成具名材料：
  `{"R": ..., "n1": 1.0, "n2_material": "BK7"}`（也支持 `n1_material`）

`GET /paths` 列全档；`POST /paths/{name}/trace` 点名追迹。

#### 多波长追迹（`POST /trace` 与 `POST /paths/{name}/trace` 同构）

```json
{
  "elements": [{"type": "lens",
                "params": {"r1": 103.36, "r2": -103.36, "glass": "BK7"}}],
  "ray": {"y": 1.0, "u": 0.0},
  "s": 300.0,
  "wavelengths": [0.4861, 0.5876, 0.6563],
  "reference_wavelengths": {"short": 0.4861, "long": 0.6563}
}
```

- `wavelengths` 非空、逐项为正的有限值、不得重复；结果按波长升序
- `reference_wavelengths` 可省，缺省取波长组的最短与最长；两基准必须在清单内且 short<long
- 响应：

```json
{
  "wavelengths": [0.4861, 0.5876, 0.6563],
  "count": 3,
  "results": [
    { "wavelength": 0.4861,
      "system_abcd": {"A": ..., "B": ..., "C": ..., "D": ...},
      "determinant": 1.0,
      "segments": [{"index": 1, "type": "lens",
                    "matrix": {...}, "indices": {"lens_index": 1.52238,
                                                 "surround_index": 1.0}}],
      "imaging": {"image_distance": 147.61, "magnification": -0.492,
                  "effective_focal_length": 98.93, "afocal": false},
      "outgoing_ray": {"y": ..., "u": ...},
      "roundtrip": {"residual": ..., "restored_ray": {...}} }
  ],
  "chromatic_aberration": {
    "reference_wavelengths": {"short": 0.4861, "long": 0.6563, "unit": "um"},
    "axial_chromatic_aberration": 3.47696,
    "axial_kind": "image_distance_difference",
    "axial_sign_convention": "s'(long) - s'(short)；正常色散为正",
    "transverse_chromatic_aberration": -0.01159,
    "short_image_distance": 147.61, "long_image_distance": 151.09
  }
}
```

- 物在无穷远（`s: null`）时每个波长独立给出后焦距，
  `axial_kind` 为 `back_focal_length_difference`
- 各波长结果对象互不共享；同一批并行请求的材料、矩阵、光线互不串扰

#### 旧单波长请求（不带 `wavelengths`）

```json
{"elements": [{"type": "lens", "params": {"f": 100}}],
 "ray": {"y": 5.0, "u": 0.01}, "s": 200}
```

响应仍是旧结构（顶层 `system_abcd`/`imaging`/`outgoing_ray`/`segments`/`roundtrip`），
数值与旧版完全一致。

### 错误

统一形如 `{"error": {"code": "...", "message": "...", ...}}`，元件类错误带 `index`
（第几个元件）。code 包括：`unknown_element`、`negative_spacing`、
`zero_focal_length`、`invalid_refractive_index`、`non_finite_number`、
`missing_field`、`unknown_material`、`invalid_material_reference`、
`invalid_dispersion`、`invalid_wavelengths`、`invalid_reference_wavelength`、
`unknown_path`、`duplicate_name`、`invalid_body` 等，全部在追迹前退回。

## 手算核对场景

- **BK7 单透镜**（对称双凸，d 线 f=100，s=300）：Sellmeier 算出
  f_F≈98.93、f_d=100.00、f_C≈100.48，像距 147.61 / 150.00 / 151.09，
  轴向色差 = 151.09−147.61 ≈ **+3.48 mm**（蓝短红长）
- **消色差双胶合**（BK7 冠片 r1=31.912, r2=−75；F2 火石 r1=−75, r2=128.696）：
  F/C 两线 f 均为 100.050（轴向色差 ~1e-10），d 线 f=100.000（二级光谱）
- **描述等价**：同一条柯西曲线分别登记为 `cauchy` 系数档与 `sampled` 采样档，
  折射率一致到 ~1e-15，追迹结果不随描述方式漂移

## 模块划分

| 文件 | 职责 |
|------|------|
| `app/dispersion.py` | 波长→折射率色散模型（constant/cauchy/sellmeier/sampled 拟合），纯数学 |
| `app/materials.py` | 材料档对象、登记体规范化、单次追迹内的材料解析器 |
| `app/elements.py` | 元件档与 2×2 矩阵运算、造镜者公式 |
| `app/systems.py` | 单波长系统连乘、正向/反向追迹、物像求解 |
| `app/multispectral.py` | 多波长：逐波长解析材料→连乘→解像面（波长隔离） |
| `app/chromatic.py` | 跨波长轴向/倍率色差归约（只消费结果，不追迹） |
| `app/storage.py` | 材料档/光路档的进程内 SQLite 存取 |
| `app/validation.py` | 入参检查与规范化（元件、波长组、基准谱线、物距） |
| `app/errors.py` | 带类型的错误 |
| `app/demo.py` | 内置材料与示范光路 |
| `app/main.py` | FastAPI 路由与追迹编排 |

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

钉死的行为：无色散材料各波长像距完全一致、色差为零；BK7 单透镜三谱线蓝短红长、
轴向色差符号与量值；消色差双胶合 F/C 色差归零、d 线留残余；采样点档与解析系数档
折射率一致、追迹不漂移；旧单波长写死折射率请求行为不变；空气系统行列式为 1、
单透镜高斯公式、共焦示范档 C≈0 放大率为负、往返还原、等折射率退化、
各类非法输入带类型与元件序号被拒；并行多组波长追迹互不污染。
