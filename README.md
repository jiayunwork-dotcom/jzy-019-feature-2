# 近轴光路追迹服务

把薄透镜、空气间隔、球面折射面串成一条近轴光路的 HTTP 服务。
给定入射高度 `y` 与近轴角 `u`，返回出射光线、每段矩阵、整段系统矩阵，
以及给定物距对应的像距与横向放大率。

折射面不再只收写死的折射率：可以点名一份**具名材料档**（玻璃），
服务按当前波长现算折射率再拼矩阵。一次请求给一串波长，
服务对每个波长独立追迹、独立解共轭像面，并在逐波长结果之上归约
**轴向色差**（长短两谱线像距之差）与**倍率色差**（两谱线放大率之差）。

- Python 3.12 + FastAPI，仅提供 HTTP 接口，无前端、无账户
- 具名光路档与具名材料档都写入进程内 SQLite（WAL），可登记 / 列出 / 点名取用
- 内置共焦望远镜示范档 `telescope_demo`（f1=100、f2=50、间距 150，放大率 -0.5，C≈0）
- 内置示范玻璃 `N-BK7` 与 `SF10`（Sellmeier 系数）
- 也可在当次请求里内联元件清单，只追一次不落库

## 约定

光线列向量全程为 `(y, u)`（高度、近轴角），不中途切换方向余弦。
**波长一律用纳米（nm）**；Sellmeier 系数沿用文献惯例（λ 以 µm 计），
模型内部自行换算。

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
- 薄透镜与空气间隔不吃色散；写死折射率的折射面等价于无色散材料，
  各波长追迹结果完全一致、色差为零

## 材料与色散

材料档描述折射率如何随波长变化，两种描述方式（登记时二选一）：

1. **采样点** `samples`：给若干 `(波长 nm, 折射率)` 点（至少 2 个）。
   服务按 Cauchy 形式 `n(λ) = A + B/λ² (+ C/λ⁴)` 最小二乘拟合
   （2 点用两项，≥3 点用三项），内插与外推都沿拟合曲线走，不做线性硬连。
2. **解析系数** `coefficients`：折射率是波长的解析函数，直接现算。
   - `{"model": "sellmeier", "B": [...], "C": [...]}`：
     `n² = 1 + Σ Bᵢλ²/(λ²−Cᵢ)`，λ 以 µm 计（文献惯例）；
   - `{"model": "cauchy", "A": ..., "B": ..., "C": ...}`（`C` 可省），λ 以 nm 计；
   - `{"model": "constant", "n": ...}`：全波段恒定（无色散）。

折射面的 `n1`、`n2` 既可以写数值（恒定折射率，老式写法不变），
也可以写材料档名字符串（也可用别名 `material1`/`material2`）。
元件里点了材料名，请求就必须给出 `wavelengths`，不静默取默认谱线。

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

### `POST /materials` — 登记材料档
```json
{
  "name": "my_crown",
  "coefficients": {"model": "sellmeier",
                   "B": [1.03961212, 0.231792344, 1.01046945],
                   "C": [0.00600069867, 0.0200179144, 103.560653]}
}
```
或采样点描述：
```json
{
  "name": "my_crown_samples",
  "samples": [{"wavelength": 486.1327, "n": 1.52238},
              {"wavelength": 587.5618, "n": 1.51680},
              {"wavelength": 656.2725, "n": 1.51432}]
}
```
重名返回 `409 duplicate_name`。

### `GET /materials` — 列出全部材料档
给出每份材料的描述方式（`samples` 或 `coefficients`）与全部系数或采样点；
采样点档同时给出拟合所得的 Cauchy 系数。

### `GET /materials/{name}/index?wavelength=587.5618` — 点名现算折射率
未登记档名返回 `404 unknown_material`。

### `POST /paths` — 登记具名光路档
```json
{
  "name": "bk7_singlet",
  "elements": [
    {"type": "refract", "params": {"R": 50,  "n1": 1.0,    "n2": "N-BK7"}},
    {"type": "refract", "params": {"R": -50, "n1": "N-BK7", "n2": 1.0}}
  ]
}
```
元件类型：`space`（参数 `L`）、`lens`（参数 `f`）、
`refract`（参数 `R`、`n1`、`n2`；`n1`/`n2` 收数值或材料档名）。
重名返回 `409 duplicate_name`。

### `GET /paths` — 列出全部光路档（元件种类与参数全文）

### `POST /paths/{name}/trace` 与 `POST /trace` — 追迹

老式单波长请求（行为与 1.x 完全一致）：
```json
{"elements": [...], "ray": {"y": 5.0, "u": 0.01}, "s": 200}
```

多波长请求：加一串 `wavelengths`（nm），可选 `chromatic` 指定色差基准谱线
（缺省取清单里的最短与最长波长）：
```json
{
  "elements": [
    {"type": "refract", "params": {"R": 50,  "n1": 1.0,    "n2": "N-BK7"}},
    {"type": "refract", "params": {"R": -50, "n1": "N-BK7", "n2": 1.0}}
  ],
  "ray": {"y": 1.0, "u": 0.0},
  "s": 300,
  "wavelengths": [486.1327, 587.5618, 656.2725],
  "chromatic": {"short_wavelength": 486.1327, "long_wavelength": 656.2725}
}
```

多波长响应：`results` 按请求波长顺序给出每个波长独立的
每段矩阵、系统 A/B/C/D、像距、放大率、出射光线与往返还原残差
（`elements` 里是按该波长现算出的数值折射率）；`chromatic` 给出：

- `axial`：轴向色差 = 短波像距 − 长波像距（物在无穷远时即后焦距之差）；
- `lateral`：倍率色差 = 短波放大率 − 长波放大率
  （物在无穷远时不适用，返回 `null` 并附说明）；
- `basis`：实际采用的长短两条基准谱线及其来源。

`s = null` 时每个波长各自给出后焦距，可直接看到不同颜色的焦点在轴上拉开多远。

错误统一形如 `{"error": {"code": "...", "message": "...", ...}}`，
带类型的 code 包括：`unknown_element`、`negative_spacing`、
`zero_focal_length`、`invalid_refractive_index`、`non_finite_number`、
`missing_field`、`unknown_path`、`unknown_material`、`duplicate_name`、
`invalid_material`、`invalid_wavelength`、`invalid_chromatic_basis`、
`dispersion_evaluation_failed`、`invalid_body` 等；
元件相关错误都带 `index` 指明错在第几个元件。

## 模块划分

| 文件 | 职责 |
|------|------|
| `app/dispersion.py` | 波长 → 折射率的色散模型（Cauchy / Sellmeier / 恒定、采样点拟合） |
| `app/materials.py` | 具名材料档的进程内 SQLite 存取 |
| `app/elements.py` | 元件矩阵与 2×2 矩阵运算 |
| `app/systems.py` | 单波长系统连乘、正向/反向追迹、物像求解 |
| `app/spectral.py` | 按波长解析材料、多波长独立追迹编排 |
| `app/chromatic.py` | 跨波长色差归约（轴向 / 倍率） |
| `app/storage.py` | 具名光路档的进程内 SQLite 存取 |
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
无色散材料各波长结果一致且色差为零、薄透镜/空气间隔不配波长色散、
色散单透镜三谱线像距蓝短红长且轴向色差符号量值可手算核对、
消色差双胶合长短谱线色差归零而中间谱线留二级光谱、
同一玻璃采样点档与解析系数档折射率一致且追迹结果不漂移、
老单波长请求行为不变、
焦距为零/未知元件/负间隔/非正折射率/非有限数/未登记材料或光路被拒、
并行多组波长追迹互不污染。
