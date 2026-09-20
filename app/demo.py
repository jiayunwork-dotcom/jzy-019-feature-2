"""内置材料档与示范光路（启动时幂等写入进程内 SQLite）。

材料：

- ``air``：恒定折射率 1.0；
- ``BK7``：Sellmeier 系数（Schott N-BK7，波长 μm）；
- ``F2``：Sellmeier 系数（Schott F2）。

光路：

- ``telescope_demo``：共焦开普勒望远镜（旧档，f1=100, 间距150, f2=50）；
- ``single_bk7_lens``：空气中对称双凸 BK7 薄透镜，d 线焦距 100，
  蓝端焦距短、红端焦距长；
- ``achromat_doublet``：BK7 冠片 + F2 火石片的密接双胶合薄透镜组，
  F/C 两谱线合成焦度重合（轴向色差归零），d 线留二级光谱残余。
"""

from __future__ import annotations

from .elements import Element
from .materials import Material
from .dispersion import ConstantIndex, SellmeierDispersion

TELESCOPE_DEMO_NAME = "telescope_demo"
SINGLE_BK7_NAME = "single_bk7_lens"
ACHROMAT_NAME = "achromat_doublet"

TELESCOPE_F1 = 100.0
TELESCOPE_F2 = 50.0
TELESCOPE_SPACING = TELESCOPE_F1 + TELESCOPE_F2

# Schott Sellmeier（B1..B3, C1..C3，波长 μm）
BK7_B = (1.03961217, 0.231792344, 1.01046945)
BK7_C = (0.00600069867, 0.0200179144, 103.560653)
F2_B = (1.34533359, 0.209073176, 0.937357162)
F2_C = (0.00997743871, 0.0470450767, 112.402912)

# 对称双凸 BK7 单透镜：R1 = +R, R2 = -R，d 线（n=1.5167985）f = 100
BK7_LENS_RADIUS = 2.0 * (1.516798488 - 1.0) * 100.0  # ≈ 103.36

# 消色差双胶合（F/C 消色差，d 线 EFL = 100）：
# 冠片 BK7：r1=31.91206, r2=-75；火石 F2：r1=-75, r2=128.69556
DOUBLET_R1 = 31.9120605
DOUBLET_R2 = -75.0
DOUBLET_R3 = 128.6955640


def seed_materials() -> list[Material]:
    return [
        Material("air", ConstantIndex(1.0)),
        Material("BK7", SellmeierDispersion(BK7_B, BK7_C)),
        Material("F2", SellmeierDispersion(F2_B, F2_C)),
    ]


def telescope_demo_elements() -> list[Element]:
    return [
        Element("lens", {"f": TELESCOPE_F1}),
        Element("space", {"L": TELESCOPE_SPACING}),
        Element("lens", {"f": TELESCOPE_F2}),
    ]


def single_bk7_lens_elements() -> list[Element]:
    return [
        Element("lens", {
            "r1": BK7_LENS_RADIUS,
            "r2": -BK7_LENS_RADIUS,
            "glass": "BK7",
        }),
    ]


def achromat_doublet_elements() -> list[Element]:
    return [
        # BK7 冠片（双凸，第二面与火石片密接）
        Element("lens", {"r1": DOUBLET_R1, "r2": DOUBLET_R2, "glass": "BK7"}),
        # F2 火石片（凹-凸，密接面曲率相同）
        Element("lens", {"r1": DOUBLET_R2, "r2": DOUBLET_R3, "glass": "F2"}),
    ]


def seed_paths() -> list[tuple[str, list[Element]]]:
    return [
        (TELESCOPE_DEMO_NAME, telescope_demo_elements()),
        (SINGLE_BK7_NAME, single_bk7_lens_elements()),
        (ACHROMAT_NAME, achromat_doublet_elements()),
    ]
