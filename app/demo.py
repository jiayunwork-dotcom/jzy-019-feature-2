"""内置示范档：共焦（无焦）开普勒式望远镜。

两块薄透镜，间距恰好等于两焦距之和：
f1 = 100, f2 = 50, d = 150。

C 分量为 0，横向放大率 = -f2/f1 = -0.5（倒立、缩小）。
"""

from __future__ import annotations

from .elements import Element

TELESCOPE_DEMO_NAME = "telescope_demo"

TELESCOPE_F1 = 100.0
TELESCOPE_F2 = 50.0
TELESCOPE_SPACING = TELESCOPE_F1 + TELESCOPE_F2


def telescope_demo_elements() -> list[Element]:
    return [
        Element("lens", {"f": TELESCOPE_F1}),
        Element("space", {"L": TELESCOPE_SPACING}),
        Element("lens", {"f": TELESCOPE_F2}),
    ]
