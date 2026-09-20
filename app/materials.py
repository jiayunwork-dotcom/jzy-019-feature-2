"""具名材料档：登记描述、按波长解析折射率。

材料档在一次追迹内只读使用；``MaterialResolver`` 维护名称 -> 材料的
请求级快照，解析结果缓存在该次请求的解析器实例上，不写回材料档，
天然避免跨请求、跨波长串扰。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .dispersion import (
    DispersionModel,
    build_dispersion,
)
from .errors import OpticsError


@dataclass(frozen=True)
class Material:
    """具名玻璃/介质档。"""

    name: str
    model: DispersionModel

    def describe(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": self.name,
            "wavelength_unit": "um",
            "model": self.model.describe(),
            "description_kind": self.model.type_name,
        }
        return body


def normalize_material_body(body: Any) -> Material:
    """把登记材料的原始 JSON 转成 :class:`Material`，不合规即拒绝。"""
    if not isinstance(body, dict):
        raise OpticsError("invalid_body", "材料登记请求必须是 JSON 对象", status_code=400)
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise OpticsError(
            "invalid_name",
            "材料档名必须是非空字符串",
            field="name",
            status_code=400,
        )
    if "dispersion" in body:
        spec = body["dispersion"]
    else:
        # 也允许把模型字段直接平铺在对象里
        spec = body.get("model")
    if spec is None:
        if "n" in body:
            spec = {"type": "constant", "n": body["n"]}
        else:
            raise OpticsError(
                "missing_field",
                "材料档缺少色散描述 dispersion（type=constant/cauchy/sellmeier/sampled）",
                field="dispersion",
            )
    model = build_dispersion(spec)
    return Material(name=name.strip(), model=model)


@dataclass
class MaterialResolver:
    """一次追迹请求范围内的材料解析器：按波长现算折射率。

    常量介质也在追迹前过一遍正定性校验；玻璃折射率按波长独立计算并
    缓存在本次解析器里，缓存不跨请求共享。
    """

    materials: dict[str, Material]
    wavelength: float
    _cache: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self._cache = {}

    def material(self, name: str) -> Material:
        material = self.materials.get(name)
        if material is None:
            raise OpticsError(
                "unknown_material",
                f"材料档 {name!r} 未登记；请先用 /materials 登记，或直接写数值折射率",
                material=name,
                status_code=404,
            )
        return material

    def index_of(self, name: str) -> float:
        assert self._cache is not None
        if name not in self._cache:
            material = self.material(name)
            n = material.model.index(self.wavelength)
            if not math.isfinite(n) or n <= 0.0:
                raise OpticsError(
                    "invalid_refractive_index",
                    f"材料 {name!r} 在波长 {self.wavelength:g} μm 处折射率 "
                    f"为 {n!r}，必须为正且有限",
                    material=name,
                    wavelength=self.wavelength,
                )
            self._cache[name] = n
        return self._cache[name]
