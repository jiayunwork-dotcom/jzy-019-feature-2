"""带类型的错误定义。

所有输入不合规的情况都抛 ``OpticsError``，携带稳定的 ``code``，
HTTP 层据此给出 4xx 状态码与 ``{"error": {...}}`` 响应体。
"""

from __future__ import annotations

import math


class OpticsError(Exception):
    """近轴光路服务的全部业务错误。"""

    def __init__(self, code: str, message: str, *, status_code: int = 422, **extra: object):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = extra

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {"code": self.code, "message": self.message}
        body.update(self.extra)
        return _json_safe(body)


def _json_safe(value: object) -> object:
    """把 inf/nan 等不可 JSON 编码的浮点替换成字符串标记。"""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value
