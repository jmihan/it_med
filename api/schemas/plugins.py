"""
Pydantic-модели для эндпоинтов плагинов.
"""

from pydantic import BaseModel
from typing import Any


class PluginInfo(BaseModel):
    name: str
    metadata: dict[str, Any]
    is_stub: bool


class PluginsListResponse(BaseModel):
    plugins: list[PluginInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    plugins_loaded: list[str]
