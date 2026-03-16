"""
GET /api/v1/plugins — информация о доступных плагинах.
"""

from fastapi import APIRouter, HTTPException

from api.dependencies import get_pipeline
from api.schemas.plugins import PluginInfo, PluginsListResponse
from core.registry import PluginRegistry

router = APIRouter()


@router.get("/plugins", response_model=PluginsListResponse)
def list_plugins():
    get_pipeline()  # убедиться, что плагины зарегистрированы
    all_meta = PluginRegistry.get_all_metadata()
    plugins = [
        PluginInfo(
            name=name,
            metadata=meta,
            is_stub=PluginRegistry.is_stub(name),
        )
        for name, meta in all_meta.items()
    ]
    return PluginsListResponse(plugins=plugins)


@router.get("/plugins/{plugin_name}")
def get_plugin_info(plugin_name: str):
    get_pipeline()
    try:
        meta = PluginRegistry.get_metadata(plugin_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Плагин '{plugin_name}' не найден")
    return PluginInfo(
        name=plugin_name,
        metadata=meta,
        is_stub=PluginRegistry.is_stub(plugin_name),
    )
