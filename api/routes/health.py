"""
GET /api/v1/health — проверка состояния сервиса.
"""

from fastapi import APIRouter

from api.dependencies import get_pipeline
from api.schemas.plugins import HealthResponse
from core.registry import PluginRegistry

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    # Убедимся, что pipeline инициализирован
    get_pipeline()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        plugins_loaded=PluginRegistry.list_available(),
    )
