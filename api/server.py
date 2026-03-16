"""
FastAPI-приложение: фабрика, CORS, lifespan.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import CORS_ORIGINS
from api.dependencies import get_pipeline
from api.routes import analyze, health, plugins


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей при старте сервера."""
    get_pipeline()
    yield


def create_app() -> FastAPI:
    """Создание и настройка FastAPI-приложения."""
    app = FastAPI(
        title="MedAI Analysis API",
        version="1.0.0",
        description=(
            "REST API для анализа медицинских изображений. "
            "Поддерживает плагинную архитектуру для различных типов диагностики."
        ),
        lifespan=lifespan,
    )

    # CORS — для интеграции с внешними МИС
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Маршруты
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(plugins.router, prefix="/api/v1", tags=["Plugins"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])

    return app
