"""
Pydantic-модели запросов и ответов для эндпоинтов анализа.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional


class KeypointData(BaseModel):
    x: float
    y: float
    confidence: float


class ClassificationData(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    prob_normal: float
    prob_pathology: float


class MetricsData(BaseModel):
    """
    Расширяемая модель метрик.

    Общие поля: valid, pathology.
    Плагин-специфичные поля передаются через extra="allow".
    """
    model_config = ConfigDict(extra="allow")

    valid: bool = False
    pathology: Optional[dict] = None


class ImagesData(BaseModel):
    annotated: Optional[str] = None
    layers: Optional[dict[str, str]] = None


class AnalysisResponse(BaseModel):
    status: str = "success"
    plugin: str
    mode: str
    pathology_detected: bool
    classification: Optional[ClassificationData] = None
    keypoints: dict[str, KeypointData] = {}
    detection_conf: float = 0.0
    metrics: MetricsData = Field(default_factory=MetricsData)
    method: str = "geometric"
    explanation_steps: list[dict] = []
    images: Optional[ImagesData] = None


class BatchResultItem(BaseModel):
    image_id: str
    filename: str
    pathology_detected: Optional[bool] = None
    classification: Optional[ClassificationData] = None
    metrics: Optional[MetricsData] = None
    error: Optional[str] = None


class BatchAnalysisResponse(BaseModel):
    status: str = "success"
    total: int
    results: list[BatchResultItem]
    csv_summary: str
