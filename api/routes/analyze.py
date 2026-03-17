"""
POST /api/v1/analyze       — анализ одного изображения.
POST /api/v1/analyze/batch — пакетный анализ.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.auth import require_api_key
from api.config import MAX_BATCH_SIZE, MAX_FILE_SIZE_MB
from api.dependencies import get_pipeline
from api.image_utils import serialize_results_images
from api.schemas.analysis import (
    AnalysisResponse,
    BatchAnalysisResponse,
    BatchResultItem,
    ClassificationData,
    ImagesData,
    KeypointData,
    MetricsData,
)
from core.image_io import load_from_bytes
from core.registry import PluginRegistry

router = APIRouter()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}


def _validate_file(file: UploadFile) -> None:
    """Проверка расширения и размера файла."""
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла: {ext}. Допустимые: {ALLOWED_EXTENSIONS}",
        )


def _parse_keypoints(raw: dict) -> dict[str, KeypointData]:
    """Конвертация keypoints из формата плагина в Pydantic-модели."""
    result = {}
    for name, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            result[name] = KeypointData(x=value[0], y=value[1], confidence=value[2])
        elif isinstance(value, dict):
            result[name] = KeypointData(**value)
    return result


def _parse_classification(raw: dict | None) -> ClassificationData | None:
    """Конвертация результата классификации."""
    if raw is None:
        return None
    return ClassificationData(
        class_id=raw.get("class_id", 0),
        class_name=raw.get("class_name", ""),
        confidence=raw.get("confidence", 0.0),
        prob_normal=raw.get("prob_normal", 0.0),
        prob_pathology=raw.get("prob_pathology", 0.0),
    )


def _parse_metrics(raw: dict) -> MetricsData:
    """Конвертация метрик. Передаёт все поля плагина (extra="allow")."""
    return MetricsData(**raw)


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_single(
    file: UploadFile = File(...),
    plugin: str = Form("hip_dysplasia"),
    mode: str = Form("doctor"),
    include_images: bool = Form(True),
    image_format: str = Form("png"),
    _api_key: str = Depends(require_api_key),
):
    """Анализ одного медицинского изображения."""
    # Валидация
    _validate_file(file)

    pipeline = get_pipeline()

    if PluginRegistry.is_stub(plugin):
        raise HTTPException(
            status_code=400,
            detail=f"Плагин '{plugin}' находится в разработке",
        )

    if mode not in ("doctor", "student"):
        raise HTTPException(status_code=400, detail="mode должен быть 'doctor' или 'student'")

    # Чтение файла
    file_bytes = file.file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой. Максимум: {MAX_FILE_SIZE_MB} МБ",
        )

    # Загрузка изображения
    try:
        image = load_from_bytes(file_bytes, file.filename or "image.png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки изображения: {e}")

    # Запуск пайплайна
    try:
        results = pipeline.run(image, plugin, mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {e}")

    # Формирование ответа
    images_data = None
    if include_images:
        raw_images = serialize_results_images(results, fmt=image_format)
        if raw_images:
            images_data = ImagesData(
                annotated=raw_images.get("annotated"),
                layers=raw_images.get("layers"),
            )

    return AnalysisResponse(
        plugin=plugin,
        mode=mode,
        pathology_detected=results.get("pathology_detected", False),
        classification=_parse_classification(results.get("classification")),
        keypoints=_parse_keypoints(results.get("keypoints", {})),
        detection_conf=results.get("detection_conf", 0.0),
        metrics=_parse_metrics(results.get("metrics", {})),
        method=results.get("method", "geometric"),
        explanation_steps=results.get("explanation_steps", []),
        images=images_data,
    )


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
def analyze_batch(
    files: list[UploadFile] = File(...),
    plugin: str = Form("hip_dysplasia"),
    mode: str = Form("doctor"),
    include_images: bool = Form(False),
    image_format: str = Form("png"),
    _api_key: str = Depends(require_api_key),
):
    """Пакетный анализ нескольких изображений."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Слишком много файлов. Максимум: {MAX_BATCH_SIZE}",
        )

    if mode not in ("doctor", "student"):
        raise HTTPException(status_code=400, detail="mode должен быть 'doctor' или 'student'")

    pipeline = get_pipeline()

    if PluginRegistry.is_stub(plugin):
        raise HTTPException(
            status_code=400,
            detail=f"Плагин '{plugin}' находится в разработке",
        )

    results_list: list[BatchResultItem] = []
    csv_lines = ["id,class"]

    for f in files:
        import os
        image_id = os.path.splitext(f.filename or "unknown")[0]

        try:
            _validate_file(f)
            file_bytes = f.file.read()
            image = load_from_bytes(file_bytes, f.filename or "image.png")
            raw = pipeline.run(image, plugin, mode=mode)

            pathology = raw.get("pathology_detected", False)
            csv_lines.append(f"{image_id},{1 if pathology else 0}")

            results_list.append(BatchResultItem(
                image_id=image_id,
                filename=f.filename or "unknown",
                pathology_detected=pathology,
                classification=_parse_classification(raw.get("classification")),
                metrics=_parse_metrics(raw.get("metrics", {})),
            ))
        except Exception as e:
            csv_lines.append(f"{image_id},-1")
            results_list.append(BatchResultItem(
                image_id=image_id,
                filename=f.filename or "unknown",
                pathology_detected=None,
                error=str(e),
            ))

    return BatchAnalysisResponse(
        total=len(results_list),
        results=results_list,
        csv_summary="\n".join(csv_lines),
    )
