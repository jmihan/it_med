"""
Клинический дашборд (режим врача).
Строгий дизайн: оригинал + разметка, таблица метрик, диагноз, экспорт.
Полностью data-driven — не содержит хардкода под конкретный плагин.
"""

import streamlit as st
from typing import Dict, Any

from ui.components import image_viewer, metrics_panel, report_export


def render(results: Dict[str, Any]):
    """
    Отрисовка клинического дашборда.

    Args:
        results: Обогащённый результат из AnalysisPipeline.run()
    """
    metadata = results.get("plugin_metadata", {})
    st.header(f"Результаты клинического анализа — {metadata.get('display_name', 'Анализ')}")

    # --- Изображения: оригинал + разметка ---
    original = results.get("original_image")
    annotated = results.get("annotated_image")

    if original is not None and annotated is not None:
        image_viewer.render_comparison(original, annotated)
    elif original is not None:
        image_viewer.render_single(original, caption="Исходный снимок")

    st.divider()

    # --- Метрики ---
    st.subheader("Медицинские показатели")
    metrics = results.get("metrics", {})
    metric_defs = metadata.get("metric_definitions", [])
    pathology = results.get("pathology_detected", False)

    metrics_panel.render(metrics, metric_defs, pathology)

    # --- Классификация (второе мнение) ---
    if metadata.get("has_classification"):
        classification = results.get("classification")
        metrics_panel.render_classification(classification)

    # --- Экспорт ---
    st.divider()
    report_export.render_download(results)
