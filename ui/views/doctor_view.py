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

    # --- Вердикты: геометрический + нейросетевой ---
    st.divider()
    st.subheader("Вердикты системы")
    col_geo, col_nn = st.columns(2)

    with col_geo:
        st.markdown("**Геометрический анализ**")
        geo_path = results.get("geometric_pathology")
        geo_conf = results.get("geometric_confidence")
        if geo_path is None:
            st.warning("Недостаточно ключевых точек для геометрического вердикта")
        elif geo_path:
            conf_str = f" ({geo_conf:.0%} уверенность)" if geo_conf is not None else ""
            st.error(f"⚠️ Патология{conf_str}")
        else:
            conf_str = f" ({geo_conf:.0%} уверенность)" if geo_conf is not None else ""
            st.success(f"✅ Норма{conf_str}")

    with col_nn:
        st.markdown("**Нейросетевой классификатор (ResNet)**")
        nn_path = results.get("resnet_pathology")
        nn_conf = results.get("resnet_confidence")
        if nn_path is None:
            st.info("Классификатор не подключён")
        elif nn_path:
            conf_str = f" ({nn_conf:.0%} уверенность)" if nn_conf is not None else ""
            st.error(f"⚠️ Патология{conf_str}")
        else:
            conf_str = f" ({nn_conf:.0%} уверенность)" if nn_conf is not None else ""
            st.success(f"✅ Норма{conf_str}")

    # --- Экспорт ---
    st.divider()
    report_export.render_download(results)
