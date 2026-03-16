"""
Страница анализа одного снимка.
Запускает pipeline и передаёт результат в нужный view (врач/студент).
"""

import streamlit as st
from typing import Any

from core.pipeline import AnalysisPipeline
from core.registry import PluginRegistry
import ui.state as state
from ui.views import doctor_view, student_view


def render(pipeline: AnalysisPipeline):
    """
    Отрисовка страницы анализа одного снимка.

    Args:
        pipeline: Экземпляр AnalysisPipeline
    """
    plugin_name = state.get("selected_plugin")
    mode = state.get("mode")
    image = state.get("uploaded_image")

    # Проверка: заглушка?
    if PluginRegistry.is_stub(plugin_name):
        meta = PluginRegistry.get_metadata(plugin_name)
        st.info(f"Модуль «{meta['display_name']}» находится в разработке. Выберите другой тип исследования.")
        return

    # Нет загруженного снимка
    if image is None:
        st.info("👈 Загрузите снимок в боковой панели для начала анализа")
        _show_instructions()
        return

    # Запуск анализа (с кэшированием)
    results = state.get("analysis_results")
    if results is None:
        with st.spinner("Анализируем снимок..."):
            try:
                results = pipeline.run(image, plugin_name, mode)
                state.set("analysis_results", results)
            except Exception as e:
                st.error(f"Ошибка анализа: {e}")
                return

    # Dispatch в нужный view
    if mode == "doctor":
        doctor_view.render(results)
    else:
        student_view.render(results)


def _show_instructions():
    """Инструкции на пустой странице."""
    st.markdown("""
    ### Как начать работу

    1. Выберите **тип исследования** в боковой панели
    2. Выберите **режим** (Врач / Студент)
    3. Загрузите **снимок** (PNG, JPG или DICOM)
    4. Дождитесь результатов анализа

    ---

    **Режим врача** — краткие результаты: снимок с разметкой, метрики, диагноз.

    **Режим студента** — пошаговое объяснение с интерактивными слоями визуализации.
    """)
