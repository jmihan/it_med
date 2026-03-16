"""
Динамический сайдбар, построенный из метаданных реестра плагинов.
Не содержит хардкода под конкретные плагины.
"""

import streamlit as st
from core.registry import PluginRegistry
from core.image_io import load_from_upload
import ui.state as state


def render():
    """Отрисовка сайдбара: выбор плагина, режим, загрузка файла, навигация."""
    st.sidebar.title("🏥 MedAI Platform")
    st.sidebar.caption("Интеллектуальная диагностика медицинских изображений")
    st.sidebar.divider()

    # --- Выбор плагина (динамически из реестра) ---
    all_metadata = PluginRegistry.get_all_metadata()
    plugin_names = list(all_metadata.keys())
    display_names = {name: meta["display_name"] for name, meta in all_metadata.items()}

    # Формат отображения: иконка + название
    def format_plugin(name):
        meta = all_metadata[name]
        icon = meta.get("icon", "🔬")
        label = meta["display_name"]
        if meta.get("stub"):
            return f"{icon} {label} (в разработке)"
        return f"{icon} {label}"

    selected = st.sidebar.selectbox(
        "Тип исследования",
        options=plugin_names,
        format_func=format_plugin,
        key="selected_plugin",
        on_change=state.reset_analysis,
    )

    # Предупреждение для заглушек
    selected_meta = all_metadata.get(selected, {})
    if selected_meta.get("stub"):
        st.sidebar.warning(f"Модуль «{selected_meta['display_name']}» находится в разработке")
        st.sidebar.info(selected_meta.get("description", ""))

    st.sidebar.divider()

    # --- Режим интерфейса ---
    mode_options = {"doctor": "👨‍⚕️ Врач (Клинический)", "student": "🎓 Студент (Образовательный)"}
    mode = st.sidebar.radio(
        "Режим интерфейса",
        options=list(mode_options.keys()),
        format_func=lambda k: mode_options[k],
        key="mode",
        on_change=state.reset_analysis,
    )

    st.sidebar.divider()

    # --- Навигация ---
    page_options = {"single": "📋 Анализ снимка", "batch": "📦 Пакетная обработка"}
    st.sidebar.radio(
        "Раздел",
        options=list(page_options.keys()),
        format_func=lambda k: page_options[k],
        key="current_page",
    )

    st.sidebar.divider()

    # --- Загрузка файла (только для страницы анализа одного снимка) ---
    if state.get("current_page") == "single" and not selected_meta.get("stub"):
        formats = selected_meta.get("supported_formats", ["png", "jpg", "jpeg"])
        uploaded = st.sidebar.file_uploader(
            "Загрузите снимок",
            type=formats,
            key="uploaded_file_widget",
        )

        if uploaded is not None:
            try:
                image = load_from_upload(uploaded)
                state.set("uploaded_image", image)
            except Exception as e:
                st.sidebar.error(f"Ошибка загрузки: {e}")
                state.set("uploaded_image", None)
        else:
            state.set("uploaded_image", None)
            state.reset_analysis()

    # --- Описание выбранного плагина ---
    if not selected_meta.get("stub") and selected_meta.get("description"):
        st.sidebar.divider()
        st.sidebar.caption(selected_meta["description"])
