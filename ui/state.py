"""
Централизованное управление состоянием Streamlit-приложения.
Все обращения к st.session_state проходят через этот модуль.
"""

import streamlit as st


DEFAULTS = {
    "selected_plugin": "hip_dysplasia",
    "mode": "doctor",
    "current_page": "single",
    "uploaded_file": None,
    "uploaded_image": None,
    "analysis_results": None,
    "active_layers": {},
}


def init_state():
    """Инициализация состояния при первом запуске."""
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str):
    """Получить значение из состояния."""
    return st.session_state.get(key, DEFAULTS.get(key))


def set(key: str, value):
    """Установить значение в состоянии."""
    st.session_state[key] = value


def reset_analysis():
    """Сброс результатов анализа (при смене плагина/режима/снимка)."""
    st.session_state["analysis_results"] = None
    st.session_state["active_layers"] = {}
