"""
Главная точка входа Streamlit-приложения MedAI Platform.

Роли:
  1. Инициализация состояния и pipeline (один раз)
  2. Отрисовка сайдбара через динамический компонент
  3. Роутинг на нужную страницу (анализ / пакетная обработка)
"""

import os
import sys
import streamlit as st

# Добавляем корень проекта в sys.path для корректных импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import AnalysisPipeline
from core.registry import register_all_plugins

import ui.state as state
from ui.components import sidebar
from ui.pages import single_analysis, batch_processing


@st.cache_resource
def init_pipeline() -> AnalysisPipeline:
    """
    Инициализация pipeline и регистрация плагинов.
    Кэшируется — модели загружаются один раз на процесс.
    """
    register_all_plugins()
    return AnalysisPipeline()


def main():
    st.set_page_config(
        layout="wide",
        page_title="MedAI Platform",
        page_icon="🏥",
    )

    # Инициализация
    state.init_state()
    pipeline = init_pipeline()

    # Сайдбар
    sidebar.render()

    # Роутинг
    page = state.get("current_page")
    if page == "batch":
        batch_processing.render(pipeline)
    else:
        single_analysis.render(pipeline)


if __name__ == "__main__":
    main()
