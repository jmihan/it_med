"""
Data-driven панель метрик.
Рендерит метрики на основе определений из метаданных плагина.
"""

import streamlit as st
from typing import Dict, List, Any


def render(metrics: Dict[str, Any], metric_definitions: List[Dict], pathology_detected: bool = False):
    """
    Отрисовка метрик в виде st.metric карточек.

    Args:
        metrics: Словарь с рассчитанными значениями
        metric_definitions: Список определений из get_ui_metadata()["metric_definitions"]
        pathology_detected: Общий результат диагностики
    """
    if not metrics.get("valid", True):
        st.warning("Недостаточно данных для расчёта метрик")
        return

    if not metric_definitions:
        return

    # Размещаем метрики в колонках (по 2 в ряду)
    cols_per_row = 2
    for i in range(0, len(metric_definitions), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(metric_definitions):
                break

            defn = metric_definitions[idx]
            key = defn["key"]
            label = defn["label"]
            unit = defn.get("unit", "")
            metric_type = defn.get("type", "numeric")
            normal_range = defn.get("normal_range")

            value = metrics.get(key)
            if value is None:
                # Попробовать вложенные ключи (pathology.left.is_pathology)
                value = _get_nested(metrics, key)

            with col:
                if metric_type == "bool":
                    _render_bool_metric(label, value)
                else:
                    _render_numeric_metric(label, value, unit, normal_range)

    # Итоговый диагноз
    st.divider()
    if pathology_detected:
        st.error("⚠️ **Заключение: Обнаружены признаки патологии**")
    else:
        st.success("✅ **Заключение: Патология не обнаружена**")


def render_classification(classification: Dict[str, Any]):
    """Отображение результатов нейросетевой классификации (второе мнение)."""
    if classification is None:
        return

    st.divider()
    st.subheader("Нейросетевая классификация (ResNet)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Предсказание", classification.get("class_name", "—"))
    with col2:
        conf = classification.get("confidence", 0)
        st.metric("Уверенность", f"{conf:.1%}")
    with col3:
        p_path = classification.get("prob_pathology", 0)
        st.metric("P(Патология)", f"{p_path:.1%}")


def _render_numeric_metric(label: str, value, unit: str, normal_range=None):
    """Отрисовка числовой метрики с индикацией нормы/патологии."""
    if value is None:
        st.metric(label, "Н/Д")
        return

    display_value = f"{value:.1f}{unit}"

    if normal_range and len(normal_range) == 2:
        lo, hi = normal_range
        if lo <= value <= hi:
            delta = "Норма"
            delta_color = "normal"
        else:
            delta = "Патология"
            delta_color = "inverse"
        st.metric(label, display_value, delta=delta, delta_color=delta_color)
    else:
        st.metric(label, display_value)


def _render_bool_metric(label: str, value):
    """Отрисовка булевой метрики."""
    if value is None:
        st.metric(label, "Н/Д")
    elif value:
        st.metric(label, "⚠️ Да", delta="Отклонение", delta_color="inverse")
    else:
        st.metric(label, "✅ Нет", delta="Норма", delta_color="normal")


def _get_nested(d: dict, key: str):
    """Попытка получить значение по ключу, в том числе из вложенных словарей."""
    if key in d:
        return d[key]
    # Рекурсивный поиск
    for v in d.values():
        if isinstance(v, dict):
            result = _get_nested(v, key)
            if result is not None:
                return result
    return None
