"""
Компонент экспорта отчётов (PDF и CSV).
"""

import io
import cv2
import numpy as np
import streamlit as st
from typing import Dict, Any


def render_download(results: Dict[str, Any]):
    """Кнопки скачивания отчёта."""
    st.subheader("Экспорт результатов")

    col1, col2 = st.columns(2)

    # --- Текстовый отчёт ---
    with col1:
        report_text = _generate_text_report(results)
        st.download_button(
            "📄 Скачать отчёт (TXT)",
            data=report_text,
            file_name="medical_report.txt",
            mime="text/plain",
        )

    # --- Размеченный снимок ---
    with col2:
        annotated = results.get("annotated_image")
        if annotated is not None:
            _, buf = cv2.imencode(".png", annotated)
            st.download_button(
                "🖼️ Скачать снимок с разметкой",
                data=buf.tobytes(),
                file_name="annotated_scan.png",
                mime="image/png",
            )


def _generate_text_report(results: Dict[str, Any]) -> str:
    """Генерация текстового отчёта."""
    metadata = results.get("plugin_metadata", {})
    metrics = results.get("metrics", {})
    classification = results.get("classification")
    pathology = results.get("pathology_detected", False)

    lines = []
    lines.append("=" * 60)
    lines.append(f"МЕДИЦИНСКИЙ ОТЧЁТ — {metadata.get('display_name', 'Анализ')}")
    lines.append("=" * 60)
    lines.append("")

    # Метрики
    lines.append("РАСЧЁТНЫЕ ПОКАЗАТЕЛИ:")
    lines.append("-" * 40)
    for defn in metadata.get("metric_definitions", []):
        key = defn["key"]
        label = defn["label"]
        unit = defn.get("unit", "")
        value = metrics.get(key)
        if value is not None:
            if defn.get("type") == "bool":
                val_str = "Да" if value else "Нет"
            else:
                val_str = f"{value:.1f}{unit}"

            normal_range = defn.get("normal_range")
            status = ""
            if normal_range and defn.get("type") != "bool":
                lo, hi = normal_range
                status = " [НОРМА]" if lo <= value <= hi else " [ПАТОЛОГИЯ]"
            elif defn.get("type") == "bool":
                status = " [ОТКЛОНЕНИЕ]" if value else " [НОРМА]"

            lines.append(f"  {label}: {val_str}{status}")

    lines.append("")

    # Классификация
    if classification:
        lines.append("НЕЙРОСЕТЕВАЯ КЛАССИФИКАЦИЯ (ResNet):")
        lines.append("-" * 40)
        lines.append(f"  Предсказание: {classification.get('class_name', '—')}")
        lines.append(f"  Уверенность: {classification.get('confidence', 0):.1%}")
        lines.append(f"  P(Норма): {classification.get('prob_normal', 0):.1%}")
        lines.append(f"  P(Патология): {classification.get('prob_pathology', 0):.1%}")
        lines.append("")

    # Заключение
    lines.append("ЗАКЛЮЧЕНИЕ:")
    lines.append("-" * 40)
    if pathology:
        lines.append("  Обнаружены признаки патологии.")
        lines.append("  Рекомендуется консультация специалиста.")
    else:
        lines.append("  Признаков патологии не обнаружено.")
        lines.append("  Показатели в пределах нормы.")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Отчёт сгенерирован платформой MedAI")
    lines.append("Данный отчёт не является медицинским заключением")
    lines.append("=" * 60)

    return "\n".join(lines)
