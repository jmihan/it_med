"""
Компонент экспорта отчётов.
"""

import io
import cv2
import numpy as np
import streamlit as st
from typing import Dict, Any


def generate_text_report(results: Dict[str, Any]) -> str:
    """Генерация текстового отчёта с геометрическим и нейросетевым вердиктами."""
    metadata = results.get("plugin_metadata", {})
    metrics = results.get("metrics", {})
    classification = results.get("classification")
    pathology = results.get("pathology_detected", False)
    geo_pathology = results.get("geometric_pathology")
    geo_confidence = results.get("geometric_confidence")
    resnet_pathology = results.get("resnet_pathology")
    resnet_confidence = results.get("resnet_confidence")

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
        if value is None:
            continue

        if isinstance(value, dict):
            if "h_mm" in value:
                value = value.get("h_mm") or value.get("h_px")
            elif "d_mm" in value:
                value = value.get("d_mm") or value.get("d_px")
            else:
                continue

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

    # Геометрический вердикт
    lines.append("ГЕОМЕТРИЧЕСКИЙ ВЕРДИКТ:")
    lines.append("-" * 40)
    if geo_pathology is None:
        lines.append("  Вердикт: Недостаточно ключевых точек для анализа")
    elif geo_pathology:
        conf_str = f" (уверенность: {geo_confidence:.0%})" if geo_confidence is not None else ""
        lines.append(f"  Вердикт: Патология{conf_str}")
    else:
        conf_str = f" (уверенность: {geo_confidence:.0%})" if geo_confidence is not None else ""
        lines.append(f"  Вердикт: Норма{conf_str}")
    lines.append("")

    # Нейросетевая классификация
    lines.append("НЕЙРОСЕТЕВАЯ КЛАССИФИКАЦИЯ (ResNet):")
    lines.append("-" * 40)
    if resnet_pathology is None:
        lines.append("  Классификатор не подключён")
    elif classification:
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


def render_download(results: Dict[str, Any]):
    """Кнопки скачивания отчёта."""
    st.subheader("Экспорт результатов")

    col1, col2 = st.columns(2)

    with col1:
        report_text = generate_text_report(results)
        st.download_button(
            "📄 Скачать отчёт (TXT)",
            data=report_text,
            file_name="medical_report.txt",
            mime="text/plain",
        )

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
