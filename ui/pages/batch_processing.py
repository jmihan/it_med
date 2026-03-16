"""
Страница пакетной обработки.
Обработка папки с изображениями → submission.csv + размеченные скриншоты.
"""

import os
import io
import csv
import cv2
import numpy as np
import streamlit as st
from typing import List

from core.pipeline import AnalysisPipeline
from core.registry import PluginRegistry
from core.image_io import load_image
import ui.state as state
from ui.components.image_viewer import _to_rgb


def render(pipeline: AnalysisPipeline):
    """Отрисовка страницы пакетной обработки."""
    st.header("📦 Пакетная обработка снимков")
    st.caption("Автоматический анализ папки с изображениями и генерация submission.csv")

    plugin_name = state.get("selected_plugin")

    if PluginRegistry.is_stub(plugin_name):
        meta = PluginRegistry.get_metadata(plugin_name)
        st.info(f"Модуль «{meta['display_name']}» находится в разработке.")
        return

    # --- Ввод пути к папке ---
    folder_path = st.text_input(
        "Путь к папке с изображениями",
        value="data/processed/test/images",
        help="Укажите путь к папке, содержащей снимки для анализа",
    )

    save_annotated = st.checkbox("Сохранять размеченные снимки", value=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("🚀 Запустить", type="primary", use_container_width=True)

    if not start_btn:
        st.markdown("""
        ### Формат результата

        - **submission.csv** — файл вида `id,class` (0 = норма, 1 = патология)
        - **Размеченные снимки** — изображения с наложенной разметкой ИИ (опционально)
        """)
        return

    # --- Сканирование папки ---
    if not os.path.isdir(folder_path):
        st.error(f"Папка не найдена: {folder_path}")
        return

    image_extensions = {".png", ".jpg", ".jpeg", ".dcm"}
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        st.warning("В указанной папке не найдено изображений")
        return

    st.info(f"Найдено {len(image_files)} изображений")

    # --- Обработка ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    results_list = []
    annotated_images = {}

    for i, (image_id, result) in enumerate(pipeline.run_batch(image_files, plugin_name)):
        progress = (i + 1) / len(image_files)
        progress_bar.progress(progress)
        status_text.text(f"Обработано: {i + 1}/{len(image_files)} — {image_id}")

        # Извлечение данных для CSV
        if "error" in result:
            pathology_class = -1  # Ошибка
        else:
            pathology_class = 1 if result.get("pathology_detected", False) else 0

        confidence = 0.0
        classification = result.get("classification")
        if classification:
            confidence = classification.get("confidence", 0.0)

        results_list.append({
            "id": image_id,
            "class": pathology_class,
            "confidence": confidence,
        })

        # Сохранение размеченного снимка
        if save_annotated and "annotated_image" in result:
            annotated_images[image_id] = result["annotated_image"]

    progress_bar.progress(1.0)
    status_text.text(f"Готово! Обработано {len(image_files)} снимков.")

    st.divider()

    # --- Таблица результатов ---
    st.subheader("Результаты")
    import pandas as pd
    df = pd.DataFrame(results_list)
    st.dataframe(df, use_container_width=True)

    # --- Статистика ---
    total = len(results_list)
    pathology_count = sum(1 for r in results_list if r["class"] == 1)
    normal_count = sum(1 for r in results_list if r["class"] == 0)
    error_count = sum(1 for r in results_list if r["class"] == -1)

    stat_cols = st.columns(4)
    stat_cols[0].metric("Всего", total)
    stat_cols[1].metric("Патология", pathology_count)
    stat_cols[2].metric("Норма", normal_count)
    stat_cols[3].metric("Ошибки", error_count)

    st.divider()

    # --- Скачивание submission.csv ---
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["id", "class"])
    for r in results_list:
        if r["class"] >= 0:  # Пропускаем ошибки
            writer.writerow([r["id"], r["class"]])

    st.download_button(
        "📥 Скачать submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
    )

    # --- Галерея размеченных снимков ---
    if annotated_images:
        st.divider()
        st.subheader("Размеченные снимки")

        cols_per_row = 3
        items = list(annotated_images.items())
        for i in range(0, len(items), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(items):
                    break
                img_id, img = items[idx]
                result = results_list[idx]
                label = "⚠️ Патология" if result["class"] == 1 else "✅ Норма"
                with col:
                    st.image(_to_rgb(img), caption=f"{img_id} — {label}", use_container_width=True)
