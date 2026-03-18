"""
Страница пакетной обработки.
Обработка папки с изображениями → submission.csv + архив с результатами анализа.
"""

import csv
import io
import os
import zipfile

import cv2
import streamlit as st

from core.pipeline import AnalysisPipeline
from core.registry import PluginRegistry
import ui.state as state
from ui.components.image_viewer import _to_rgb
from ui.components.report_export import generate_text_report


def _build_archive(archive_data: dict, results_list: list) -> bytes:
    """
    Формирует ZIP-архив с результатами анализа всех снимков.

    Структура архива:
      submission.csv
      reports/{id}_report.txt
      reports/{id}_annotated.png
      reports/{id}_gradcam.png   (если доступен GradCAM)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # submission.csv
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["id", "class"])
        for row in results_list:
            if row["class"] >= 0:
                writer.writerow([row["id"], row["class"]])
        zf.writestr("submission.csv", csv_buf.getvalue())

        # По каждому снимку
        for image_id, data in archive_data.items():
            # Текстовый отчёт
            report_text = generate_text_report(data)
            zf.writestr(f"reports/{image_id}_report.txt", report_text)

            # Размеченный снимок (геометрическая разметка)
            annotated = data.get("annotated_image")
            if annotated is not None:
                _, png = cv2.imencode(".png", annotated)
                zf.writestr(f"reports/{image_id}_annotated.png", png.tobytes())

            # Тепловая карта GradCAM
            heatmap = data.get("heatmap_overlay")
            if heatmap is not None:
                _, png = cv2.imencode(".png", heatmap)
                zf.writestr(f"reports/{image_id}_gradcam.png", png.tobytes())

    buf.seek(0)
    return buf.read()


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

    save_annotated = st.checkbox("Показывать размеченные снимки в галерее", value=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("🚀 Запустить", type="primary", use_container_width=True)

    if not start_btn:
        st.markdown("""
        ### Формат результата

        - **submission.csv** — файл вида `id,class` (0 = норма, 1 = патология)
        - **Архив результатов (.zip)** — для каждого снимка: текстовый отчёт,
          размеченное изображение и тепловая карта GradCAM (если классификатор загружен)
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

    # --- Обработка (student mode — нужен GradCAM для архива) ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    results_list = []   # для таблицы и CSV
    archive_data = {}   # image_id → данные для архива (без тяжёлых layer_images)
    annotated_images = {}  # для галереи

    for i, (image_id, result) in enumerate(
        pipeline.run_batch(image_files, plugin_name, mode='student')
    ):
        progress = (i + 1) / len(image_files)
        progress_bar.progress(progress)
        status_text.text(f"Обработано: {i + 1}/{len(image_files)} — {image_id}")

        error_msg = ""
        if "error" in result:
            pathology_class = -1
            error_msg = result["error"]
        else:
            pathology_class = 1 if result.get("pathology_detected", False) else 0

        confidence = 0.0
        classification = result.get("classification")
        if classification:
            confidence = classification.get("confidence", 0.0)

        results_list.append({
            "id": image_id,
            "class": pathology_class,
            "confidence": round(confidence, 3),
            "error": error_msg,
        })

        if "error" not in result:
            # Сохраняем только нужное для архива (экономим память)
            archive_data[image_id] = {
                "plugin_metadata":    result.get("plugin_metadata"),
                "metrics":            result.get("metrics"),
                "classification":     result.get("classification"),
                "pathology_detected": result.get("pathology_detected"),
                "geometric_pathology":  result.get("geometric_pathology"),
                "geometric_confidence": result.get("geometric_confidence"),
                "resnet_pathology":   result.get("resnet_pathology"),
                "resnet_confidence":  result.get("resnet_confidence"),
                "annotated_image":    result.get("annotated_image"),
                "heatmap_overlay":    result.get("heatmap_overlay"),
            }

        if save_annotated and "annotated_image" in result:
            annotated_images[image_id] = result["annotated_image"]

    progress_bar.progress(1.0)
    status_text.text(f"Готово! Обработано {len(image_files)} снимков.")

    st.divider()

    # --- Таблица результатов ---
    st.subheader("Результаты")
    import pandas as pd
    df = pd.DataFrame(results_list)
    st.dataframe(df[["id", "class", "confidence"]], use_container_width=True)

    errors = [r for r in results_list if r["class"] == -1]
    if errors:
        with st.expander(f"⚠️ Детали ошибок ({len(errors)})"):
            for r in errors:
                st.text(f"{r['id']}: {r['error']}")

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

    # --- Скачивание ---
    dl_col1, dl_col2 = st.columns(2)

    # submission.csv
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["id", "class"])
    for r in results_list:
        if r["class"] >= 0:
            writer.writerow([r["id"], r["class"]])

    with dl_col1:
        st.download_button(
            "📥 Скачать submission.csv",
            data=csv_buf.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
        )

    # Архив результатов
    with dl_col2:
        if archive_data:
            archive_bytes = _build_archive(archive_data, results_list)
            st.download_button(
                "📦 Скачать архив результатов (.zip)",
                data=archive_bytes,
                file_name="results_archive.zip",
                mime="application/zip",
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
                result_row = results_list[idx]
                label = "⚠️ Патология" if result_row["class"] == 1 else "✅ Норма"
                with col:
                    st.image(_to_rgb(img), caption=f"{img_id} — {label}", use_container_width=True)
