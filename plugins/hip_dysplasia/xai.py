"""
Генератор текстовых объяснений для образовательного режима.
Формирует пошаговое описание на основе реальных метрик анализа.
"""

from typing import Dict, List

_QUADRANT_LABELS = {
    "inner_lower": "внутренний нижний (норма)",
    "outer_lower": "наружный нижний (подвывих)",
    "outer_upper": "наружный верхний (вывих)",
    "inner_upper": "внутренний верхний",
}

_DIAGNOSIS_LABELS = {
    "norm": "Норма",
    "subluxation": "Подвывих",
    "dislocation": "Вывих",
}


def generate_explanation(results: dict) -> List[Dict[str, str]]:
    """
    Генерация пошагового объяснения на основе результатов анализа.

    Args:
        results: Полный результат из pipeline (содержит metrics, keypoints, classification)

    Returns:
        Список словарей [{title, text}] — шаги объяснения
    """
    metrics = results.get("metrics", {})
    keypoints = results.get("keypoints", {})
    classification = results.get("classification")
    pathology_detected = results.get("pathology_detected", False)
    method = results.get("method", "geometric")

    steps = []

    # --- Шаг 1: Детекция ключевых точек ---
    detected = []
    for name, (x, y, conf) in keypoints.items():
        if conf >= 0.3:
            detected.append(f"  - **{name}**: координаты ({int(x)}, {int(y)}), уверенность {conf:.0%}")

    kp_text = f"Модель-детектор (YOLO Pose) обнаружила **{len(detected)} из 10** ключевых точек:\n"
    kp_text += "\n".join(detected) if detected else "  Точки не найдены."
    kp_text += "\n\nТочки разделены на 5 групп: Y-хрящи (TRC, зелёные), края крыши впадины (ACE, красные), центры головок бедра (FHC, синие), метафизы (FMM, оранжевые), верхние края метафизов (FMP, жёлто-зелёные)."

    steps.append({
        "title": "Шаг 1: Поиск ключевых анатомических ориентиров",
        "text": kp_text,
    })

    # --- Шаг 2: Расчёт углов и расстояний ---
    if metrics.get("valid"):
        angle_l = metrics.get("hilgenreiner_angle_left", 0)
        angle_r = metrics.get("hilgenreiner_angle_right", 0)
        threshold = metrics.get("pathology", {}).get("threshold", 25)
        path_l = metrics.get("pathology", {}).get("left", {}).get("is_pathology", False)
        path_r = metrics.get("pathology", {}).get("right", {}).get("is_pathology", False)

        angle_text = "Через Y-хрящи (L\\_TRC и R\\_TRC) проведена **линия Хильгенрейнера** — базовая горизонтальная ось таза.\n\n"
        angle_text += "От неё измерены ацетабулярные углы — углы наклона крыши вертлужной впадины:\n"
        angle_text += f"  - **Левый**: {angle_l:.1f}° — {'⚠️ ПАТОЛОГИЯ' if path_l else '✅ Норма'}\n"
        angle_text += f"  - **Правый**: {angle_r:.1f}° — {'⚠️ ПАТОЛОГИЯ' if path_r else '✅ Норма'}\n\n"
        angle_text += f"Порог нормы для ~6 мес: **{threshold:.0f}°**. Превышение на 5°+ указывает на дисплазию."

        # Расстояние между Y-хрящами
        trc_mm = metrics.get("trc_distance_mm")
        trc_px = metrics.get("trc_distance_px")
        if trc_mm is not None:
            angle_text += f"\n\n**Расстояние между Y-хрящами** (L\\_TRC — R\\_TRC): **{trc_mm:.1f} мм** ({trc_px:.0f} px)"
        elif trc_px is not None:
            angle_text += f"\n\n**Расстояние между Y-хрящами**: {trc_px:.0f} px (масштаб мм/px недоступен)"

        # Расстояния h и d
        for side, label in [("left", "Левая"), ("right", "Правая")]:
            h_data = metrics.get(f"h_distance_{side}", {})
            d_data = metrics.get(f"d_distance_{side}", {})

            if h_data.get("h_px") is not None or d_data.get("d_px") is not None:
                angle_text += f"\n\n**Расстояния h/d ({label} сторона):**"

            if h_data.get("h_px") is not None:
                h_val = f"{h_data['h_mm']:.1f} мм" if h_data.get("h_mm") else f"{h_data['h_px']:.0f} px"
                h_status = ""
                if h_data.get("is_normal") is True:
                    h_status = " ✅ Норма (8–12 мм)"
                elif h_data.get("is_normal") is False:
                    h_status = " ⚠️ Отклонение (норма 8–12 мм)"
                angle_text += f"\n  - Высота h: **{h_val}**{h_status}"

            if d_data.get("d_px") is not None:
                d_val = f"{d_data['d_mm']:.1f} мм" if d_data.get("d_mm") else f"{d_data['d_px']:.0f} px"
                d_status = ""
                if d_data.get("is_normal") is True:
                    d_status = " ✅ Норма (10–15 мм)"
                elif d_data.get("is_normal") is False:
                    d_status = " ⚠️ Отклонение (норма 10–15 мм)"
                angle_text += f"\n  - Дистанция d: **{d_val}**{d_status}"

        # Линии Перкина с квадрантным анализом
        has_perkin = False
        for side, label in [("left", "Левая"), ("right", "Правая")]:
            perkin = metrics.get(f"perkin_{side}")
            if perkin is not None:
                if not has_perkin:
                    angle_text += "\n\n**Линии Перкина** (квадрантный анализ):\n"
                    has_perkin = True
                quadrant_label = _QUADRANT_LABELS.get(perkin.get("quadrant", ""), "?")
                diagnosis_label = _DIAGNOSIS_LABELS.get(perkin.get("diagnosis", ""), "?")
                icon = "✅" if perkin["diagnosis"] == "norm" else "⚠️"
                angle_text += f"  - {label}: {icon} квадрант — {quadrant_label}, диагноз: **{diagnosis_label}**\n"

        # Триада Путти
        putti = metrics.get("putti_triad")
        if putti:
            angle_text += f"\n\n**Триада Путти** ({putti['signs_present']}/3 признаков):\n"
            angle_text += f"  - Скошенность крыши: {'⚠️ Да' if putti['sign_1_roof_slope'] else '✅ Нет'}\n"
            angle_text += f"  - Смещение бедренной кости: {'⚠️ Да' if putti['sign_2_displacement'] else '✅ Нет'}\n"
            angle_text += f"  - Гипоплазия ядра окостенения: {'⚠️ Да' if putti['sign_3_ossification'] else '✅ Нет'}\n"
            if putti["triad_complete"]:
                angle_text += "  **Полная триада Путти — высокая вероятность дисплазии.**"
    else:
        angle_text = "Недостаточно точек для расчёта углов. Необходимы как минимум Y-хрящи (L\\_TRC, R\\_TRC) и края крыши (L\\_ACE, R\\_ACE)."

    steps.append({
        "title": "Шаг 2: Построение осей, расчёт углов и расстояний",
        "text": angle_text,
    })

    # --- Шаг 3: Классификация и GradCAM ---
    if classification:
        cls_name = classification.get("class_name", "?")
        conf = classification.get("confidence", 0)
        p_norm = classification.get("prob_normal", 0)
        p_path = classification.get("prob_pathology", 0)

        cam_text = f"Нейросеть-классификатор (ResNet) дала независимую оценку:\n"
        cam_text += f"  - **Предсказание**: {cls_name} (уверенность {conf:.0%})\n"
        cam_text += f"  - Вероятность нормы: {p_norm:.0%}\n"
        cam_text += f"  - Вероятность патологии: {p_path:.0%}\n\n"
        if method == "resnet_primary":
            cam_text += "**ResNet является основным методом принятия решения.** Геометрические метрики используются для визуализации и дополнительного обоснования.\n\n"
        cam_text += "Тепловая карта (Grad-CAM) показывает области ROI-кропа (выделенная YOLO область таза), на которые нейросеть обращала наибольшее внимание при классификации. Яркие области — зоны максимального влияния на решение."
    else:
        cam_text = "Классификатор ResNet не подключён. Тепловая карта Grad-CAM недоступна."
        if method == "geometric":
            cam_text += "\nРешение принято на основе геометрического анализа."

    steps.append({
        "title": "Шаг 3: Зоны внимания нейросети (Grad-CAM)",
        "text": cam_text,
    })

    # --- Итоговый вывод ---
    if pathology_detected:
        conclusion = "⚠️ **Заключение**: Обнаружены признаки дисплазии тазобедренного сустава.\n\n"
        if method == "resnet_primary":
            conclusion += "Решение принято классификатором ResNet. "
        if metrics.get("valid"):
            path_info = metrics.get("pathology", {})
            if path_info.get("any_pathology"):
                conclusion += "Ацетабулярный угол превышает пороговое значение, что подтверждает недоразвитие крыши вертлужной впадины. "
            putti = metrics.get("putti_triad", {})
            if putti.get("signs_present", 0) >= 2:
                conclusion += f"Выявлено {putti['signs_present']}/3 признаков триады Путти. "
        conclusion += "Рекомендуется консультация ортопеда."
    elif method == "insufficient_data":
        conclusion = "⚠️ **Заключение**: Невозможно сделать вывод — недостаточно данных для анализа."
    else:
        conclusion = "✅ **Заключение**: Признаков дисплазии не обнаружено.\n\n"
        if method == "resnet_primary":
            conclusion += "Решение принято классификатором ResNet. "
        if metrics.get("valid"):
            conclusion += "Ацетабулярные углы в пределах нормы, анатомические ориентиры соответствуют нормальному развитию тазобедренного сустава."

    steps.append({
        "title": "Итоговое заключение",
        "text": conclusion,
    })

    return steps
