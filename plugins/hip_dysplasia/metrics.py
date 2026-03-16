"""
Расчёт диагностических метрик дисплазии тазобедренного сустава.

Реализованные метрики:
  - Ацетабулярный индекс (угол Хильгенрейнера)
  - Проверка нарушения линии Перкина
  - Диагностика патологии по пороговым значениям
"""

import numpy as np
from typing import Dict, Tuple


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Угол между двумя 2D-векторами в градусах (0..180).

    Args:
        v1: Первый вектор (2,)
        v2: Второй вектор (2,)

    Returns:
        Угол в градусах
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_hilgenreiner_angle(keypoints: dict, side: str = "left") -> float:
    """
    Расчёт ацетабулярного индекса (угла Хильгенрейнера).

    Линия Хильгенрейнера проходит через оба Y-образных хряща (L_TRC → R_TRC).
    Ацетабулярный угол — угол между линией Хильгенрейнера и линией от
    Y-хряща до края крыши вертлужной впадины (TRC → ACE) на соответствующей стороне.

    Норма: < 30° у новорожденных.
    Патология: > 30° указывает на дисплазию.

    Args:
        keypoints: dict {имя_точки: (x, y, conf)} — обнаруженные ключевые точки
        side: "left" или "right"

    Returns:
        Угол в градусах
    """
    l_trc = np.array(keypoints["L_TRC"][:2])
    r_trc = np.array(keypoints["R_TRC"][:2])

    # Вектор линии Хильгенрейнера (базовая горизонтальная линия)
    hilgenreiner_vec = r_trc - l_trc

    if side == "left":
        ace = np.array(keypoints["L_ACE"][:2])
        ace_vec = ace - l_trc
    else:
        ace = np.array(keypoints["R_ACE"][:2])
        ace_vec = ace - r_trc
        # Инвертируем направление для правой стороны
        hilgenreiner_vec = l_trc - r_trc

    return calculate_angle_between_vectors(hilgenreiner_vec, ace_vec)


def check_pathology(angle_left: float, angle_right: float,
                    threshold: float = 30.0) -> Dict:
    """
    Определение патологии по ацетабулярным индексам обеих сторон.

    Args:
        angle_left: Ацетабулярный угол левой стороны (градусы)
        angle_right: Ацетабулярный угол правой стороны (градусы)
        threshold: Пороговое значение (default: 30°)

    Returns:
        dict с результатами диагностики:
          - left: {angle, is_pathology}
          - right: {angle, is_pathology}
          - any_pathology: bool — есть ли патология хотя бы с одной стороны
    """
    return {
        "left": {
            "angle": round(angle_left, 1),
            "is_pathology": angle_left > threshold,
        },
        "right": {
            "angle": round(angle_right, 1),
            "is_pathology": angle_right > threshold,
        },
        "any_pathology": angle_left > threshold or angle_right > threshold,
        "threshold": threshold,
    }


def calculate_perkin_line_violation(keypoints: dict, side: str = "left") -> Dict:
    """
    Проверка положения головки бедра относительно линии Перкина.

    Линия Перкина — вертикаль через край крыши вертлужной впадины (ACE),
    перпендикулярная линии Хильгенрейнера.
    Норма: центр головки бедра (или метафиза) медиальнее линии Перкина.

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        side: "left" или "right"

    Returns:
        dict с результатом:
          - violation: bool — True если головка латеральнее линии Перкина
          - femoral_point: (x, y) — использованная точка головки/метафиза
          - perkin_x: float — x-координата линии Перкина
    """
    l_trc = np.array(keypoints["L_TRC"][:2])
    r_trc = np.array(keypoints["R_TRC"][:2])

    # Вектор Хильгенрейнера
    h_vec = r_trc - l_trc
    # Перпендикуляр (повернуть на 90°)
    perp = np.array([-h_vec[1], h_vec[0]])
    perp_norm = perp / (np.linalg.norm(perp) + 1e-8)

    if side == "left":
        ace = np.array(keypoints["L_ACE"][:2])
        # Используем головку бедра или метафиз
        fhc_conf = keypoints.get("L_FHC", (0, 0, 0))[2]
        if fhc_conf > 0.3:
            femoral = np.array(keypoints["L_FHC"][:2])
        else:
            femoral = np.array(keypoints["L_FMM"][:2])
    else:
        ace = np.array(keypoints["R_ACE"][:2])
        fhc_conf = keypoints.get("R_FHC", (0, 0, 0))[2]
        if fhc_conf > 0.3:
            femoral = np.array(keypoints["R_FHC"][:2])
        else:
            femoral = np.array(keypoints["R_FMM"][:2])

    # Проекция вектора (femoral - ace) на перпендикуляр Хильгенрейнера
    # Положительная проекция = латеральнее (нарушение на левой стороне)
    vec_to_femoral = femoral - ace
    projection = np.dot(vec_to_femoral, perp_norm)

    # Для левой стороны: латерально = отрицательный x (влево)
    # Для правой стороны: латерально = положительный x (вправо)
    if side == "left":
        violation = projection < 0  # Головка левее ACE
    else:
        violation = projection > 0  # Головка правее ACE

    return {
        "violation": bool(violation),
        "femoral_point": femoral.tolist(),
        "perkin_x": float(ace[0]),
    }


def calculate_all_metrics(keypoints: dict, threshold: float = 30.0) -> Dict:
    """
    Расчёт всех диагностических метрик.

    Args:
        keypoints: dict {имя_точки: (x, y, conf)}
        threshold: Пороговое значение ацетабулярного угла

    Returns:
        dict со всеми метриками
    """
    # Проверяем наличие минимально необходимых точек
    required = ["L_TRC", "R_TRC", "L_ACE", "R_ACE"]
    for name in required:
        if name not in keypoints or keypoints[name][2] < 0.1:
            return {
                "error": f"Не обнаружена ключевая точка: {name}",
                "valid": False,
            }

    # Ацетабулярные углы
    angle_left = calculate_hilgenreiner_angle(keypoints, side="left")
    angle_right = calculate_hilgenreiner_angle(keypoints, side="right")
    pathology = check_pathology(angle_left, angle_right, threshold)

    result = {
        "valid": True,
        "hilgenreiner_angle_left": round(angle_left, 1),
        "hilgenreiner_angle_right": round(angle_right, 1),
        "pathology": pathology,
    }

    # Линия Перкина (если доступны головки/метафизы)
    for side in ["left", "right"]:
        prefix = "L" if side == "left" else "R"
        fhc_key = f"{prefix}_FHC"
        fmm_key = f"{prefix}_FMM"
        has_femoral = (
            (fhc_key in keypoints and keypoints[fhc_key][2] > 0.3) or
            (fmm_key in keypoints and keypoints[fmm_key][2] > 0.3)
        )
        if has_femoral:
            perkin = calculate_perkin_line_violation(keypoints, side=side)
            result[f"perkin_violation_{side}"] = perkin["violation"]

    return result
