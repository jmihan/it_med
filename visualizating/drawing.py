"""
Утилиты для наложения графики на медицинские снимки.

Визуализация:
  - Ключевые точки с цветовой кодировкой
  - Линия Хильгенрейнера
  - Ацетабулярные углы с дугами
  - Линии Перкина
  - Heatmap (GradCAM)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ImageAnnotator:
    """Утилиты для наложения графики на медицинские снимки."""

    # Цветовая кодировка точек (BGR для OpenCV)
    KEYPOINT_COLORS = {
        "L_TRC": (0, 255, 0),     # Зелёный — Y-хрящи
        "R_TRC": (0, 255, 0),
        "L_ACE": (0, 0, 255),     # Красный — края крыши
        "R_ACE": (0, 0, 255),
        "L_FHC": (255, 0, 0),     # Синий — головки бедра
        "R_FHC": (255, 0, 0),
        "L_FMM": (0, 165, 255),   # Оранжевый — метафизы
        "R_FMM": (0, 165, 255),
    }

    # Русские подписи для точек
    KEYPOINT_LABELS = {
        "L_TRC": "Y-хрящ Л",
        "R_TRC": "Y-хрящ П",
        "L_ACE": "Крыша Л",
        "R_ACE": "Крыша П",
        "L_FHC": "Головка Л",
        "R_FHC": "Головка П",
        "L_FMM": "Метафиз Л",
        "R_FMM": "Метафиз П",
    }

    @staticmethod
    def draw_keypoints(image: np.ndarray, keypoints: Dict,
                       radius: int = 5, show_labels: bool = True,
                       min_conf: float = 0.3) -> np.ndarray:
        """
        Отрисовка ключевых точек с цветовой кодировкой и подписями.

        Args:
            image: Изображение BGR
            keypoints: dict {имя: (x, y, conf)}
            radius: Радиус точки
            show_labels: Показывать подписи
            min_conf: Минимальная уверенность для отрисовки
        """
        img = image.copy()

        for name, (x, y, conf) in keypoints.items():
            if conf < min_conf:
                continue

            pt = (int(x), int(y))
            color = ImageAnnotator.KEYPOINT_COLORS.get(name, (255, 255, 255))

            # Точка
            cv2.circle(img, pt, radius, color, -1)
            cv2.circle(img, pt, radius + 1, (255, 255, 255), 1)

            # Подпись
            if show_labels:
                label = ImageAnnotator.KEYPOINT_LABELS.get(name, name)
                label_text = f"{label} ({conf:.0%})"
                cv2.putText(img, label_text, (pt[0] + radius + 3, pt[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return img

    @staticmethod
    def draw_hilgenreiner_line(image: np.ndarray, keypoints: Dict,
                                color: Tuple = (0, 255, 0),
                                thickness: int = 2,
                                extend: float = 0.1) -> np.ndarray:
        """
        Отрисовка линии Хильгенрейнера (через L_TRC и R_TRC).

        Args:
            image: Изображение BGR
            keypoints: dict {имя: (x, y, conf)}
            color: Цвет линии (BGR)
            thickness: Толщина линии
            extend: Коэффициент удлинения линии за пределы точек
        """
        img = image.copy()

        l_trc = keypoints.get("L_TRC")
        r_trc = keypoints.get("R_TRC")
        if l_trc is None or r_trc is None:
            return img
        if l_trc[2] < 0.3 or r_trc[2] < 0.3:
            return img

        p1 = np.array(l_trc[:2])
        p2 = np.array(r_trc[:2])

        # Удлиняем линию в обе стороны
        direction = p2 - p1
        p1_ext = p1 - direction * extend
        p2_ext = p2 + direction * extend

        cv2.line(img, (int(p1_ext[0]), int(p1_ext[1])),
                 (int(p2_ext[0]), int(p2_ext[1])), color, thickness, cv2.LINE_AA)

        # Подпись
        mid = ((p1 + p2) / 2).astype(int)
        cv2.putText(img, "Hilgenreiner", (mid[0] - 40, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return img

    @staticmethod
    def draw_acetabular_angles(image: np.ndarray, keypoints: Dict,
                                angles: Dict,
                                color_normal: Tuple = (0, 255, 0),
                                color_pathology: Tuple = (0, 0, 255),
                                thickness: int = 2,
                                arc_radius: int = 40) -> np.ndarray:
        """
        Отрисовка ацетабулярных углов с дугами и числовыми значениями.

        Args:
            image: Изображение BGR
            keypoints: dict {имя: (x, y, conf)}
            angles: dict с полями:
              - hilgenreiner_angle_left: float
              - hilgenreiner_angle_right: float
              - pathology: dict с is_pathology для каждой стороны
            color_normal: Цвет для нормы
            color_pathology: Цвет для патологии
        """
        img = image.copy()

        l_trc = keypoints.get("L_TRC")
        r_trc = keypoints.get("R_TRC")
        if l_trc is None or r_trc is None:
            return img

        for side in ["left", "right"]:
            prefix = "L" if side == "left" else "R"
            ace_key = f"{prefix}_ACE"
            trc_key = f"{prefix}_TRC"

            ace = keypoints.get(ace_key)
            trc = keypoints.get(trc_key)
            if ace is None or trc is None:
                continue
            if ace[2] < 0.3 or trc[2] < 0.3:
                continue

            angle = angles.get(f"hilgenreiner_angle_{side}", 0)
            is_pathology = angles.get("pathology", {}).get(side, {}).get("is_pathology", False)
            color = color_pathology if is_pathology else color_normal

            trc_pt = (int(trc[0]), int(trc[1]))
            ace_pt = (int(ace[0]), int(ace[1]))

            # Линия TRC → ACE
            cv2.line(img, trc_pt, ace_pt, color, thickness, cv2.LINE_AA)

            # Дуга угла
            h_vec = np.array(r_trc[:2]) - np.array(l_trc[:2])
            if side == "right":
                h_vec = -h_vec
            h_angle = np.degrees(np.arctan2(-h_vec[1], h_vec[0]))

            ace_vec = np.array(ace[:2]) - np.array(trc[:2])
            ace_angle = np.degrees(np.arctan2(-ace_vec[1], ace_vec[0]))

            start_angle = min(h_angle, ace_angle)
            end_angle = max(h_angle, ace_angle)

            cv2.ellipse(img, trc_pt, (arc_radius, arc_radius),
                        0, -end_angle, -start_angle, color, 1, cv2.LINE_AA)

            # Подпись с углом
            label = f"{angle:.1f}{chr(176)}"
            status = "PATHO" if is_pathology else "NORM"
            label_full = f"{label} [{status}]"

            # Позиция подписи — по дуге
            mid_angle = np.radians((start_angle + end_angle) / 2)
            label_x = int(trc[0] + (arc_radius + 15) * np.cos(mid_angle))
            label_y = int(trc[1] - (arc_radius + 15) * np.sin(mid_angle))

            cv2.putText(img, label_full, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return img

    @staticmethod
    def draw_perkin_lines(image: np.ndarray, keypoints: Dict,
                          color: Tuple = (0, 255, 255),
                          thickness: int = 1,
                          line_length: int = 200) -> np.ndarray:
        """
        Отрисовка линий Перкина (вертикали через ACE точки, перпендикулярные Хильгенрейнеру).

        Args:
            image: Изображение BGR
            keypoints: dict {имя: (x, y, conf)}
            color: Цвет линии (BGR)
            thickness: Толщина линии
            line_length: Длина линии в пикселях
        """
        img = image.copy()

        l_trc = keypoints.get("L_TRC")
        r_trc = keypoints.get("R_TRC")
        if l_trc is None or r_trc is None:
            return img

        # Вектор Хильгенрейнера
        h_vec = np.array(r_trc[:2]) - np.array(l_trc[:2])
        # Перпендикуляр (вниз)
        perp = np.array([h_vec[1], -h_vec[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-8) * line_length

        for ace_key in ["L_ACE", "R_ACE"]:
            ace = keypoints.get(ace_key)
            if ace is None or ace[2] < 0.3:
                continue

            ace_pt = np.array(ace[:2])
            p1 = ace_pt - perp * 0.2  # Немного выше
            p2 = ace_pt + perp        # Вниз

            cv2.line(img, (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])),
                     color, thickness, cv2.LINE_AA)

            cv2.putText(img, "Perkin", (int(ace[0]) + 5, int(ace[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        return img

    @staticmethod
    def draw_full_analysis(image: np.ndarray, keypoints: Dict,
                            metrics: Dict,
                            show_labels: bool = True) -> np.ndarray:
        """
        Полная отрисовка: точки + линии + углы + линии Перкина.

        Args:
            image: Изображение BGR
            keypoints: dict {имя: (x, y, conf)}
            metrics: dict из calculate_all_metrics()
            show_labels: Показывать подписи точек
        """
        if not metrics.get("valid"):
            return ImageAnnotator.draw_keypoints(image, keypoints,
                                                  show_labels=show_labels)

        # Последовательно накладываем слои
        img = ImageAnnotator.draw_hilgenreiner_line(image, keypoints)
        img = ImageAnnotator.draw_perkin_lines(img, keypoints)
        img = ImageAnnotator.draw_acetabular_angles(img, keypoints, metrics)
        img = ImageAnnotator.draw_keypoints(img, keypoints,
                                             show_labels=show_labels)

        return img

    @staticmethod
    def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
        """
        Наложение XAI heatmap поверх исходного изображения.

        Args:
            image: Исходное изображение BGR
            heatmap: Карта внимания (float32, [0, 1])
            alpha: Прозрачность наложения
        """
        if heatmap is None:
            return image.copy()

        # Масштабируем heatmap под размер изображения
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Конвертируем в цветную карту
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Накладываем
        return cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
