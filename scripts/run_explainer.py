"""
Скрипт для визуализации областей влияния на решение классификатора (XAI / GradCAM).

Запуск:
    Базовый запуск — показывает GradCAM для предсказанного класса:

    python scripts/run_explainer.py --image data/processed/test/images/scan.png

    С обученными весами:
    python scripts/run_explainer.py --image scan.png --weights weights/classifier.pt

    Принудительно для класса Pathology:
    python scripts/run_explainer.py --image scan.png --class_id 1

    Сравнить Normal vs Pathology рядом:
    python scripts/run_explainer.py --image scan.png --both

    Сохранить в файл (без показа окна):
    python scripts/run_explainer.py --image scan.png --both --save result.png

    Сменить метод и backbone:
    python scripts/run_explainer.py --image scan.png --method gradcam++ --backbone resnet50

"""
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import ResNetClassifier
from visualizating.explainers import ModelExplainer


def parse_args():
    parser = argparse.ArgumentParser(description="GradCAM для медицинских снимков")
    parser.add_argument("--image",    required=True,          help="Путь к изображению (PNG/JPG/DICOM)")
    parser.add_argument("--weights",  default=None,           help="Путь к весам модели (.pt)")
    parser.add_argument("--backbone", default="resnet18",     help="resnet18 | resnet50")
    parser.add_argument("--method",   default="gradcam",      help="gradcam | gradcam++ | eigencam")
    parser.add_argument("--class_id", default=None, type=int, help="0=Normal, 1=Pathology. По умолчанию — предсказанный класс")
    parser.add_argument("--alpha",    default=0.5,  type=float, help="Прозрачность наложения (0..1)")
    parser.add_argument("--both",     action="store_true",    help="Показать карты для обоих классов")
    parser.add_argument("--save",     default=None,           help="Сохранить результат в файл вместо показа")
    parser.add_argument("--device",   default="cpu",          help="cpu | cuda")
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    """Загружает PNG/JPG или DICOM, возвращает BGR np.ndarray."""
    if path.lower().endswith(".dcm") or not path.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            import pydicom
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(float)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) * (255.0 / (img_max - img_min))
            img = img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        except Exception as e:
            print(f"[DICOM] Не удалось прочитать как DICOM: {e}, пробуем через OpenCV...")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


MAX_DISPLAY = 900  # максимальная сторона окна при первом показе


def show_or_save(title: str, image: np.ndarray, save_path: str = None):
    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Сохранено: {save_path}")
    else:
        h, w = image.shape[:2]
        scale = min(1.0, MAX_DISPLAY / max(h, w))
        display = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, display)
        cv2.resizeWindow(title, display.shape[1], display.shape[0])


def main():
    args = parse_args()

    # --- Загрузка изображения ---
    print(f"Загрузка изображения: {args.image}")
    image = load_image(args.image)
    print(f"  Размер: {image.shape[1]}x{image.shape[0]}")

    # --- Загрузка модели ---
    print(f"Инициализация модели: backbone={args.backbone}, device={args.device}")
    classifier = ResNetClassifier(
        weights_path=args.weights,
        device=args.device,
        backbone=args.backbone,
    )

    # --- Предсказание ---
    prediction = classifier.predict(image)
    print(
        f"  Предсказание: {prediction['class_name']} "
        f"(confidence={prediction['confidence']:.1%}, "
        f"p_normal={prediction['prob_normal']:.1%}, "
        f"p_pathology={prediction['prob_pathology']:.1%})"
    )

    # --- Explainer ---
    explainer = ModelExplainer(
        model=classifier.model,
        device=args.device,
        method=args.method,
    )
    print(f"  Метод XAI: {args.method}")

    if args.both:
        # Два оверлея: для класса Normal и Pathology
        result = explainer.explain_both_classes(image, alpha=args.alpha)

        # Собираем в одну картинку рядом
        label_normal    = _add_label(result["normal"],    "Normal (class 0)")
        label_pathology = _add_label(result["pathology"], "Pathology (class 1)")
        combined = np.hstack([label_normal, label_pathology])

        if args.save:
            base, ext = os.path.splitext(args.save)
            show_or_save("Both classes", combined, args.save)
        else:
            show_or_save("GradCAM — Normal vs Pathology", combined)
    else:
        overlay = explainer.explain(image, class_id=args.class_id, alpha=args.alpha)
        class_label = (
            prediction["class_name"] if args.class_id is None
            else ("Normal" if args.class_id == 0 else "Pathology")
        )
        overlay = _add_label(overlay, f"{args.method.upper()} → {class_label}")

        show_or_save(f"GradCAM [{class_label}]", overlay, args.save)

    if not args.save:
        print("Нажмите любую клавишу в окне OpenCV для выхода.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _add_label(image: np.ndarray, text: str) -> np.ndarray:
    """Добавляет текстовую подпись снизу изображения."""
    bar = np.zeros((36, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([image, bar])


if __name__ == "__main__":
    main()
