"""
Оценка качества диагностической системы на тестовой выборке.

Использование:
  python scripts/evaluate_system.py [--data-dir data/processed/test] [--geometric-only] [--classifier-only]

Считает: Accuracy, Precision, Recall, F1, Specificity, Confusion Matrix.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def collect_images(data_dir: Path):
    """Собрать изображения с ground truth из {normal,pathology}/ директорий."""
    images = []
    for label_dir, gt_label in [("normal", 0), ("pathology", 1)]:
        d = data_dir / label_dir
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                images.append((str(f), gt_label))
    return images


def evaluate(args):
    from core.image_io import load_image
    from core.pipeline import AnalysisPipeline
    from core.registry import register_all_plugins

    register_all_plugins()
    pipeline = AnalysisPipeline()

    data_dir = Path(args.data_dir)
    images = collect_images(data_dir)

    if not images:
        print(f"[ERROR] Не найдены изображения в {data_dir}/{{normal,pathology}}/")
        sys.exit(1)

    print(f"Найдено {len(images)} изображений ({sum(1 for _, g in images if g == 0)} норма, "
          f"{sum(1 for _, g in images if g == 1)} патология)")
    print("=" * 60)

    y_true = []
    y_pred_geometric = []
    y_pred_classifier = []
    errors = []

    for path, gt in images:
        image_id = Path(path).stem
        try:
            image = load_image(path)
            results = pipeline.run(image, "hip_dysplasia", mode="doctor")

            # Геометрический метод
            pathology_geo = results.get("pathology_detected", False)
            y_true.append(gt)
            y_pred_geometric.append(1 if pathology_geo else 0)

            # Классификатор
            cls = results.get("classification")
            if cls is not None:
                y_pred_classifier.append(cls.get("class_id", 0))
            else:
                y_pred_classifier.append(None)

            metrics = results.get("metrics", {})
            angle_l = metrics.get("hilgenreiner_angle_left", "?")
            angle_r = metrics.get("hilgenreiner_angle_right", "?")
            status = "ПАТОЛ" if pathology_geo else "НОРМА"
            gt_str = "ПАТОЛ" if gt == 1 else "НОРМА"
            match = "OK" if (1 if pathology_geo else 0) == gt else "MISS"

            print(f"  {image_id}: L={angle_l}\u00b0 R={angle_r}\u00b0 -> {status} (GT: {gt_str}) [{match}]")

        except Exception as e:
            errors.append((image_id, str(e)))
            print(f"  {image_id}: ERROR — {e}")

    print("=" * 60)

    if errors:
        print(f"\nОшибки: {len(errors)} из {len(images)}")

    # Метрики для геометрического метода
    print("\n--- Геометрический метод (YOLO + углы) ---")
    print_metrics(y_true, y_pred_geometric)

    # Метрики для классификатора
    y_cls_true = [y_true[i] for i in range(len(y_pred_classifier)) if y_pred_classifier[i] is not None]
    y_cls_pred = [p for p in y_pred_classifier if p is not None]
    if y_cls_pred:
        print("\n--- Классификатор (ResNet) ---")
        print_metrics(y_cls_true, y_cls_pred)
    else:
        print("\nКлассификатор ResNet: не загружен, метрики недоступны")

    # Сохранение результатов
    output = {
        "data_dir": str(data_dir),
        "total_images": len(images),
        "errors": len(errors),
        "geometric": compute_metrics_dict(y_true, y_pred_geometric),
    }
    if y_cls_pred:
        output["classifier"] = compute_metrics_dict(y_cls_true, y_cls_pred)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {output_path}")


def compute_metrics_dict(y_true, y_pred):
    """Вычислить метрики и вернуть как dict."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def print_metrics(y_true, y_pred):
    """Вывести метрики в консоль."""
    m = compute_metrics_dict(y_true, y_pred)
    cm = m["confusion_matrix"]

    print(f"  Accuracy:    {m['accuracy']:.1%}")
    print(f"  Precision:   {m['precision']:.1%}")
    print(f"  Recall:      {m['recall']:.1%}")
    print(f"  F1-score:    {m['f1']:.1%}")
    print(f"  Specificity: {m['specificity']:.1%}")
    print(f"  Confusion matrix: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")


def main():
    parser = argparse.ArgumentParser(
        description="Оценка качества диагностической системы"
    )
    parser.add_argument("--data-dir", default="data/processed/test",
                        help="Директория с тестовыми данными (default: data/processed/test)")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Путь для сохранения результатов (default: evaluation_results.json)")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
