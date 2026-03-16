import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List


def _compute_metrics(
    all_targets: List[int],
    all_preds: List[int],
    all_probs: List[float],
    loss: float,
) -> Dict[str, float]:
    """
    Вычисляет метрики для бинарной классификации.
    Метрики:
    - loss       — среднее значение функции потерь
    - accuracy   — доля правильных предсказаний
    - precision  — точность (из всех предсказанных патологий, сколько реальных)
    - recall     — полнота / чувствительность (из всех реальных патологий, сколько найдено)
    - f1         — гармоническое среднее precision и recall
    - specificity— специфичность (из всех норм, сколько верно определено как норма)
    - auc_roc    — площадь под ROC-кривой (приближение через трапеции)
    """
    n = len(all_targets)
    tp = sum(p == 1 and t == 1 for p, t in zip(all_preds, all_targets))
    tn = sum(p == 0 and t == 0 for p, t in zip(all_preds, all_targets))
    fp = sum(p == 1 and t == 0 for p, t in zip(all_preds, all_targets))
    fn = sum(p == 0 and t == 1 for p, t in zip(all_preds, all_targets))

    accuracy    = (tp + tn) / n if n > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC-ROC через трапециевидное правило
    paired = sorted(zip(all_probs, all_targets), key=lambda x: -x[0])
    pos_total = sum(all_targets)
    neg_total = n - pos_total
    auc = 0.0
    if pos_total > 0 and neg_total > 0:
        tp_cur = fp_cur = 0
        prev_tpr = prev_fpr = 0.0
        for _, label in paired:
            if label == 1:
                tp_cur += 1
            else:
                fp_cur += 1
            tpr = tp_cur / pos_total
            fpr = fp_cur / neg_total
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_tpr, prev_fpr = tpr, fpr

    return {
        "loss":        loss,
        "accuracy":    accuracy * 100,
        "precision":   precision * 100,
        "recall":      recall * 100,
        "f1":          f1 * 100,
        "specificity": specificity * 100,
        "auc_roc":     auc * 100,
    }


def _format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"Loss: {metrics['loss']:.4f}  "
        f"Acc: {metrics['accuracy']:.1f}%  "
        f"P: {metrics['precision']:.1f}%  "
        f"R: {metrics['recall']:.1f}%  "
        f"F1: {metrics['f1']:.1f}%  "
        f"Spec: {metrics['specificity']:.1f}%  "
        f"AUC: {metrics['auc_roc']:.1f}%"
    )


class MedicalTrainer:
    """
    Универсальный класс для цикла обучения.
    Скрывает в себе бойлерплейт код PyTorch.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Обучение за одну эпоху, возвращает метрики на train."""
        self.model.train()
        running_loss = 0.0
        all_targets: List[int] = []
        all_preds:   List[int] = []
        all_probs:   List[float] = []

        pbar = tqdm(self.train_loader, desc="Train")
        for images, targets in pbar:
            images  = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            probs = F.softmax(outputs.detach(), dim=1)[:, 1]
            preds = torch.argmax(outputs.detach(), dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.train_loader)
        return _compute_metrics(all_targets, all_preds, all_probs, avg_loss)

    def validate_epoch(self) -> Dict[str, float]:
        """Валидация за одну эпоху, возвращает метрики на val."""
        self.model.eval()
        running_loss = 0.0
        all_targets: List[int] = []
        all_preds:   List[int] = []
        all_probs:   List[float] = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Val  ")
            for images, targets in pbar:
                images  = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                probs = F.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_targets.extend(targets.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.val_loader)
        return _compute_metrics(all_targets, all_preds, all_probs, avg_loss)

    def fit(self, epochs: int, save_path: str):
        """Основной цикл обучения."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")

            train_metrics = self.train_epoch()
            val_metrics   = self.validate_epoch()

            print(f"  Train  | {_format_metrics(train_metrics)}")
            print(f"  Val    | {_format_metrics(val_metrics)}")

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                torch.save(self.model.state_dict(), save_path)
                print(f"  >> Best model saved (val_loss={self.best_val_loss:.4f})")
