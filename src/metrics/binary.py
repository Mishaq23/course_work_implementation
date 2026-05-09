import torch
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_average_precision,
    binary_f1_score,
    binary_precision,
    binary_recall,
    binary_specificity,
)

from src.metrics.base_metric import BaseMetric


def logits_to_binary_probs(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()

    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)

    if logits.ndim == 2 and logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[:, 1]

    return torch.sigmoid(logits)


class BinaryMetric(BaseMetric):
    def __init__(self, device="auto", threshold: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._probs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        probs = logits_to_binary_probs(logits).detach().cpu()
        labels = labels.detach().long().cpu()

        if probs.ndim != 1:
            probs = probs.reshape(-1)

        if labels.ndim != 1:
            labels = labels.reshape(-1)

        self._probs.append(probs)
        self._labels.append(labels)

    def compute(self):
        if len(self._probs) == 0:
            return float("nan")

        probs = torch.cat(self._probs, dim=0)
        labels = torch.cat(self._labels, dim=0)
        value = self._compute_metric(probs, labels)

        if isinstance(value, torch.Tensor):
            value = value.item()

        return float(value)

    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        raise NotImplementedError()


class AccuracyMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        return binary_accuracy(probs, labels, threshold=self.threshold)


class PrecisionMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        return binary_precision(probs, labels, threshold=self.threshold)


class RecallMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        return binary_recall(probs, labels, threshold=self.threshold)


class SpecificityMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        if labels.unique().numel() < 2:
            return float("nan")

        return binary_specificity(probs, labels, threshold=self.threshold)


class F1Metric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        return binary_f1_score(probs, labels, threshold=self.threshold)


class BalancedAccuracyMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        if labels.unique().numel() < 2:
            return float("nan")

        recall = binary_recall(probs, labels, threshold=self.threshold)
        specificity = binary_specificity(probs, labels, threshold=self.threshold)
        return (recall + specificity) / 2.0


class AUROCMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        if labels.unique().numel() < 2:
            return float("nan")

        return binary_auroc(probs, labels)


class AveragePrecisionMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        if labels.unique().numel() < 2:
            return float("nan")

        return binary_average_precision(probs, labels)
