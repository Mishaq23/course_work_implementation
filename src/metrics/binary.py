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


def _prepare_binary_probs_and_labels(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = probs.reshape(-1).detach().cpu().to(torch.float64)
    labels = labels.reshape(-1).detach().cpu().to(torch.long)
    return probs, labels


def _binary_confusion_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs, labels = _prepare_binary_probs_and_labels(probs, labels)

    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        nan_tensor = torch.tensor([float("nan")], dtype=torch.float64)
        return nan_tensor, nan_tensor, nan_tensor, nan_tensor, nan_tensor

    sorted_indices = torch.argsort(probs, descending=True)
    sorted_probs = probs[sorted_indices]
    sorted_labels = labels[sorted_indices]

    tp_cumsum = torch.cumsum((sorted_labels == 1).to(torch.float64), dim=0)
    fp_cumsum = torch.cumsum((sorted_labels == 0).to(torch.float64), dim=0)

    threshold_ends = torch.ones_like(sorted_probs, dtype=torch.bool)
    threshold_ends[:-1] = sorted_probs[:-1] != sorted_probs[1:]

    thresholds = sorted_probs[threshold_ends]
    true_positives = tp_cumsum[threshold_ends]
    false_positives = fp_cumsum[threshold_ends]

    # Candidate above the largest score corresponds to predicting every sample as negative.
    all_negative_threshold = torch.nextafter(
        sorted_probs[:1],
        torch.full_like(sorted_probs[:1], float("inf")),
    )
    thresholds = torch.cat([all_negative_threshold, thresholds], dim=0)
    true_positives = torch.cat(
        [torch.zeros(1, dtype=torch.float64), true_positives],
        dim=0,
    )
    false_positives = torch.cat(
        [torch.zeros(1, dtype=torch.float64), false_positives],
        dim=0,
    )

    true_negatives = negatives - false_positives
    false_negatives = positives - true_positives
    return thresholds, true_positives, false_positives, true_negatives, false_negatives


def _threshold_objective_values(
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    true_negatives: torch.Tensor,
    false_negatives: torch.Tensor,
    objective: str,
) -> torch.Tensor:
    if objective == "accuracy":
        total = true_positives + false_positives + true_negatives + false_negatives
        return (true_positives + true_negatives) / total.clamp_min(1e-12)

    if objective == "balanced_accuracy":
        recall = true_positives / (true_positives + false_negatives).clamp_min(1e-12)
        specificity = true_negatives / (true_negatives + false_positives).clamp_min(1e-12)
        return (recall + specificity) / 2.0

    if objective == "f1":
        denom = (2.0 * true_positives) + false_positives + false_negatives
        values = torch.zeros_like(denom)
        valid = denom > 0
        values[valid] = (2.0 * true_positives[valid]) / denom[valid]
        return values

    raise ValueError(
        f"Unknown threshold objective={objective!r}. "
        "Expected one of: ['accuracy', 'balanced_accuracy', 'f1']."
    )


def find_optimal_threshold(
    probs: torch.Tensor,
    labels: torch.Tensor,
    objective: str = "balanced_accuracy",
) -> tuple[float, float]:
    thresholds, tp, fp, tn, fn = _binary_confusion_curve(probs, labels)
    if torch.isnan(thresholds).all():
        return float("nan"), float("nan")

    objective_values = _threshold_objective_values(tp, fp, tn, fn, objective=objective)
    best_value = torch.max(objective_values)
    best_candidates = torch.nonzero(
        torch.isclose(
            objective_values,
            best_value,
            atol=1e-12,
            rtol=0.0,
        ),
        as_tuple=False,
    ).reshape(-1)

    if best_candidates.numel() == 1:
        best_index = int(best_candidates[0].item())
    else:
        candidate_thresholds = thresholds[best_candidates]
        # Prefer a threshold close to the common default when several are equally good.
        tie_break_index = torch.argmin(torch.abs(candidate_thresholds - 0.5))
        best_index = int(best_candidates[tie_break_index].item())

    return float(thresholds[best_index].item()), float(best_value.item())


def compute_equal_error_rate(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Equal Error Rate from binary probabilities and labels.

    The implementation aggregates operating points over unique score
    thresholds, then linearly interpolates the crossing between
    False Positive Rate and False Negative Rate.
    """
    if labels.unique().numel() < 2:
        return float("nan")

    probs = probs.reshape(-1).detach().cpu().to(torch.float64)
    labels = labels.reshape(-1).detach().cpu().to(torch.long)

    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return float("nan")

    sorted_indices = torch.argsort(probs, descending=True)
    sorted_probs = probs[sorted_indices]
    sorted_labels = labels[sorted_indices]

    tp_cumsum = torch.cumsum((sorted_labels == 1).to(torch.float64), dim=0)
    fp_cumsum = torch.cumsum((sorted_labels == 0).to(torch.float64), dim=0)

    threshold_ends = torch.ones_like(sorted_probs, dtype=torch.bool)
    threshold_ends[:-1] = sorted_probs[:-1] != sorted_probs[1:]

    true_positive_rate = tp_cumsum[threshold_ends] / positives
    false_positive_rate = fp_cumsum[threshold_ends] / negatives
    false_negative_rate = 1.0 - true_positive_rate

    false_positive_rate = torch.cat(
        [torch.tensor([0.0], dtype=torch.float64), false_positive_rate]
    )
    false_negative_rate = torch.cat(
        [torch.tensor([1.0], dtype=torch.float64), false_negative_rate]
    )

    difference = false_positive_rate - false_negative_rate
    exact_match = torch.isclose(
        difference,
        torch.zeros_like(difference),
        atol=1e-12,
        rtol=0.0,
    )
    if exact_match.any():
        return float(false_positive_rate[exact_match][0].item())

    sign_change = difference[:-1] * difference[1:] < 0
    if sign_change.any():
        left_idx = int(torch.nonzero(sign_change, as_tuple=False)[0].item())
        right_idx = left_idx + 1

        left_diff = difference[left_idx]
        right_diff = difference[right_idx]
        alpha = left_diff / (left_diff - right_diff)

        eer_fpr = false_positive_rate[left_idx] + alpha * (
            false_positive_rate[right_idx] - false_positive_rate[left_idx]
        )
        eer_fnr = false_negative_rate[left_idx] + alpha * (
            false_negative_rate[right_idx] - false_negative_rate[left_idx]
        )
        return float(((eer_fpr + eer_fnr) / 2.0).item())

    best_idx = int(torch.argmin(torch.abs(difference)).item())
    return float(
        (
            false_positive_rate[best_idx] + false_negative_rate[best_idx]
        ).div(2.0).item()
    )


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

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

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


class BestBalancedAccuracyMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        if labels.unique().numel() < 2:
            return float("nan")

        _, best_value = find_optimal_threshold(
            probs,
            labels,
            objective="balanced_accuracy",
        )
        return best_value


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


class EERMetric(BinaryMetric):
    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        return compute_equal_error_rate(probs, labels)


class OptimalThresholdMetric(BinaryMetric):
    def __init__(
        self,
        objective: str = "balanced_accuracy",
        *args,
        **kwargs,
    ):
        self.objective = objective
        super().__init__(*args, **kwargs)

    def _compute_metric(self, probs: torch.Tensor, labels: torch.Tensor):
        if labels.unique().numel() < 2:
            return float("nan")

        threshold, _ = find_optimal_threshold(
            probs,
            labels,
            objective=self.objective,
        )
        return threshold
