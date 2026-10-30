"""Manual classification metrics for the lesson on evaluation."""

import numpy as np


def confusion(target_values: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    result = np.zeros((classes, classes), dtype=int)
    np.add.at(result, (target_values, predictions), 1)
    return result


def precision_recall_f1(confusion_matrix: np.ndarray, average: str) -> tuple[float, float, float]:
    true_positive = np.diag(confusion_matrix)
    predicted_positive = confusion_matrix.sum(axis=0)
    actual_positive = confusion_matrix.sum(axis=1)
    if average == "micro":
        precision = true_positive.sum() / predicted_positive.sum() if predicted_positive.sum() else 0.0
        recall = true_positive.sum() / actual_positive.sum() if actual_positive.sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return float(precision), float(recall), float(f1)
    precision = np.divide(
        true_positive, predicted_positive, out=np.zeros(len(true_positive), float),
        where=predicted_positive > 0,
    )
    recall = np.divide(
        true_positive, actual_positive, out=np.zeros(len(true_positive), float),
        where=actual_positive > 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(len(precision)),
        where=precision + recall > 0,
    )
    return float(precision.mean()), float(recall.mean()), float(f1.mean())


def roc_auc(target_values: np.ndarray, scores: np.ndarray) -> float:
    """Integrate ROC steps at each distinct score; tied scores move together."""
    positives = np.sum(target_values == 1)
    negatives = np.sum(target_values == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both classes")
    false_positive_rates = [0.0]
    true_positive_rates = [0.0]
    for threshold in np.sort(np.unique(scores))[::-1]:
        selected = scores >= threshold
        false_positive_rates.append(np.sum(selected & (target_values == 0)) / negatives)
        true_positive_rates.append(np.sum(selected & (target_values == 1)) / positives)
    return float(np.trapezoid(true_positive_rates, false_positive_rates))
