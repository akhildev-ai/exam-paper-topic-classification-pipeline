from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from src.classify_topics import HybridTopicClassifier

LOGGER = logging.getLogger(__name__)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1_score(precision: float, recall: float) -> float:
    return _safe_divide(2 * precision * recall, precision + recall)


def compute_classification_metrics(
    expected_labels: Sequence[Sequence[str]], predicted_labels: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Compute exact-match accuracy and micro-averaged precision, recall, and F1."""
    if len(expected_labels) != len(predicted_labels):
        raise ValueError("Expected and predicted label lists must have the same length")

    exact_matches = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for expected, predicted in zip(expected_labels, predicted_labels):
        expected_set = set(expected)
        predicted_set = set(predicted)
        if expected_set == predicted_set:
            exact_matches += 1

        true_positive += len(expected_set & predicted_set)
        false_positive += len(predicted_set - expected_set)
        false_negative += len(expected_set - predicted_set)

    accuracy = _safe_divide(exact_matches, len(expected_labels))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1_score = _f1_score(precision, recall)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
    }


def validate_predictions(
    classifier: HybridTopicClassifier, validation_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Run the classifier against labeled validation records and compute metrics."""
    evaluated_records: list[dict[str, Any]] = []
    expected_labels: list[Sequence[str]] = []
    predicted_labels: list[Sequence[str]] = []

    for record in validation_records:
        prediction = classifier.classify_question_record(record)
        gold_labels = tuple(record.get("labels", []))
        predicted = tuple(topic["name"] for topic in prediction.get("topics", []))
        expected_labels.append(gold_labels)
        predicted_labels.append(predicted)
        evaluated_records.append(
            {
                **record,
                "predicted_topics": list(predicted),
                "predicted_topic_details": prediction.get("topics", []),
                "matched": set(gold_labels) == set(predicted),
                "classification_status": prediction.get("classification_status", "unclassified"),
            }
        )

    metrics = compute_classification_metrics(expected_labels, predicted_labels)
    report = {
        "total_records": len(validation_records),
        "metrics": metrics,
        "evaluated_records": evaluated_records,
    }
    LOGGER.info("Validation completed for %s labeled questions", len(validation_records))
    return report
