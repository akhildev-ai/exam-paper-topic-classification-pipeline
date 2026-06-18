from src.validate import compute_classification_metrics


def test_compute_classification_metrics_supports_multi_label_scores() -> None:
    expected = [["Calculus"], ["Geometry", "Calculus"], ["Statistics"]]
    predicted = [["Calculus"], ["Geometry"], ["Probability"]]

    metrics = compute_classification_metrics(expected, predicted)

    assert metrics == {
        "accuracy": 0.3333,
        "precision": 0.6667,
        "recall": 0.5,
        "f1_score": 0.5714,
    }
