"""Metrics include minority categories as well as overall accuracy."""

from fault_data import FAULT_LABELS


def classification_metrics(records):
    if not records:
        raise ValueError("No predictions to evaluate.")
    matrix = {label: {prediction: 0 for prediction in (*FAULT_LABELS, "unparsed")}
              for label in FAULT_LABELS}
    for record in records:
        matrix[record["expected_diagnosis"]][record["predicted_diagnosis"]] += 1
    per_label = {}
    for label in FAULT_LABELS:
        support = sum(matrix[label].values())
        true_positive = matrix[label][label]
        predicted = sum(matrix[expected][label] for expected in FAULT_LABELS)
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    present = [metrics for metrics in per_label.values() if metrics["support"]]
    return {
        "examples": len(records),
        "accuracy": sum(row["expected_diagnosis"] == row["predicted_diagnosis"]
                        for row in records) / len(records),
        "macro_f1": sum(metrics["f1"] for metrics in present) / len(present),
        "balanced_accuracy": sum(metrics["recall"] for metrics in present) / len(present),
        "parsed_rate": sum(row["predicted_diagnosis"] != "unparsed" for row in records) / len(records),
        "per_label": per_label, "confusion_matrix": matrix,
    }
