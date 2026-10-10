"""Convert observed Deep4ge logs to inputs, without using their fault labels."""

import csv
import hashlib
import json
import math


FAULT_LABELS = (
    "no_injected_fault", "hyperparameter_fault", "layer_fault", "loss_fault",
    "activation_fault", "optimization_fault", "weight_fault", "regularization_fault",
)
CATEGORY_LABELS = {
    "Hyperparameter": "hyperparameter_fault", "Layer": "layer_fault",
    "Loss": "loss_fault", "Activation": "activation_fault",
    "Optimization": "optimization_fault", "Weight": "weight_fault",
    "Regularization": "regularization_fault",
}
CURVES = ("train_loss", "val_loss", "train_acc", "val_acc")
DYNAMIC_FEATURES = (
    "large_weight_count", "acc_gap_too_big", "loss_oscillation", "dying_relu",
    "gradient_vanish", "gradient_explode", "decrease_acc_count", "increase_loss_count",
    "cons_mean_weight_count", "cons_std_weight_count", "nan_weight_count",
    "nan_gradients_count", "saturated_activation", "mean_gradient", "gradient_std",
    "gradient_max", "gradient_min", "gradient_median", "adjusted_lr",
    "mean_activation", "std_activation", "mean_grad", "std_grad",
)
# Hardware utilization is deliberately excluded: it can identify the collection
# environment rather than the training fault. These are the only input columns.
INPUT_COLUMNS = ("architecture", "num_epochs", "sampled_epochs", *CURVES, "dynamic_features")
OUTPUT_COLUMNS = (
    "example_id", "source", "source_group", "source_file", "source_sha256",
    "input_sha256", "split", *INPUT_COLUMNS, "diagnosis",
)


def parse_number(value):
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return float(text == "true")
    if text.startswith("tf.tensor("):
        return parse_number(text[len("tf.tensor("):].split(",", 1)[0])
    if not text:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def compact_number(value):
    return None if value is None else float(f"{value:.5g}")


def read_log(stream):
    reader = csv.DictReader(stream)
    required = {"epoch", *CURVES, *DYNAMIC_FEATURES}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"Training log is missing columns: {sorted(missing)}")
    rows = list(reader)
    if not rows:
        raise ValueError("Training log has no epochs.")
    epochs = [parse_number(row["epoch"]) for row in rows]
    if any(epoch is None or epoch < 0 or int(epoch) != epoch for epoch in epochs):
        raise ValueError("Epochs must be nonnegative integers.")
    if any(left >= right for left, right in zip(epochs, epochs[1:])):
        raise ValueError("Epochs must be strictly increasing.")
    return rows


def log_to_input(log, architecture="unknown", curve_points=12):
    if curve_points < 2:
        raise ValueError("At least two curve points are required.")
    if architecture not in {"FNN", "CNN", "RNN", "unknown"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    count = len(log)
    indices = sorted({round(i * (count - 1) / (min(count, curve_points) - 1))
                      for i in range(min(count, curve_points))}) if count > 1 else [0]
    result = {
        "architecture": architecture,
        "num_epochs": count,
        "sampled_epochs": json.dumps([int(float(log[i]["epoch"])) for i in indices]),
    }
    for column in CURVES:
        result[column] = json.dumps([compact_number(parse_number(log[i][column]))
                                     for i in indices], allow_nan=False)
    features = {}
    for column in DYNAMIC_FEATURES:
        values = [parse_number(row[column]) for row in log]
        finite = [value for value in values if value is not None]
        features[column] = {
            "first": compact_number(values[0]), "last": compact_number(values[-1]),
            "min": compact_number(min(finite)) if finite else None,
            "max": compact_number(max(finite)) if finite else None,
            "mean": compact_number(sum(finite) / len(finite)) if finite else None,
            "missing": len(values) - len(finite),
        }
    result["dynamic_features"] = json.dumps(features, separators=(",", ":"), allow_nan=False)
    return result


def input_hash(row):
    # Match what actually reaches the model, including CSV string conversion.
    content = json.dumps({column: str(row[column]) for column in INPUT_COLUMNS}, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_splits(splits):
    seen_ids, seen_groups, seen_inputs = set(), set(), set()
    for name in ("train", "validation", "test"):
        rows = splits[name]
        if not rows:
            raise ValueError(f"Empty {name} split.")
        labels = {row["diagnosis"] for row in rows}
        if labels != set(FAULT_LABELS):
            raise ValueError(f"{name} must contain all eight labels; found {sorted(labels)}.")
        ids = {row["example_id"] for row in rows}
        groups = {row["source_group"] for row in rows}
        inputs = {input_hash(row) for row in rows}
        if len(ids) != len(rows) or seen_ids & ids:
            raise ValueError("Duplicate example IDs.")
        if seen_groups & groups:
            raise ValueError("Source programs overlap between splits.")
        if seen_inputs & inputs:
            raise ValueError("Identical model inputs overlap between splits.")
        for row in rows:
            if row["split"] != name or input_hash(row) != row["input_sha256"]:
                raise ValueError("Split metadata or input checksum mismatch.")
        seen_ids.update(ids)
        seen_groups.update(groups)
        seen_inputs.update(inputs)


def load_splits(directory):
    splits = {}
    for name in ("train", "validation", "test"):
        path = directory / f"modeldoctor_{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python prepare_fault_data.py --download first.")
        with path.open(newline="", encoding="utf-8") as file:
            splits[name] = list(csv.DictReader(file))
    validate_splits(splits)
    return splits
