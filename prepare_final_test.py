import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "v3"
FINAL_FILE = DATA_DIR / "modeldoctor_final_test.csv"
MANIFEST_FILE = RESULTS_DIR / "final_test_manifest.json"

LABELS = [
    "healthy",
    "overfitting",
    "underfitting",
    "learning_rate_too_high",
    "learning_rate_too_low",
    "data_leakage_suspected",
    "insufficient_evidence",
]

COLUMNS = [
    "example_id",
    "source",
    "split",
    "model_name",
    "task",
    "train_loss",
    "validation_loss",
    "train_metric",
    "validation_metric",
    "learning_rate",
    "epochs",
    "additional_information",
    "diagnosis",
]

INPUT_COLUMNS = [
    "model_name",
    "task",
    "train_loss",
    "validation_loss",
    "train_metric",
    "validation_metric",
    "learning_rate",
    "epochs",
    "additional_information",
]

MODELS_AND_TASKS = [
    ("final_mobile_cnn", "image_classification"),
    ("final_text_adapter", "intent_classification"),
    ("final_tabular_stack", "risk_classification"),
    ("final_sequence_model", "sequence_classification"),
    ("final_forecaster", "time_series_forecasting"),
]


def rounded(values):
    return [round(value, 3) for value in values]


def smooth(start, end, length, power=1.35):
    values = []
    for index in range(length):
        fraction = index / (length - 1)
        progress = 1 - (1 - fraction) ** power
        values.append(start + (end - start) * progress)
    return rounded(values)


def piecewise(start, middle, end, first_length, total_length):
    first = smooth(start, middle, first_length)
    second = smooth(middle, end, total_length - first_length + 1, power=1.1)[1:]
    return first + second


def make_row(
    label,
    index,
    train_loss,
    validation_loss,
    train_metric,
    validation_metric,
    learning_rate,
    additional_information,
):
    model_name, task = MODELS_AND_TASKS[index % len(MODELS_AND_TASKS)]
    epochs = list(range(1, len(train_loss) + 1))

    return {
        "example_id": f"final_{label}_{index + 1:02d}",
        "source": "untouched_final_v1",
        "split": "final",
        "model_name": model_name,
        "task": task,
        "train_loss": json.dumps(rounded(train_loss)),
        "validation_loss": json.dumps(rounded(validation_loss)),
        "train_metric": json.dumps(rounded(train_metric)),
        "validation_metric": json.dumps(rounded(validation_metric)),
        "learning_rate": learning_rate,
        "epochs": json.dumps(epochs),
        "additional_information": additional_information,
        "diagnosis": label,
    }


def healthy_rows():
    configurations = [
        (7, 1.28, 0.18, 0.46, 0.95, 0.0012, "The result remained stable when evaluation was repeated."),
        (8, 1.05, 0.22, 0.55, 0.93, 0.0007, "One minor validation fluctuation recovered in the following epoch."),
        (6, 1.44, 0.25, 0.38, 0.91, 0.0018, "Training completed normally and the final gap stayed small."),
        (10, 1.62, 0.16, 0.34, 0.96, 0.0009, "A separate split audit reported no shared records."),
        (9, 1.18, 0.20, 0.50, 0.94, 0.0014, "Both curves settled near their best values at the end."),
    ]
    rows = []
    for index, (length, loss_start, loss_end, metric_start, metric_end, rate, information) in enumerate(configurations):
        train_loss = smooth(loss_start, loss_end, length)
        validation_loss = smooth(loss_start + 0.06, loss_end + 0.06, length)
        train_metric = smooth(metric_start, metric_end, length)
        validation_metric = smooth(metric_start - 0.02, metric_end - 0.03, length)
        if index == 1:
            middle = 5
            validation_loss[middle] = round(validation_loss[middle - 1] + 0.012, 3)
            validation_metric[middle] = round(validation_metric[middle - 1] - 0.004, 3)
        rows.append(make_row("healthy", index, train_loss, validation_loss, train_metric, validation_metric, rate, information))
    return rows


def overfitting_rows():
    configurations = [
        (8, 4, 1.31, 0.10, 0.42, 0.88, "Regularization settings stayed unchanged after validation peaked."),
        (9, 5, 1.08, 0.08, 0.36, 0.73, "The final checkpoint was used even though an earlier checkpoint validated better."),
        (10, 6, 1.52, 0.12, 0.48, 0.92, "The late generalization gap also appeared with a different seed."),
        (7, 3, 0.96, 0.07, 0.31, 0.76, "No duplicated records or preprocessing overlap were found."),
        (11, 7, 1.40, 0.09, 0.39, 0.84, "Training was allowed to continue after validation stopped improving."),
    ]
    rows = []
    for index, (length, best_epoch, start, train_end, val_best, val_end, information) in enumerate(configurations):
        train_loss = smooth(start, train_end, length)
        validation_loss = piecewise(start + 0.07, val_best, val_end, best_epoch, length)
        train_metric = smooth(0.42, 0.98, length)
        validation_metric = piecewise(0.40, 0.86, 0.62, best_epoch, length)
        rows.append(make_row("overfitting", index, train_loss, validation_loss, train_metric, validation_metric, 0.001, information))
    return rows


def underfitting_rows():
    configurations = [
        (8, 2.08, 1.42, 0.24, 0.48, "The optimizer was stable, but both splits stopped improving at weak values."),
        (10, 1.91, 1.31, 0.31, 0.54, "Extra epochs made almost no difference after the early plateau."),
        (7, 2.30, 1.58, 0.15, 0.39, "The small network could not fit even the training examples well."),
        (9, 1.76, 1.27, 0.36, 0.57, "Training and validation remained close and similarly poor."),
        (11, 2.14, 1.49, 0.20, 0.45, "Loss stopped changing meaningfully near the end of the run."),
    ]
    rows = []
    for index, (length, start, end, metric_start, metric_end, information) in enumerate(configurations):
        train_loss = smooth(start, end, length, power=2.8)
        validation_loss = smooth(start + 0.06, end + 0.07, length, power=2.8)
        train_metric = smooth(metric_start, metric_end, length, power=2.8)
        validation_metric = smooth(metric_start - 0.02, metric_end - 0.03, length, power=2.8)
        rows.append(make_row("underfitting", index, train_loss, validation_loss, train_metric, validation_metric, 0.0011, information))
    return rows


def learning_rate_high_rows():
    low_high_pairs = [
        ([1.10, 2.70, 0.88, 3.20, 1.25, 2.85, 0.96, 3.05], 0.09, "Gradient norms spiked repeatedly despite a fixed batch size."),
        ([1.65, 0.91, 2.95, 1.12, 3.41, 0.84, 2.62, 1.30], 0.06, "The run never settled into a consistent downward direction."),
        ([0.82, 4.10, 1.45, 3.62, 0.76, 4.55, 1.28], 0.15, "Clipping reduced individual spikes but did not produce convergence."),
        ([1.34, 2.46, 0.99, 3.08, 1.18, 2.77, 0.86, 3.31, 1.07], 0.08, "A ten-times smaller rate was stable in a short pilot."),
        ([2.05, 0.74, 3.20, 1.09, 2.88, 0.81, 3.47, 1.21], 0.12, "Large alternating loss changes continued through the final epoch."),
    ]
    rows = []
    for index, (train_loss, rate, information) in enumerate(low_high_pairs):
        validation_loss = [value + 0.12 for value in train_loss]
        train_metric = [0.58 if value < 1.0 else 0.29 if value > 2.5 else 0.43 for value in train_loss]
        validation_metric = [value - 0.025 for value in train_metric]
        rows.append(make_row("learning_rate_too_high", index, train_loss, validation_loss, train_metric, validation_metric, rate, information))
    return rows


def learning_rate_low_rows():
    configurations = [
        (12, 1.64, 0.08, 0.34, 0.05, 0.0000006, "The run was stable, but each epoch produced a barely visible update."),
        (15, 1.91, 0.11, 0.22, 0.07, 0.0000012, "The epoch budget ended while performance was still far from useful."),
        (10, 1.37, 0.06, 0.39, 0.04, 0.0000004, "There were no oscillations, only very slow progress."),
        (14, 2.04, 0.13, 0.18, 0.08, 0.0000015, "A larger rate made faster progress in a separate short pilot."),
        (16, 1.55, 0.09, 0.30, 0.06, 0.0000008, "Both splits improved smoothly by only a small amount."),
    ]
    rows = []
    for index, (length, start, decrease, metric_start, metric_gain, rate, information) in enumerate(configurations):
        train_loss = smooth(start, start - decrease, length, power=1.0)
        validation_loss = smooth(start + 0.05, start - decrease + 0.07, length, power=1.0)
        train_metric = smooth(metric_start, metric_start + metric_gain, length, power=1.0)
        validation_metric = smooth(metric_start - 0.01, metric_start + metric_gain - 0.02, length, power=1.0)
        rows.append(make_row("learning_rate_too_low", index, train_loss, validation_loss, train_metric, validation_metric, rate, information))
    return rows


def leakage_rows():
    information = [
        "An audit found 317 product IDs represented in both training and validation groups.",
        "Normalization statistics were calculated on the complete dataset before splitting.",
        "The input field outcome_after_30_days directly revealed the target label.",
        "Augmentation happened before the split, leaving 126 near-duplicate recordings across both splits.",
        "Validation reached 0.997, then fell to 0.704 on a clean holdout split by source document.",
    ]
    rows = []
    for index, text in enumerate(information):
        length = 5 + index % 2
        train_loss = smooth(0.58, 0.04, length)
        validation_loss = smooth(0.53, 0.02, length)
        train_metric = smooth(0.76, 0.985, length)
        validation_metric = smooth(0.80, 0.997, length)
        rows.append(make_row("data_leakage_suspected", index, train_loss, validation_loss, train_metric, validation_metric, 0.0008, text))
    return rows


def insufficient_rows():
    rows = [
        make_row("insufficient_evidence", 0, [1.03], [1.08], [0.57], [0.54], 0.001, "The run completed without logging errors."),
        make_row("insufficient_evidence", 1, [1.31, 1.02, 0.79, 0.61, 0.49], [], [0.42, 0.55, 0.66, 0.74, 0.80], [], 0.0009, "Only training-side values were saved."),
        make_row("insufficient_evidence", 2, [1.18, 0.90], [1.22, 0.95], [], [], "not_recorded", "The optimizer configuration and metric logs are unavailable."),
        make_row("insufficient_evidence", 3, [1.24, 1.01, 0.83, 0.69, 0.58], [1.28, 1.06, 0.88, 0.74, 0.64], [0.66, 0.61, 0.56, 0.51, 0.46], [0.64, 0.59, 0.54, 0.49, 0.44], 0.001, "Loss improves while an accuracy metric gets worse, so the logs disagree."),
        make_row("insufficient_evidence", 4, [0.92, 0.63, 0.47, 0.38], [0.96, 0.68, 0.52, 0.44], [0.60, 0.74, 0.82, 0.87], [], 0.0006, "The validation metric definition and values were not recorded."),
    ]
    return rows


def input_signature(row):
    return tuple(str(row[column]) for column in INPUT_COLUMNS)


def validate(rows):
    counts = Counter(row["diagnosis"] for row in rows)
    expected_counts = {label: 5 for label in LABELS}
    if counts != expected_counts:
        raise ValueError(f"Wrong label balance: {counts}")

    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate final example IDs were found.")

    signatures = {input_signature(row) for row in rows}
    if len(signatures) != len(rows):
        raise ValueError("Duplicate inputs were found inside the final test.")

    for row in rows:
        epochs = json.loads(row["epochs"])
        for column in ["train_loss", "validation_loss", "train_metric", "validation_metric"]:
            values = json.loads(row[column])
            if values and len(values) != len(epochs):
                raise ValueError(f"Wrong {column} length in {row['example_id']}.")

    previous_signatures = set()
    for filename in [
        "modeldoctor_train.csv",
        "modeldoctor_validation.csv",
        "modeldoctor_test.csv",
        "modeldoctor_challenge.csv",
    ]:
        with (DATA_DIR / filename).open(newline="", encoding="utf-8") as file:
            for old_row in csv.DictReader(file):
                previous_signatures.add(input_signature(old_row))

    overlap = signatures.intersection(previous_signatures)
    if overlap:
        raise ValueError("The final test overlaps an earlier dataset.")


def main():
    if FINAL_FILE.exists():
        raise FileExistsError(
            "The final test already exists and will not be overwritten."
        )

    rows = [
        *healthy_rows(),
        *overfitting_rows(),
        *underfitting_rows(),
        *learning_rate_high_rows(),
        *learning_rate_low_rows(),
        *leakage_rows(),
        *insufficient_rows(),
    ]
    validate(rows)

    with FINAL_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    file_hash = hashlib.sha256(FINAL_FILE.read_bytes()).hexdigest()
    manifest = {
        "dataset": "untouched_final_v1",
        "examples": len(rows),
        "examples_per_diagnosis": 5,
        "exact_overlap_with_previous_data": 0,
        "sha256": file_hash,
        "status": "created_before_first_model evaluation",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_FILE.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print("FINAL TEST CREATED")
    print("-" * 40)
    print(f"Examples: {len(rows)}")
    print("Examples per diagnosis: 5")
    print("Exact overlap with earlier data: 0")
    print(f"SHA-256: {file_hash}")
    print("The file is now frozen and ready for one evaluation.")


if __name__ == "__main__":
    main()
