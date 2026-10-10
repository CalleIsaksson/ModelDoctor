"""Train a fixed Random Forest baseline and evaluate on validation only."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from analysis_features import enable_local_analysis_dependencies, observed_features
from fault_data import FAULT_LABELS, load_splits
from fault_metrics import classification_metrics
from train import data_provenance


PROJECT_DIR = Path(__file__).resolve().parent


def save_confusion_plot(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array([[metrics["confusion_matrix"][label][prediction] for prediction in FAULT_LABELS]
                       for label in FAULT_LABELS])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    names = [label.replace("_fault", "").replace("no_injected", "no injected") for label in FAULT_LABELS]
    figure, axis = plt.subplots(figsize=(10, 8), layout="constrained")
    display = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if normalized[i, j] > 0.6 else "black", fontsize=9)
    axis.set(xticks=range(8), yticks=range(8), xticklabels=names, yticklabels=names,
             xlabel="Predicted category", ylabel="Published category", title="CPU baseline — validation only")
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    figure.colorbar(display, ax=axis, label="Fraction within published category")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "deep4ge")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "deep4ge_cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    enable_local_analysis_dependencies()
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    splits, provenance = load_splits(args.data_dir), data_provenance(args.data_dir)
    train_rows, val_rows = splits["train"], splits["validation"]
    train_features, val_features = [[observed_features(row) for row in rows] for rows in (train_rows, val_rows)]
    settings = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 3,
                "class_weight": "balanced_subsample", "random_state": args.seed, "n_jobs": 2}
    pipeline = Pipeline([
        ("vectorize", DictVectorizer(sparse=False)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", RandomForestClassifier(**settings)),
    ])
    print(f"Fitting CPU baseline on {len(train_rows)} examples; validating on {len(val_rows)}.", flush=True)
    pipeline.fit(train_features, [row["diagnosis"] for row in train_rows])
    predictions = pipeline.predict(val_features)
    records = [{"example_id": row["example_id"], "source_group": row["source_group"],
                "expected_diagnosis": row["diagnosis"], "predicted_diagnosis": prediction}
               for row, prediction in zip(val_rows, predictions)]
    metrics = classification_metrics(records)
    majority = Counter(row["diagnosis"] for row in train_rows).most_common(1)[0][0]
    dummy_metrics = classification_metrics([{**record, "predicted_diagnosis": majority} for record in records])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    feature_names = pipeline[:-1].get_feature_names_out()
    importance = sorted(zip(feature_names, pipeline[-1].feature_importances_), key=lambda item: item[1], reverse=True)
    report = {"model": "RandomForestClassifier", "evaluation_split": "validation", "test_evaluated": False,
              "training_examples": len(train_rows), "validation_examples": len(val_rows),
              "settings": settings, "sklearn_version": sklearn.__version__,
              "dataset_csv_sha256": provenance["csv_sha256"], "source_revision": provenance["revision"],
              "preprocessing": "Observed inputs only; vectorizer and median imputer fitted exclusively on training.",
              "feature_count": len(feature_names), "random_forest": metrics,
              "majority_class": majority, "majority_baseline": dummy_metrics,
              "feature_importance": [{"feature": str(name), "importance": float(value)} for name, value in importance],
              "note": "Fixed baseline settings, no validation tuning. Feature importance describes association, not causation. No Llama training or final test evaluation."}
    (args.output_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_path = PROJECT_DIR / "models" / "modeldoctor_cpu_baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "report": report}, model_path)
    save_confusion_plot(metrics, args.output_dir / "validation_confusion.png")
    text = ["# CPU baseline: validation results", "", "Fixed Random Forest trained on the training split only.",
            "The final test split was not evaluated.", "", "| Model | Accuracy | Macro F1 | Balanced accuracy |",
            "| --- | ---: | ---: | ---: |"]
    for name, result in (("Majority category", dummy_metrics), ("Random Forest", metrics)):
        text.append(f"| {name} | {result['accuracy']:.2%} | {result['macro_f1']:.4f} | {result['balanced_accuracy']:.2%} |")
    text.extend(["", "| Published category | Validation examples | Recall | F1 |", "| --- | ---: | ---: | ---: |"])
    for label, result in metrics["per_label"].items():
        text.append(f"| {label} | {result['support']} | {result['recall']:.2%} | {result['f1']:.4f} |")
    text.extend(["", "These scores are a reference for future Llama validation results, not Llama performance.",
                 "Both models must use the same held-out programs for a meaningful comparison.",
                 "Published mutation labels may be ambiguous from observations alone.", ""])
    (args.output_dir / "SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Accuracy {metrics['accuracy']:.2%}; macro F1 {metrics['macro_f1']:.4f}; balanced accuracy {metrics['balanced_accuracy']:.2%}.")
    print(f"Reports: {args.output_dir}; baseline model: {model_path}")


if __name__ == "__main__":
    main()
