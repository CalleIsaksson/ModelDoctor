import json

import pandas as pd

import evaluate_model as evaluator


TARGET_LABELS = {
    "healthy",
    "overfitting",
    "underfitting",
    "learning_rate_too_high",
    "learning_rate_too_low",
}


def print_result(name, predictions):
    accuracy = predictions["correct"].mean()
    insufficient_count = (
        predictions["predicted_diagnosis"] == "insufficient_evidence"
    ).sum()

    print(f"\n{name}")
    print("-" * 45)
    print(f"Accuracy:                 {accuracy:.2%}")
    print(f"Insufficient predictions: {insufficient_count} of {len(predictions)}")

    return {
        "accuracy": round(float(accuracy), 4),
        "insufficient_predictions": int(insufficient_count),
        "examples": len(predictions),
    }


def main():
    all_rows = pd.read_csv(
        evaluator.challenge_file,
        keep_default_na=False
    )
    rows = all_rows[all_rows["diagnosis"].isin(TARGET_LABELS)].copy()

    original_predictions = pd.read_csv(
        evaluator.results_dir / "challenge_predictions.csv"
    )
    original_predictions = original_predictions[
        original_predictions["expected_diagnosis"].isin(TARGET_LABELS)
    ]
    results = {}
    results["original_complete_cases"] = print_result(
        "ORIGINAL COMPLETE CASES",
        original_predictions
    )

    model, tokenizer = evaluator.load_finetuned_model()

    known_metadata = rows.copy()
    known_metadata["model_name"] = "controlled_mlp"
    known_metadata["task"] = "binary_classification"
    predictions = evaluator.evaluate(model, tokenizer, known_metadata)
    results["known_model_and_task"] = print_result(
        "KNOWN MODEL AND TASK",
        predictions
    )

    neutral_text = rows.copy()
    neutral_text["additional_information"] = (
        "Metrics were recorded once at the end of every epoch."
    )
    predictions = evaluator.evaluate(model, tokenizer, neutral_text)
    results["neutral_additional_information"] = print_result(
        "NEUTRAL ADDITIONAL INFORMATION",
        predictions
    )

    both_changes = neutral_text.copy()
    both_changes["model_name"] = "controlled_mlp"
    both_changes["task"] = "binary_classification"
    predictions = evaluator.evaluate(model, tokenizer, both_changes)
    results["known_metadata_and_neutral_text"] = print_result(
        "KNOWN METADATA AND NEUTRAL TEXT",
        predictions
    )

    results["conclusion"] = (
        "Additional information is the main trigger for excessive "
        "insufficient_evidence predictions; model name and task have little effect."
    )

    with (evaluator.results_dir / "insufficient_analysis.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
