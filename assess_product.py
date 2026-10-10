"""Compare completed validation predictions with the existing CPU reference.

No weights are loaded, no training is run, and the final test split is not read.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from fault_data import FAULT_LABELS
from fault_metrics import classification_metrics

ROOT = Path(__file__).resolve().parent


def read_records(path, expected):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("Predictions must cover each validation example exactly once. Limited/smoke predictions cannot establish an advantage.")
    for row in rows:
        reference = expected[row["example_id"]]
        if row["expected_diagnosis"] != reference["diagnosis"] or row["source_group"] != reference["source_group"]:
            raise ValueError("Prediction IDs, labels or source groups do not match the validation dataset.")
        if row["predicted_diagnosis"] not in (*FAULT_LABELS, "unparsed"):
            raise ValueError("Unsupported predicted category.")
    return rows


def summarize(records):
    result = classification_metrics(records)
    baselines = [row for row in records if row["expected_diagnosis"] == "no_injected_fault"]
    faults = [row for row in records if row["expected_diagnosis"] != "no_injected_fault"]
    result["baseline_flag_rate"] = sum(row["predicted_diagnosis"] not in {"no_injected_fault", "unparsed"} for row in baselines) / len(baselines) if baselines else None
    result["mutation_miss_rate"] = sum(row["predicted_diagnosis"] == "no_injected_fault" for row in faults) / len(faults) if faults else None
    return result


def assess(candidate, reference, bootstrap_samples=300):
    model, cpu = summarize(candidate), summarize(reference)
    by_id = {row["example_id"]: row for row in reference}
    groups = defaultdict(list)
    for row in candidate:
        groups[row["source_group"]].append(row)
    rng = random.Random(42)
    keys = sorted(groups)
    differences = []
    for _ in range(bootstrap_samples):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]]
        matching = [by_id[row["example_id"]] for row in sample]
        differences.append(classification_metrics(sample)["macro_f1"] - classification_metrics(matching)["macro_f1"])
    differences.sort()
    interval = [differences[int((len(differences)-1)*q)] for q in (.025,.975)] if differences else None
    advantage = model["macro_f1"] > cpu["macro_f1"] and model["parsed_rate"] == 1 and model["baseline_flag_rate"] <= cpu["baseline_flag_rate"]
    return {"candidate": model, "cpu_reference": cpu, "macro_f1_delta": model["macro_f1"] - cpu["macro_f1"],
        "program_groups": len(groups), "exploratory_cluster_bootstrap": {"samples": bootstrap_samples,
        "seed": 42, "macro_f1_delta_interval_95": interval,
        "note": "Resampling source programs, not individual mutations. Few groups and validation model selection limit this evidence."},
        "advantage_on_this_validation_set": advantage, "commercial_readiness_established": False,
        "interpretation": "Baseline flag rate refers to no-injected-fault source programs, not verified healthy production training. Independent real-user evaluation and workflow benefit remain required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/deep4ge")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "results/deep4ge_cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/product_assessment")
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    args = parser.parse_args()
    if not 0 <= args.bootstrap_samples <= 5000:
        parser.error("Bootstrap samples must be between 0 and 5,000.")
    dataset = args.data_dir / "modeldoctor_validation.csv"
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    reference_report = json.loads((args.reference_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    if reference_report["evaluation_split"] != "validation" or reference_report["dataset_csv_sha256"][dataset.name] != dataset_hash:
        raise ValueError("CPU reference belongs to another dataset or split.")
    with dataset.open(encoding="utf-8", newline="") as stream:
        expected = {row["example_id"]: row for row in csv.DictReader(stream)}
    candidate = read_records(args.predictions, expected)
    reference = read_records(args.reference_dir / "validation_predictions.csv", expected)
    report = {"evaluation_split": "validation", "test_evaluated": False,
        "dataset_sha256": dataset_hash, "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        **assess(candidate, reference, args.bootstrap_samples)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assessment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = ["# Model advantage: validation only", "", "| Measure | Candidate | CPU reference |", "| --- | ---: | ---: |"]
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "parsed_rate", "baseline_flag_rate", "mutation_miss_rate"):
        text.append(f"| {key} | {report['candidate'][key]:.4f} | {report['cpu_reference'][key]:.4f} |")
    text.extend(["", f"Advantage under the recorded comparison criteria: **{report['advantage_on_this_validation_set']}**.",
        "Criteria: higher macro F1, all responses parsed, no higher baseline flag rate than the CPU reference.",
        "", report["interpretation"], "", "This is a validation comparison, not commercial certification. The final test remains untouched.", ""])
    (args.output_dir / "ASSESSMENT.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Macro F1 difference: {report['macro_f1_delta']:+.4f}. Reports: {args.output_dir}")


if __name__ == "__main__":
    main()
