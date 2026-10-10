"""Review published targets and plot original epoch curves from training only."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import zipfile

from fault_data import CURVES, DYNAMIC_FEATURES, FAULT_LABELS, load_splits, log_to_input, parse_number, read_log
from prepare_fault_data import ARCHIVE_PATH, sha256_file
from train import data_provenance


PROJECT_DIR = Path(__file__).resolve().parent


def select_examples(rows, per_category, seed):
    """Cover distinct source programs before choosing additional variants."""
    rng = random.Random(seed)
    selected = []
    for label in FAULT_LABELS:
        candidates = rng.sample([row for row in rows if row["diagnosis"] == label],
                                sum(row["diagnosis"] == label for row in rows))
        group_seen, chosen = set(), []
        for row in candidates:
            if row["source_group"] not in group_seen:
                chosen.append(row)
                group_seen.add(row["source_group"])
                if len(chosen) == per_category:
                    break
        chosen_ids = {row["example_id"] for row in chosen}
        chosen.extend(row for row in candidates if row["example_id"] not in chosen_ids)
        selected.extend(chosen[:per_category])
    return selected


def values(log, column):
    return [number if (number := parse_number(row[column])) is not None else math.nan for row in log]


def plot_case(row, log, pyplot):
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 7), layout="constrained")
    epochs = values(log, "epoch")
    panels = [(("train_loss", "val_loss"), "Recorded loss"),
              (("train_acc", "val_acc"), "Recorded metric (source definition)"),
              (("mean_gradient", "gradient_max"), "Gradient measurements"),
              (("adjusted_lr",), "Recorded learning rate")]
    for axis, (columns, title) in zip(axes.flat, panels):
        any_finite = False
        for column in columns:
            measured = values(log, column)
            if any(math.isfinite(number) for number in measured):
                axis.plot(epochs, measured, label=column, linewidth=1.4)
                any_finite = True
        if any_finite:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No finite measurements", ha="center", va="center", transform=axis.transAxes)
        axis.set(title=title, xlabel="Epoch (source numbering)")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        axis.grid(alpha=0.2)
    figure.suptitle(f"Published target: {row['diagnosis']} | {row['example_id']}\n"
                   f"Program {row['source_group']} | {row['architecture']} | training split")
    return figure


def audit_rows(rows):
    summary = {"examples": len(rows), "categories": dict(Counter(row["diagnosis"] for row in rows)),
               "curves": {}, "dynamic_fully_missing_examples": {}}
    for column in CURVES:
        curves = [json.loads(row[column]) for row in rows]
        summary["curves"][column] = {
            "examples_with_missing_sample": sum(any(value is None for value in curve) for curve in curves),
            "fully_missing_examples": sum(all(value is None for value in curve) for curve in curves),
            "flat_sampled_examples": sum(len(set(value for value in curve if value is not None)) == 1 for curve in curves),
        }
    dynamics = [json.loads(row["dynamic_features"]) for row in rows]
    for feature in DYNAMIC_FEATURES:
        summary["dynamic_fully_missing_examples"][feature] = sum(dynamic[feature]["mean"] is None for dynamic in dynamics)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "deep4ge")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "data_review")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be positive.")
    rows = load_splits(args.data_dir)["train"]
    provenance = data_provenance(args.data_dir)
    if sha256_file(args.archive) != provenance["archive_sha256"]:
        raise ValueError("Archive checksum differs from the prepared dataset.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, records = select_examples(rows, args.per_category, args.seed), []
    overview, axes = plt.subplots(4, 2, figsize=(13, 12), layout="constrained")
    with zipfile.ZipFile(args.archive) as archive, PdfPages(args.output_dir / "training_examples.pdf") as pdf:
        root = f"deep4ge-{provenance['revision']}/"
        for index, row in enumerate(selected):
            raw = archive.read(root + row["source_file"])
            if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
                raise ValueError("Selected source log checksum differs.")
            log = read_log(io.StringIO(raw.decode("utf-8-sig")))
            converted = log_to_input(log, row["architecture"], provenance["curve_points"])
            if any(str(value) != str(row[column]) for column, value in converted.items()):
                raise ValueError("Source log does not reproduce prepared input.")
            figure = plot_case(row, log, plt)
            plot_name = f"{row['example_id']}.png"
            figure.savefig(args.output_dir / plot_name, dpi=150)
            pdf.savefig(figure)
            plt.close(figure)
            records.append({"example_id": row["example_id"], "diagnosis": row["diagnosis"],
                            "source_group": row["source_group"], "source_file": row["source_file"],
                            "source_sha256": row["source_sha256"], "epochs": len(log), "plot": plot_name})
            if index % args.per_category == 0:
                axis = axes.flat[index // args.per_category]
                for column in ("train_loss", "val_loss"):
                    axis.plot(values(log, "epoch"), values(log, column), label=column)
                axis.set(title=f"{row['diagnosis']} | {row['example_id']}", xlabel="Epoch", ylabel="Recorded loss")
                axis.legend(fontsize=8)
                axis.grid(alpha=0.2)
    overview.suptitle("Original training logs: one published example per category")
    overview.savefig(args.output_dir / "overview.png", dpi=150)
    plt.close(overview)
    with (args.output_dir / "reviewed_examples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {"review_split": "train", "test_evaluated": False,
              "source_revision": provenance["revision"], "dataset_csv_sha256": provenance["csv_sha256"],
              "plotted_examples": len(records), "source_to_input_checks_passed": True,
              "selection": f"Seed {args.seed}; {args.per_category} examples per category; diverse source programs first.",
              "audit": audit_rows(rows)}
    (args.output_dir / "data_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Training data review", "", f"Plotted {len(records)} original logs, covering all eight published categories.",
             "Each selected log reproduces its prepared input and matches its source checksum.",
             "Only training examples were reviewed. No validation tuning or final test scoring.", "",
             "Open [overview.png](overview.png) for loss curves or [training_examples.pdf](training_examples.pdf) for all four panels per example.",
             "Each PNG also shows source metrics, gradients and learning rate. Missing data appear as gaps.", "",
             "| Curve | Examples with missing samples | Fully missing | Flat sampled curve |", "| --- | ---: | ---: | ---: |"]
    for column, counts in report["audit"]["curves"].items():
        lines.append(f"| {column} | {counts['examples_with_missing_sample']} | {counts['fully_missing_examples']} | {counts['flat_sampled_examples']} |")
    lines.extend(["", "The source metric can be accuracy or a regression proxy. Scales and losses differ between programs.",
                  "The displayed targets name published mutations; a curve alone cannot verify a unique causal diagnosis.",
                  "Flat curves and missing signals remain in the benchmark rather than being assigned invented targets.",
                  "Keep the original category labels. Check per-category recall when comparing models, especially rare categories.", ""])
    (args.output_dir / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reviewed {len(records)} original training logs. Plots, PDF and audit: {args.output_dir}")


if __name__ == "__main__":
    main()
