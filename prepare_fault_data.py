"""Download and prepare published labels. Uses only Python's standard library."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import urllib.request
import zipfile

from fault_data import (CATEGORY_LABELS, FAULT_LABELS, OUTPUT_COLUMNS, input_hash,
                        log_to_input, read_log, validate_splits)


SOURCE_REVISION = "1d895afac1985876f289fbcdc2ee56c62f612071"
SOURCE_URL = f"https://codeload.github.com/SigmaJahan/deep4ge/zip/{SOURCE_REVISION}"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "deep4ge"
ARCHIVE_PATH = DATA_DIR / "raw" / "deep4ge-1d895af.zip"


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_archive(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing archive: {destination}", flush=True)
        return
    partial = destination.with_suffix(".zip.partial")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ModelDoctor-data-import"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
    with zipfile.ZipFile(partial) as archive:
        if f"deep4ge-{SOURCE_REVISION}/data/manifest.csv" not in archive.namelist():
            raise ValueError("Downloaded archive does not contain the pinned dataset manifest.")
    partial.replace(destination)


def read_archive(archive_path, curve_points):
    rows = []
    skipped = []
    with zipfile.ZipFile(archive_path) as archive:
        root = f"deep4ge-{SOURCE_REVISION}/"
        manifest_bytes = archive.read(root + "data/manifest.csv")
        manifest = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"))))
        metadata_bytes = archive.read(root + "data/seed_programs/seed_metadata.csv")
        metadata = list(csv.DictReader(io.StringIO(metadata_bytes.decode("utf-8-sig"))))
        architectures = {row["so_id"]: row["architecture"] for row in metadata}
        for index, entry in enumerate(manifest):
            path = PurePosixPath(entry["csv_path"])
            if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("data", "training_logs"):
                raise ValueError(f"Unexpected log path in manifest: {path}")
            faulty_text = entry["is_faulty"].lower()
            if faulty_text not in {"true", "false"}:
                raise ValueError("Invalid is_faulty value.")
            faulty = faulty_text == "true"
            if entry["subset"] != ("buggy" if faulty else "correct"):
                raise ValueError("Inconsistent source subset and fault status.")
            label = CATEGORY_LABELS[entry["fault_category"]] if faulty else "no_injected_fault"
            content = archive.read(root + str(path))
            try:
                log = read_log(io.StringIO(content.decode("utf-8-sig")))
                if len(log) != int(entry["num_epochs"]):
                    raise ValueError("Manifest epoch count does not match log length.")
                observed = log_to_input(log, architectures[entry["so_id"]], curve_points)
            except ValueError as error:
                skipped.append({"source_file": str(path), "diagnosis": label, "reason": str(error)})
                continue
            row = {
                "example_id": f"deep4ge_{index:05d}", "source": "deep4ge_v1",
                "source_group": entry["so_id"], "source_file": str(path),
                "source_sha256": hashlib.sha256(content).hexdigest(), "split": "",
                **observed,
                "diagnosis": label,
            }
            row["input_sha256"] = input_hash(row)
            rows.append(row)
            if (index + 1) % 2000 == 0:
                print(f"Read {index + 1}/{len(manifest)} training logs", flush=True)
        documents = {name: archive.read(root + name) for name in (
            "README.md", "CITATION.cff", "LICENSE", "docs/PROVENANCE.md",
            "docs/DATA_DICTIONARY.md", "docs/SEED_ATTRIBUTION.md",
            "data/seed_programs/ATTRIBUTION.csv",
        )}
    return rows, manifest_bytes, metadata_bytes, documents, skipped, len(manifest)


def remove_conflicting_inputs(rows):
    labels = defaultdict(set)
    for row in rows:
        labels[row["input_sha256"]].add(row["diagnosis"])
    ambiguous = {key for key, values in labels.items() if len(values) > 1}
    # Identical observations with different source labels have no single
    # learnable answer. Keep their removal visible in the preparation report.
    return [row for row in rows if row["input_sha256"] not in ambiguous], ambiguous


def split_by_program(rows, seed):
    groups = {row["source_group"] for row in rows}
    parents = {group: group for group in groups}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    seen = {}
    for row in rows:
        key = row["input_sha256"]
        if key in seen:
            parents[find(row["source_group"])] = find(seen[key])
        else:
            seen[key] = row["source_group"]
    components = defaultdict(list)
    for row in rows:
        components[find(row["source_group"])].append(row)
    keys = sorted(components)
    counts = {key: Counter(row["diagnosis"] for row in components[key]) for key in keys}
    total = Counter(row["diagnosis"] for row in rows)
    rng = random.Random(seed)
    best_score, best = float("inf"), None
    train_end, validation_end = round(len(keys) * 0.7), round(len(keys) * 0.85)
    # Search deterministic group assignments for balanced label coverage and
    # roughly 70/15/15 example counts, without ever splitting a source program.
    for _ in range(2000):
        shuffled = rng.sample(keys, len(keys))
        assignment = {"train": shuffled[:train_end],
                      "validation": shuffled[train_end:validation_end],
                      "test": shuffled[validation_end:]}
        score = 0.0
        for name, target in (("train", 0.7), ("validation", 0.15), ("test", 0.15)):
            actual = Counter()
            for key in assignment[name]:
                actual.update(counts[key])
            if any(not actual[label] for label in FAULT_LABELS):
                score = float("inf")
                break
            score += sum(abs(actual[label] / total[label] - target) for label in FAULT_LABELS)
            score += abs(sum(actual.values()) / len(rows) - target)
        if score < best_score:
            best_score, best = score, assignment
    if best is None:
        raise ValueError("Cannot create program-separated splits with every label in each split.")
    splits = {}
    for name, assigned in best.items():
        split_rows = [dict(row, split=name) for key in assigned for row in components[key]]
        rng.shuffle(split_rows)
        splits[name] = split_rows
    validate_splits(splits)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--curve-points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.curve_points < 2:
        parser.error("--curve-points must be at least two.")
    if args.download:
        download_archive(args.archive)
    if not args.archive.exists():
        parser.error("No Deep4ge archive found. Add --download to retrieve it.")
    rows, manifest, metadata, documents, skipped, source_count = read_archive(args.archive, args.curve_points)
    retained, ambiguous = remove_conflicting_inputs(rows)
    splits = split_by_program(retained, args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    (source_dir / "manifest.csv").write_bytes(manifest)
    (source_dir / "seed_metadata.csv").write_bytes(metadata)
    for name, content in documents.items():
        (source_dir / name.replace("/", "_")).write_bytes(content)
    with (output_dir / "excluded_logs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("source_file", "diagnosis", "reason"))
        writer.writeheader()
        writer.writerows(skipped)
    files = {}
    for name, split_rows in splits.items():
        path = output_dir / f"modeldoctor_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(split_rows)
        files[path.name] = sha256_file(path)
    report = {
        "dataset": "deep4ge_v1", "source_url": SOURCE_URL, "revision": SOURCE_REVISION,
        "doi": "10.5281/zenodo.20337241", "dataset_license": "CC-BY-4.0",
        "archive_sha256": sha256_file(args.archive),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "source_examples": source_count, "valid_log_examples": len(rows), "retained_examples": len(retained),
        "excluded_invalid_logs": len(skipped),
        "invalid_log_reasons": dict(Counter(row["reason"] for row in skipped)),
        "removed_conflicting_examples": len(rows) - len(retained),
        "conflicting_input_patterns": len(ambiguous), "curve_points": args.curve_points,
        "seed": args.seed, "labels": list(FAULT_LABELS), "csv_sha256": files,
        "split_method": "source-program groups; identical inputs across programs stay together",
        "splits": {name: {"examples": len(values),
                          "source_groups": sorted({row["source_group"] for row in values}),
                          "labels": dict(sorted(Counter(row["diagnosis"] for row in values).items()))}
                   for name, values in splits.items()},
        "target_note": "Published baseline/mutation-category labels only; no human-verified explanation or repair targets.",
        "evaluation_note": "Program-separated mutation benchmark; source labels are not proof of a unique observable cause.",
    }
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, values in report["splits"].items():
        print(f"{name}: {values['examples']} examples, {len(values['source_groups'])} source programs")
        print(values["labels"])
    print(f"Excluded {len(skipped)} malformed logs and {len(rows) - len(retained)} contradictory observations. Data saved to {output_dir}")


if __name__ == "__main__":
    main()
