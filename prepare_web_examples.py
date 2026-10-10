"""Export three public training examples for the website; never use final test."""

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from fault_data import CURVES, DYNAMIC_FEATURES, load_splits, read_log
from prepare_fault_data import ARCHIVE_PATH, sha256_file
from train import data_provenance


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    directory = PROJECT_DIR / "data" / "deep4ge"
    report = data_provenance(directory)
    rows = {row["example_id"]: row for row in load_splits(directory)["train"]}
    selected = [
        ("deep4ge_06106", "generalization", "Kurvor som går åt olika håll", "rnn_training_log.csv", "Träningsfel sjunker medan valideringsfel ökar."),
        ("deep4ge_12583", "baseline", "En publicerad referenskörning", "baseline_training_log.csv", "Uppmätta resultat från ett experiment utan injicerad ändring."),
        ("deep4ge_06244", "loss", "Träningsresultat på en annan skala", "loss_training_log.csv", "Undersök hur skalan påverkar tolkningen av kurvorna."),
    ]
    output = PROJECT_DIR / "web" / "dist" / "examples"
    output.mkdir(parents=True, exist_ok=True)
    columns = ["epoch", *CURVES, *DYNAMIC_FEATURES]
    if sha256_file(ARCHIVE_PATH) != report["archive_sha256"]:
        raise ValueError("Source archive checksum mismatch.")
    examples = []
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        for example_id, slug, name, filename, description in selected:
            row = rows[example_id]
            content = archive.read(f"deep4ge-{report['revision']}/{row['source_file']}")
            if hashlib.sha256(content).hexdigest() != row["source_sha256"]:
                raise ValueError("Example source checksum mismatch.")
            log = read_log(io.StringIO(content.decode("utf-8-sig")))
            with (output / filename).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(log)
            examples.append({"id": slug, "name": name, "file": filename, "description": description,
                             "architecture": row["architecture"], "category": row["diagnosis"],
                             "epochs": len(log), "source_example_id": example_id,
                             "source_sha256": row["source_sha256"]})
    (output / "index.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "template.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"epoch": epoch} for epoch in range(3))
    (output / "ATTRIBUTION.md").write_text(
        f"# Public example data\n\nThree logs from the Deep4ge training split.\n"
        f"Source: https://github.com/SigmaJahan/deep4ge/tree/{report['revision']}\n\n"
        "Dataset license: CC BY 4.0. See https://creativecommons.org/licenses/by/4.0/\n\n"
        "Only observed input columns are included in the CSV exports. Published categories\n"
        "and source hashes remain in index.json for explicitly labelled demonstrations.\n"
        "No validation or final-test examples are exported.\n", encoding="utf-8")
    print(f"Exported {len(examples)} training examples to {output}")


if __name__ == "__main__":
    main()
