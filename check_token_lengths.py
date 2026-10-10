"""Check all examples with the actual instruction tokenizer, without model weights."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from fault_data import load_splits
from fault_prompts import create_input_messages, create_training_messages
from train import DEFAULT_MODEL, data_provenance


PROJECT_DIR = Path(__file__).resolve().parent


def check_row(row, tokenizer, max_length, generation_tokens=24):
    messages = create_training_messages(row)
    full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    prompt = tokenizer.apply_chat_template(create_input_messages(row), tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
    prefix_ok = full[:len(prompt)] == prompt
    supervised = len(full) - len(prompt)
    return {"example_id": row["example_id"], "split": row["split"],
            "training_tokens": len(full), "prompt_tokens": len(prompt),
            "generation_budget_tokens": len(prompt) + generation_tokens,
            "answer_tokens": supervised, "prefix_ok": prefix_ok,
            "fits": prefix_ok and supervised > 0 and max(len(full), len(prompt) + generation_tokens) <= max_length}


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "deep4ge")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "token_lengths")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    splits, provenance = load_splits(args.data_dir), data_provenance(args.data_dir)
    from transformers import AutoTokenizer
    from tqdm import tqdm
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_only,
                                                 padding_side="left", trust_remote_code=False,
                                                 cache_dir=PROJECT_DIR / "models" / "tokenizer_cache")
    except OSError as error:
        # Report the blocker honestly; another tokenizer must not masquerade as
        # a successful check of the requested Llama model.
        status = {"status": "tokenizer_unavailable", "model_id": args.model_id,
                  "checked_examples": 0, "max_length": args.max_length,
                  "dataset_csv_sha256": provenance["csv_sha256"],
                  "reason": str(error), "next_step": "Authenticate with hf auth login using an account with model access, then rerun this command."}
        (args.output_dir / "token_length_report.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise SystemExit("Tokenizer unavailable. For the gated Llama repository, use hf auth login with model access. No token length check has passed.") from error
    if not tokenizer.chat_template:
        raise ValueError("The selected tokenizer has no instruction chat template.")
    records, summary = [], {}
    for name, rows in splits.items():
        selected = [check_row(row, tokenizer, args.max_length) for row in tqdm(rows, desc=f"Checking {name}")]
        records.extend(selected)
        lengths = [max(row["training_tokens"], row["generation_budget_tokens"]) for row in selected]
        summary[name] = {"examples": len(selected), "maximum_required_tokens": max(lengths),
                         "median_required_tokens": percentile(lengths, 0.5), "p95_required_tokens": percentile(lengths, 0.95),
                         "over_budget": sum(length > args.max_length for length in lengths),
                         "invalid_answer_masking": sum(not row["prefix_ok"] or row["answer_tokens"] <= 0 for row in selected)}
    with (args.output_dir / "example_lengths.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    passed = all(row["fits"] for row in records)
    tokenizer_digest = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    report = {"status": "passed" if passed else "failed", "model_id": args.model_id,
              "tokenizer_sha256": tokenizer_digest,
              "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "checked_examples": len(records), "max_length": args.max_length, "reserved_generation_tokens": 24,
              "dataset_csv_sha256": provenance["csv_sha256"], "splits": summary,
              "note": "No model weights loaded; no training, predictions, or test labels scored."}
    (args.output_dir / "token_length_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Some examples exceed the context budget or have invalid answer masking. See example_lengths.csv.")


if __name__ == "__main__":
    main()
