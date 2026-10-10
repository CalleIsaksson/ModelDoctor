"""Classify a new observed log with the saved Deep4ge adapter."""

import argparse
from pathlib import Path

from fault_data import log_to_input, read_log
from fault_prompts import extract_diagnosis
from train import load_saved_adapter, predict_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="CSV with Deep4ge's observed log columns.")
    parser.add_argument("--architecture", choices=("FNN", "CNN", "RNN", "unknown"), default="unknown")
    parser.add_argument("--adapter-dir", type=Path, default=Path(__file__).resolve().parent / "models" / "modeldoctor_deep4ge")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()
    import torch

    model, tokenizer, settings = load_saved_adapter(
        args.adapter_dir, args.cpu or not torch.cuda.is_available(), not args.no_quantization,
    )
    with args.log.open(newline="", encoding="utf-8-sig") as file:
        row = log_to_input(read_log(file), args.architecture, settings["curve_points"])
    answer = predict_rows(model, tokenizer, [row], settings["max_length"])[0]
    label = extract_diagnosis(answer)
    if label == "unparsed":
        raise ValueError(f"Model did not return a valid category: {answer!r}")
    print(f"Diagnosis: {label}")


if __name__ == "__main__":
    main()
