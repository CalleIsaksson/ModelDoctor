import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation_metrics import score_evidence_numbers
from modeldoctor_prompts import create_input_messages, extract_diagnosis


# Files and model settings
project_dir = Path(__file__).resolve().parent
challenge_file = project_dir / "data" / "modeldoctor_challenge.csv"
adapter_dir = project_dir / "models" / "modeldoctor_lora_v3"
results_dir = project_dir / "results" / "v3"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
batch_size = 4


def load_finetuned_model():
    if not adapter_dir.exists():
        raise FileNotFoundError(
            "The saved LoRA adapter was not found. Run python train.py first."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, tokenizer


def evaluate(model, tokenizer, dataframe, description="Testing examples"):
    records = []

    for start in tqdm(
        range(0, len(dataframe), batch_size),
        desc=description
    ):
        batch_dataframe = dataframe.iloc[start:start + batch_size]
        batch_messages = [
            create_input_messages(row)
            for _, row in batch_dataframe.iterrows()
        ]

        model_inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        answers = tokenizer.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        for (_, row), answer in zip(batch_dataframe.iterrows(), answers):
            predicted = extract_diagnosis(answer)
            expected = row["diagnosis"]
            evidence_score = score_evidence_numbers(row, answer)
            clean_answer = answer.lower().replace("*", "")
            format_ok = all(
                heading in clean_answer
                for heading in ["diagnosis:", "evidence:", "next experiment:"]
            )

            records.append({
                "example_id": row["example_id"],
                "expected_diagnosis": expected,
                "predicted_diagnosis": predicted,
                "correct": predicted == expected,
                "format_ok": format_ok,
                "answer": answer.strip(),
                **evidence_score,
            })

    return pd.DataFrame(records)


def save_results(predictions, evaluation_name, dataset_name):
    accuracy = predictions["correct"].mean()
    format_rate = predictions["format_ok"].mean()
    parsed_rate = (predictions["predicted_diagnosis"] != "unparsed").mean()
    evidence_with_numbers = predictions["evidence_has_numbers"]
    evidence_number_rate = evidence_with_numbers.mean()
    if evidence_with_numbers.any():
        numeric_grounding_rate = predictions.loc[
            evidence_with_numbers,
            "evidence_numbers_supported"
        ].mean()
    else:
        numeric_grounding_rate = 0.0

    per_label_accuracy = {
        diagnosis: round(group["correct"].mean(), 4)
        for diagnosis, group in predictions.groupby("expected_diagnosis")
    }

    metrics = {
        "dataset": dataset_name,
        "examples": len(predictions),
        "diagnosis_accuracy": round(accuracy, 4),
        "parsed_rate": round(parsed_rate, 4),
        "format_rate": round(format_rate, 4),
        "evidence_with_numbers_rate": round(evidence_number_rate, 4),
        "evidence_numeric_grounding_rate": round(numeric_grounding_rate, 4),
        "per_label_accuracy": per_label_accuracy,
        "note": "This is a controlled evaluation set, not a real-world benchmark."
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        results_dir / f"{evaluation_name}_predictions.csv",
        index=False
    )

    with (results_dir / f"{evaluation_name}_metrics.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    comparison_file = results_dir / "comparison.json"

    if comparison_file.exists():
        with comparison_file.open("r", encoding="utf-8") as file:
            comparison = json.load(file)

        comparison[evaluation_name] = metrics

        with comparison_file.open("w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=4)

    return metrics


def print_summary(metrics, evaluation_name):
    print(f"\n{evaluation_name.upper()} TEST RESULT")
    print("-" * 35)
    print(f"Examples:           {metrics['examples']}")
    print(f"Diagnosis accuracy: {metrics['diagnosis_accuracy']:.2%}")
    print(f"Answer format:      {metrics['format_rate']:.2%}")
    print(f"Numeric grounding:  {metrics['evidence_numeric_grounding_rate']:.2%}")
    print("\nAccuracy per diagnosis:")

    for diagnosis, accuracy in metrics["per_label_accuracy"].items():
        print(f"  {diagnosis:<28} {accuracy:.2%}")

    print(f"\nSaved to {results_dir / f'{evaluation_name}_metrics.json'}")
    print(
        "Full answers saved to "
        f"{results_dir / f'{evaluation_name}_predictions.csv'}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=challenge_file,
        help="CSV file to evaluate."
    )
    parser.add_argument(
        "--name",
        default="challenge",
        help="Name used for the saved result files."
    )
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Allow a frozen final test to be run again."
    )
    args = parser.parse_args()

    data_file = args.data
    if not data_file.is_absolute():
        data_file = project_dir / data_file

    if args.name == "final":
        final_result_file = results_dir / "final_metrics.json"
        if final_result_file.exists() and not args.allow_repeat:
            raise FileExistsError(
                "The frozen final test has already been evaluated. "
                "Use --allow-repeat only if you intentionally want another run."
            )

        manifest_file = results_dir / "final_test_manifest.json"
        with manifest_file.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        actual_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
        if actual_hash != manifest["sha256"]:
            raise ValueError("The frozen final test file has changed.")

    dataframe = pd.read_csv(data_file, keep_default_na=False)
    dataset_name = dataframe["source"].iloc[0]

    print(f"{args.name.upper()} DATA")
    print("-" * 35)
    print(f"Examples: {len(dataframe)}")
    print("Examples per diagnosis:")
    print(dataframe["diagnosis"].value_counts().sort_index().to_string())
    print("\nLoading the saved QLoRA adapter...")

    model, tokenizer = load_finetuned_model()
    predictions = evaluate(
        model,
        tokenizer,
        dataframe,
        description=f"Testing {args.name} examples"
    )
    metrics = save_results(
        predictions,
        evaluation_name=args.name,
        dataset_name=dataset_name
    )
    print_summary(metrics, args.name)


if __name__ == "__main__":
    main()
