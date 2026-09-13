import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation_metrics import score_evidence_numbers
from modeldoctor_prompts import (
    create_input_messages,
    create_training_messages,
    extract_diagnosis,
)


# 1. Load the tokenizer and QLoRA model
model_id = "meta-llama/Llama-3.2-1B-Instruct"
experiment_name = "v3"
device = "cuda" if torch.cuda.is_available() else "cpu"

project_dir = Path(__file__).resolve().parent
train_file = project_dir / "data" / "modeldoctor_train.csv"
validation_file = project_dir / "data" / "modeldoctor_validation.csv"
test_file = project_dir / "data" / "modeldoctor_test.csv"
adapter_dir = project_dir / "models" / "modeldoctor_lora_v3"
results_dir = project_dir / "results" / experiment_name

batch_size = 4
evaluation_batch_size = 4
learning_rate = 2e-4
num_epochs = 4
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    }
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.print_trainable_parameters()


# 2. Turn one chat into tokens and labels
def tokenize_training_messages(training_messages):
    tokenized = tokenizer.apply_chat_template(
        training_messages,
        tokenize=True,
        return_tensors="pt"
    )

    prompt_messages = training_messages[:2]

    prompt_tokenized = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    prompt_length = prompt_tokenized["input_ids"].shape[1]

    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    labels = input_ids.clone()
    labels[:prompt_length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": prompt_length
    }


# 5. Pad examples so they fit in one batch
def collate_batch(examples):
    input_ids = pad_sequence(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    attention_mask = pad_sequence(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 6. Build a DataLoader from one CSV file
def build_dataloader(csv_file, shuffle):
    dataframe = pd.read_csv(csv_file)

    training_messages = [
        create_training_messages(row)
        for _, row in dataframe.iterrows()
    ]

    tokenized_examples = [
        tokenize_training_messages(messages)
        for messages in training_messages
    ]

    dataloader = DataLoader(
        tokenized_examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch
    )

    return dataframe, dataloader


# 7. Calculate loss only for the answer positions
def calculate_answer_loss(batch):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    first_answer_tokens = (labels != -100).int().argmax(dim=1)
    first_logit_position = max(first_answer_tokens.min().item() - 1, 0)
    logit_positions = torch.arange(
        first_logit_position,
        labels.shape[1],
        device=labels.device
    )

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logit_positions
    )

    shifted_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
    answer_targets = shifted_labels[:, logit_positions]

    return F.cross_entropy(
        out.logits.float().reshape(-1, model.config.vocab_size),
        answer_targets.reshape(-1),
        ignore_index=-100
    )


# 8. Calculate loss without changing the model
def calculate_average_loss(dataloader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            loss = calculate_answer_loss(batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# 8. Generate answers and save the test result
def evaluate_answers(test_dataframe, stage_name):
    model.eval()
    model.config.use_cache = True
    records = []

    for start in tqdm(
        range(0, len(test_dataframe), evaluation_batch_size),
        desc=f"Generating {stage_name} answers"
    ):
        batch_dataframe = test_dataframe.iloc[start:start + evaluation_batch_size]
        batch_messages = [
            create_input_messages(row)
            for _, row in batch_dataframe.iterrows()
        ]

        model_inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        answer_ids = generated_ids[:, prompt_length:]
        answers = tokenizer.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        for (_, row), answer in zip(batch_dataframe.iterrows(), answers):
            expected = row["diagnosis"]
            predicted = extract_diagnosis(answer)
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

    predictions = pd.DataFrame(records)
    accuracy = predictions["correct"].mean()
    parsed_rate = (predictions["predicted_diagnosis"] != "unparsed").mean()
    format_rate = predictions["format_ok"].mean()
    evidence_with_numbers = predictions["evidence_has_numbers"]
    evidence_number_rate = evidence_with_numbers.mean()
    if evidence_with_numbers.any():
        numeric_grounding_rate = predictions.loc[
            evidence_with_numbers,
            "evidence_numbers_supported"
        ].mean()
    else:
        numeric_grounding_rate = 0.0

    per_label_accuracy = {}
    for diagnosis, group in predictions.groupby("expected_diagnosis"):
        per_label_accuracy[diagnosis] = round(group["correct"].mean(), 4)

    metrics = {
        "examples": len(predictions),
        "diagnosis_accuracy": round(accuracy, 4),
        "parsed_rate": round(parsed_rate, 4),
        "format_rate": round(format_rate, 4),
        "evidence_with_numbers_rate": round(evidence_number_rate, 4),
        "evidence_numeric_grounding_rate": round(numeric_grounding_rate, 4),
        "per_label_accuracy": per_label_accuracy
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / f"{stage_name}_predictions.csv", index=False)

    with (results_dir / f"{stage_name}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"{stage_name} diagnosis accuracy: {accuracy:.2%}")
    print(f"{stage_name} answer format rate: {format_rate:.2%}")
    print(f"{stage_name} numeric evidence grounding: {numeric_grounding_rate:.2%}")

    model.config.use_cache = False
    return metrics


# 10. Train and check validation loss after every epoch
def train_adapters(train_dataloader, validation_dataloader):
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate
    )

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_parameters = None

    for epoch in range(num_epochs):
        model.train()
        model.config.use_cache = False
        total_loss = 0.0

        progress = tqdm(
            train_dataloader,
            desc=f"Training epoch {epoch + 1}/{num_epochs}"
        )

        for batch in progress:
            optimizer.zero_grad()

            loss = calculate_answer_loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        training_loss = total_loss / len(train_dataloader)
        validation_loss = calculate_average_loss(validation_dataloader)

        history.append({
            "epoch": epoch + 1,
            "training_loss": round(training_loss, 4),
            "validation_loss": round(validation_loss, 4)
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"training loss: {training_loss:.4f}, "
            f"validation loss: {validation_loss:.4f}"
        )

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_parameters[name].to(parameter.device))

    print(
        f"Restored best adapter from epoch {best_epoch} "
        f"with validation loss {best_validation_loss:.4f}"
    )

    return history, best_epoch


# 11. Run the full before-and-after experiment
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run the first test without training."
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Use the already saved baseline and start training directly."
    )
    args = parser.parse_args()

    if args.baseline_only and args.skip_baseline:
        parser.error("Choose either --baseline-only or --skip-baseline, not both.")

    train_dataframe, train_dataloader = build_dataloader(train_file, shuffle=True)
    validation_dataframe, validation_dataloader = build_dataloader(
        validation_file,
        shuffle=False
    )
    test_dataframe = pd.read_csv(test_file)

    print(f"\nMODELDOCTOR EXPERIMENT {experiment_name.upper()}")
    print("-" * 42)
    print(f"Device:              {device}")
    print(f"Epochs:              {num_epochs}")
    print(f"Learning rate:       {learning_rate}")
    print("Training examples:", len(train_dataframe))
    print("Validation examples:", len(validation_dataframe))
    print("Test examples:", len(test_dataframe))
    print("Examples per training diagnosis:")
    print(train_dataframe["diagnosis"].value_counts().sort_index().to_string())

    baseline_file = results_dir / "baseline_metrics.json"

    if args.skip_baseline:
        if not baseline_file.exists():
            raise FileNotFoundError(
                "No saved baseline was found. Run python train.py --baseline-only first."
            )

        with baseline_file.open("r", encoding="utf-8") as file:
            baseline_metrics = json.load(file)

        print("Using the already saved baseline result.")
    else:
        baseline_validation_loss = calculate_average_loss(validation_dataloader)
        baseline_metrics = evaluate_answers(test_dataframe, "baseline")
        baseline_metrics["validation_loss"] = round(baseline_validation_loss, 4)

        with baseline_file.open("w", encoding="utf-8") as file:
            json.dump(baseline_metrics, file, indent=4)

        print(f"Baseline validation loss: {baseline_validation_loss:.4f}")

    if args.baseline_only:
        print("Baseline test finished. No training was performed.")
        return

    history, best_epoch = train_adapters(
        train_dataloader,
        validation_dataloader
    )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to: {adapter_dir}")

    final_validation_loss = calculate_average_loss(validation_dataloader)
    finetuned_metrics = evaluate_answers(test_dataframe, "finetuned")
    finetuned_metrics["validation_loss"] = round(final_validation_loss, 4)

    with (results_dir / "finetuned_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(finetuned_metrics, file, indent=4)

    comparison = {
        "experiment": experiment_name,
        "dataset_version": "controlled_synthetic_v3",
        "base_model": model_id,
        "adapter_directory": str(adapter_dir),
        "training_examples": len(train_dataframe),
        "best_epoch": best_epoch,
        "baseline": baseline_metrics,
        "training_history": history,
        "finetuned": finetuned_metrics
    }

    with (results_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    print(
        "Accuracy change: "
        f"{baseline_metrics['diagnosis_accuracy']:.2%} -> "
        f"{finetuned_metrics['diagnosis_accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
