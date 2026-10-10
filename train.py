"""Fine-tune ModelDoctor on published Deep4ge fault-category labels."""

import argparse
from collections import Counter
import csv
import hashlib
import inspect
import json
from pathlib import Path
import random
import tempfile

from fault_data import FAULT_LABELS, INPUT_COLUMNS, load_splits
from fault_metrics import classification_metrics
from fault_prompts import create_input_messages, create_training_messages, extract_diagnosis
from training_checkpoints import (adapter_state, load_checkpoint, resolve_checkpoint,
                                  restore_adapter, restore_random_state, save_checkpoint)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "deep4ge")
    parser.add_argument("--adapter-dir", type=Path, default=PROJECT_DIR / "models" / "modeldoctor_deep4ge")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results" / "deep4ge")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--check-data", action="store_true", help="Validate CSVs and provenance without loading a model.")
    parser.add_argument("--smoke-test", action="store_true", help="Offline CPU test using a tiny random model and real rows; no usable adapter.")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the saved Deep4ge adapter without training.")
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="validation",
                        help="Use validation for development; select test explicitly for the final benchmark.")
    parser.add_argument("--cpu", action="store_true", help="Explicitly allow slow CPU training without quantization.")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing adapter.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Defaults to an adapter-specific checkpoint directory.")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Save every N optimizer steps, and after each epoch.")
    parser.add_argument("--resume-from", type=Path, help="A latest.pt checkpoint or its containing directory.")
    parser.add_argument("--stop-after-steps", type=int, help="Stop and checkpoint after this many additional optimizer steps.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.max_length) < 1:
        parser.error("Epochs, batch size, accumulation, and max length must be positive.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")
    if args.checkpoint_every < 1 or (args.stop_after_steps is not None and args.stop_after_steps < 1):
        parser.error("Checkpoint interval and step limit must be positive.")
    if args.resume_from and (args.baseline_only or args.evaluate_only or args.check_data or args.smoke_test):
        parser.error("--resume-from applies to normal training only.")
    if args.resume_from and args.with_baseline:
        parser.error("A baseline belongs to a fresh run, not a resumed adapter.")
    if any(limit is not None and limit < len(FAULT_LABELS)
           for limit in (args.max_train_examples, args.max_eval_examples)):
        parser.error("Example limits must be at least eight, to include every category.")
    if sum((args.smoke_test, args.baseline_only, args.evaluate_only, args.check_data)) > 1:
        parser.error("Choose only one of --smoke-test, --baseline-only, --evaluate-only, --check-data.")
    return args


def representative_subset(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    shuffled = rng.sample(rows, len(rows))
    selected, selected_ids = [], set()
    for label in FAULT_LABELS:
        first = next(row for row in shuffled if row["diagnosis"] == label)
        selected.append(first)
        selected_ids.add(first["example_id"])
    selected.extend(row for row in shuffled if row["example_id"] not in selected_ids)
    return selected[:limit]


def data_provenance(directory):
    report_path = directory / "preparation_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Missing preparation_report.json. Run prepare_fault_data.py first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for filename, expected in report["csv_sha256"].items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Dataset file has changed since preparation: {filename}")
    return report


def tiny_model_and_tokenizer(rows, directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    splitter = pre_tokenizers.Whitespace()
    words = set()
    for row in rows:
        for message in create_training_messages(row):
            words.update(word for word, _ in splitter.pre_tokenize_str(message["content"]))
    words.update(("system", "user", "assistant"))
    vocab = {token: index for index, token in enumerate(["[PAD]", "[EOS]", "[UNK]", *sorted(words)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = splitter
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       eos_token="[EOS]", unk_token="[UNK]", padding_side="left")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ':\\n' + message['content'] + '\\n' }}"
        "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant:\\n' }}{% endif %}"
    )
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=8192, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    base_dir = directory / "base"
    model.save_pretrained(base_dir)
    return model, tokenizer, str(base_dir)


def load_base_model(model_id, cpu=False, quantize=True):
    import torch
    from transformers import AutoModelForCausalLM

    device = "cpu" if cpu else "cuda"
    dtype = torch.float32 if cpu else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    kwargs = {"device_map": {"": device}, "dtype": dtype}
    if not cpu and quantize:
        from transformers import BitsAndBytesConfig
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise RuntimeError("QLoRA requires bitsandbytes. Install requirements-gpu.txt, or use --no-quantization.") from error
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def load_saved_adapter(adapter_dir, cpu=False, quantize=True):
    from peft import PeftModel
    from transformers import AutoTokenizer

    settings = json.loads((adapter_dir / "modeldoctor_config.json").read_text(encoding="utf-8"))
    if settings["task"] != "deep4ge_fault_category" or settings["labels"] != list(FAULT_LABELS):
        raise ValueError("This adapter is not a Deep4ge ModelDoctor classifier.")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, padding_side="left")
    base = load_base_model(settings["base_model"], cpu, quantize)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer, settings


def tokenize_row(row, tokenizer, max_length):
    messages = create_training_messages(row)
    full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, return_dict=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template does not preserve the prompt prefix for answer masking.")
    if len(full) > max_length:
        raise ValueError(f"Example {row['example_id']} needs {len(full)} tokens; increase --max-length or reprepare shorter curves.")
    if len(full) <= len(prompt):
        raise ValueError("Training example has no answer tokens.")
    return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):]}


def build_loader(rows, tokenizer, args, training):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm

    encoded = [tokenize_row(row, tokenizer, args.max_length)
               for row in tqdm(rows, desc="Tokenizing training" if training else "Tokenizing validation")]

    def collate(examples):
        length = max(len(example["input_ids"]) for example in examples)
        return {
            "input_ids": torch.tensor([example["input_ids"] + [tokenizer.pad_token_id] * (length - len(example["input_ids"])) for example in examples]),
            "attention_mask": torch.tensor([[1] * len(example["input_ids"]) + [0] * (length - len(example["input_ids"])) for example in examples]),
            "labels": torch.tensor([example["labels"] + [-100] * (length - len(example["labels"])) for example in examples]),
        }

    sampler = None
    if training:
        counts = Counter(row["diagnosis"] for row in rows)
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler([1 / counts[row["diagnosis"]] for row in rows],
                                        num_samples=len(rows), replacement=True, generator=generator)
    return DataLoader(encoded, batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
                      generator=torch.Generator().manual_seed(args.seed + (1 if training else 2)))


def answer_loss(model, batch):
    import torch
    from torch.nn import functional as F

    batch = {key: value.to(model.device) for key, value in batch.items()}
    labels = batch.pop("labels")
    # For Llama, omit prompt-only vocabulary logits to keep GPU memory modest.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if "logits_to_keep" in inspect.signature(base.forward).parameters:
        first = max((labels != -100).int().argmax(dim=1).min().item() - 1, 0)
        positions = torch.arange(first, labels.shape[1] - 1, device=model.device)
        logits = model(**batch, logits_to_keep=positions).logits
        targets = labels[:, positions + 1]
    else:
        logits = model(**batch).logits[:, :-1]
        targets = labels[:, 1:]
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100)
    return loss, (targets != -100).sum().item()


def validation_loss(model, loader):
    import torch

    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss, count = answer_loss(model, batch)
            total += loss.item() * count
            tokens += count
    return total / tokens


def predict_rows(model, tokenizer, rows, max_length, batch_size=1):
    import torch
    from tqdm import tqdm

    model.eval()
    model.config.use_cache = True
    answers = []
    for start in tqdm(range(0, len(rows), batch_size), desc="Classifying logs"):
        messages = [create_input_messages(row) for row in rows[start:start + batch_size]]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              padding=True, return_tensors="pt", return_dict=True)
        if inputs["input_ids"].shape[1] + 24 > max_length:
            raise ValueError("Evaluation prompt exceeds context budget; increase --max-length.")
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        answers.extend(tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    model.config.use_cache = False
    return answers


def evaluate(model, tokenizer, rows, args, stage):
    answers = predict_rows(model, tokenizer, rows, args.max_length, args.batch_size)
    records = [{"example_id": row["example_id"], "source_group": row["source_group"],
                "expected_diagnosis": row["diagnosis"], "predicted_diagnosis": extract_diagnosis(answer),
                "answer": answer.strip()} for row, answer in zip(rows, answers)]
    metrics = classification_metrics(records)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / f"{stage}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.results_dir / f"{stage}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{stage}: accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.4f}, parsed={metrics['parsed_rate']:.2%}")
    return metrics


def checkpoint_signature(args, comparison, train_rows, val_rows):
    return {
        "task": comparison["task"], "base_model": comparison["base_model"],
        "dataset_csv_sha256": comparison["dataset_csv_sha256"], "seed": args.seed,
        "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate, "max_length": args.max_length,
        "runtime": comparison["runtime"], "base_model_revision": comparison["base_model_revision"],
        "tokenizer_sha256": comparison["tokenizer_sha256"],
        "chat_template_sha256": comparison["chat_template_sha256"],
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": "all-linear"},
        "train_ids_sha256": hashlib.sha256(json.dumps([row["example_id"] for row in train_rows]).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256(json.dumps([row["example_id"] for row in val_rows]).encode()).hexdigest(),
    }


def train_adapters(model, train_loader, val_loader, args, signature):
    import torch
    from tqdm import tqdm

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=model.device.type == "cuda" and not torch.cuda.is_bf16_supported())
    checkpoint_path = args.checkpoint_dir / "latest.pt"
    progress_state = {"epoch": 0, "next_batch": 0, "global_step": 0, "loss_sum": 0.0,
                      "answer_tokens": 0, "history": [], "best_loss": float("inf"),
                      "best_adapter": None, "best_epoch": None}
    if args.resume_from:
        state = load_checkpoint(args.resume_from, signature)
        restore_adapter(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        progress_state = state["progress"]
        if args.epochs < progress_state["epoch"] or (args.epochs == progress_state["epoch"] and progress_state["next_batch"]):
            raise ValueError("--epochs is smaller than the checkpoint's training position.")
        restore_random_state(state["rng"])
        print(f"Resuming optimizer step {progress_state['global_step']}, epoch {progress_state['epoch'] + 1}, batch {progress_state['next_batch']}")
    else:
        save_checkpoint(checkpoint_path, model, optimizer, scaler, progress_state, signature)
    initial_step = progress_state["global_step"]
    for epoch in range(progress_state["epoch"], args.epochs):
        model.train()
        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        # Sampling is reproducible for each epoch, including after a restart.
        train_loader.sampler.generator.manual_seed(args.seed + epoch)
        next_batch = progress_state["next_batch"]
        total, tokens = progress_state["loss_sum"], progress_state["answer_tokens"]
        for index, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if index < next_batch:
                continue
            window_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(train_loader) - window_start)
            loss, count = answer_loss(model, batch)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is nonfinite.")
            scaler.scale(loss / window_size).backward()
            total += loss.item() * count
            tokens += count
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                progress_state.update(epoch=epoch, next_batch=index + 1,
                                      global_step=progress_state["global_step"] + 1,
                                      loss_sum=total, answer_tokens=tokens)
                stop = args.stop_after_steps is not None and progress_state["global_step"] - initial_step >= args.stop_after_steps
                if stop or progress_state["global_step"] % args.checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, model, optimizer, scaler, progress_state, signature)
                if stop:
                    print(f"Stopped after a complete optimizer step. Resume from {checkpoint_path}")
                    return {"history": progress_state["history"], "best_epoch": progress_state["best_epoch"],
                            "global_step": progress_state["global_step"], "stopped": True,
                            "checkpoint": str(checkpoint_path)}
        val_loss = validation_loss(model, val_loader)
        if not torch.isfinite(torch.tensor(val_loss)):
            raise ValueError("Validation loss is nonfinite.")
        progress_state["history"].append({"epoch": epoch + 1, "train_loss": total / tokens, "validation_loss": val_loss})
        print(progress_state["history"][-1])
        if val_loss < progress_state["best_loss"]:
            progress_state.update(best_loss=val_loss, best_epoch=epoch + 1, best_adapter=adapter_state(model))
        progress_state.update(epoch=epoch + 1, next_batch=0, loss_sum=0.0, answer_tokens=0)
        save_checkpoint(checkpoint_path, model, optimizer, scaler, progress_state, signature)
    if progress_state["best_adapter"] is None:
        raise ValueError("No completed validation checkpoint exists.")
    restore_adapter(model, progress_state["best_adapter"])
    return {"history": progress_state["history"], "best_epoch": progress_state["best_epoch"],
            "global_step": progress_state["global_step"], "stopped": False,
            "checkpoint": str(checkpoint_path)}


def run(args, splits, report, smoke_dir=None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cpu = args.cpu or not torch.cuda.is_available()
    if cpu and not (args.cpu or args.smoke_test):
        raise RuntimeError("CUDA is unavailable here. Use a CUDA GPU for QLoRA, --smoke-test for an offline check, or explicitly --cpu for slow full-model CPU training.")
    train_rows = representative_subset(splits["train"], args.max_train_examples, args.seed)
    val_rows = representative_subset(splits["validation"], args.max_eval_examples, args.seed)
    evaluation_rows = representative_subset(splits[args.evaluation_split], args.max_eval_examples, args.seed)
    if args.evaluate_only:
        model, tokenizer, settings = load_saved_adapter(args.adapter_dir, cpu, not args.no_quantization)
        if settings["dataset_csv_sha256"] != report["csv_sha256"]:
            raise ValueError("Saved adapter belongs to different prepared CSVs.")
        return evaluate(model, tokenizer, evaluation_rows, args, f"saved_adapter_{args.evaluation_split}")
    if args.adapter_dir.exists() and not args.overwrite and not args.baseline_only and not args.resume_from:
        raise FileExistsError("Adapter directory already exists; choose --adapter-dir or explicitly --overwrite.")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.adapter_dir.with_name(args.adapter_dir.name + "_checkpoints")
    if (args.checkpoint_dir / "latest.pt").exists() and not (args.resume_from or args.overwrite or args.baseline_only):
        raise FileExistsError("Training checkpoint already exists. Use --resume-from, a new --checkpoint-dir, or --overwrite.")
    if args.smoke_test:
        torch.set_num_threads(2)
        base, tokenizer, model_id = tiny_model_and_tokenizer(train_rows, smoke_dir)
    else:
        model_id = args.model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not tokenizer.chat_template:
            raise ValueError("Choose an instruction model with a chat template.")
        base = load_base_model(model_id, cpu, not args.no_quantization)
        if not cpu and not args.no_quantization:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                                  gradient_checkpointing_kwargs={"use_reentrant": False})
        elif not cpu:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                          bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    model.config.use_cache = False
    model.print_trainable_parameters()
    train_loader = build_loader(train_rows, tokenizer, args, True)
    val_loader = build_loader(val_rows, tokenizer, args, False)
    if args.smoke_test:
        from torch.nn import functional as F
        model.eval()
        batch = next(iter(val_loader))
        with torch.no_grad():
            selected_loss, _ = answer_loss(model, batch)
            full_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            reference_loss = F.cross_entropy(full_logits[:, :-1].float().reshape(-1, full_logits.shape[-1]),
                                             batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        if not torch.allclose(selected_loss, reference_loss, atol=1e-6):
            raise ValueError("Optimized answer loss differs from full next-token loss.")
    comparison = {"task": "deep4ge_fault_category", "base_model": model_id,
                  "labels": list(FAULT_LABELS), "source_revision": report["revision"],
                  "dataset_csv_sha256": report["csv_sha256"], "input_columns": list(INPUT_COLUMNS),
                  "curve_points": report["curve_points"], "max_length": args.max_length,
                  "training_examples": len(train_rows), "validation_examples": len(val_rows),
                  "evaluation_split": args.evaluation_split, "evaluation_examples": len(evaluation_rows),
                  "test_evaluated": args.evaluation_split == "test", "seed": args.seed,
                  "base_model_revision": getattr(base.config, "_commit_hash", None),
                  "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                  "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                  "runtime": {"torch_version": str(torch.__version__), "device": model.device.type,
                              "dtype": str(base.dtype), "quantized_4bit": bool(getattr(base, "is_loaded_in_4bit", False))},
                  "epochs": args.epochs, "batch_size": args.batch_size,
                  "gradient_accumulation": args.gradient_accumulation, "learning_rate": args.learning_rate,
                  "training_sampler": "inverse class frequency, training split only",
                  "evaluation_note": report["evaluation_note"], "smoke_test": args.smoke_test}
    if args.baseline_only or args.with_baseline:
        comparison["baseline"] = evaluate(model, tokenizer, evaluation_rows, args, f"baseline_{args.evaluation_split}")
    if args.baseline_only:
        return comparison
    signature = checkpoint_signature(args, comparison, train_rows, val_rows)
    try:
        training = train_adapters(model, train_loader, val_loader, args, signature)
    except KeyboardInterrupt:
        print(f"Training interrupted. Resume the latest completed checkpoint with --resume-from {args.checkpoint_dir}")
        return {**comparison, "stopped": True, "checkpoint": str(args.checkpoint_dir / 'latest.pt')}
    comparison.update(training)
    if training["stopped"]:
        return comparison
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    (args.adapter_dir / "modeldoctor_config.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    comparison["finetuned"] = evaluate(model, tokenizer, evaluation_rows, args, f"finetuned_{args.evaluation_split}")
    if args.smoke_test:
        reloaded, restored_tokenizer, _ = load_saved_adapter(args.adapter_dir, cpu=True, quantize=False)
        before = predict_rows(model, tokenizer, evaluation_rows[:1], args.max_length)
        after = predict_rows(reloaded, restored_tokenizer, evaluation_rows[:1], args.max_length)
        if before != after:
            raise ValueError("Saved adapter does not reproduce the same output after reload.")
        comparison["adapter_reload_passed"] = True
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "training_report.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved adapter: {args.adapter_dir}")
    return comparison


def main():
    args = parse_args()
    splits = load_splits(args.data_dir)
    report = data_provenance(args.data_dir)
    for name, rows in splits.items():
        print(f"{name}: {len(rows)} examples, {len({row['source_group'] for row in rows})} programs; {dict(Counter(row['diagnosis'] for row in rows))}")
    if args.check_data:
        print("PASS: all labels present, source groups and inputs separated, checksums match.")
        return
    if args.smoke_test:
        args.epochs, args.batch_size, args.gradient_accumulation = 1, 1, 2
        args.max_train_examples = args.max_eval_examples = 8
        with tempfile.TemporaryDirectory(prefix="modeldoctor-smoke-") as directory:
            smoke_dir = Path(directory)
            args.adapter_dir, args.results_dir = smoke_dir / "adapter", smoke_dir / "results"
            args.checkpoint_dir, args.stop_after_steps = smoke_dir / "checkpoints", None
            result = run(args, splits, report, smoke_dir)
            print("PASS: real rows, answer masking, LoRA backward/optimizer, validation, generation, adapter save/reload.")
            print("Tiny random model only; no production model trained or saved.")
            output = PROJECT_DIR / "results" / "deep4ge_smoke"
            output.mkdir(parents=True, exist_ok=True)
            (output / "smoke_test.json").write_text(json.dumps({
                "passed": True, "production_training": False, "adapter_reload_passed": result["adapter_reload_passed"],
                "history": result["history"], "source_revision": report["revision"],
            }, indent=2), encoding="utf-8")
    else:
        run(args, splits, report)


if __name__ == "__main__":
    main()
