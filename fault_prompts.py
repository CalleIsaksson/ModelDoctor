"""Supervision is the published mutation category only, never invented prose."""

import re

from fault_data import FAULT_LABELS, INPUT_COLUMNS


SYSTEM_PROMPT = """You are ModelDoctor. Classify a neural-network training log into one fault category.
Use the observed curves and internal training statistics. null means missing or nonfinite.
Epoch numbers are zero-based; curves are sampled and statistics summarize the complete log.
Recorded train_acc/val_acc follow the source metric definitions; some regression runs use
1 - MAPE/100 instead of classification accuracy.
Categories:
no_injected_fault: baseline run without an injected mutation; not a guarantee of good performance.
hyperparameter_fault: batch size, learning rate, epoch count, or batching fault.
layer_fault: network architecture or layer configuration fault.
loss_fault: loss function selection fault.
activation_fault: activation function fault.
optimization_fault: optimizer or gradient clipping fault.
weight_fault: weight initialization or bias fault.
regularization_fault: dropout or regularizer fault.
Predict the category supported best by the supplied log. The exact cause may be ambiguous.
Answer with exactly one line: Diagnosis: <category>
"""


def create_input_messages(row):
    # Explicit allowlist: IDs, filenames, source_group, is_faulty and labels must
    # never enter the prompt, even if callers pass the complete manifest record.
    content = "Classify this training run.\n" + "\n".join(
        f"{column}: {row[column]}" for column in INPUT_COLUMNS
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


def create_training_messages(row):
    if row["diagnosis"] not in FAULT_LABELS:
        raise ValueError(f"Unsupported label: {row['diagnosis']}")
    return create_input_messages(row) + [
        {"role": "assistant", "content": f"Diagnosis: {row['diagnosis']}"},
    ]


def extract_diagnosis(answer):
    match = re.fullmatch(r"\s*Diagnosis:\s*([a-z_]+)\s*", answer)
    return match.group(1) if match and match.group(1) in FAULT_LABELS else "unparsed"
