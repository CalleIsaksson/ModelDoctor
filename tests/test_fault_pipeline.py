import csv
import io
import json
import math
import unittest

from analysis_features import observed_features
from check_token_lengths import check_row
from fault_data import (CURVES, DYNAMIC_FEATURES, FAULT_LABELS, input_hash,
                        log_to_input, parse_number, read_log, validate_splits)
from fault_metrics import classification_metrics
from fault_prompts import create_input_messages, create_training_messages, extract_diagnosis
from prepare_fault_data import remove_conflicting_inputs, split_by_program


def example(group, label_index=0):
    log = [{"epoch": str(epoch), **{column: "0" for column in (*CURVES, *DYNAMIC_FEATURES)}}
           for epoch in range(2)]
    log[0]["train_loss"] = str(group * 10 + label_index)
    row = {"example_id": f"case_{group}_{label_index}", "source_group": str(group),
           "source_file": "secret_label_in_filename.csv", "split": "",
           **log_to_input(log, "FNN"), "diagnosis": FAULT_LABELS[label_index]}
    row["input_sha256"] = input_hash(row)
    return row


class FaultPipelineTests(unittest.TestCase):
    def test_prompt_does_not_depend_on_answer_or_source_identity(self):
        row = example(1)
        changed = dict(row, diagnosis="activation_fault", source_group="different_program",
                       source_file="different_label.csv", example_id="different_id", is_faulty=True)
        self.assertEqual(create_input_messages(row), create_input_messages(changed))
        self.assertNotEqual(create_training_messages(row), create_training_messages(changed))

    def test_cpu_baseline_features_exclude_target_and_source_identity(self):
        row = example(1)
        changed = dict(row, diagnosis="activation_fault", source_group="other", source_file="other.csv",
                       example_id="other", split="test", source_sha256="other", input_sha256="other")
        self.assertEqual(json.dumps(observed_features(row), sort_keys=True),
                         json.dumps(observed_features(changed), sort_keys=True))
        row["val_loss"] = "[null, null]"
        self.assertTrue(math.isnan(observed_features(row)["val_loss_last"]))
        self.assertEqual(observed_features(row)["val_loss_missing_rate"], 1)

    def test_length_check_reserves_generation_tokens_and_checks_answer_prefix(self):
        class ControlledTokenizer:
            def __init__(self, invalid_prefix=False):
                self.invalid_prefix = invalid_prefix

            def apply_chat_template(self, messages, **kwargs):
                if len(messages) == 3:
                    return ([999] if self.invalid_prefix else [0]) + list(range(1, 104))
                return list(range(100))

        row = example(1)
        self.assertFalse(check_row(row, ControlledTokenizer(), max_length=123)["fits"])
        self.assertTrue(check_row(row, ControlledTokenizer(), max_length=124)["fits"])
        self.assertFalse(check_row(row, ControlledTokenizer(True), max_length=4096)["fits"])

    def test_nonfinite_observations_remain_missing_not_zero(self):
        self.assertIsNone(parse_number("nan"))
        self.assertIsNone(parse_number("inf"))
        self.assertEqual(parse_number("tf.Tensor(False, shape=(), dtype=bool)"), 0)
        self.assertEqual(parse_number("True"), 1)
        log = [{"epoch": "0", **{column: "nan" for column in (*CURVES, *DYNAMIC_FEATURES)}}]
        observed = log_to_input(log)
        self.assertEqual(json.loads(observed["train_loss"]), [None])
        self.assertEqual(json.loads(observed["dynamic_features"])["mean_gradient"]["missing"], 1)
        self.assertNotIn("NaN", observed["dynamic_features"])

    def test_interleaved_epoch_logs_are_rejected(self):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["epoch", *CURVES, *DYNAMIC_FEATURES])
        writer.writeheader()
        for epoch in (0, 0, 1, 1):
            writer.writerow({"epoch": epoch, **{column: 0 for column in (*CURVES, *DYNAMIC_FEATURES)}})
        stream.seek(0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            read_log(stream)

    def test_programs_and_cross_program_duplicates_stay_in_one_split(self):
        rows = [example(group, label) for group in range(20) for label in range(8)]
        rows[8] = dict(rows[0], example_id="cross_program_duplicate", source_group="1")
        splits = split_by_program(rows, seed=42)
        validate_splits(splits)
        assignments = {row["source_group"]: name for name, values in splits.items() for row in values}
        self.assertEqual(assignments["0"], assignments["1"])
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        self.assertEqual(splits, split_by_program(rows, seed=42))
        leaked = {name: [dict(row) for row in values] for name, values in splits.items()}
        leaked["test"][0]["source_group"] = leaked["train"][0]["source_group"]
        with self.assertRaisesRegex(ValueError, "Source programs overlap"):
            validate_splits(leaked)

    def test_conflicting_identical_observations_are_excluded(self):
        row = example(0)
        conflict = dict(row, diagnosis="weight_fault", example_id="conflict")
        retained, patterns = remove_conflicting_inputs([row, conflict, example(1)])
        self.assertEqual(len(retained), 1)
        self.assertEqual(patterns, {row["input_sha256"]})

    def test_accuracy_cannot_hide_missed_minority_class(self):
        records = [{"expected_diagnosis": "no_injected_fault", "predicted_diagnosis": "no_injected_fault"}] * 9
        records += [{"expected_diagnosis": "weight_fault", "predicted_diagnosis": "no_injected_fault"}]
        metrics = classification_metrics(records)
        self.assertEqual(metrics["accuracy"], 0.9)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertLess(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["per_label"]["weight_fault"]["recall"], 0)

    def test_output_must_be_exactly_one_supported_category(self):
        self.assertEqual(extract_diagnosis("Diagnosis: layer_fault\n"), "layer_fault")
        self.assertEqual(extract_diagnosis("Diagnosis: layer_fault or loss_fault"), "unparsed")
        self.assertEqual(extract_diagnosis("Diagnosis: overfitting"), "unparsed")
        self.assertEqual(extract_diagnosis("Diagnosis: layer_fault\nEvidence: invented"), "unparsed")


if __name__ == "__main__":
    unittest.main()
