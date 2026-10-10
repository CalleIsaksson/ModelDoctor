"""Exercise real LoRA/AdamW/dropout recovery at optimizer and epoch boundaries."""

from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from test_fault_pipeline import example
from train import build_loader, tiny_model_and_tokenizer, train_adapters
from training_checkpoints import adapter_state, load_checkpoint


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def run_training(self, directory, name, stop=None, resume=None, epochs=2):
        import torch
        from peft import LoraConfig, get_peft_model
        random.seed(42)
        torch.manual_seed(42)
        rows = [example(1, label) for label in range(8)]
        model, tokenizer, _ = tiny_model_and_tokenizer(rows, directory / "base_model")
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                               bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
        args = SimpleNamespace(seed=42, max_length=4096, batch_size=3, gradient_accumulation=2,
                               learning_rate=2e-4, epochs=epochs, checkpoint_every=1,
                               checkpoint_dir=directory / name, resume_from=resume, stop_after_steps=stop)
        result = train_adapters(model, build_loader(rows, tokenizer, args, True),
                                build_loader(rows, tokenizer, args, False), args, {"test": "resume", "seed": 42})
        return result, adapter_state(model)

    def assert_matches(self, expected_result, expected_state, actual_result, actual_state):
        import torch
        self.assertEqual(actual_result["history"], expected_result["history"])
        self.assertEqual(actual_result["best_epoch"], expected_result["best_epoch"])
        self.assertEqual(actual_result["global_step"], expected_result["global_step"])
        for name, value in expected_state.items():
            self.assertTrue(torch.equal(value, actual_state[name]), name)

    def test_resume_matches_uninterrupted_including_short_accumulation_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected, expected_state = self.run_training(directory, "uninterrupted")
            for step in (1, 2, 3):
                with self.subTest(stop_after_step=step):
                    name = f"interrupted_{step}"
                    stopped, _ = self.run_training(directory, name, stop=step)
                    self.assertTrue(stopped["stopped"])
                    resumed, resumed_state = self.run_training(directory, name, resume=directory / name)
                    self.assertFalse(resumed["stopped"])
                    self.assert_matches(expected, expected_state, resumed, resumed_state)

    def test_completed_epoch_can_resume_with_more_epochs_and_changed_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected, expected_state = self.run_training(directory, "uninterrupted")
            self.run_training(directory, "extended", epochs=1)
            resumed, resumed_state = self.run_training(directory, "extended", resume=directory / "extended")
            self.assert_matches(expected, expected_state, resumed, resumed_state)
            with self.assertRaisesRegex(ValueError, "seed"):
                load_checkpoint(directory / "extended", {"test": "resume", "seed": 99})


if __name__ == "__main__":
    unittest.main()
