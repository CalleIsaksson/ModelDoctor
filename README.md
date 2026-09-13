# ModelDoctor

ModelDoctor is a project that uses an existing language model and fine-tunes it into an SLM specialized in diagnosing machine-learning training problems, such as overfitting, underfitting, data leakage, and learning rates that are too high or too low.

The current starting model is:

`meta-llama/Llama-3.2-1B-Instruct`

The training dataset consists of structured examples from controlled training runs. Each example connects training metrics and configurations to a diagnosis, explanation, and suggested next experiment.

## Current progress

- Loaded the pretrained Llama model from Hugging Face
- Implemented tokenization
- Applied Llama's chat template
- Generated a response from the model
- Have imported data from a large dataset of machine-learning training results
- Have created an initial labeling guide for the diagnoses the model will later learn to identify
- Have started defining the CSV structure for labeled training examples
- Structured the system, user, and output templates
- Implemented next-token prediction and calculated loss for the expected answer
- Made the data preparation work for every row in the CSV file
- Added batching and padding with a PyTorch DataLoader
- Implemented QLoRA and moved the training to my GPU
- Added a simple training loop for the LoRA adapters
- Created balanced examples for every diagnosis in the labeling guide
- Split the examples into separate train, validation, and test files
- Added a baseline test that saves the model's answers before training
- Added validation during training and a comparison after training
- Added saving for the trained LoRA adapter
- Added a separate challenge test for checking generalization
- Created v2 training data with varied wording and shared neutral contexts
- Completed and evaluated the improved v3 model

The v1 results remain directly in `results`, v2 is in `results/v2`, and v3 is
in `results/v3`. The finished v3 adapter is saved as
`models/modeldoctor_lora_v3`.

## First results

The v1 adapter reached 100% accuracy on the controlled test set, but only
52.38% on the separate challenge set. This showed that the first test data was
too similar to the training data.

For v2, the training examples received more varied wording and shared neutral
context. The v2 adapter reached 77.14% on its controlled test set and 38.10%
on the challenge set. Overfitting improved from 0% to 33.33% on the challenge
set, but the model started predicting `insufficient_evidence` too often. This
is the next problem to investigate.

An ablation test showed that unfamiliar model names were not the cause. On 15
complete challenge cases, the model predicted `insufficient_evidence` 9 times.
Replacing only `Additional information` with neutral text reduced this to zero
and increased accuracy from 13.33% to 46.67%. This suggests that the model is
using that text as a shortcut instead of relying enough on the metric curves.

## V3 improvements completed

- Added the diagnosis rules from `LABELING_GUIDE.md` to the shared system
  prompt and added automatic checks for generated labels.
- Increased the controlled dataset to 350 examples: 252 training, 49
  validation, and 49 test examples.
- Added varied and relevant additional information for every diagnosis.
- Added neutral `insufficient_evidence` examples where missing or contradictory
  metrics must reveal the answer.
- Added contrast examples where the same wording occurs with different curve
  patterns and labels.
- Kept the old challenge set as a development test instead of treating it as a
  final benchmark.
- Added a numeric grounding check for values written in `Evidence`.
- Optimized training so logits are only created for answer positions. This
  produces exactly the same loss while using less GPU memory.
- Added validation-based checkpoint selection and saved the best LoRA adapter.
- Created and hashed a new final test after v3 training was complete, then ran
  it once without making more model changes.

## V3 results

- Controlled test: 97.96% diagnosis accuracy on 49 examples.
- Development challenge: 85.71% diagnosis accuracy on 21 examples.
- Frozen final test: 80.00% diagnosis accuracy on 35 examples.
- Final answer format rate: 100%.
- Final numeric grounding rate: 76.67% among answers containing numbers.
- `insufficient_evidence` reached 100% on both the development and final tests,
  so the excessive fallback seen in v2 was fixed.

## Remaining limitations

- The final test is controlled and manually designed, not a real-world
  benchmark.
- Data leakage only reached 20% on the final test. The model recognized familiar
  wording such as `near-duplicate`, but missed new descriptions of entity
  overlap, preprocessing leakage, and target-revealing features.
- Two final overfitting examples were incorrectly called healthy.
- One final underfitting example was confused with a learning rate that was too
  low.
- Numeric grounding only checks whether a number exists in the input or is a
  supported difference. It cannot always detect when a real number is attached
  to the wrong metric.
- The final test has now been evaluated and must not be used as an untouched
  benchmark for future model changes. A future version needs a new final set.

Run only the first test without training:

`python train.py --baseline-only`

Run baseline, training, and the controlled after-test:

`python train.py`

Run the development challenge with the saved adapter:

`python evaluate_model.py`

The frozen final test has already been evaluated once. The evaluation script
blocks an accidental second final run unless `--allow-repeat` is explicitly
used.
