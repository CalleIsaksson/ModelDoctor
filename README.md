# ModelDoctor

ModelDoctor fine-tunes an instruction language model to classify neural-network
training faults from observed training logs. The current training task uses
**published Deep4ge baseline/mutation-category labels**. ModelDoctor does not
have to produce its own answer key.

The base model is `meta-llama/Llama-3.2-1B-Instruct`. `train.py` trains it on
Deep4ge and saves the resulting adapter to `models/modeldoctor_deep4ge`.

## Current status

The labeled dataset is prepared. Data validation, unit tests and an
offline training test with a tiny random model have passed, including adapter
saving/reloading and exact CPU recovery after interrupted training.

All 13,980 examples passed the official Llama tokenizer check. The largest
context requirement is 1,669 tokens, including 24 reserved generation tokens,
within the default 4,096-token budget. A CPU Random Forest baseline and a
visual review of 24 original training logs are available below.

The Deep4ge Llama adapter has not yet been trained or evaluated in this checkout.
There are no diagnosis accuracy results for it yet. The checks ran on CPU;
the production 4-bit GPU training path still needs to be verified during a
CUDA training run.

## Website

The website begins with a plain-language product introduction, followed by an
optional interactive demo and then setup for your own training. The interface
uses rounded cards, pill controls and soft mint, peach and lavender accents.
It includes automatic Python collection, saved history, project filters,
comparisons of curves/settings, inspection checks and CSV/JSON export.

Three illustrative scenarios explain diverging, improving and flat curves.
Play/pause reveals successive training rounds; they are clearly labelled UI
illustrations, trigger no model inference and never enter training data. Three
original Deep4ge training logs are also available through the technical details.
Their published labels remain separate from the visible assessment and export.

Manual upload accepts a simple `epoch,train_loss,val_loss` CSV; `val_loss` can be
empty or omitted, although that prevents assessing validation development.
The importer expands missing telemetry as blanks for the existing API contract.
Complete original logs are preserved. The model type is optional and hidden
under advanced settings; terms such as loss and validation are explained at the graph.

The main result describes an observed pattern, such as increasing validation
loss after the best recorded epoch. Explore loss, metrics, gradients and learning
rate, toggle curves, select epochs, or focus the best validation epoch. The
investigation plan offers interactive checklists with what to inspect and how
to follow up. Choosing a task or an area adapts these human-written guides only;
it never changes model inputs or training targets. Review marks last for the
current page session and are included in JSON exports; they do not certify a fix.

**Hyperparameters** and **loss function** are not shown as the customer's diagnosis.
The result card describes a measured training pattern and a concrete next check.
`assessment.mjs` supplies this versioned pattern assessment independently of the
experimental classifier. Raw model/dataset categories are available only in
technical details and remain separate in exports. The current answer
key supplies no verified cause or repair; detailed guides help users investigate
without claiming the classifier has learned explanations. Task-specific PyTorch
loss checks link to the official documentation.

Private browser-only preview:
[ModelDoctor](https://modeldoctor-lab.mysticmacaw64.chatgpt.site).

```powershell
python web_server.py --disable-model
```

Open http://127.0.0.1:8000. No frontend installation is needed. `--disable-model`
keeps logging and the UI available without loading Llama or taking GPU memory,
including during your later CUDA training. Collection never triggers inference.
The training scripts, dataset, checkpoints and GPU requirements are unchanged.
Illustrative demos supply no classifier label, and your own uploads inherit no
demo answer. Once the Deep4ge adapter is trained, the same Python server
detects it and uses the existing inference code when started without
`--disable-model`. Inference is explicitly requested for completed saved runs;
curve/LR-only logs receive no learned classification. The hosted static interface
saves manually selected runs in the current browser; automatic Python collection
uses the local server and its separate database under ignored `runs/`.

Try collection without training any model:

```powershell
python examples/collect_run.py
```

For your own training, install the independent client with
`python -m pip install ./sdk`, or download its standalone module from the UI.
Existing CSV producers can use `python watch_logs.py logs`: create a
`filename.csv.done` marker after closing each completed log. Simple loss logs
are expanded with missing telemetry rather than invented measurements.
See [PRODUCT_PILOT.md](PRODUCT_PILOT.md) for integration, replay, measurement
semantics, customer pilot criteria and honest product claims;
[web/README.md](web/README.md) documents the API and web checks.

After training, compare complete validation predictions with the CPU reference:

```powershell
python assess_product.py --predictions results/deep4ge/finetuned_validation_predictions.csv
```

This loads no model weights and does not read the final test. It checks complete
coverage and reports macro F1, baseline flag rate, missed mutations and uncertainty
over source programs. A learned-model advantage and production accuracy have not
yet been demonstrated. [web_checks.json](results/web_checks.json) records the
website checks.

## Ready training data

The source archive has 14,227 training logs. Importing it produced **13,980
usable examples**. We excluded 247 files with repeated/decreasing epoch numbers,
which appear to interleave trajectories. Every exclusion is recorded in
`data/deep4ge/excluded_logs.csv`; the original archive is preserved under `raw`.

| Split | Examples | Independent source programs | File |
| --- | ---: | ---: | --- |
| Train | 10,014 | 41 | `data/deep4ge/modeldoctor_train.csv` |
| Validation | 1,827 | 8 | `data/deep4ge/modeldoctor_validation.csv` |
| Test | 2,139 | 9 | `data/deep4ge/modeldoctor_test.csv` |

All eight categories occur in each split. Every variant of a source program
stays in one split, and identical model inputs cannot cross split boundaries.
The split search balances label coverage and counts approximately; it does not
force an exact 70/15/15 ratio. Repeated inputs with contradictory labels would
be excluded, with counts reported (none were found in this import).

The `diagnosis` column is the answer key:

- `no_injected_fault`: baseline without an injected mutation
- `hyperparameter_fault`: batch size, learning rate, epochs, or batching
- `layer_fault`: architecture/layer configuration
- `loss_fault`: loss function
- `activation_fault`: activation functions
- `optimization_fault`: optimizer/gradient clipping
- `weight_fault`: weight initialization/bias
- `regularization_fault`: dropout/regularizer

Inputs contain architecture family, 12 sampled points from each loss/metric
curve, and summaries of observed internal gradient, activation, weight and
learning-rate signals over the complete log. Missing/nonfinite measurements
remain `null`. Hardware utilization, filenames, source IDs and labels never
enter the prompt. The source's `train_acc`/`val_acc` metric is not always
classification accuracy: some regression runs use `1 - MAPE/100`.

The output is **only** `Diagnosis: <category>`. Deep4ge has no reviewed prose
explanations or repair targets. `Hyperparameter` does not specify a learning
rate direction; baseline does not guarantee healthy training. These classes
are not equivalent to the previous seven symptom diagnoses. See
[FAULT_LABELING_GUIDE.md](FAULT_LABELING_GUIDE.md) for the exact target meanings.

Source: [Deep4ge](https://github.com/SigmaJahan/deep4ge),
[archived release](https://doi.org/10.5281/zenodo.20337241), pinned repository
revision `1d895afac1985876f289fbcdc2ee56c62f612071`.
The original index is `data/deep4ge/source/manifest.csv`. Source documentation,
citation and attribution are preserved in the same folder. Generated dataset
files use CC BY 4.0; adapted seed programs have separate attribution rules.
We import data files only and do not execute the source programs.

## Train

Use your Python environment with a CUDA-enabled PyTorch installation for
QLoRA. If needed, select the appropriate installation command on
[PyTorch's official setup page](https://pytorch.org/get-started/locally/).
Installing the requirements does not convert an existing CPU-only PyTorch
installation into a CUDA build.

```powershell
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available())"
python train.py --check-data
python train.py
```

The Llama model requires access on its
[Hugging Face model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
If it is not cached, authenticate using `hf auth login` with an account that
has access. You can choose another instruction model with `--model-id`.

Default settings are 4-bit QLoRA, rank 8, batch size 1, gradient accumulation
8, three epochs, and a context budget of 4,096 tokens. Only answer tokens
contribute to loss. Inverse-frequency sampling balances the training categories;
validation/test retain their observed distributions. The best validation-loss
adapter is saved, followed by validation accuracy, macro F1, balanced accuracy,
per-class results and a confusion matrix in `results/deep4ge`. Development runs
use validation by default; the final test must be requested explicitly.

An existing adapter is protected from accidental replacement. Use a new
`--adapter-dir` or explicitly `--overwrite` when starting another experiment.
The new adapter starts from the base model, not from the old synthetic adapter.

For a shorter first experiment, optional before/after comparison, and evaluation:

```powershell
python train.py --epochs 1 --max-train-examples 512 --max-eval-examples 64
python train.py --with-baseline --adapter-dir models/modeldoctor_deep4ge_comparison
python train.py --evaluate-only
python train.py --evaluate-only --evaluation-split test
```

Example limits are recorded in the training report; limited evaluation is not
the full benchmark. Each limited subset includes every category. `--baseline-only`
runs only the untrained model on the chosen evaluation split. Compare against
the CPU baseline using the complete validation split, without `--max-eval-examples`.
Test results should not guide checkpoint selection; further development after
inspecting them requires a fresh final benchmark for an unbiased final estimate.

## Save and resume training

Training saves `latest.pt` every 200 optimizer steps and after each epoch to
`models/modeldoctor_deep4ge_checkpoints/`. `--checkpoint-every` changes the interval.
Each checkpoint includes adapter weights, optimizer/scaler, random states,
sampler position, loss history and the best validation adapter. Writes replace
the previous file only after the new file is complete.

```powershell
python train.py --checkpoint-every 50
python train.py --resume-from models/modeldoctor_deep4ge_checkpoints
```

Use the same model, data, subset limits, seed, batch size, accumulation, learning
rate, tokenizer and runtime settings when resuming; mismatches are rejected.
Repeat any custom arguments from the original run. `--epochs` can be increased.
Ctrl+C resumes from the last completed checkpoint, so steps since that save
will be repeated. For a planned short run, stop at a saved optimizer boundary:

```powershell
python train.py --stop-after-steps 20
python train.py --resume-from models/modeldoctor_deep4ge_checkpoints
```

The CPU tests compare uninterrupted training with recovery midway through an
epoch, immediately before validation, and after a completed epoch. They recover
identical adapter tensors and loss history with LoRA dropout, weighted sampling
and a partial final accumulation window. CUDA recovery still needs verification
on the target GPU.

## CPU baseline, token lengths and data review

```powershell
python -m pip install -r requirements-analysis.txt
python check_token_lengths.py
python cpu_baseline.py
python review_training_data.py
```

The token check downloads only the selected model's tokenizer and checks every
training/validation/test input, answer masking and generation budget. It does
not score the final test. The report is
[token_length_report.json](results/token_lengths/token_length_report.json).
Rerun it if inputs, model or context length change.

The fixed Random Forest uses numerical summaries from the same observed inputs.
Its vectorizer and missing-value imputer are fitted on training only, with no
label or source identity in the features. It evaluates the 1,827 validation
examples from eight separate programs; the final test is not evaluated.

| Validation baseline | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Always predict the training majority category | 34.87% | 0.0646 | 12.50% |
| Random Forest | 34.32% | 0.2638 | 38.94% |

The forest misses all `layer_fault` and `activation_fault` examples and nearly
all baseline examples in validation. Its higher macro F1 does not establish
reliable diagnosis. Use per-category results to assess future Llama runs. These
numbers describe the CPU baseline, not Llama. Full results and the confusion
plot are in [results/deep4ge_cpu/SUMMARY.md](results/deep4ge_cpu/SUMMARY.md).
The saved CPU model is `models/modeldoctor_cpu_baseline.joblib`.

The visual review selects three training logs per category, prioritizing distinct
source programs, and verifies each source checksum and prepared input. It plots
the original epoch curves for loss, source metric, gradients and learning rate.
See [overview.png](results/data_review/overview.png),
[training_examples.pdf](results/data_review/training_examples.pdf) and
[REVIEW.md](results/data_review/REVIEW.md). The audit records missing and flat
curves; targets are kept as published mutation categories. Specific observations
from the plotted examples are in [OBSERVATIONS.md](results/data_review/OBSERVATIONS.md).

## Offline pipeline checks

CPU preparation and testing need no GPU or model downloads:

```powershell
python -m pip install -r requirements.txt
python train.py --check-data
python -m unittest discover -s tests -v
python train.py --smoke-test
```

The smoke test trains a tiny randomly initialized model on actual rows, checks
answer masking and optimized loss, generates answers, saves/reloads its LoRA
adapter, and verifies reload consistency. It discards the temporary adapter and
saves only a check report under `results/deep4ge_smoke`. It does not train Llama
or establish classification quality. Full CPU training is available explicitly
through `python train.py --cpu`, but is slow and uses unquantized weights.

## Rebuild data or classify a new log

Data preparation uses only Python's standard library:

```powershell
python prepare_fault_data.py --download
```

This reuses the local pinned archive when present. The preparation report records
archive/CSV hashes, exclusions, classes and split groups. `train.py` verifies the
checksums before training. If you change the data preparation options, train a
new adapter for the resulting dataset.

After training, classify a CSV with the observed columns documented in
`data/deep4ge/source/docs_DATA_DICTIONARY.md`:

```powershell
python diagnose_fault.py path/to/training_log.csv --architecture CNN
```

This command applies the saved adapter and the same input transformation. The
log filename is not passed to the model. Deep4ge logs remain available in the
source ZIP; it is not necessary to extract or run its training code.

## Scope of the answer key

The labels identify the experiment's baseline/injected mutation category. They
are independent of ModelDoctor, but do not prove a unique cause can be inferred
from the log or that every mutation harmed performance. This is a controlled
mutation benchmark. Real-world diagnosis, exact learning-rate directions,
data leakage, explanations and validated fixes require additional supervision.

## Previous experiments

The files directly under `data/`, such as `modeldoctor_all.csv` and
`modeldoctor_validation.csv`, belong to the earlier synthetic experiments.
The current trainer reads the labeled files under `data/deep4ge/`.

The old trainer is preserved as `train_synthetic.py`. `evaluate_model.py`
evaluates the old v3 adapter. Historical results remain in `results/`,
`results/v2/` and `results/v3/`; they do not measure the Deep4ge classifier.
