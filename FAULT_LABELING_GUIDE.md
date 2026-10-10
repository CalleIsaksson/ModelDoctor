# ModelDoctor: published fault-category targets

The current task uses Deep4ge's published baseline and mutation-category labels.
These labels come from the experiment setup, not from ModelDoctor's predictions.

| Source annotation | Training target | Meaning |
| --- | --- | --- |
| `is_faulty=False` | `no_injected_fault` | Baseline without an injected mutation; performance may still be poor |
| `Hyperparameter` | `hyperparameter_fault` | Batch size, learning rate, epoch count, or batching |
| `Layer` | `layer_fault` | Architecture or layer configuration |
| `Loss` | `loss_fault` | Loss function selection |
| `Activation` | `activation_fault` | Activation functions |
| `Optimization` | `optimization_fault` | Optimizer or gradient clipping |
| `Weight` | `weight_fault` | Weight initialization or bias |
| `Regularization` | `regularization_fault` | Dropout or regularizer |

The assistant target is exactly `Diagnosis: <training target>`. The dataset does
not provide reviewed prose explanations or recommended repairs. Those fields
are not generated or supervised in this task.

The website keeps these raw categories in **technical details**, rather than
presenting hyperparameters or a loss function as the customer's diagnosis. The
main result describes measured patterns, and a separate checklist supplies
human-written checks and ways to verify a change. Choosing a task or area only
changes that guide; it does not supply a new model diagnosis or modify these
training targets. An inspected checkbox is not evidence of a successful repair.

The first demo's three scenarios are illustrations built only for the UI. They
have no published category or model prediction and never enter training data.

`Hyperparameter` does not identify whether the learning rate was too high or too
low. None of these labels can be translated directly into the old v3 classes
`overfitting`, `underfitting`, or `data_leakage_suspected`. A baseline is not proof
that a model generalizes well. The old `LABELING_GUIDE.md` belongs to the separate
synthetic experiment.

Inputs contain architecture family, sampled observed loss/metric curves and
summaries of internal gradient, activation, weight and learning-rate signals.
They exclude filenames, source IDs, fault labels and subset names. Hardware
utilization fields are excluded because they can identify the collection
environment. The source metric named accuracy can instead represent
`1 - MAPE/100` on regression runs; it is not always classification accuracy.

The import excludes logs with repeated or decreasing epoch numbers: several
released files appear to interleave multiple trajectories, and they cannot be
treated as one unambiguous training curve. Every exclusion is listed in
`data/deep4ge/excluded_logs.csv`. Nonfinite observations are represented as
missing values, rather than being changed to zero.

Published mutation labels do not guarantee that the fault is uniquely inferable
from its log, or that a mutation caused the observed performance. Validate
diagnosis quality on held-out programs. Real-world claims require an independent
expert-reviewed benchmark. Explanations and proposed fixes would need their own
supervision and evaluation.

Sources: [Deep4ge](https://github.com/SigmaJahan/deep4ge),
[data provenance](https://github.com/SigmaJahan/deep4ge/blob/main/docs/PROVENANCE.md),
[CSV dictionary](https://github.com/SigmaJahan/deep4ge/blob/main/docs/DATA_DICTIONARY.md).
