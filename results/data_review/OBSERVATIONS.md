# Notes from the visual review

These observations refer to the default seed-42 training selection in
[reviewed_examples.csv](reviewed_examples.csv), with source hashes recorded there.
Published targets were not changed.

- In [deep4ge_06106](deep4ge_06106.png), training loss falls while validation
  loss rises. The recorded training metric improves while the validation metric
  remains around its initial level. This is consistent with poor generalization;
  the published `hyperparameter_fault` category does not specify a unique cause
  or learning-rate correction.
- In [deep4ge_08924](deep4ge_08924.png), both losses fall and flatten despite the
  `layer_fault` target. A decreasing loss curve cannot establish that a program
  has no injected mutation. Its recorded metrics are negative, so treating all
  `train_acc`/`val_acc` values as classification percentages would be incorrect.
- In [overview.png](overview.png), losses span very different scales, including
  roughly 100 million for one loss mutation. Input preparation preserves these
  source measurements; common absolute thresholds would need task-specific
  supervision.

The full training audit finds 818 examples with completely missing sampled
training metrics and 826 with completely missing validation metrics. Those
measurements remain missing. The CPU baseline uses a training-fitted imputer;
the language-model prompts retain explicit `null` values.

For the next CUDA run, compare macro F1 and per-category recall on the complete
validation split. The current CPU baseline misses both layer and activation
categories there. Use the final test once the experiment settings are fixed.
