# Training data review

Plotted 24 original logs, covering all eight published categories.
Each selected log reproduces its prepared input and matches its source checksum.
Only training examples were reviewed. No validation tuning or final test scoring.

Open [overview.png](overview.png) for loss curves or [training_examples.pdf](training_examples.pdf) for all four panels per example.
Each PNG also shows source metrics, gradients and learning rate. Missing data appear as gaps.

| Curve | Examples with missing samples | Fully missing | Flat sampled curve |
| --- | ---: | ---: | ---: |
| train_loss | 122 | 84 | 96 |
| val_loss | 122 | 82 | 450 |
| train_acc | 848 | 818 | 1476 |
| val_acc | 848 | 826 | 3001 |

The source metric can be accuracy or a regression proxy. Scales and losses differ between programs.
The displayed targets name published mutations; a curve alone cannot verify a unique causal diagnosis.
Flat curves and missing signals remain in the benchmark rather than being assigned invented targets.
Keep the original category labels. Check per-category recall when comparing models, especially rare categories.
