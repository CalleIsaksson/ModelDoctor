# CPU baseline: validation results

Fixed Random Forest trained on the training split only.
The final test split was not evaluated.

| Model | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Majority category | 34.87% | 0.0646 | 12.50% |
| Random Forest | 34.32% | 0.2638 | 38.94% |

| Published category | Validation examples | Recall | F1 |
| --- | ---: | ---: | ---: |
| no_injected_fault | 637 | 0.94% | 0.0185 |
| hyperparameter_fault | 224 | 5.80% | 0.0515 |
| layer_fault | 193 | 0.00% | 0.0000 |
| loss_fault | 306 | 95.75% | 0.6720 |
| activation_fault | 24 | 0.00% | 0.0000 |
| optimization_fault | 180 | 50.00% | 0.2691 |
| weight_fault | 225 | 88.00% | 0.9231 |
| regularization_fault | 38 | 71.05% | 0.1765 |

These scores are a reference for future Llama validation results, not Llama performance.
Both models must use the same held-out programs for a meaningful comparison.
Published mutation labels may be ambiguous from observations alone.
