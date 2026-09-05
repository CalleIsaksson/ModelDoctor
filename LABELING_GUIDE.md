# ModelDoctor Labeling Guide

## overfitting

Use this label when:
- Training loss continues decreasing.
- Validation loss starts increasing.
- The gap grows over multiple epochs.

Do not use this label when:
- Only one epoch is available.
- Both training and validation results are poor.

Evidence should mention:
- The epoch where validation performance started getting worse.
- How training and validation moved in different directions.

Suggested next experiment:
- Test early stopping at the best validation epoch.

## underfitting

Use this label when:
- Training loss remains high.
- Training accuracy is low.
- Validation accuracy is low, often times similar accuracy scores as well.
- Validation error is high, and ususally close to training error.

Dont use this label when:
- Only one epoch is available
- If training and validation still are actively improving at the last epoch.

Evidence should mention:
- Training preformance remains bad.
- Train / validation have similar values.

Suggested next experiment:
- Train more epochs

## learning_rate_too_high

Use this label when:
- The loss value jumps up and down with large, reccuring values without an obvious downward trend instead of going down smoothly
- The accuracy drops and the model fails to converge.

Dont use this label when:
- Only one epoch is available
- The loss function, despite lower variations, decreases steadily.

Evidence should mention:
- The exact epoch and values when these reaccuring jumps happen.


Suggested next experiment:
- Lower the learning rate, for example 10 fold, and run the same experiment.

## learning_rate_too_low

Use this label when:
- The loss decreases consistently, but only by a very small amount over many epochs.
- The accuracy improves very slowly.
- The training does not reach a useful result within the available epoch budget.

Dont use this label when:
- Only one epoch is available.
- When the loss value converges quickly but at a low value.

Evidence should mention:
- How little the loss decreased and over how many epochs.

Suggested next experiment:
- Increase the learning rate, about 10x and run the same experiment.

## healthy

Use this label when:
- The loss function decreases over a reasonable amount of epochs and converges to a low value.
- The model has a high validation and training accuracy, while also having a low validation and training loss.

Dont use this label when:
- Any of the metrics are missing, which results in the diagnosis "insufficient_evidence".
- Data leakage is suspected or any of the metrics above are incorrect.

Evidence should mention:
- That both the training and validation accuracy improved, that the losses decreased and that the gap remained small.

Suggested next experiment:
- Try to recreate the same result with another seed to check if the result is accurate and reproducible.

## data_leakage_suspected

Use this label when:
- Duplicate or highly similar examples occur in both the training and validation splits.
- The same entities, users or source documents occur in both splits when they should be separated.
- Preprocessing was fitted on the complete dataset before creating the splits.
- An input feature contains or directly reveals information about the target.
- Validation performance is unexpectedly high but drops substantially on a clean, independently created holdout set.

Do not use this label when:
- Validation performance is high but no evidence of leakage is available.
- The task may simply be easy.
- Performance is worse on a dataset with a different distribution.
- Only training and validation metrics are available without information about how the data was split.

Evidence should mention:
- The exact overlap, duplicate count or preprocessing problem that was found.
- Which splits were affected.
- The difference between the original validation result and the clean holdout result.

Suggested next experiment:
- Create a new deduplicated split before preprocessing, train the same configuration again and evaluate it on an untouched holdout set.


## insufficient_evidence

Use this label when:
- Only one or very few epochs are available.
- Validation metrics or other important information are missing.
- The available metrics are contradictory or could support multiple diagnoses.
- There is not enough information to distinguish one suspected cause from another.

Dont use this label when:
- The available metrics show a clear pattern for one of the other diagnoses.
- Direct evidence supports a specific diagnosis.

Evidence should mention:
- Which information is missing or unclear.
- Why the available results are not enough for a more specific diagnosis.

Suggested next experiment:
- Collect the missing metrics or continue training for more epochs, then diagnose the run again.
