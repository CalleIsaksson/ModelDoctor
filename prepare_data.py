import csv
import json
import random
from pathlib import Path


# Build the same dataset every time
RANDOM_SEED = 42
DATASET_VERSION = "v3"
EXAMPLES_PER_LABEL = 50
TRAIN_PER_LABEL = 36
VALIDATION_PER_LABEL = 7
TEST_PER_LABEL = 7

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SEED_FILE = DATA_DIR / "modeldoctor_v0.csv"

LABELS = [
    "healthy",
    "overfitting",
    "underfitting",
    "learning_rate_too_high",
    "learning_rate_too_low",
    "data_leakage_suspected",
    "insufficient_evidence",
]

BASE_COLUMNS = [
    "model_name",
    "task",
    "train_loss",
    "validation_loss",
    "train_metric",
    "validation_metric",
    "learning_rate",
    "epochs",
    "additional_information",
    "diagnosis",
    "evidence",
    "next_experiment",
]

INPUT_COLUMNS = [
    "model_name",
    "task",
    "train_loss",
    "validation_loss",
    "train_metric",
    "validation_metric",
    "learning_rate",
    "epochs",
    "additional_information",
]

OUTPUT_COLUMNS = ["example_id", "source", "split", *BASE_COLUMNS]

EXPERIMENT_TYPES = [
    ("controlled_mlp", "binary_classification"),
    ("controlled_mlp", "multiclass_classification"),
    ("controlled_cnn", "image_classification"),
    ("controlled_tabular_net", "binary_classification"),
    ("controlled_encoder", "text_classification"),
    ("controlled_sequence_net", "sequence_classification"),
]

# These sentences are shared by several labels. They describe the run but do
# not reveal the answer, so the model still has to read the metric curves.
NEUTRAL_CONTEXTS = [
    "The run completed without hardware or logging errors.",
    "Batch size and optimizer settings stayed fixed during the run.",
    "Metrics were recorded once at the end of every epoch.",
    "The same architecture was used for the complete experiment.",
    "No learning-rate scheduler was enabled.",
    "The optimizer state was reset before training started.",
    "Gradient clipping was enabled with the same limit in every epoch.",
    "Data augmentation settings stayed unchanged during training.",
    "The experiment used a fixed random seed.",
    "No checkpoints were removed from the recorded history.",
]

RELEVANT_CONTEXTS = {
    "healthy": [
        "The gap between the two splits stayed small throughout the run.",
        "A second seed produced a similar smooth improvement on both splits.",
        "Validation had one small wobble before returning to its improving trend.",
        "The final checkpoint kept strong results on both training and validation.",
        "A split audit found no overlap and both curves converged normally.",
    ],
    "overfitting": [
        "The best validation checkpoint occurred before the final epoch.",
        "Training kept improving after validation had started getting worse.",
        "The train-validation gap became larger near the end of the run.",
        "No early stopping was used after the validation score peaked.",
        "A second seed showed the same late divergence between the two splits.",
    ],
    "underfitting": [
        "Both splits reached a similarly weak plateau near the end.",
        "The model still performed poorly on its own training examples.",
        "Adding a few extra epochs caused almost no change after the plateau.",
        "Training and validation remained close but both results were weak.",
        "Optimization was stable, yet the small architecture could not fit the training data.",
    ],
    "learning_rate_too_high": [
        "Gradient norms repeatedly spiked after optimizer updates.",
        "The same optimizer converged when a much smaller rate was used in a pilot run.",
        "Large loss jumps continued even though the batch size stayed fixed.",
        "Neither split settled into a downward trend during the run.",
        "Gradient clipping did not remove the recurring oscillations.",
    ],
    "learning_rate_too_low": [
        "Parameter updates were very small and progress remained slow but stable.",
        "A short pilot with a larger rate made more progress in the same number of epochs.",
        "The curves moved in the correct direction at an impractically slow pace.",
        "The available epoch budget ended before the model reached a useful result.",
        "There were no spikes, but each epoch changed the metrics only slightly.",
    ],
}


def choose(options, index):
    return options[index % len(options)]


def neutral_context(index):
    return choose(NEUTRAL_CONTEXTS, index)


def mixed_context(label, index):
    # Even examples reuse the exact same neutral text across different labels.
    # Odd examples use varied information that is relevant to their curves.
    if index % 2 == 0:
        return neutral_context(index // 2)

    return choose(RELEVANT_CONTEXTS[label], index // 2)


def rounded(values):
    return [round(value, 3) for value in values]


def smooth_curve(start, end, length, power=1.4):
    values = []

    for index in range(length):
        fraction = index / (length - 1)
        progress = 1 - (1 - fraction) ** power
        values.append(start + (end - start) * progress)

    return rounded(values)


def as_text(values):
    return json.dumps(rounded(values))


def base_row(label, index):
    model_name, task = EXPERIMENT_TYPES[index % len(EXPERIMENT_TYPES)]

    return {
        "example_id": f"synthetic_{DATASET_VERSION}_{label}_{index:03d}",
        "source": f"controlled_synthetic_{DATASET_VERSION}",
        "split": "",
        "model_name": model_name,
        "task": task,
        "diagnosis": label,
    }


def make_healthy(index, rng):
    row = base_row("healthy", index)
    length = rng.randint(8, 12)
    epochs = list(range(1, length + 1))

    train_loss = smooth_curve(rng.uniform(1.2, 1.8), rng.uniform(0.12, 0.28), length)
    validation_loss = smooth_curve(
        train_loss[0] + rng.uniform(0.03, 0.12),
        train_loss[-1] + rng.uniform(0.03, 0.09),
        length,
    )
    train_metric = smooth_curve(rng.uniform(0.35, 0.55), rng.uniform(0.90, 0.97), length)
    validation_metric = smooth_curve(
        train_metric[0] - rng.uniform(0.0, 0.03),
        train_metric[-1] - rng.uniform(0.01, 0.05),
        length,
    )

    # Some healthy runs contain one small wobble but keep improving overall.
    if index % 3 == 0:
        middle = length // 2
        validation_loss[middle] = round(validation_loss[middle - 1] + 0.015, 3)
        validation_metric[middle] = round(validation_metric[middle - 1] - 0.005, 3)

    evidence_options = [
        (
            f"Across {length} epochs, training loss decreased from {train_loss[0]:.3f} "
            f"to {train_loss[-1]:.3f} and validation loss decreased from "
            f"{validation_loss[0]:.3f} to {validation_loss[-1]:.3f}. The final "
            f"training and validation metrics remained close at {train_metric[-1]:.3f} "
            f"and {validation_metric[-1]:.3f}."
        ),
        (
            f"Both splits improved over {length} epochs. Final losses were "
            f"{train_loss[-1]:.3f} for training and {validation_loss[-1]:.3f} for "
            f"validation, with final metrics of {train_metric[-1]:.3f} and "
            f"{validation_metric[-1]:.3f}."
        ),
        (
            f"Training and validation followed the same improving trend: losses ended at "
            f"{train_loss[-1]:.3f} and {validation_loss[-1]:.3f}, while their metrics "
            f"reached {train_metric[-1]:.3f} and {validation_metric[-1]:.3f}."
        ),
    ]

    next_options = [
        "Repeat the run with another random seed to check that the result is reproducible.",
        "Run the same configuration with a second seed and compare the final metrics.",
        "Repeat the experiment once more to confirm that the stable result is reproducible.",
    ]

    row.update(
        train_loss=as_text(train_loss),
        validation_loss=as_text(validation_loss),
        train_metric=as_text(train_metric),
        validation_metric=as_text(validation_metric),
        learning_rate=round(rng.uniform(0.0003, 0.003), 6),
        epochs=json.dumps(epochs),
        additional_information=mixed_context("healthy", index),
        evidence=choose(evidence_options, index),
        next_experiment=choose(next_options, index),
    )
    return row


def make_overfitting(index, rng):
    row = base_row("overfitting", index)
    length = rng.randint(8, 12)
    best_epoch = rng.randint(3, length - 3)
    epochs = list(range(1, length + 1))

    train_loss = smooth_curve(rng.uniform(1.1, 1.7), rng.uniform(0.05, 0.18), length)
    first_validation = smooth_curve(
        train_loss[0] + rng.uniform(0.03, 0.12),
        rng.uniform(0.30, 0.50),
        best_epoch,
    )
    last_validation = smooth_curve(
        first_validation[-1],
        rng.uniform(0.75, 1.25),
        length - best_epoch + 1,
        power=1.1,
    )[1:]
    validation_loss = first_validation + last_validation

    train_metric = smooth_curve(rng.uniform(0.40, 0.55), rng.uniform(0.95, 0.99), length)
    first_metric = smooth_curve(
        train_metric[0] - 0.01,
        rng.uniform(0.76, 0.88),
        best_epoch,
    )
    last_metric = smooth_curve(
        first_metric[-1],
        rng.uniform(0.55, 0.72),
        length - best_epoch + 1,
        power=1.1,
    )[1:]
    validation_metric = first_metric + last_metric

    evidence_options = [
        (
            f"Validation loss reached its lowest value of {validation_loss[best_epoch - 1]:.3f} "
            f"at epoch {best_epoch}, then increased to {validation_loss[-1]:.3f}, while "
            f"training loss continued decreasing to {train_loss[-1]:.3f}."
        ),
        (
            f"After epoch {best_epoch}, training continued improving to "
            f"{train_loss[-1]:.3f} loss, but validation loss moved in the opposite "
            f"direction and finished at {validation_loss[-1]:.3f}."
        ),
        (
            f"The generalization gap grew after epoch {best_epoch}: the final training "
            f"metric was {train_metric[-1]:.3f}, while the validation metric fell to "
            f"{validation_metric[-1]:.3f}."
        ),
        (
            f"Validation performance was best at epoch {best_epoch} and became worse "
            f"afterward, even though training loss kept falling to {train_loss[-1]:.3f}."
        ),
    ]

    next_options = [
        f"Restore the checkpoint from epoch {best_epoch} and test early stopping.",
        f"Use epoch {best_epoch} as the early-stopping point and compare it with the final checkpoint.",
        f"Stop training near epoch {best_epoch} and evaluate that checkpoint on a separate holdout.",
    ]

    row.update(
        train_loss=as_text(train_loss),
        validation_loss=as_text(validation_loss),
        train_metric=as_text(train_metric),
        validation_metric=as_text(validation_metric),
        learning_rate=round(rng.uniform(0.0003, 0.003), 6),
        epochs=json.dumps(epochs),
        additional_information=mixed_context("overfitting", index),
        evidence=choose(evidence_options, index),
        next_experiment=choose(next_options, index),
    )
    return row


def make_underfitting(index, rng):
    row = base_row("underfitting", index)
    length = rng.randint(8, 12)
    epochs = list(range(1, length + 1))

    train_end = rng.uniform(1.25, 1.65)
    train_loss = smooth_curve(rng.uniform(1.8, 2.3), train_end, length, power=2.4)
    validation_loss = smooth_curve(
        train_loss[0] + rng.uniform(0.02, 0.10),
        train_loss[-1] + rng.uniform(0.03, 0.10),
        length,
        power=2.4,
    )
    train_metric = smooth_curve(rng.uniform(0.25, 0.38), rng.uniform(0.43, 0.55), length, power=2.4)
    validation_metric = smooth_curve(
        train_metric[0] - rng.uniform(0.0, 0.02),
        train_metric[-1] - rng.uniform(0.01, 0.04),
        length,
        power=2.4,
    )

    evidence_options = [
        (
            f"Training remained poor after {length} epochs: training loss was "
            f"{train_loss[-1]:.3f} and training metric was {train_metric[-1]:.3f}. "
            f"Validation values were similarly poor at {validation_loss[-1]:.3f} "
            f"loss and {validation_metric[-1]:.3f} metric."
        ),
        (
            f"Both splits reached a weak plateau. Training and validation losses ended "
            f"close together at {train_loss[-1]:.3f} and {validation_loss[-1]:.3f}, "
            f"and neither metric exceeded {max(train_metric[-1], validation_metric[-1]):.3f}."
        ),
        (
            f"The model still performed poorly on its training data after {length} epochs, "
            f"with a {train_metric[-1]:.3f} training metric. Validation was similarly "
            f"weak at {validation_metric[-1]:.3f}."
        ),
    ]

    next_options = [
        "Train for more epochs and, if training remains poor, test a larger model.",
        "Increase the training budget first, then try a model with more capacity if the plateau remains.",
        "Continue training and compare the result with a larger architecture using the same split.",
    ]

    row.update(
        train_loss=as_text(train_loss),
        validation_loss=as_text(validation_loss),
        train_metric=as_text(train_metric),
        validation_metric=as_text(validation_metric),
        learning_rate=round(rng.uniform(0.0005, 0.003), 6),
        epochs=json.dumps(epochs),
        additional_information=mixed_context("underfitting", index),
        evidence=choose(evidence_options, index),
        next_experiment=choose(next_options, index),
    )
    return row


def make_learning_rate_too_high(index, rng):
    row = base_row("learning_rate_too_high", index)
    length = rng.randint(8, 12)
    epochs = list(range(1, length + 1))

    train_loss = []
    for epoch_index in range(length):
        if epoch_index % 2 == 0:
            train_loss.append(rng.uniform(0.75, 1.45))
        else:
            train_loss.append(rng.uniform(1.90, 3.40))

    train_loss = rounded(train_loss)
    validation_loss = rounded([value + rng.uniform(0.08, 0.20) for value in train_loss])
    train_metric = rounded([
        rng.uniform(0.45, 0.60) if epoch_index % 2 == 0 else rng.uniform(0.20, 0.38)
        for epoch_index in range(length)
    ])
    validation_metric = rounded([value - rng.uniform(0.01, 0.04) for value in train_metric])

    evidence_options = [
        (
            f"Training loss repeatedly jumped from {train_loss[0]:.3f} at epoch 1 "
            f"to {train_loss[1]:.3f} at epoch 2 and from {train_loss[2]:.3f} "
            f"at epoch 3 to {train_loss[3]:.3f} at epoch 4, with no stable downward trend."
        ),
        (
            f"Loss oscillated throughout {length} epochs, including increases from "
            f"{train_loss[0]:.3f} to {train_loss[1]:.3f} and from "
            f"{train_loss[2]:.3f} to {train_loss[3]:.3f}; the run did not converge."
        ),
        (
            f"The large recurring loss spikes were paired with metric drops, such as "
            f"{train_metric[0]:.3f} to {train_metric[1]:.3f} between epochs 1 and 2."
        ),
    ]

    next_options = [
        "Lower the learning rate by about 10x and run the same experiment again.",
        "Repeat the configuration with a learning rate ten times smaller.",
        "Reduce the learning rate first and check whether the loss begins to converge smoothly.",
    ]

    row.update(
        train_loss=as_text(train_loss),
        validation_loss=as_text(validation_loss),
        train_metric=as_text(train_metric),
        validation_metric=as_text(validation_metric),
        learning_rate=round(rng.uniform(0.05, 0.30), 4),
        epochs=json.dumps(epochs),
        additional_information=mixed_context("learning_rate_too_high", index),
        evidence=choose(evidence_options, index),
        next_experiment=choose(next_options, index),
    )
    return row


def make_learning_rate_too_low(index, rng):
    row = base_row("learning_rate_too_low", index)
    length = rng.randint(12, 16)
    epochs = list(range(1, length + 1))

    train_start = rng.uniform(1.4, 1.9)
    train_end = train_start - rng.uniform(0.05, 0.14)
    train_loss = smooth_curve(train_start, train_end, length, power=1.0)
    validation_loss = smooth_curve(train_start + 0.06, train_end + 0.07, length, power=1.0)
    metric_start = rng.uniform(0.30, 0.42)
    train_metric = smooth_curve(metric_start, metric_start + rng.uniform(0.03, 0.08), length, power=1.0)
    validation_metric = smooth_curve(
        metric_start - 0.01,
        train_metric[-1] - rng.uniform(0.01, 0.03),
        length,
        power=1.0,
    )

    loss_change = train_loss[0] - train_loss[-1]
    metric_change = train_metric[-1] - train_metric[0]
    evidence_options = [
        (
            f"Training loss decreased consistently by only {loss_change:.3f}, from "
            f"{train_loss[0]:.3f} to {train_loss[-1]:.3f}, across {length} epochs, "
            f"while the training metric improved by only {metric_change:.3f}."
        ),
        (
            f"The curves moved in the right direction but very slowly: over {length} "
            f"epochs training loss changed by just {loss_change:.3f} and the metric "
            f"changed by {metric_change:.3f}."
        ),
        (
            f"Despite {length} stable epochs, training loss only moved from "
            f"{train_loss[0]:.3f} to {train_loss[-1]:.3f}, leaving performance poor "
            "within the available budget."
        ),
    ]

    next_options = [
        "Increase the learning rate by about 10x and run the same experiment again.",
        "Try a learning rate ten times larger while keeping the other settings fixed.",
        "Raise the learning rate and compare how much progress is made in the same epoch budget.",
    ]

    row.update(
        train_loss=as_text(train_loss),
        validation_loss=as_text(validation_loss),
        train_metric=as_text(train_metric),
        validation_metric=as_text(validation_metric),
        learning_rate=round(rng.uniform(0.000001, 0.00001), 7),
        epochs=json.dumps(epochs),
        additional_information=mixed_context("learning_rate_too_low", index),
        evidence=choose(evidence_options, index),
        next_experiment=choose(next_options, index),
    )
    return row


def make_data_leakage(index, rng):
    row = base_row("data_leakage_suspected", index)
    length = 6
    epochs = list(range(1, length + 1))
    train_loss = smooth_curve(0.42, rng.uniform(0.02, 0.06), length)
    validation_loss = smooth_curve(0.38, rng.uniform(0.01, 0.04), length)
    train_metric = smooth_curve(0.84, rng.uniform(0.98, 0.995), length)
    validation_metric = smooth_curve(0.88, rng.uniform(0.99, 0.999), length)

    scenario = index % 6
    if scenario == 0:
        duplicate_count = rng.randint(80, 420)
        additional_information = (
            f"A split audit found {duplicate_count} identical samples in both the training "
            "and validation sets. Accuracy on a deduplicated holdout was 0.68."
        )
        evidence = (
            f"The split audit found {duplicate_count} exact duplicates shared by the training "
            f"and validation sets. Validation metric was {validation_metric[-1]:.3f}, but the "
            "deduplicated holdout metric was 0.680."
        )
    elif scenario == 1:
        additional_information = (
            "The scaler and feature selector were fitted on the complete dataset before the "
            "train-validation split. Metric on a clean holdout was 0.64."
        )
        evidence = (
            f"Preprocessing was fitted before the split, so validation information entered the "
            f"training pipeline. Validation metric was {validation_metric[-1]:.3f}, compared "
            "with 0.640 on a clean holdout."
        )
    elif scenario == 2:
        additional_information = (
            "A feature audit found that the input column named target_copy exactly matched the "
            "target for every training and validation sample. Clean holdout metric was 0.61."
        )
        evidence = (
            f"The target_copy input feature directly revealed the target in both splits. "
            f"Validation metric was {validation_metric[-1]:.3f}, but it fell to 0.610 after "
            "removing that feature on a clean holdout."
        )
    elif scenario == 3:
        additional_information = (
            "A group audit found that records from the same patients appeared in both "
            "training and validation instead of being split by patient ID."
        )
        evidence = (
            "The same patients occurred in both splits, so related records could leak "
            "patient-specific information into validation."
        )
    elif scenario == 4:
        duplicate_count = rng.randint(50, 250)
        additional_information = (
            f"Augmentation was applied before splitting and created {duplicate_count} "
            "near-identical image pairs across training and validation."
        )
        evidence = (
            f"The audit found {duplicate_count} near-duplicate augmented images shared "
            "between the training and validation splits."
        )
    else:
        additional_information = (
            "Source documents were divided by individual rows; paragraphs from the same "
            "documents occurred in both splits. Performance dropped to 0.66 when split by document."
        )
        evidence = (
            f"Paragraphs from the same source documents crossed the split boundary. The "
            f"validation metric was {validation_metric[-1]:.3f}, but a document-level "
            "holdout reached only 0.660."
        )

    next_options = [
        (
            "Create a new clean split before preprocessing, remove the leakage source, "
            "and evaluate the same configuration on an untouched holdout set."
        ),
        "Rebuild the splits around independent groups, then rerun preprocessing and training.",
        "Remove the overlap, retrain from scratch, and verify the result on a clean holdout.",
    ]

    row.update(
        train_loss=as_text(train_loss),
        validation_loss=as_text(validation_loss),
        train_metric=as_text(train_metric),
        validation_metric=as_text(validation_metric),
        learning_rate=0.001,
        epochs=json.dumps(epochs),
        additional_information=additional_information,
        evidence=evidence,
        next_experiment=choose(next_options, index),
    )
    return row


def make_insufficient_evidence(index, rng):
    row = base_row("insufficient_evidence", index)
    scenario = index % 5

    if scenario == 0:
        epochs = [1]
        train_loss = [round(rng.uniform(0.8, 1.8), 3)]
        validation_loss = [round(train_loss[0] + rng.uniform(-0.05, 0.15), 3)]
        train_metric = [round(rng.uniform(0.4, 0.7), 3)]
        validation_metric = [round(train_metric[0] + rng.uniform(-0.05, 0.05), 3)]
        additional_information = "Only one epoch was recorded."
        evidence = "Only one epoch is available, so no training or validation trend can be established."
        next_experiment = "Continue training for more epochs and record both training and validation metrics."
    elif scenario == 1:
        epochs = list(range(1, 7))
        train_loss = smooth_curve(
            rng.uniform(1.2, 1.7),
            rng.uniform(0.40, 0.65),
            len(epochs),
        )
        validation_loss = []
        train_metric = smooth_curve(
            rng.uniform(0.35, 0.48),
            rng.uniform(0.75, 0.86),
            len(epochs),
        )
        validation_metric = []
        additional_information = "Validation loss and validation metric were not recorded."
        evidence = (
            "Validation loss and validation metric are missing, so generalization cannot be "
            "compared with training performance."
        )
        next_experiment = "Run the experiment again with validation loss and validation metric logging enabled."
    elif scenario == 2:
        epochs = [1, 2]
        train_start = rng.uniform(0.9, 1.3)
        train_loss = rounded([train_start, train_start - rng.uniform(0.20, 0.38)])
        validation_loss = rounded(
            [train_loss[0] + rng.uniform(0.02, 0.08), train_loss[1] + rng.uniform(0.02, 0.08)]
        )
        train_metric = []
        validation_metric = []
        additional_information = "Only two epochs and no accuracy metrics were recorded."
        evidence = (
            "Only two epochs are available and both training and validation metrics are missing, "
            "so the early loss decrease does not support a specific diagnosis."
        )
        next_experiment = "Collect more epochs and record training and validation metrics before diagnosing the run again."
    elif scenario == 3:
        epochs = [1, 2, 3, 4, 5]
        loss_scale = rng.uniform(0.90, 1.10)
        train_loss = rounded([value * loss_scale for value in [1.20, 0.98, 0.81, 0.69, 0.60]])
        validation_loss = rounded([value + rng.uniform(0.03, 0.07) for value in train_loss])
        metric_start = rng.uniform(0.58, 0.68)
        train_metric = rounded([metric_start - 0.04 * step for step in range(len(epochs))])
        validation_metric = rounded([value - rng.uniform(0.01, 0.03) for value in train_metric])
        additional_information = "Loss improved while the recorded accuracy moved in the opposite direction."
        evidence = (
            "The losses improve but both recorded metrics become worse, so the logs are "
            "contradictory and may use an unknown metric direction or contain a logging error."
        )
        next_experiment = "Verify the metric definition and logging code, then repeat the run before diagnosing it."
    else:
        epochs = [1, 2, 3, 4, 5, 6]
        train_loss = smooth_curve(rng.uniform(1.30, 1.60), rng.uniform(0.35, 0.50), len(epochs))
        validation_loss = smooth_curve(
            train_loss[0] + rng.uniform(0.03, 0.08),
            train_loss[-1] + rng.uniform(0.04, 0.09),
            len(epochs),
        )
        train_metric = smooth_curve(rng.uniform(0.34, 0.43), rng.uniform(0.78, 0.86), len(epochs))
        validation_metric = []
        additional_information = "Validation loss exists, but the validation metric and metric definition are missing."
        evidence = (
            "The validation metric and its definition were not recorded, so the available "
            "information is incomplete and does not support a reliable diagnosis."
        )
        next_experiment = "Record the validation metric and its definition, then rerun the evaluation."

    # Half of these examples use the same neutral text as the other labels.
    # The missing or contradictory values must therefore reveal the diagnosis.
    if index % 2 == 0:
        additional_information = neutral_context(index // 2)

    row.update(
        train_loss=as_text(train_loss),
        validation_loss=as_text(validation_loss),
        train_metric=as_text(train_metric),
        validation_metric=as_text(validation_metric),
        learning_rate=0.001,
        epochs=json.dumps(epochs),
        additional_information=additional_information,
        evidence=evidence,
        next_experiment=next_experiment,
    )
    return row


GENERATORS = {
    "healthy": make_healthy,
    "overfitting": make_overfitting,
    "underfitting": make_underfitting,
    "learning_rate_too_high": make_learning_rate_too_high,
    "learning_rate_too_low": make_learning_rate_too_low,
    "data_leakage_suspected": make_data_leakage,
    "insufficient_evidence": make_insufficient_evidence,
}


def load_seed_rows():
    with SEED_FILE.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for index, row in enumerate(rows):
        row["example_id"] = f"human_seed_{index:03d}"
        row["source"] = "human_seed"
        row["split"] = ""

    return rows


def build_rows():
    rng = random.Random(RANDOM_SEED)
    rows = load_seed_rows()

    for label in LABELS:
        current_count = sum(row["diagnosis"] == label for row in rows)
        missing_count = EXAMPLES_PER_LABEL - current_count

        if missing_count < 0:
            raise ValueError(f"There are more than {EXAMPLES_PER_LABEL} seed rows for {label}.")

        for index in range(missing_count):
            rows.append(GENERATORS[label](index, rng))

    return rows


def split_rows(rows):
    rng = random.Random(RANDOM_SEED)
    splits = {"train": [], "validation": [], "test": []}

    for label in LABELS:
        label_rows = [row for row in rows if row["diagnosis"] == label]
        seed_rows = [row for row in label_rows if row["source"] == "human_seed"]
        generated_rows = [row for row in label_rows if row["source"] != "human_seed"]
        rng.shuffle(generated_rows)

        train_needed = TRAIN_PER_LABEL - len(seed_rows)
        if train_needed < 0:
            raise ValueError(f"Too many human seed rows to keep all {label} examples in train.")

        train_rows = seed_rows + generated_rows[:train_needed]
        validation_start = train_needed
        validation_end = validation_start + VALIDATION_PER_LABEL
        validation_rows = generated_rows[validation_start:validation_end]
        test_rows = generated_rows[validation_end:validation_end + TEST_PER_LABEL]

        if not (
            len(train_rows) == TRAIN_PER_LABEL
            and len(validation_rows) == VALIDATION_PER_LABEL
            and len(test_rows) == TEST_PER_LABEL
        ):
            raise ValueError(f"Could not create the requested split for {label}.")

        for row in train_rows:
            row["split"] = "train"
        for row in validation_rows:
            row["split"] = "validation"
        for row in test_rows:
            row["split"] = "test"

        splits["train"].extend(train_rows)
        splits["validation"].extend(validation_rows)
        splits["test"].extend(test_rows)

    for split_rows_list in splits.values():
        rng.shuffle(split_rows_list)

    return splits


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_values(row, column):
    return json.loads(row[column])


def validate_labeling_rule(row):
    if row["source"] == "human_seed":
        return

    label = row["diagnosis"]
    train_loss = read_values(row, "train_loss")
    validation_loss = read_values(row, "validation_loss")
    train_metric = read_values(row, "train_metric")
    validation_metric = read_values(row, "validation_metric")

    if label == "healthy":
        valid = (
            train_loss and validation_loss and train_metric and validation_metric
            and train_loss[-1] < train_loss[0]
            and validation_loss[-1] < validation_loss[0]
            and train_metric[-1] > train_metric[0]
            and validation_metric[-1] > validation_metric[0]
            and abs(validation_loss[-1] - train_loss[-1]) < 0.15
        )
    elif label == "overfitting":
        best_index = validation_loss.index(min(validation_loss))
        valid = (
            best_index < len(validation_loss) - 1
            and train_loss[-1] < train_loss[best_index]
            and validation_loss[-1] > validation_loss[best_index] + 0.20
            and validation_metric[-1] < max(validation_metric)
        )
    elif label == "underfitting":
        valid = (
            train_loss[-1] > 1.20
            and validation_loss[-1] > 1.20
            and train_metric[-1] < 0.60
            and validation_metric[-1] < 0.60
            and abs(validation_loss[-1] - train_loss[-1]) < 0.15
        )
    elif label == "learning_rate_too_high":
        large_jumps = sum(
            current > previous + 0.30
            for previous, current in zip(train_loss, train_loss[1:])
        )
        valid = large_jumps >= 2
    elif label == "learning_rate_too_low":
        loss_change = train_loss[0] - train_loss[-1]
        metric_change = train_metric[-1] - train_metric[0]
        valid = (
            len(train_loss) >= 10
            and 0 < loss_change < 0.20
            and 0 < metric_change < 0.12
            and all(current <= previous for previous, current in zip(train_loss, train_loss[1:]))
        )
    elif label == "data_leakage_suspected":
        information = row["additional_information"].lower()
        leakage_terms = [
            "identical samples",
            "complete dataset",
            "target_copy",
            "same patients",
            "near-identical",
            "same documents",
        ]
        valid = any(term in information for term in leakage_terms)
    elif label == "insufficient_evidence":
        few_epochs = len(read_values(row, "epochs")) <= 2
        missing_values = not validation_loss or not validation_metric
        contradictory_logs = (
            len(train_loss) > 1
            and len(train_metric) > 1
            and train_loss[-1] < train_loss[0]
            and train_metric[-1] < train_metric[0]
        )
        valid = few_epochs or missing_values or contradictory_logs
    else:
        valid = False

    if not valid:
        raise ValueError(
            f"{row['example_id']} does not follow the labeling rule for {label}."
        )


def validate_splits(splits):
    all_rows = splits["train"] + splits["validation"] + splits["test"]
    example_ids = [row["example_id"] for row in all_rows]
    input_signatures = [tuple(row[column] for column in INPUT_COLUMNS) for row in all_rows]

    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate example IDs were found.")

    if len(input_signatures) != len(set(input_signatures)):
        raise ValueError("Duplicate model inputs were found.")

    for row in all_rows:
        validate_labeling_rule(row)
        epochs = json.loads(row["epochs"])

        for column in ["train_loss", "validation_loss", "train_metric", "validation_metric"]:
            values = json.loads(row[column])
            if values and len(values) != len(epochs):
                raise ValueError(f"{row['example_id']} has the wrong length in {column}.")

    for label in LABELS:
        label_train_rows = [
            row
            for row in splits["train"]
            if row["diagnosis"] == label
        ]
        unique_contexts = {
            row["additional_information"]
            for row in label_train_rows
        }

        if len(unique_contexts) < 10:
            raise ValueError(f"Not enough text variation for {label}.")

    context_labels = {}
    for row in splits["train"]:
        context = row["additional_information"]
        context_labels.setdefault(context, set()).add(row["diagnosis"])

    shared_contexts = [
        context
        for context, labels in context_labels.items()
        if len(labels) >= 5
    ]

    if len(shared_contexts) < 5:
        raise ValueError("Not enough contrast examples with shared wording were found.")


def main():
    rows = build_rows()
    splits = split_rows(rows)
    validate_splits(splits)

    all_rows = splits["train"] + splits["validation"] + splits["test"]
    write_csv(DATA_DIR / "modeldoctor_all.csv", all_rows)
    write_csv(DATA_DIR / "modeldoctor_train.csv", splits["train"])
    write_csv(DATA_DIR / "modeldoctor_validation.csv", splits["validation"])
    write_csv(DATA_DIR / "modeldoctor_test.csv", splits["test"])

    print(f"All examples: {len(all_rows)}")
    print(f"Train examples: {len(splits['train'])}")
    print(f"Validation examples: {len(splits['validation'])}")
    print(f"Test examples: {len(splits['test'])}")

    for label in LABELS:
        counts = {
            split_name: sum(row["diagnosis"] == label for row in split_rows_list)
            for split_name, split_rows_list in splits.items()
        }
        train_contexts = {
            row["additional_information"]
            for row in splits["train"]
            if row["diagnosis"] == label
        }
        print(f"{label}: {counts}, unique training contexts: {len(train_contexts)}")


if __name__ == "__main__":
    main()
