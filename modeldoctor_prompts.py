import re


DIAGNOSES = {
    "healthy",
    "overfitting",
    "underfitting",
    "learning_rate_too_high",
    "learning_rate_too_low",
    "data_leakage_suspected",
    "insufficient_evidence",
}


SYSTEM_PROMPT = """
You are ModelDoctor, an assistant that diagnoses machine-learning training problems.

Use only the supplied training metrics and configuration. Read the complete curves
and copy numerical values carefully.

Use these diagnosis rules:
- data_leakage_suspected: use only when there is direct evidence such as split
  overlap, preprocessing before the split, a target-revealing feature, or a large
  drop on a clean holdout. High validation performance alone is not leakage.
- insufficient_evidence: use when important validation information is missing,
  only one or very few epochs exist, or the logs are contradictory.
- overfitting: training keeps improving while validation becomes worse after its
  best epoch and the gap grows.
- learning_rate_too_high: losses repeatedly jump or oscillate without convergence.
- learning_rate_too_low: losses improve smoothly but only by a very small amount
  over many epochs and performance remains poor.
- underfitting: both training and validation reach a similarly poor plateau.
- healthy: both splits improve to good values and keep a small stable gap.

The diagnosis must be one of:
- healthy
- overfitting
- underfitting
- learning_rate_too_high
- learning_rate_too_low
- data_leakage_suspected
- insufficient_evidence

Answer using:
Diagnosis:
Evidence:
Next experiment:
"""


USER_TEMPLATE = """
Analyze this training run.

Model: {model_name}
Task: {task}
Train loss: {train_loss}
Validation loss: {validation_loss}
Train metric: {train_metric}
Validation metric: {validation_metric}
Learning rate: {learning_rate}
Epochs: {epochs}
Additional information: {additional_information}
"""


OUTPUT_TEMPLATE = """

Diagnosis: {diagnosis}
Evidence: {evidence}
Next experiment: {next_experiment}
"""


def create_input_messages(row):
    user_content = USER_TEMPLATE.format(
        model_name=row["model_name"],
        task=row["task"],
        train_loss=row["train_loss"],
        validation_loss=row["validation_loss"],
        train_metric=row["train_metric"],
        validation_metric=row["validation_metric"],
        learning_rate=row["learning_rate"],
        epochs=row["epochs"],
        additional_information=row["additional_information"],
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def create_training_messages(row):
    messages = create_input_messages(row)
    output_content = OUTPUT_TEMPLATE.format(
        diagnosis=row["diagnosis"],
        evidence=row["evidence"],
        next_experiment=row["next_experiment"],
    )

    return messages + [
        {"role": "assistant", "content": output_content},
    ]


def extract_section(answer, heading, next_heading=None):
    clean_answer = answer.replace("*", "")

    if next_heading:
        pattern = rf"(?ims)^\s*{re.escape(heading)}\s*:\s*(.*?)(?=^\s*{re.escape(next_heading)}\s*:|\Z)"
    else:
        pattern = rf"(?ims)^\s*{re.escape(heading)}\s*:\s*(.*)"

    match = re.search(pattern, clean_answer)
    return match.group(1).strip() if match else ""


def extract_diagnosis(answer):
    diagnosis_text = extract_section(answer, "Diagnosis", "Evidence").lower()

    for diagnosis in sorted(DIAGNOSES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(diagnosis)}\b", diagnosis_text):
            return diagnosis

    return "unparsed"


def extract_evidence(answer):
    return extract_section(answer, "Evidence", "Next experiment")
