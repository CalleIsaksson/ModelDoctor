import json
import re

from modeldoctor_prompts import extract_evidence


NUMBER_PATTERN = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
CURVE_COLUMNS = [
    "train_loss",
    "validation_loss",
    "train_metric",
    "validation_metric",
]


def find_numbers(text):
    return [
        float(value)
        for value in re.findall(NUMBER_PATTERN, str(text).lower())
    ]


def values_are_close(first, second):
    tolerance = max(0.0006, abs(second) * 0.002)
    return abs(first - second) <= tolerance


def allowed_evidence_numbers(row):
    allowed = []

    for column in [*CURVE_COLUMNS, "epochs", "learning_rate", "additional_information"]:
        allowed.extend(find_numbers(row[column]))

    for column in CURVE_COLUMNS:
        try:
            values = json.loads(row[column])
        except (TypeError, json.JSONDecodeError):
            values = []

        if len(values) >= 2:
            derived_values = [
                abs(values[0] - values[-1]),
                abs(min(values) - values[-1]),
                abs(max(values) - values[-1]),
            ]
            allowed.extend(round(value, 3) for value in derived_values)

    return allowed


def score_evidence_numbers(row, answer):
    evidence = extract_evidence(answer)
    evidence_numbers = find_numbers(evidence)
    allowed_numbers = allowed_evidence_numbers(row)
    unsupported_numbers = [
        number
        for number in evidence_numbers
        if not any(values_are_close(number, allowed) for allowed in allowed_numbers)
    ]

    return {
        "evidence_has_numbers": bool(evidence_numbers),
        "evidence_numbers_supported": bool(evidence_numbers) and not unsupported_numbers,
        "evidence_numbers": json.dumps(evidence_numbers),
        "unsupported_evidence_numbers": json.dumps(unsupported_numbers),
    }
