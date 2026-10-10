"""CPU baseline features derived solely from the classifier's observed inputs."""

import json
import math
from pathlib import Path
import sys

from fault_data import CURVES, DYNAMIC_FEATURES


def enable_local_analysis_dependencies():
    # Optional workspace-only dependency installation for the current session.
    local = Path(__file__).resolve().parent / ".analysis-deps"
    if local.exists():
        sys.path.insert(0, str(local))


def finite_or_nan(value):
    if value is None:
        return float("nan")
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def observed_features(row):
    result = {"num_epochs": float(row["num_epochs"])}
    for architecture in ("FNN", "CNN", "RNN", "unknown"):
        result[f"architecture_{architecture}"] = float(row["architecture"] == architecture)
    for column in CURVES:
        values = [finite_or_nan(value) for value in json.loads(row[column])]
        finite = [value for value in values if math.isfinite(value)]
        result.update({f"{column}_first": values[0], f"{column}_last": values[-1],
                       f"{column}_min": min(finite) if finite else float("nan"),
                       f"{column}_max": max(finite) if finite else float("nan"),
                       f"{column}_mean": sum(finite) / len(finite) if finite else float("nan"),
                       f"{column}_change": values[-1] - values[0],
                       f"{column}_missing_rate": 1 - len(finite) / len(values)})
    result["final_loss_gap"] = result["val_loss_last"] - result["train_loss_last"]
    result["final_metric_gap"] = result["train_acc_last"] - result["val_acc_last"]
    dynamic = json.loads(row["dynamic_features"])
    for feature in DYNAMIC_FEATURES:
        for statistic in ("first", "last", "min", "max", "mean", "missing"):
            result[f"{feature}_{statistic}"] = finite_or_nan(dynamic[feature][statistic])
    return result
