"""Generate a clearly synthetic onboarding log, without training a model."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))
from modeldoctor_client import Run


def main():
    with Run("Onboarding example (synthetic)", project="onboarding", architecture="FNN",
             dataset="synthetic-demo", config={"learning_rate": .001}, metric_name="not collected") as run:
        for epoch in range(12):
            run.log(epoch, train_loss=1 / (epoch + 1), val_loss=.35 + abs(epoch - 5) * .04,
                    signals={"adjusted_lr": .001})
    print(f"Run: {run.run_id}. Open http://127.0.0.1:8000 and choose Körningar.")
    if run.last_error:
        print(run.last_error)


if __name__ == "__main__":
    main()
