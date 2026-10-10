"""No ML dependencies, hooks, optimizer changes, or synchronous network in log()."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
import uuid
import warnings

__version__ = "0.2.0"
CURVES = ("train_loss", "val_loss", "train_acc", "val_acc")
SIGNALS = ("large_weight_count", "acc_gap_too_big", "loss_oscillation", "dying_relu",
    "gradient_vanish", "gradient_explode", "decrease_acc_count", "increase_loss_count",
    "cons_mean_weight_count", "cons_std_weight_count", "nan_weight_count", "nan_gradients_count",
    "saturated_activation", "mean_gradient", "gradient_std", "gradient_max", "gradient_min",
    "gradient_median", "adjusted_lr", "mean_activation", "std_activation", "mean_grad", "std_grad")
COLUMNS = ("epoch", *CURVES, *SIGNALS)


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def normalize_csv(text):
    """Expand a simple loss log; absent telemetry stays absent, never fabricated."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not {"epoch", "train_loss"}.issubset(reader.fieldnames or ()):
        raise ValueError("CSV needs epoch and train_loss columns.")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CSV headers must be unique.")
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row has the wrong number of columns.")
        rows.append({name: row.get(name, "") for name in COLUMNS})
    if not 0 < len(rows) <= 10000:
        raise ValueError("CSV needs between 1 and 10,000 epochs.")
    return csv_text(rows)


def post_snapshot(payload, endpoint="http://127.0.0.1:8000", timeout=.5, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(endpoint.rstrip("/") + "/api/runs",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Run:
    """Log once per epoch. Failures to transmit never propagate into training."""
    def __init__(self, name, *, project="default", architecture="unknown", config=None,
                 dataset="unspecified", metric_name="source metric", endpoint="http://127.0.0.1:8000",
                 data_dir="runs/outbox", signal_schema="unverified", token=None, timeout=.5):
        self.run_id = uuid.uuid4().hex
        self.metadata = {"api_version": 1, "run_id": self.run_id, "name": name, "project": project,
            "architecture": architecture, "config": config or {}, "dataset": dataset,
            "metric_name": metric_name, "source": "python-client", "signal_schema": signal_schema}
        # Validate serializability before starting a worker. Explicitly supplied config only.
        self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        if architecture not in {"FNN", "CNN", "RNN", "unknown"}:
            raise ValueError("Unsupported architecture.")
        self.endpoint, self.token, self.timeout = endpoint, token, timeout
        self.directory = Path(data_dir)
        self.rows, self.revision = [], 0
        self.condition = threading.Condition()
        self.pending, self.sending, self.stopping = None, False, False
        self.last_error, self.last_sent_revision, self.closed = None, 0, False
        self.disk_warning = False
        self.worker = threading.Thread(target=self._send_loop, daemon=True, name="modeldoctor-uploader")
        self.worker.start()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.finish(status="failed" if kind else "completed")
        return False

    def log(self, epoch, *, train_loss=None, val_loss=None, train_acc=None, val_acc=None, signals=None):
        if self.closed:
            raise ValueError("Run has already finished.")
        if type(epoch) is not int or epoch < 0 or (self.rows and epoch <= self.rows[-1]["epoch"]):
            raise ValueError("Epochs must be nonnegative, strictly increasing integers.")
        if len(self.rows) >= 10000:
            raise ValueError("Maximum 10,000 epochs per run.")
        signals = signals or {}
        if set(signals) - set(SIGNALS):
            raise ValueError("Unsupported training signal column.")
        values = {"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc, **signals}
        row = {name: "" for name in COLUMNS}
        row["epoch"] = epoch
        for name, value in values.items():
            if value is not None:
                # Caller passes scalars; we do not read GPU tensors or weights ourselves.
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be a Python numeric scalar or None.")
                row[name] = value if math.isfinite(value) else ""
        self.rows.append(row)
        self._snapshot("running")

    def _snapshot(self, status):
        self.revision += 1
        payload = {**self.metadata, "revision": self.revision, "status": status, "csv": csv_text(self.rows)}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            for suffix, text in (("csv", payload["csv"]), ("json", json.dumps(payload, allow_nan=False))):
                target = self.directory / f"{self.run_id}.{suffix}"
                temporary = target.with_suffix(f".{suffix}.tmp")
                temporary.write_text(text, encoding="utf-8", newline="")
                temporary.replace(target)
        except OSError as error:
            self.last_error = f"Local log could not be saved: {type(error).__name__}"
            if not self.disk_warning:
                try:
                    warnings.warn(self.last_error, RuntimeWarning, stacklevel=2)
                except RuntimeWarning:
                    # Logging still must not abort training if warnings are promoted to exceptions.
                    pass
                self.disk_warning = True
        with self.condition:
            self.pending = payload
            self.condition.notify_all()

    def _send_loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.stopping)
                if self.pending is None:
                    return
                payload, self.pending = self.pending, None
                self.sending = True
            try:
                post_snapshot(payload, self.endpoint, self.timeout, self.token)
                self.last_sent_revision = payload["revision"]
                self.last_error = None
            except Exception as error:
                # Do not disclose credentials/URLs in errors, and never fail the user's training.
                self.last_error = f"Upload pending ({type(error).__name__}); local snapshot can be replayed."
                with self.condition:
                    if self.pending is None:
                        self.pending = payload
                    if not self.stopping:
                        self.condition.wait(timeout=2)
            finally:
                with self.condition:
                    self.sending = False
                    self.condition.notify_all()
                    stopping = self.stopping
            if stopping:
                # A failed final attempt remains on disk; do not delay process exit.
                return

    def flush(self, timeout=1):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.last_sent_revision < self.revision and time.monotonic() < deadline:
                self.condition.wait(timeout=max(0, deadline - time.monotonic()))
        return self.last_sent_revision == self.revision

    def finish(self, status="completed", timeout=1):
        if self.closed:
            return self.last_sent_revision == self.revision
        if status not in {"completed", "failed"}:
            raise ValueError("Final status must be completed or failed.")
        if self.rows:
            self._snapshot(status)
        self.closed = True
        uploaded = self.flush(timeout) if self.rows else True
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.worker.join(timeout=self.timeout + .1)
        return uploaded


def upload_csv(path, *, endpoint="http://127.0.0.1:8000", project="default", architecture="unknown",
               dataset="unspecified", metric_name="source metric", token=None):
    path = Path(path)
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("CSV exceeds 5 MB.")
    text = normalize_csv(path.read_text(encoding="utf-8"))
    # Stable identity makes repeated completed-file delivery idempotent.
    run_id = hashlib.sha256((str(path.resolve()) + "\0" + text).encode()).hexdigest()[:40]
    return post_snapshot({"api_version": 1, "run_id": run_id, "revision": 1, "csv": text,
        "name": path.name, "project": project, "architecture": architecture, "dataset": dataset,
        "metric_name": metric_name, "status": "completed", "source": "folder-watcher",
        "signal_schema": "unverified", "config": {}}, endpoint, timeout=5, token=token)


def replay(directory="runs/outbox", *, endpoint="http://127.0.0.1:8000", token=None):
    results = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.stat().st_size > 6 * 1024 * 1024:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Never silently finalize an interrupted/running snapshot.
        result = post_snapshot(payload, endpoint, timeout=5, token=token)
        results.append({"file": path.name, "accepted": result["accepted"]})
    return results
