"""Versioned website API using the exact training input transformation."""

import io
import csv
import hashlib
import json
from pathlib import Path
import threading

from fault_data import CURVES, DYNAMIC_FEATURES, FAULT_LABELS, log_to_input, parse_number, read_log
from inspection import inspect_report


PROJECT_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EPOCHS = 10000


def prepare_upload(csv_text, architecture):
    if not isinstance(csv_text, str) or len(csv_text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV-filen får vara högst 5 MB.")
    if architecture not in {"FNN", "CNN", "RNN", "unknown"}:
        raise ValueError("Välj en giltig arkitektur.")
    reader = csv.reader(io.StringIO(csv_text.lstrip("\ufeff")))
    headers = next(reader, [])
    if len(headers) != len(set(headers)) or any(len(row) != len(headers) for row in reader if row):
        raise ValueError("CSV-filen har dubbla kolumnnamn eller fel antal kolumner på en rad.")
    log = read_log(io.StringIO(csv_text.lstrip("\ufeff")))
    if len(log) > MAX_EPOCHS:
        raise ValueError("CSV-filen får innehålla högst 10 000 epoker.")
    observed = log_to_input(log, architecture)
    return log, observed


def report_for_log(log, observed):
    epochs = [int(float(row["epoch"])) for row in log]
    curves = {column: [parse_number(row[column]) for row in log] for column in CURVES}
    missing = sum(parse_number(row[column]) is None for row in log for column in (*CURVES, *DYNAMIC_FEATURES))
    total = len(log) * (len(CURVES) + len(DYNAMIC_FEATURES))
    result = {"api_version": 1, "architecture": observed["architecture"], "epochs": epochs,
            "curves": curves, "num_epochs": len(log),
            "signals": {column: [parse_number(row[column]) for row in log]
                        for column in ("mean_gradient", "gradient_max", "adjusted_lr")},
            "data_quality": {"missing_values": missing, "total_values": total,
                             "missing_rate": missing / total},
            "final": {column: values[-1] for column, values in curves.items()},
            "diagnosis": None, "mode": "preview", "model": None,
            "warnings": ["Källans train_acc/val_acc kan vara ett regressionsmått och visas som registrerade värden."]}
    result["inspection"] = inspect_report(result)
    return result


class ModelDoctorService:
    def __init__(self, adapter_dir=None, cpu=False, quantize=True, run_store=None, inference_enabled=True):
        self.adapter_dir = Path(adapter_dir or PROJECT_DIR / "models" / "modeldoctor_deep4ge")
        self.cpu, self.quantize = cpu, quantize
        self.lock = threading.Lock()
        self.loaded = None
        self.run_store = run_store
        self.inference_enabled = inference_enabled
        self.loaded_fingerprint = None

    def status(self):
        settings_path = self.adapter_dir / "modeldoctor_config.json"
        weights_exist = any((self.adapter_dir / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
        available = False
        if settings_path.is_file() and weights_exist and self.inference_enabled:
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                available = settings.get("task") == "deep4ge_fault_category" and settings.get("labels") == list(FAULT_LABELS)
            except (ValueError, OSError):
                pass
        return {"api_version": 1, "available": available, "mode": "llama" if available else "preview",
                "model": "ModelDoctor · Llama" if available else None, "max_upload_bytes": MAX_UPLOAD_BYTES,
                "required_columns": ["epoch", *CURVES, *DYNAMIC_FEATURES],
                "history_enabled": self.run_store is not None, "inference_enabled": self.inference_enabled,
                "product_version": "0.2.0"}

    def ingest(self, payload):
        if self.run_store is None:
            raise ValueError("Körningshistoriken är inte aktiverad.")
        self.run_store.validate_metadata(payload)
        log, observed = prepare_upload(payload.get("csv"), payload.get("architecture", "unknown"))
        result = report_for_log(log, observed)
        result["collection_status"] = payload.get("status", "completed")
        # Collection never triggers GPU inference, even at the end of a run.
        payload = {**payload, "csv": payload["csv"].replace("\r\n", "\n").rstrip("\n") + "\n"}
        return self.run_store.upsert(payload, result)

    def analyze_run(self, run_id):
        run = self.run_store.get(run_id)
        if run["status"] != "completed":
            raise ValueError("Modellanalys kräver en avslutad körning. Kurvor och kontroller fungerar under träningen.")
        result = self.analyze(run["csv"], run["architecture"])
        return self.run_store.save_analysis(run_id, run["revision"], result)

    def adapter_fingerprint(self):
        names = ("modeldoctor_config.json", "adapter_model.safetensors", "adapter_model.bin")
        return tuple((name, path.stat().st_mtime_ns, path.stat().st_size)
                     for name in names if (path := self.adapter_dir / name).is_file())

    def provenance(self):
        manifest = {}
        for name, _, _ in self.adapter_fingerprint():
            digest = hashlib.sha256()
            with (self.adapter_dir / name).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            manifest[name] = digest.hexdigest()
        return {"adapter_files_sha256": manifest, "task": "deep4ge_fault_category",
                "product_version": "0.2.0", "scope": "Experimental classification of Deep4ge mutation categories"}

    def analyze(self, csv_text, architecture):
        log, observed = prepare_upload(csv_text, architecture)
        result = report_for_log(log, observed)
        if not self.status()["available"]:
            return result
        if not any(parse_number(row[key]) is not None for row in log for key in DYNAMIC_FEATURES if key != "adjusted_lr"):
            result["model_skip_reason"] = "insufficient_training_signals"
            result["warnings"].append("Enbart kurvor och learning rate räcker inte som underlag för den tränade felklassificeraren. Träningssignaler saknas.")
            return result
        from fault_prompts import extract_diagnosis
        from train import load_saved_adapter, predict_rows
        import torch
        with self.lock:
            fingerprint = self.adapter_fingerprint()
            if self.loaded is None or fingerprint != self.loaded_fingerprint:
                self.loaded = load_saved_adapter(self.adapter_dir, self.cpu or not torch.cuda.is_available(), self.quantize)
                self.loaded_fingerprint = fingerprint
                self.loaded_provenance = self.provenance()
            model, tokenizer, settings = self.loaded
            if settings["curve_points"] != 12:
                observed = log_to_input(log, architecture, settings["curve_points"])
            answer = predict_rows(model, tokenizer, [observed], settings["max_length"])[0]
            provenance = self.loaded_provenance
        category = extract_diagnosis(answer)
        if category == "unparsed":
            raise ValueError("Modellen kunde inte ge en giltig kategori. Prova igen eller kontrollera loggen.")
        result.update(diagnosis=category, mode="llama", model="ModelDoctor · Llama")
        result["model_provenance"] = provenance
        return result
