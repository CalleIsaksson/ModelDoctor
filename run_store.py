"""Local pilot run storage. Never writes training datasets or adapters."""

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as database:
            database.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, revision INTEGER NOT NULL, digest TEXT NOT NULL,
                metadata TEXT NOT NULL, csv TEXT NOT NULL, report TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                feedback TEXT)""")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def validate_metadata(payload):
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ValueError("Ogiltigt körnings-id.")
        revision = payload.get("revision", 1)
        if type(revision) is not int or revision < 1:
            raise ValueError("Revisionen måste vara ett positivt heltal.")
        metadata = {"name": payload.get("name", run_id), "project": payload.get("project", "default"),
                    "architecture": payload.get("architecture", "unknown"),
                    "status": payload.get("status", "completed"), "source": payload.get("source", "upload"),
                    "config": payload.get("config", {}), "metric_name": payload.get("metric_name", "source metric"),
                    "dataset": payload.get("dataset", "unspecified"),
                    "signal_schema": payload.get("signal_schema", "unverified")}
        for key in ("name", "project", "source", "metric_name", "dataset", "signal_schema"):
            if not isinstance(metadata[key], str) or not 0 < len(metadata[key]) <= 160:
                raise ValueError(f"Ogiltigt värde för {key}.")
        if metadata["status"] not in {"running", "completed", "failed"}:
            raise ValueError("Körningsstatusen stöds inte.")
        if not isinstance(metadata["config"], dict) or len(json.dumps(metadata["config"], allow_nan=False).encode()) > 16384:
            raise ValueError("Konfigurationen måste vara ett JSON-objekt på högst 16 KB.")
        return run_id, revision, metadata

    @staticmethod
    def decode(row, full=True):
        if row is None:
            raise KeyError("Körningen finns inte.")
        report = json.loads(row["report"])
        result = {"run_id": row["id"], "revision": row["revision"], **json.loads(row["metadata"]),
                  "created_at": row["created_at"], "updated_at": row["updated_at"],
                  "input_sha256": row["digest"], "feedback": json.loads(row["feedback"]) if row["feedback"] else None}
        if full:
            result.update(csv=row["csv"], report=report)
        else:
            result.update(num_epochs=report["num_epochs"], final=report["final"],
                          findings=len(report["inspection"]["findings"]), diagnosis=report["diagnosis"])
        return result

    def get(self, run_id):
        with self.connect() as database:
            return self.decode(database.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())

    def list(self, limit=200):
        with self.connect() as database:
            rows = database.execute("SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.decode(row, full=False) for row in rows]

    def upsert(self, payload, report):
        run_id, revision, metadata = self.validate_metadata(payload)
        csv_text = payload["csv"]
        digest = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
        canonical = json.dumps(metadata, sort_keys=True, allow_nan=False)
        with self.connect() as database:
            database.execute("BEGIN IMMEDIATE")
            previous = database.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if previous:
                if revision <= previous["revision"]:
                    if revision == previous["revision"] and (digest != previous["digest"] or canonical != previous["metadata"]):
                        raise ValueError("Samma revision innehåller olika data.")
                    return {"accepted": False, "run": self.decode(previous)}
                old = json.loads(previous["metadata"])
                if any(metadata[key] != old[key] for key in metadata if key != "status"):
                    raise ValueError("En körnings identitet och inställningar får inte ändras. Skapa en ny körning.")
                if old["status"] != "running":
                    raise ValueError("En avslutad körning får inte skrivas över. Skapa en ny körning.")
                # Require an immutable prefix; a new snapshot may only append epochs.
                if not csv_text.startswith(previous["csv"].rstrip("\r\n") + "\n") and csv_text != previous["csv"]:
                    raise ValueError("Tidigare epoker får inte ändras i en pågående körning.")
            timestamp = utc_now()
            database.execute("""INSERT INTO runs (id, revision, digest, metadata, csv, report, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET revision=excluded.revision, digest=excluded.digest,
                metadata=excluded.metadata, csv=excluded.csv, report=excluded.report, updated_at=excluded.updated_at,
                feedback=NULL""", (run_id, revision, digest, canonical, csv_text,
                json.dumps(report, allow_nan=False), previous["created_at"] if previous else timestamp, timestamp))
            row = database.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return {"accepted": True, "run": self.decode(row)}

    def save_analysis(self, run_id, revision, report):
        with self.connect() as database:
            database.execute("BEGIN IMMEDIATE")
            changed = database.execute("UPDATE runs SET report=?, feedback=NULL WHERE id=? AND revision=?",
                (json.dumps(report, allow_nan=False), run_id, revision)).rowcount
            if not changed:
                raise ValueError("Körningen ändrades medan analysen pågick. Läs in den igen.")
        return self.get(run_id)

    def feedback(self, run_id, verdict, note):
        if verdict not in {"useful", "incorrect", "unsure"} or not isinstance(note, str) or len(note) > 2000:
            raise ValueError("Ogiltig återkoppling.")
        run = self.get(run_id)
        if not run["report"]["diagnosis"]:
            raise ValueError("Återkoppling på modellresultat kräver en modellprediktion.")
        value = {"verdict": verdict, "note": note, "created_at": utc_now(),
                 "revision": run["revision"], "model_provenance": run["report"].get("model_provenance")}
        with self.connect() as database:
            changed = database.execute("UPDATE runs SET feedback=? WHERE id=? AND revision=? AND report=?",
                (json.dumps(value), run_id, run["revision"], json.dumps(run["report"], allow_nan=False))).rowcount
            if not changed:
                raise ValueError("Analysen ändrades. Läs in körningen igen.")
        return self.get(run_id)

    def delete(self, run_id):
        with self.connect() as database:
            if not database.execute("DELETE FROM runs WHERE id=?", (run_id,)).rowcount:
                raise KeyError("Körningen finns inte.")
