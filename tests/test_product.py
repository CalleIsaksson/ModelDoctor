import copy
import csv
from functools import partial
import http.client
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))
from modeldoctor_client import COLUMNS, Run, normalize_csv, replay
from assess_product import assess, read_records
from run_store import RunStore
from web_server import Handler
from web_service import ModelDoctorService
from watch_logs import scan


def payload(run_id="test-run", revision=1, status="running", epochs=3):
    return {"api_version": 1, "run_id": run_id, "revision": revision, "name": "test",
        "project": "pilot", "architecture": "FNN", "status": status, "config": {"lr": .01},
        "csv": normalize_csv("epoch,train_loss,val_loss\n" + "".join(f"{i},{1/(i+1)},{.5+i*.1}\n" for i in range(epochs)))}


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = RunStore(Path(self.temporary.name) / "runs.sqlite3")
        self.service = ModelDoctorService(Path(self.temporary.name) / "no-adapter", run_store=self.store)

    def tearDown(self):
        self.temporary.cleanup()

    def test_collection_is_durable_idempotent_and_does_not_load_a_model(self):
        with patch.object(self.service, "analyze", side_effect=AssertionError("No inference during collection")):
            first = self.service.ingest(payload())
            self.assertTrue(first["accepted"])
            self.assertFalse(self.service.ingest(payload())["accepted"])
        restored = RunStore(self.store.path).get("test-run")
        self.assertEqual(restored["report"]["num_epochs"], 3)
        self.assertIsNone(restored["report"]["diagnosis"])

    def test_stale_updates_and_changed_epochs_cannot_overwrite_newer_data(self):
        self.service.ingest(payload())
        self.service.ingest(payload(revision=2, epochs=4))
        self.assertFalse(self.service.ingest(payload())["accepted"])
        changed = payload(revision=3, epochs=5)
        changed["csv"] = changed["csv"].replace("0,1.0,", "0,2.0,")
        with self.assertRaisesRegex(ValueError, "epoker"):
            self.service.ingest(changed)
        self.assertEqual(self.store.get("test-run")["revision"], 2)
        with self.assertRaisesRegex(ValueError, "Samma revision"):
            self.service.ingest(payload(revision=2, epochs=5))

    def test_terminal_runs_are_immutable_and_partial_runs_cannot_be_classified(self):
        self.service.ingest(payload())
        with self.assertRaisesRegex(ValueError, "avslutad"):
            self.service.analyze_run("test-run")
        self.service.ingest(payload(revision=2, status="completed"))
        with self.assertRaisesRegex(ValueError, "avslutad"):
            self.service.ingest(payload(revision=3, status="running", epochs=4))
        result = self.service.analyze_run("test-run")
        self.assertIsNone(result["report"]["diagnosis"])

    def test_inspection_is_factual_and_incomplete_telemetry_abstains(self):
        run = self.service.ingest(payload(status="completed", epochs=10))["run"]
        findings = run["report"]["inspection"]
        self.assertEqual(findings["best_validation"], {"epoch": 0, "value": .5})
        self.assertIn("validation_regression", [item["id"] for item in findings["findings"]])
        with patch.object(self.service, "status", return_value={"available": True}):
            result = self.service.analyze(run["csv"], "FNN")
        self.assertEqual(result["model_skip_reason"], "insufficient_training_signals")
        self.assertIsNone(result["diagnosis"])

    def test_compare_and_save_analysis_rejects_stale_revision_and_feedback_is_separate(self):
        run = self.service.ingest(payload())["run"]
        self.service.ingest(payload(revision=2, epochs=4))
        with self.assertRaisesRegex(ValueError, "ändrades"):
            self.store.save_analysis("test-run", 1, run["report"])
        report = self.store.get("test-run")["report"]
        report["diagnosis"] = "weight_fault"
        self.store.save_analysis("test-run", 2, report)
        self.store.feedback("test-run", "incorrect", "manual review needed")
        self.assertEqual(self.store.get("test-run")["feedback"]["verdict"], "incorrect")
        self.assertNotIn("incorrect", self.store.get("test-run")["csv"])
        self.assertEqual(report["diagnosis"], "weight_fault")

    def test_folder_watcher_requires_done_marker_and_retries_failure(self):
        directory = Path(self.temporary.name)
        (directory / "train.csv").write_text("epoch,train_loss\n0,1\n", encoding="utf-8")
        delivered = set()
        with patch("watch_logs.upload_csv", return_value={"run": {"run_id": "x"}}) as upload:
            scan(directory, delivered)
            upload.assert_not_called()
            (directory / "train.csv.done").touch()
            upload.side_effect = OSError("offline")
            scan(directory, delivered)
            self.assertFalse(delivered)
            upload.side_effect = None
            scan(directory, delivered)
            self.assertEqual(len(delivered), 1)
            scan(directory, delivered)
            self.assertEqual(upload.call_count, 2)


class ClientTests(unittest.TestCase):
    def test_collection_does_not_change_model_training_or_random_state(self):
        import torch
        def train(collect, directory):
            torch.manual_seed(42)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            run = Run("training", data_dir=directory) if collect else None
            for epoch in range(3):
                optimizer.zero_grad()
                inputs, targets = torch.randn(8, 2), torch.randn(8, 1)
                loss = torch.nn.functional.mse_loss(model(inputs), targets)
                loss.backward()
                optimizer.step()
                if run:
                    run.log(epoch, train_loss=loss.item())
            if run:
                run.finish()
            return {key: value.detach().clone() for key, value in model.state_dict().items()}, torch.get_rng_state()
        with tempfile.TemporaryDirectory() as directory, patch("modeldoctor_client.post_snapshot", return_value={"accepted": True}):
            plain, plain_rng = train(False, directory)
            collected, collected_rng = train(True, directory)
        for key in plain:
            self.assertTrue(torch.equal(plain[key], collected[key]))
        self.assertTrue(torch.equal(plain_rng, collected_rng))

    def test_log_never_waits_for_network_and_offline_snapshot_survives(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(*args):
            entered.set()
            release.wait(3)
            raise OSError("offline")
        with tempfile.TemporaryDirectory() as directory, patch("modeldoctor_client.post_snapshot", side_effect=blocked):
            run = Run("offline", data_dir=directory, timeout=.01)
            try:
                run.log(0, train_loss=1)
                self.assertTrue(entered.wait(1))
                worker = run.worker
                started = time.monotonic()
                run.log(1, train_loss=.5)
                self.assertLess(time.monotonic() - started, .5)
                self.assertTrue(worker.is_alive())
                self.assertFalse(run.finish(timeout=.01))
                snapshot = json.loads((Path(directory) / f"{run.run_id}.json").read_text())
                self.assertEqual(snapshot["status"], "completed")
                self.assertEqual(snapshot["revision"], 3)
                self.assertEqual(len(list(csv.DictReader(io.StringIO(snapshot["csv"])))), 2)
            finally:
                release.set()
                run.worker.join(2)

    def test_replay_preserves_status_and_input_columns_without_creating_labels(self):
        with tempfile.TemporaryDirectory() as directory, patch("modeldoctor_client.post_snapshot", return_value={"accepted": True}) as send:
            run = Run("normal", data_dir=directory)
            run.log(0, train_loss=1, signals={"adjusted_lr": .001})
            run.finish(status="failed")
            send.reset_mock()
            replay(directory)
            snapshot = send.call_args.args[0]
            self.assertEqual(snapshot["status"], "failed")
            reader = csv.DictReader(io.StringIO(snapshot["csv"]))
            self.assertEqual(tuple(reader.fieldnames), COLUMNS)
            self.assertEqual(next(reader)["gradient_max"], "")
            self.assertNotIn("diagnosis", snapshot)

    def test_client_preserves_training_exception(self):
        with tempfile.TemporaryDirectory() as directory, patch("modeldoctor_client.post_snapshot", return_value={"accepted": True}):
            with self.assertRaisesRegex(RuntimeError, "training failure"):
                with Run("failure", data_dir=directory) as run:
                    run.log(0, train_loss=1)
                    raise RuntimeError("training failure")
            data = json.loads((Path(directory) / f"{run.run_id}.json").read_text())
            self.assertEqual(data["status"], "failed")


class ProductHTTPTests(unittest.TestCase):
    def test_real_client_http_delivery_replay_auth_and_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            service = ModelDoctorService(Path(directory)/"no-adapter", run_store=RunStore(Path(directory)/"runs.sqlite3"))
            server = ThreadingHTTPServer(("127.0.0.1",0), partial(Handler,service=service,api_token="test-only-key"))
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}"
            try:
                with Run("integration", endpoint=endpoint, token="test-only-key", data_dir=Path(directory)/"outbox") as run:
                    run.log(0, train_loss=1, val_loss=1.2)
                    run.log(1, train_loss=.8, val_loss=1.1)
                stored = service.run_store.get(run.run_id)
                self.assertEqual(stored["status"], "completed")
                self.assertEqual(stored["report"]["num_epochs"], 2)
                self.assertFalse(replay(Path(directory)/"outbox", endpoint=endpoint,token="test-only-key")[0]["accepted"])
                connection = http.client.HTTPConnection("127.0.0.1",server.server_port)
                connection.request("GET","/api/runs")
                response = connection.getresponse()
                self.assertEqual(response.status,401)
                response.read()
                connection.request("DELETE",f"/api/runs/{run.run_id}",headers={"Authorization":"Bearer test-only-key"})
                response = connection.getresponse()
                self.assertEqual(response.status,200)
                response.read()
                connection.close()
                self.assertEqual(service.run_store.list(), [])
            finally:
                server.shutdown(); server.server_close(); thread.join()


class ProductAssessmentTests(unittest.TestCase):
    def test_perfect_candidate_beats_reference_and_partial_predictions_are_rejected(self):
        from fault_data import FAULT_LABELS
        expected = {f"id-{i}": {"diagnosis":label,"source_group":f"group-{i%2}"} for i,label in enumerate(FAULT_LABELS)}
        candidate = [{"example_id":key,"source_group":row["source_group"],"expected_diagnosis":row["diagnosis"],"predicted_diagnosis":row["diagnosis"]} for key,row in expected.items()]
        reference = [{**row,"predicted_diagnosis":"loss_fault"} for row in candidate]
        result = assess(candidate,reference,bootstrap_samples=20)
        self.assertTrue(result["advantage_on_this_validation_set"])
        self.assertFalse(result["commercial_readiness_established"])
        self.assertEqual(result["candidate"]["baseline_flag_rate"],0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"predictions.csv"
            with path.open("w",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(candidate[0]))
                writer.writeheader(); writer.writerows(candidate[:-1])
            with self.assertRaisesRegex(ValueError,"exactly once"):
                read_records(path,expected)

    def test_same_reference_does_not_establish_a_learned_model_advantage(self):
        records=[{"example_id":"id","source_group":"g","expected_diagnosis":"no_injected_fault","predicted_diagnosis":"no_injected_fault"}]
        result=assess(records,records,bootstrap_samples=10)
        self.assertEqual(result["macro_f1_delta"],0)
        self.assertFalse(result["advantage_on_this_validation_set"])


if __name__ == "__main__":
    unittest.main()
