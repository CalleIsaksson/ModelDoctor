import csv
from functools import partial
import http.client
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fault_data import CURVES, DYNAMIC_FEATURES, INPUT_COLUMNS, log_to_input, read_log
from web_server import Handler
from web_service import ModelDoctorService, prepare_upload


ROOT = Path(__file__).resolve().parents[1]
CSV = (ROOT / "web" / "dist" / "examples" / "rnn_training_log.csv").read_text(encoding="utf-8")


class WebServiceTests(unittest.TestCase):
    def test_upload_reuses_training_converter_and_model_sees_only_observations(self):
        log, observed = prepare_upload(CSV, "RNN")
        self.assertEqual(observed, log_to_input(read_log(io.StringIO(CSV)), "RNN"))
        self.assertEqual(set(observed), set(INPUT_COLUMNS))
        self.assertEqual(len(log), 50)

    def test_preview_does_not_invent_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            result = ModelDoctorService(directory).analyze(CSV, "RNN")
        self.assertEqual(result["mode"], "preview")
        self.assertIsNone(result["diagnosis"])
        self.assertEqual(result["num_epochs"], 50)
        self.assertEqual(result["curves"]["train_loss"][-1], result["final"]["train_loss"])

    def test_missing_values_remain_null_and_interleaved_epochs_are_rejected(self):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["epoch", *CURVES, *DYNAMIC_FEATURES])
        writer.writeheader()
        writer.writerows({"epoch": epoch, **{key: "nan" for key in (*CURVES, *DYNAMIC_FEATURES)}} for epoch in (0, 1))
        with tempfile.TemporaryDirectory() as directory:
            result = ModelDoctorService(directory).analyze(output.getvalue(), "unknown")
        self.assertEqual(result["curves"]["train_loss"], [None, None])
        self.assertEqual(result["data_quality"]["missing_rate"], 1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            prepare_upload(output.getvalue().replace("\n1,", "\n0,"), "RNN")

    def test_saved_adapter_connection_uses_strict_category_parser_and_caches_model(self):
        with tempfile.TemporaryDirectory() as directory:
            service = ModelDoctorService(directory)
            with patch.object(service, "status", return_value={"available": True}), \
                 patch("train.load_saved_adapter", return_value=("model", "tokenizer", {"curve_points": 12, "max_length": 4096})) as loader, \
                 patch("train.predict_rows", return_value=["Diagnosis: weight_fault"]) as predict:
                result = service.analyze(CSV, "RNN")
                service.analyze(CSV, "RNN")
                self.assertEqual(result["mode"], "llama")
                self.assertEqual(result["diagnosis"], "weight_fault")
                self.assertEqual(set(predict.call_args.args[2][0]), set(INPUT_COLUMNS))
                loader.assert_called_once()
                predict.return_value = ["Diagnosis: overfitting"]
                with self.assertRaisesRegex(ValueError, "giltig kategori"):
                    service.analyze(CSV, "RNN")


class WebHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, service=ModelDoctorService(cls.temporary.name)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temporary.cleanup()

    def request(self, method, path, value=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            connection.request(method, path, body=json.dumps(value) if value is not None else None,
                               headers=headers or {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8")
        finally:
            connection.close()

    def test_status_upload_and_local_configuration(self):
        status, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["available"])
        status, body = self.request("POST", "/api/analyze", {"api_version": 1, "csv": CSV, "architecture": "RNN", "diagnosis": "loss_fault"})
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body)["diagnosis"])
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertIn("ENABLED = true", self.request("GET", "/config.js")[1])

    def test_invalid_requests_and_cross_origin_writes_are_rejected(self):
        self.assertEqual(self.request("POST", "/api/analyze", {"csv": "bad"})[0], 400)
        self.assertEqual(self.request("POST", "/api/analyze", {"api_version": 2})[0], 400)
        self.assertEqual(self.request("POST", "/api/analyze", {}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/analyze", {}, {"Content-Type": "application/json", "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/no-such-endpoint", {})[0], 404)


if __name__ == "__main__":
    unittest.main()
