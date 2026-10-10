"""Run the ModelDoctor website locally; no extra web-framework dependencies."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import hmac
import os
import socket
from pathlib import Path
from urllib.parse import urlsplit

from web_service import MAX_UPLOAD_BYTES, ModelDoctorService
from run_store import RunStore


PROJECT_DIR = Path(__file__).resolve().parent


class PilotHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can silently bind two processes to the same port.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, service, api_token=None, **kwargs):
        self.service = service
        self.api_token = api_token
        super().__init__(*args, directory=str(PROJECT_DIR / "web" / "dist"), **kwargs)

    def json_response(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith("/api/") and path != "/api/status" and not self.authorized():
            return
        if path == "/api/status":
            self.json_response({**self.service.status(), "auth_required": bool(self.api_token)})
        elif path == "/api/runs" and self.service.run_store:
            self.json_response({"api_version": 1, "runs": self.service.run_store.list()})
        elif path.startswith("/api/runs/") and self.service.run_store:
            try:
                self.json_response(self.service.run_store.get(path.split("/")[-1]))
            except KeyError:
                self.json_response({"error": "Körningen finns inte."}, 404)
        elif self.path == "/config.js":
            payload = b'window.MODELDOCTOR_API_BASE = ""; window.MODELDOCTOR_API_ENABLED = true;'
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.startswith("/api/"):
            self.json_response({"error": "Endpoint saknas."}, 404)
        else:
            super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        parts = path.strip("/").split("/")
        if path not in {"/api/analyze", "/api/runs"} and not (len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[-1] in {"analyze", "feedback"}):
            self.json_response({"error": "Endpoint saknas."}, 404)
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self.json_response({"error": "Skicka application/json."}, 415)
            return
        # Reject cross-origin browser writes to this local model service.
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and origin not in {f"http://{host}", f"https://{host}"}:
            self.json_response({"error": "Anropet måste komma från hemsidans egen adress."}, 403)
            return
        if not self.authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD_BYTES + 1024 * 1024:
                self.json_response({"error": "Filen är tom eller för stor."}, 413)
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Ogiltigt analysanrop.")
            if payload.get("api_version", 1) != 1:
                raise ValueError("API-versionen stöds inte. Använd version 1.")
            if path == "/api/analyze":
                result = self.service.analyze(payload.get("csv"), payload.get("architecture", "unknown"))
            elif path == "/api/runs":
                result = self.service.ingest(payload)
            elif self.service.run_store is None:
                raise ValueError("Körningshistoriken är inte aktiverad.")
            elif parts[-1] == "analyze":
                result = self.service.analyze_run(parts[2])
            else:
                result = self.service.run_store.feedback(parts[2], payload.get("verdict"), payload.get("note", ""))
            self.json_response(result)
        except (ValueError, TypeError, UnicodeError, KeyError) as error:
            self.json_response({"error": str(error)}, 400)
        except Exception:
            self.json_response({"error": "Analysen kunde inte slutföras. Kontrollera modellinstallationen på servern."}, 500)

    def authorized(self):
        if self.api_token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.api_token):
            self.json_response({"error": "API-nyckel krävs."}, 401)
            return False
        return True

    def do_DELETE(self):
        parts = urlsplit(self.path).path.strip("/").split("/")
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and origin not in {f"http://{host}", f"https://{host}"}:
            self.json_response({"error": "Anropet måste komma från hemsidans egen adress."}, 403)
            return
        if not self.authorized():
            return
        if len(parts) != 3 or parts[:2] != ["api", "runs"] or self.service.run_store is None:
            self.json_response({"error": "Endpoint saknas."}, 404)
            return
        try:
            self.service.run_store.delete(parts[2])
            self.json_response({"deleted": True})
        except KeyError:
            self.json_response({"error": "Körningen finns inte."}, 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--runs-db", type=Path, default=PROJECT_DIR / "runs" / "modeldoctor.sqlite3")
    parser.add_argument("--disable-model", action="store_true", help="Keep logging/UI available without loading a model or using GPU memory.")
    args = parser.parse_args()
    token = os.environ.get("MODELDOCTOR_API_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("Set MODELDOCTOR_API_TOKEN before binding outside localhost. Internet deployment also requires authenticated HTTPS.")
    service = ModelDoctorService(args.adapter_dir, args.cpu, not args.no_quantization,
                                 run_store=RunStore(args.runs_db), inference_enabled=not args.disable_model)
    with PilotHTTPServer((args.host, args.port), partial(Handler, service=service, api_token=token)) as server:
        print(f"ModelDoctor: http://{args.host}:{server.server_port}", flush=True)
        print(f"Sparade körningar: {args.runs_db}. Insamling startar ingen modellinferens. Ctrl+C avslutar servern.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
