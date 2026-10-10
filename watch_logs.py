"""Deliver finished CSV logs marked with <filename>.csv.done; retry on failure."""

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "sdk"))
from modeldoctor_client import replay, upload_csv


def scan(directory, delivered, **settings):
    for marker in sorted(Path(directory).glob("*.csv.done")):
        csv_path = marker.with_suffix("")
        if not csv_path.is_file():
            continue
        signature = (str(csv_path.resolve()), csv_path.stat().st_mtime_ns, csv_path.stat().st_size)
        if signature in delivered:
            continue
        try:
            result = upload_csv(csv_path, **settings)
            delivered.add(signature)
            print(f"Ready: {csv_path.name} -> {result['run']['run_id']}", flush=True)
        except Exception as error:
            print(f"Pending: {csv_path.name} ({type(error).__name__}); will retry.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("logs"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--project", default="default")
    parser.add_argument("--architecture", choices=("FNN", "CNN", "RNN", "unknown"), default="unknown")
    parser.add_argument("--dataset", default="unspecified")
    parser.add_argument("--metric-name", default="source metric")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Replay client snapshots in the directory instead.")
    args = parser.parse_args()
    token = os.environ.get("MODELDOCTOR_API_TOKEN")
    if args.replay:
        print(replay(args.directory, endpoint=args.endpoint, token=token))
        return
    if not args.directory.is_dir():
        parser.error("The log directory does not exist.")
    delivered = set()
    try:
        while True:
            scan(args.directory, delivered, endpoint=args.endpoint, project=args.project,
                 architecture=args.architecture, dataset=args.dataset, metric_name=args.metric_name, token=token)
            if args.once:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
