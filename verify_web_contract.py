"""Compare browser CSV reports with the Python API on exported original logs."""

import argparse
import json
from pathlib import Path
import subprocess

from web_service import ModelDoctorService


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    examples = json.loads((ROOT / "web/dist/examples/index.json").read_text(encoding="utf-8"))
    examples.append({"file": "template.csv", "architecture": "unknown"})
    service = ModelDoctorService(ROOT / "models" / "contract-test-no-adapter")
    for example in examples:
        csv_text = (ROOT / "web/dist/examples" / example["file"]).read_text(encoding="utf-8")
        javascript = "import('./web/dist/csv.mjs').then(async ({analyzeCsv}) => { const fs = await import('node:fs'); console.log(JSON.stringify(analyzeCsv(fs.readFileSync(0,'utf8'),process.argv[1]))); });"
        completed = subprocess.run([args.node, "-e", javascript, example["architecture"]], input=csv_text,
                                   text=True, encoding="utf-8", capture_output=True, check=True, cwd=ROOT)
        browser = json.loads(completed.stdout)
        python = service.analyze(csv_text, example["architecture"])
        if python != browser:
            raise ValueError(f"Browser and Python disagree on {example['file']}")
        print(f"PASS: browser/Python report equivalence: {example['file']}")


if __name__ == "__main__":
    main()
