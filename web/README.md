# ModelDoctor website

The website starts with an introduction, optional demo, then your own training.
Rounded cards and pill controls use soft mint, peach and lavender accents. All assets are local:
HTML, CSS, JavaScript and SVG. No frontend build or npm installation is needed
to run the app. The source stays in this repository under `web/dist/`.

From the repository root:

```powershell
python web_server.py --disable-model
```

Open http://127.0.0.1:8000. The server binds to localhost by default. The static
site can also run without Python; CSV parsing and curves then run in the browser.
The local server serves the same interface and adds the model API.

CSV selection validates the file and displays loss, source metrics, gradient
and learning-rate curves. Toggle series, show the last ten epochs, select an
epoch with a click or keyboard-accessible slider, and jump to the lowest
recorded validation loss. The analysis button calls Python in local mode. Files are processed in
memory unless **Spara körning** is selected. Saved local-server runs are persisted
in `runs/modeldoctor.sqlite3`; automatic client snapshots are stored in
`runs/outbox`. Both are ignored by Git and separate from training data/adapters.
In the hosted static demo, manually saved history uses browser IndexedDB. Downloads
and report exports work in both modes. Neither static mode nor Sites receives
Python training updates. See [PRODUCT_PILOT.md](../PRODUCT_PILOT.md) for setup.

Upload accepts at least `epoch,train_loss`, with `val_loss` recommended. A simple
three-column example template is linked at the upload step. `normalizeUploadCsv`
expands missing observed columns as empty fields, preserving actual zero values
and rejecting malformed rows; unknown annotation columns do not become inputs.
Full original logs remain unchanged. The API's existing 28-column schema is
unchanged. In the UI, architecture and detailed telemetry are optional advanced
context, rather than prerequisites to see a graph.

The **Körningar** view filters projects, displays live collected snapshots and
compares two validation curves plus changed configuration keys. It warns about
different/unspecified datasets and cannot establish causal effects. Inspection
checks expose their evidence and explicit heuristic thresholds, independently
of the classifier. Missing measurements stay missing. Feedback on saved model
predictions is kept for manual review and is never fed into training automatically.

The first demo uses three clearly labelled **illustrative scenarios**, generated
only for the interface by `demo.mjs`. Play/pause progressively reveals each
scenario's 24 rounds; the displayed assessment covers the full scenario, as
stated beside playback. No model inference is requested and no mutation label
is assigned. They do not enter training or evaluation data. Switching back to
your own training restores the last open log in the current page session.

Three original public Deep4ge training logs remain available in technical details.
Published mutation labels appear only there and separately in report exports,
never as the customer's diagnosis. Uploaded logs inherit no example target.
Attribution is in `dist/examples/ATTRIBUTION.md`.

## Connect the trained model

Finish training with `python train.py`, then start/restart the website server without
`--disable-model` and
refresh the page. The service detects `models/modeldoctor_deep4ge`, loads its
adapter on the first analysis and reuses it for subsequent requests. It uses
`fault_data.log_to_input`, `train.load_saved_adapter` and `train.predict_rows`,
the same converter and inference path as the CLI. Model requests are serialized.

```powershell
python web_server.py --adapter-dir models/modeldoctor_deep4ge
python web_server.py --cpu --no-quantization
```

Use `--disable-model` while training so the service does not load weights or
allocate GPU memory. Collection never triggers inference, and saved running/failed
runs cannot receive model classification. Curve/LR-only logs abstain from learned
classification; compatible internal training measurements are also required.
Each prediction records weight/config SHA-256 in `model_provenance`.

The second command explicitly uses slow CPU inference. The demo UI works before
training; no Llama weights are downloaded just to open the website.

For future internet inference, host the Python API behind the website's own
authenticated HTTPS origin. The hosted static version currently performs
browser-only data inspection. Sites hosting does not run this local Python
model server. `dist/config.js` reserves an API-origin setting for a later
authenticated deployment. The local server overrides that file to enable its
same-origin API. Cross-origin browser writes are rejected by the local server.
The server is a single-user local pilot. Non-loopback binding requires
`MODELDOCTOR_API_TOKEN`; authenticated callers send `Authorization: Bearer ...`.
The UI accepts the token through **Koppla in träningen** and keeps it in page
memory. This does not replace multi-user identity/permissions or HTTPS for an
internet product.

## API contract (version 1)

`GET /api/status` returns model availability, mode, required columns, history
availability, inference enabled/disabled and whether API authentication is required.

`POST /api/analyze` accepts `Content-Type: application/json`:

```json
{
  "api_version": 1,
  "architecture": "RNN",
  "csv": "epoch,train_loss,...\n0,1.0,..."
}
```

The example above abbreviates the CSV. Use all 28 columns listed in the template:
`epoch`, four curves and 23 dynamic measurements. Filenames, targets and source
identifiers are never passed to the model. Limits: 5 MB of CSV and 10,000 epochs.

The response contains `api_version`, `architecture`, `num_epochs`, `epochs`,
`curves`, `signals`, `data_quality`, `final`, `mode`, `model`, `diagnosis` and
`warnings`. Missing/nonfinite measurements stay JSON `null`. `mode` is `preview`
until inference succeeds, then `llama`. `diagnosis` is either `null` or one of
the eight supported categories. No confidence or invented repair recommendation
is returned. The additive `inspection` field contains versioned pattern checks,
best validation epoch, evidence and suggestions to investigate. These are distinct
from a learned diagnosis. Errors use `{"error": "..."}` and an HTTP error status.

## Investigation guides

`assessment.mjs` produces a readable result from logged patterns, independently
of classifier output. It does not map a mutation label to overfitting or a repair.
For example, falling training loss plus validation regression is described as
diverging development, with overadaptation mentioned as a possibility to check.
Fewer than five rounds or missing endpoints cannot certify an improving trend.
The classifier's raw category is confined to the technical-details dialog.
`guidance.mjs` provides human-written checks with a reason, an
inspection step and a follow-up, keyed by observations and the selected area.
These are not generated explanations or validated repairs. Unknown task type
is the default; classification/regression selections adapt the guide, not the
classifier input. Loss guides conditionally describe PyTorch loss contracts and
link to official documentation. Hyperparameter guides request actual settings
and controlled comparisons rather than guessing a learning-rate direction.

Users may select an area themselves before an adapter is trained; this is
labelled as their choice and creates no prediction. A published example target,
learned proposal, user choice and observed pattern retain their separate sources.
Checklist marks mean reviewed, not repaired; they reset on a new log or changed
guide context and last for the current page session. JSON exports include the
versioned `assessment`, `investigation` guide and reviewed IDs separately from `diagnosis` and
`published_example_target`. The Python API and training targets are unchanged.

### Saved runs

`POST /api/runs` collects a snapshot and **never** invokes the model:

```json
{
  "api_version": 1,
  "run_id": "unique-run-id",
  "revision": 1,
  "name": "experiment-01",
  "project": "my-project",
  "architecture": "CNN",
  "status": "running",
  "dataset": "images-v2 / split-42",
  "metric_name": "accuracy",
  "config": {"learning_rate": 0.001},
  "csv": "epoch,train_loss,...\n0,1.0,...\n"
}
```

It returns `{ "accepted": true, "run": { ... } }`. Repeated identical revisions
are idempotent, older revisions are ignored, and changed data at the same revision
is rejected. New revisions may append epochs and finalize to `completed`/`failed`;
run identity/config and previous epochs remain immutable. Terminal runs cannot
be replaced. Start a new ID for a new/resumed job.

- `GET /api/runs`: latest 200 run summaries.
- `GET /api/runs/<id>`: full saved CSV, metadata, report and input hash.
- `POST /api/runs/<id>/analyze` with `{}`: explicit inference on a completed run.
  A result cannot replace a newer revision.
- `POST /api/runs/<id>/feedback`: `{ "verdict": "useful|incorrect|unsure", "note": "optional" }`.
  Requires a model result; kept separately from inputs/targets.
- `DELETE /api/runs/<id>`: deletes server history, not client outbox copies.

No uploaded config, filename, project, target or source identifier enters the
classifier input. No model weights or dataset itself are collected by the SDK.

## Checks

```powershell
python -m unittest discover -s tests -v
node --test web/tests/csv.test.mjs web/tests/pilot.test.mjs web/tests/guidance.test.mjs web/tests/onboarding.test.mjs
python verify_web_contract.py
```

The first command also checks the actual HTTP contract and model adapter wiring
with a mocked model. The JavaScript tests check CSV validity and target isolation.
The last command requires Node and compares browser reports with Python for all
three original demo logs and the missing-value template. Python-only app use
does not require Node. Regenerate the public examples with
`python prepare_web_examples.py`.

The interface was checked in headless Edge at desktop and mobile sizes, including
file selection, invalid uploads, demo switching, all chart tabs, report export,
history persistence, comparison, client download, deletion and live snapshots.
Guide checks also cover task selection, reviewed-state reset, source separation,
curve visibility, best-epoch navigation and keyboard epoch selection. Responsive
layout was checked at 360, 390, 620, 768, 1024, 1280 and 1440 pixels. The onboarding
checks cover the initial explanation, all three demos, play/pause, simple log
normalization, upload preservation when changing mode and zero demo inference.
Real Llama inference still requires the trained adapter and target hardware.
