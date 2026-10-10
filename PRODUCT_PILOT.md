# ModelDoctor product pilot

## The offer today

**Collect training logs automatically, keep runs together, and investigate
changes with curves, settings and explicit evidence.** Start with smaller teams
running epoch-based experiments in Python. The pilot runs on their own machine;
their logs do not need to be sent to a cloud service.

The useful workflow is training → background collection → saved run → checks →
comparison with the next run. The product currently provides automation and a
focused investigation workflow. A commercial advantage for the learned
classifier has **not** been demonstrated yet.

We can demonstrate automatic collection, durable history, configuration
comparison, versioned checks, CSV/JSON export and a prepared adapter connection.
We cannot yet claim production diagnosis accuracy, correct root causes, working
repair recommendations, quantified GPU savings or support for every model type.
Deep4ge labels describe injected mutations; a baseline label is not a guarantee
of healthy training. Rule checks and demo answer keys are labelled separately
from predictions.

## Try the workflow in five minutes

```powershell
python web_server.py --disable-model
```

In another terminal:

```powershell
python examples/collect_run.py
```

Open http://127.0.0.1:8000 → **Körningar**. The onboarding run is explicitly
synthetic and has no model answer key. Inspect it, collect another run, select
two runs and compare. Server history is in `runs/modeldoctor.sqlite3`; this
directory is ignored by Git. The hosted private demo instead saves manually
selected runs in the current browser's IndexedDB. It cannot receive Python logs.

## Add collection to an existing training script

Install the independent client, with no PyTorch/Transformers dependencies:

```powershell
python -m pip install ./sdk
```

Alternatively, download `web/dist/downloads/modeldoctor_client.py` and place it
beside the script. Both versions expose the same API.

```python
from modeldoctor_client import Run

with Run("cnn-lr-001", project="image-classifier", architecture="CNN",
         dataset="images-v2 / validation-split-42", metric_name="accuracy",
         config={"learning_rate": 0.001, "batch_size": 32, "seed": 42}) as run:
    for epoch in range(num_epochs):
        # Keep your existing train and validation steps here.
        # Pass the Python scalars they already calculate.
        run.log(epoch, train_loss=train_loss, val_loss=val_loss,
                train_acc=train_accuracy, val_acc=validation_accuracy,
                signals={"adjusted_lr": learning_rate})
```

The client logs once per epoch, makes no model calls, installs no hooks and never
changes the model, optimizer, seed, gradients, checkpoint or CUDA settings.
Network requests run in a separate thread with a timeout; failed delivery leaves
a replayable local snapshot in `runs/outbox`. Logging still does local disk work
and rewrites the current snapshot; it is not zero-overhead. Do not log each batch
or pass GPU tensors just to collect metrics. Use scalars you already calculate.
The existing training-loop operations remain under your control, as illustrated
in the [official PyTorch training tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

Each run has a new ID. It streams only CSV measurements and the metadata/config
you explicitly provide. It never uploads model weights, source code, environment
variables or the dataset itself. Avoid putting credentials in the config.

The context manager marks normal completion as `completed` and propagates
training exceptions while recording `failed`. A hard process termination can
leave a run `running`; replay does not invent completion. Failed/running runs
remain viewable but do not receive model classification.

Replay retained snapshots after connectivity is restored:

```powershell
python watch_logs.py runs/outbox --replay
```

Repeated deliveries are idempotent. Older revisions do not overwrite newer
ones; previous epochs and terminal runs cannot be changed. Start a new run when
resuming a training job with a changed configuration.

## Existing CSV logs without code integration

```powershell
python watch_logs.py logs --project my-project --architecture CNN --dataset "images-v2 / split-42"
```

After the training process closes `run.csv`, create the empty marker
`run.csv.done` beside it. The watcher sends only marked files and retries failed
requests. No completion is inferred from a pause between writes. `--once` scans
once. The watcher requires `epoch` and `train_loss`, expands absent curve/signal
columns as missing measurements, and validates the log on the server. It never
fabricates the 23 internal measurements required by the classifier.

## What checks mean

Version `inspection-1` flags validation loss ≥15% above its recorded minimum
when at least five values and two later epochs exist; a last-five-epoch train
loss range <1% of the first value after at least eight epochs; and three logged
max-gradient values equal to zero. It also points out missing validation and
missing measurements. These explicit thresholds are initial heuristics, not
trained or calibrated probabilities. Best-validation epoch and all triggering
values are included in the report. Suggestions are investigation steps, not
validated fixes. Live checks describe partial curves; live diagnosis has not
been evaluated.

Comparison shows validation curves and changed configuration keys. Different or
unspecified datasets produce a warning. The user must still confirm the same
validation split and loss definition; neither a lower loss nor changing one
setting establishes causation.

## Prove a classifier advantage after CUDA training

Training is unchanged. Keep the site in `--disable-model` mode during training
so it cannot load Llama or allocate GPU memory. No training requirement or
dependency was added to `requirements-gpu.txt`.

1. Train on the prepared training split using the existing `train.py` commands.
2. Evaluate the **complete validation split**, without `--max-eval-examples`.
3. Compare saved predictions with the existing fixed CPU reference:

   ```powershell
   python assess_product.py --predictions results/deep4ge/finetuned_validation_predictions.csv
   ```

   Use your actual results directory. This command loads no weights, runs no
   training, and does not read the final test. It requires every validation ID
   exactly once and verifies source groups, labels and the reference's dataset
   hash. It reports macro F1, per-category results, baseline flag rate, missed
   mutation rate and an exploratory bootstrap over source programs. A recorded
   advantage requires higher macro F1, all outputs parsed and no higher baseline
   flag rate than the CPU reference. This is a comparison criterion, not a
   guarantee of production quality; eight validation source programs provide
   limited evidence.
4. Once model choices are frozen, perform the existing explicit final test once.
5. Test independent real-user logs with human-reviewed targets, including runs
   with no known fault, absent telemetry and model/data types outside Deep4ge.
   Benchmark against the customer's present workflow and the rule checks alone.

Start the website server without `--disable-model` after training. Saved,
completed runs receive inference only when requested. Curve/LR-only logs abstain
from learned classification. Additional measurements must match Deep4ge's
definitions, such as **mean of per-layer gradient norms** for `mean_gradient`;
arbitrary PyTorch gradient summaries are not interchangeable. Measurement
compatibility and accuracy on the customer's logs still require evaluation.
Predictions record SHA-256 of adapter weights/config. Feedback is stored for
manual review and never changes targets or feeds training automatically.

## Run a customer pilot before a broad launch

Recruit one or two teams with an identifiable training-debugging problem. Agree
on success measures before testing: setup time, logging overhead, time to a
correct investigation, useful/incorrect flags and whether they would pay for
continued use. Review outcomes with an ML practitioner. Keep timings and GPU
cost savings as measurements, not marketing estimates. Use the same tasks and
data to compare their existing process, ModelDoctor's rules, and rules plus the
trained classifier; this separates the model's contribution from the UI's.

The current server is a **local, single-user pilot**, not a multi-tenant service.
External binding requires `MODELDOCTOR_API_TOKEN`; the client and the UI support
this key without storing it in source or downloaded logs. Any internet offering
also needs HTTPS, user identity, project permissions, quotas and operational
backup/retention policies. A hosted GPU API is still separate from static Sites
hosting. Keep the existing demo private until you choose to invite pilot users.
