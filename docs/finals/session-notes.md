# Finals session notes (gotchas for the next sessions)

Written at the end of Phase 0 (correctness pass, 2026-10-11). Prompt A (3-D Digital Twin) and
Prompt B (Bolt Designer) build on this. Keep these true.

## How health and predictions reach the UI now

- `GET /fleet` rows (`FleetAsset`): `health_score` = 100 × (1 − `failure_probability`) from the
  newest row in the `predictions` table for that asset, with `prediction_ts` (plant time of the
  scored window) and `model_version`. **`health_score` is `null` when there is no prediction**
  and the UI must say "no prediction yet". There is no fallback of any kind.
- Predictions come from two writers into the same table: the edge container (`source="edge"`,
  every MQTT row once its 12-row buffer is full) and the control plane (`source="api"`:
  `POST /pipeline/score` for the whole fleet, and every "Run analysis" for its asset).
  `apps/api/scoring.py` is the one place that scores with the production model centrally.
- `regime` on fleet/asset responses is **inferred** from the latest rpm/load against the
  production model's regime centroids (`apps/api/routes/fleet.py::infer_regimes`,
  `agents/ml_architect/features.py::assign_regime_ids`). It is `null` until a model is in
  production. The simulator's `regime` column is never read for display.
- `GET /assets/{id}`: `latest` is an `ObservedTelemetryRow` (eight sensors only), plus `regime`,
  `prediction`, `prediction_ts`, `model_version`, `health_score`, `thresholds`.
  `GET /assets/{id}/telemetry` returns `ObservedTelemetryRow[]`.
- `tests/test_ground_truth_isolation.py` fails if `health` or `failure_within_h` appears in any
  of those responses or in a published replay row. Keep it passing; extend it for new routes
  (a 3-D twin endpoint must go through the same scrub).
- Thresholds for any chart: `/assets/{id}.thresholds` and `/system/health.thresholds`
  (`classification` 0.5, `early_warning` 0.3, `sustained_rows` 3). Status encoding for the twin:
  healthy / warning / critical should key off `failure_probability` against those two values,
  never off brand colours.

## Run modes

- `POST /pipeline/run` defaults to `mode: "analyze"`: load production model → explain → cost →
  Decision Contract → approval gate. Seconds. 409 when no production model exists.
- `mode: "train"`: profile (split fixed first) → train → validate (+ parity) → register → stage
  `trained`. No contract. The promotion request is then decided in ModelOps.
- `GraphState.mode`, `GraphState.model_ref` (analyze) and `GraphState.split` (train) are in the
  run JSON; `RunSummary` has `mode` and `model_version_used`. The web stepper is mode-aware.

## Two decisions, never one

- `POST /approvals/{contract_id}/decision` records the maintenance decision; `finalize` only
  sets the stage and logs "promotes_model: false". The replay's simulated CMMS acts on it.
- `POST /promotions/{id}/decision` is the only path to production (`decide_promotion_db`). It
  needs a named approver; `lifecycle.promote()` refuses `validation_passed != True` and
  `parity_passed == False` (registry payload). Reject works now (it used to crash).

## Training facts the slides must match

- Numbers: `docs/finals/metrics.md` (leak-free run v57257: recall 0.865, precision 1.0, lead
  time 41.5 h, IMS 0.844) and `docs/finals/model-card.md`. The record is
  `docs/finals/prep-report.json`. Never quote the prototype's 91.5 % / 43.7 h / 0.858.
- Split for 24 motors: train 12 / validation 6 / calibration 6 (same motors as validation,
  documented) / test 6, seed 0, MTR-042 forced into test. Four completed failures only.
- Right-censoring: the last 48 h of a motor with no failure ahead has no label.
- Calibration: isotonic on LightGBM log-odds. The sidecar carries `"input":
  "decision_function"` and the edge takes the logit of the ONNX probability first. The parity
  gate caught this on the first real prep run (max |Δp| 0.90); it is fixed and tested for all
  three families.

## Demo prep and reset

- `scripts/demo_prep.py` talks only to the API (train run, promotion approval as
  `demo-prep@thor`, `POST /pipeline/score`). Compose runs it as the one-shot `prep` service;
  `edge` and `replay` wait for it (`service_completed_successfully`), so a plain
  `docker compose up` ends with a promoted model and a scored fleet before anything streams.
- `scripts/demo_reset.ps1` / `.sh` use the root-level compose project (`thor-main`) and follow
  the prep logs; the old scripts targeted project `thor`.
- Natively: run uvicorn on port 8010 with scratch `DATABASE_URL`/`MODELS_DIR`/`CHROMA_PATH`
  (see CLAUDE.md §16), then `python -m scripts.demo_prep --api http://127.0.0.1:8010`.
- At the 80 % seed point MTR-021 is already deep in its degradation (it fails live at 85 %), so
  it scores p ≈ 1.0 immediately after prep. That is correct and a good story ("the old way":
  nobody acted on it), not a bug.

## Things to be careful with

- `PipelineStage.trained` is terminal; any new poll loop must include it (the e2e test lost
  four minutes to a stale terminal set).
- `tests/test_orchestrator.py::install_fake_toolbox(production=True)` fakes a promoted model so
  analyze runs work in unit tests; pass `production=False` to test the 409 path.
- `edge/` must never import `apps` or `agents` (`tests/test_edge.py::test_edge_never_imports_control_plane`).
- The Bolt eval (`evals/bolt`) now seeds one prediction for MTR-042 because fleet answers
  without a prediction honestly say "none scored yet". 53/53 pass in template mode.
- `docs/DEMO.md` shot list was corrected; `docs/BUILD_STATUS.md` still holds prototype numbers
  and is marked superseded in DEMO.md.
