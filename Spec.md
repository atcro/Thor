# Thor — System Specification (reverse-engineered blueprint)

> Derived from the codebase on `main` as of 2026-09-13 (commit `4e8c6a3`). Where this file
> and the code disagree, the code wins and this file is stale — fix it in the same PR.
> `CLAUDE.md` holds positioning and rules; `docs/INTERFACES.md` holds the function-level build
> contract; `docs/BUILD_STATUS.md` holds measured results. This document is the *what*: the
> requirements the system satisfies, stated so they can be checked.

---

## 0. One-line definition

Thor turns raw industrial sensor telemetry into a validated, explainable, cost-based,
human-approved predictive-maintenance decision, recorded as an immutable Decision Contract
that hands off to the plant's existing MES/CMMS. Thor is a companion to the MES, never an MES.

## 1. Actors

| Actor | Role |
|---|---|
| Plant / reliability engineer | Opens the UI, runs an analysis, reads the evidence, **approves or rejects**. The only actor that can cause an action. |
| MQTT replay / plant historian | Publishes telemetry rows to `telemetry/{asset_id}`. |
| Edge inference container | Consumes telemetry directly from MQTT, scores with the deployed ONNX model, publishes `predictions/{asset_id}`. |
| Orchestrator (the single LLM-backed agent) | Chooses the next pipeline step and drafts the explanation prose from computed evidence. Never computes a number. |
| Bolt (copilot) | Read-only conversational access to the same evidence. Cannot approve. |

## 2. System context and services

`docker compose up --build` (root `compose.yaml` → `infra/docker-compose.yml`) starts:

| service | image / build | port | responsibility |
|---|---|---|---|
| `timescale` | `timescale/timescaledb:latest-pg16` | 5432 | PostgreSQL + TimescaleDB; `telemetry` hypertable + all relational tables |
| `mqtt` | `eclipse-mosquitto:2` (`streaming/mqtt/mosquitto.conf`) | 1883, 9001 (ws) | anonymous broker, no persistence |
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.1.0` | 5000 | experiment tracking + registry, artifacts on volume `mlflow-data` |
| `api` | `apps/api/Dockerfile` | 8000 | FastAPI control plane + LangGraph orchestrator (one process) |
| `edge` | `edge/Dockerfile` | 8001 | ONNX Runtime inference, subscribes to MQTT directly; **no imports from `apps/` or `agents/`** |
| `replay` | `streaming/replay/Dockerfile` | — | time-accelerated historical replay over MQTT (HTTP fallback) |
| `web` | `apps/web/Dockerfile` (node build → nginx) | 5173→80 | React SPA, five screens |

Volumes: `timescale-data`, `mlflow-data`, `models` (shared by `api` and `edge`: ONNX + sidecar),
`chroma-data` (manual index). Local dev runs everything against SQLite with no containers.

### 2.1 Configuration (`apps/api/settings.py`, env / `.env`)

| variable | default | meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./thor.db` | SQLAlchemy URL; compose sets `postgresql+psycopg://thor:thor@timescale:5432/thor` |
| `MLFLOW_TRACKING_URI` | `./mlruns` | file store locally (needs `MLFLOW_ALLOW_FILE_STORE=true` on MLflow 3), server URL in compose |
| `MQTT_BROKER_URL` | `mqtt://localhost:1883` | |
| `CHROMA_PATH` | `./.chroma` | persistent Chroma index |
| `MANUALS_DIR` | `data/manuals` | corpus of `*.md` / `*.txt` / `*.pdf` |
| `MODELS_DIR` | `./models` | candidate joblibs (`candidates/`, `artifacts/<run_id>/`) and deployed `*.onnx` + `*.json` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | `""`, `claude-sonnet-5` | read **only** in `apps/api/orchestrator.py`; blank = template mode |
| `SEED_ON_START` | unset (`1` in compose) | generate the synthetic fleet if `data/simulator/out/` is missing |
| `SEED_FRACTION` | `0.80` | fraction of the timeline bulk-loaded at startup |
| `REPLAY_SPEED`, `REPLAY_START_FRACTION` | `1800`, `0.80` | replay acceleration and where the first pass starts |
| `THOR_RAG_EMBEDDING` | `default` | `hashed` forces the offline embedding (tests) |
| `THOR_SKIP_BACKGROUND` | unset | `1` skips MQTT subscriber + RAG index thread (tests) |
| `cost_planned_maintenance` … `downtime_unplanned_h` | 4200 / 18500 / 3100 / 3 h / 14 h | plant-cost assumptions — configurable, never presented as fact |
| edge: `EDGE_MQTT_ENABLED`, `EDGE_BUFFER_ROWS`, `EDGE_MODEL_POLL_S`, `EDGE_MQTT_RETRY_S`, `CONTROL_PLANE_URL` | | edge container tuning |

## 3. Data model

### 3.1 Telemetry (unit of ingest; MQTT payload = `TelemetryRow.model_dump_json()`)

`asset_id, ts (tz-aware UTC ISO-8601), vibration_rms, vibration_kurtosis, vibration_crest,
bearing_temp_c, motor_temp_c, current_a, rpm, load_pct, regime?, health?, failure_within_h?`

One row per asset per 10 minutes. `health` (1→0) and `failure_within_h` are simulator ground
truth: allowed **only** to build training labels, never as features.

### 3.2 Relational tables (`apps/api/db.py`; SQLAlchemy Core, dialect-agnostic)

| table | key | notes |
|---|---|---|
| `assets` | `asset_id` | name, site, line, type, rated_kw, criticality |
| `telemetry` | (`asset_id`, `ts`) | hypertable on Postgres; duplicate rows ignored |
| `predictions` | id | edge/orchestrator predictions per asset |
| `pipeline_runs` | `run_id` | serialized `GraphState` JSON, updated after every node (mutable) |
| `data_quality_contracts`, `validation_reports`, `drift_reports` | id | JSON payloads of the agent outputs |
| `decision_contracts` | `contract_id` | **insert-only**; JSON payload = `DecisionContract`; `evidence_hash` = sha256 of the evidence bundle |
| `approvals` | `approval_id` | **insert-only**; FK → contract or promotion; decision, approver, note, time |
| `model_registry` | (`name`, `version`) unique | stage enum `candidate → validated → shadow → production → archived`, `onnx_path` |
| `promotion_requests` | `promotion_id` | from/to stage, status, comparison payload |

Contract status is *derived* (`pending` until an approvals row exists) — never stored, never
updated in place.

### 3.3 Contracts between agents (`apps/api/schemas.py`)

`DataQualityContract` → `CandidateSet` (`TaskSpec`, `FeatureSpec` incl. per-regime `baseline`,
`CandidateModel[]` with `IndustrialModelScore`) → `ValidationReport` (`LeakageReport`,
`BacktestFold[]`, `CalibrationReport`, `LeadTimeReport`, `RULInterval`) → `EvidenceBundle`
(`Explanation` with `ShapFeature[]`, `ManualPassage[]`, `CostComparison`, `MaintenanceWindow`)
→ `DecisionContract` → `Approval`. Orchestrator state = `GraphState` (stage, events, all of the
above, `llm_calls`, `error`).

## 4. Pipeline (functional requirements)

Stages are LangGraph nodes over `GraphState`; each persists state + a `StageEvent` before the
next runs. Stage names: `queued → profiling → training → validating → explaining →
awaiting_approval → approved | rejected | failed`.

### 04 Data Reliability — `agents/data_agent/profiling.py`
- **R4.1** `validate_schema`: required columns, units/ranges, duplicate timestamps → `SchemaIssue[]`.
- **R4.2** `detect_missingness`: per-column missing fraction, gap windows, flat-lined sensors.
- **R4.3** `detect_regime`: KMeans (k=3) on standardised `rpm`, `load_pct`; regimes named R1..R3 ascending by rpm×load; per-row assignment; silhouette on a ≤5000-row sample.
- **R4.4** `profile_dataset`: `quality_score` 0–100 minus penalties; `trainable` iff score ≥ 60 and ≥ 500 rows and ≥ 4 assets. A non-trainable contract stops the run.

### 05 ML Architect — `agents/ml_architect/{features,automl}.py`
- **R5.1** Feature pipeline: rolling window of 12 rows (2 h) per asset producing exactly
  `vib_rms_mean, vib_rms_slope, vib_kurt_mean, vib_crest_mean, bearing_temp_mean, bearing_temp_slope, temp_delta, current_mean, rpm_mean, load_mean, vib_rms_z, bearing_temp_z, vib_rms_resid`.
  z-scores/residuals are against the **per-row regime's healthy baseline** (first 20 % of each asset's rows), so a shift change is never a fault.
- **R5.2** `compute_features_np` is the numpy reference the edge container mirrors (`edge/features.py`); parity is tested to 1e-9.
- **R5.3** Label = `failure_within_h ≤ horizon_h` (default 48 h) and **only for assets whose failure completes inside the data window**; in-progress degradations are unknown.
- **R5.4** `infer_task` → classification (rationale recorded).
- **R5.5** `launch_trial` per family (random_forest, xgboost, lightgbm): Optuna TPE, seeded, objective = IMS on an asset-grouped, time-ordered holdout (last 25 % of sorted asset ids). Never a random shuffle. Params/metrics logged to MLflow best-effort; artifact saved via joblib.
- **R5.6** Industrial Model Score = 0.35·recall + 0.25·min(1, lead_time_h/72) + 0.15·precision + 0.15·(1 − min(1, brier/0.25)) + 0.10·(1 − min(1, latency_ms/50)).
- **R5.7** `train_candidates` runs all three families and returns `CandidateSet.ranked`.

### 06 Validation — `agents/validation/checks.py`
- **R6.1** `detect_leakage`: forbidden columns (`health`, `failure_within_h`, …), |corr(feature,label)| > 0.98, asset overlap, temporal overlap.
- **R6.2** `run_backtest`: 3 asset-level folds, round-robin, time-ordered; recall/precision/f1/auroc/brier per fold; undefined metrics omitted (JSON-safe).
- **R6.3** `calibrate_probabilities`: isotonic (≥ 200 rows) else sigmoid, on held-out assets; Brier before/after; reliability bins.
- **R6.4** `compute_lead_time`: hours from the first sustained alarm (3 consecutive rows with P ≥ **0.3**) to failure, per completed held-out failure; median/p10/p90.
- **R6.5** `estimate_rul_interval`: quantile GBM p10/p50/p90 on the target asset.
- **R6.6** `validate` picks the champion by IMS recomputed from backtest means, saves `<candidate>_calibrated.joblib`, records `champion_artifact=<path>` in `notes[0]`; `passed` iff leakage ok ∧ recall ≥ 0.6 ∧ median lead time ≥ 12 h. A failed gate does **not** stop the run — it is surfaced as "human review required".

### 07 Reliability — `agents/reliability/{explain,rag,cost,contract}.py`
- **R7.1** `explain`: SHAP for the latest row; TreeExplainer with an additivity check, falling back to a model-agnostic explainer over the tree model with a stratified background (healthy fleet rows + the asset's recent rows). Output is structured `ShapFeature[]`, never prose.
- **R7.2** `retrieve_manual_context`: Chroma persistent index over `data/manuals`, local embedding (MiniLM via onnxruntime; deterministic hashed bag-of-words fallback offline); builds the index if missing; one passage per (source, section); returns `ManualPassage` with source/section/score. Retrieval never raises.
- **R7.3** `calculate_failure_cost`: hazard `P(fail by h) = 1 − (1 − p_now)^(1 + h/lead)` (lead floored at 12 h); expected cost of maintain_now / maintain_later (next planned window, 72 h) / run_to_failure from the configured plant costs; `recommended = argmin`; assumptions echoed in the output.
- **R7.4** `find_maintenance_window`: earliest low-load window (02:00–05:00 UTC) ending before P(fail) exceeds 0.3, else now.
- **R7.5** `run_whatif`: re-score with overridden feature values.
- **R7.6** `create_decision_contract`: id `dc_<asset>_<UTC timestamp>`, `evidence_hash` = sha256(bundle JSON), inserted once. `template_explanation` = deterministic 4–6-sentence prose citing only numbers and sections present in the bundle.

### 03 Orchestrator — `apps/api/orchestrator.py`
- **R3.1** Only module that reads `ANTHROPIC_API_KEY` (guarded by a test).
- **R3.2** At most two LLM calls per run: (1) `route()` — a tool-choice call restricted to the fixed node names, unable to skip `validate` or `await_approval`; (2) `draft_explanation()` — prose from the evidence JSON with a system prompt forbidding new numbers/citations. Any failure or blank key → deterministic route / template prose. `llm_calls` recorded in state.
- **R3.3** Runs execute in a background thread; `GraphState` persisted after every node; node exceptions → `stage=failed`, `error` set.
- **R3.4** The graph interrupts before `finalize`; `resume(run_id, approval)` continues only with a recorded `Approval`; `finalize` raises `PermissionError` without one.
- **R3.5** After validation the champion is registered (`stage=candidate`) and a `PromotionRequest` to `production` is created; on contract approval the request is promoted and the model exported to ONNX (best effort, failures become events).

### 08 MLOps — `agents/mlops/{lifecycle,drift}.py`
- **R8.1** `register_model` (idempotent on name+version; git sha, dataset hash; MLflow best effort), `compare_champion`, `request_promotion`, `list_models`, `get_production_model`.
- **R8.2** `promote(promotion_id, approval)` raises `PermissionError` unless `approval.decision == "approved"` and `approval.promotion_id` matches; enforces stage order; archives the previous production model. `reject_promotion` marks the request rejected.
- **R8.3** `deploy_edge`: ONNX via skl2onnx (RF) / onnxmltools converters registered with skl2onnx (XGB, LGBM); calibration wrappers unwrapped (`calibration: "none"`); sidecar `{model_name, version, features, window_rows, calibration, family, deployed_at, baseline}`; verified with onnxruntime.
- **R8.4** `detect_drift`: PSI per feature (10 quantile bins), `drifted` if PSI > 0.2; report persisted.

### 09 Human Approval Gate — `apps/api/routes/approvals.py`
- **R9.1** `GET /approvals/{contract_id}` presents the contract, its evidence bundle, and derived status.
- **R9.2** `POST /approvals/{contract_id}/decision` inserts an `Approval` and resumes the run; a second decision for the same contract → **409**. Same shape for `POST /promotions/{id}/decision`.

## 5. Interfaces

### 5.1 Control-plane API (`apps/api`, all at root, CORS open)

| method | path | purpose |
|---|---|---|
| GET | `/health`, `/system/health` | liveness; db/mqtt/mlflow/edge status, row counts, last ingest |
| POST | `/ingest` | one `TelemetryRow` or a list |
| GET | `/fleet` | `FleetAsset[]` sorted by risk (health = 100·(1−p) if a prediction exists, else ground-truth health, else a vibration z-score proxy) |
| GET | `/assets/{id}`, `/assets/{id}/telemetry?hours&limit`, `/assets/{id}/predictions` | asset detail, series |
| POST | `/predictions` | edge → control plane |
| POST | `/pipeline/run` | `{asset_id, horizon_h, n_trials}` → `{run_id, stage}` |
| GET | `/pipeline`, `/pipeline/{run_id}` | run summaries; full `GraphState` |
| WS | `/ws/runs/{run_id}` | `GraphState` every 1 s until terminal |
| POST | `/whatif` | `{run_id, scenario}` → `WhatIfResult` |
| GET | `/approvals/pending`, `/approvals/{contract_id}` | gate views |
| POST | `/approvals/{contract_id}/decision`, `/promotions/{promotion_id}/decision` | record a decision |
| GET | `/models`, `/models/promotions`, `/models/drift` | registry views |
| POST | `/models/drift/check` | run PSI for an asset |
| POST | `/copilot/chat` | Bolt; `{reply, tool_calls, source}` |

### 5.2 MQTT topics
`telemetry/{asset_id}` (replay → api, edge) · `predictions/{asset_id}` (edge → subscribers).

### 5.3 Edge service (`edge/main.py`, :8001)
`GET /health` (model version, assets seen, uptime) · `GET|POST /predict` · `GET /features` ·
`POST /ingest` · `POST /models/refresh`. Loads the newest `*.onnx` + sidecar from
`MODELS_DIR`; without a model it stays healthy and silent.

### 5.4 CLI entrypoints
`python -m data.simulator.generate [--days --seed --step-min --out]` ·
`python -m streaming.replay.replay [--speed --once --transport --start-fraction --asset]` ·
`python -m data.benchmark.ai4i_benchmark [--csv --n-trials --seed --out]` ·
`uvicorn apps.api.main:app` · `npm --prefix apps/web run dev|build|test`.

## 6. Frontend (`apps/web`, React 18 + TypeScript + Vite + recharts)

| route | screen | must show |
|---|---|---|
| `/` | Fleet | risk-sorted asset cards (health pill ≥85 green / 60–85 amber / <60 ember), regime badge, open-contract flag, System panel (db/mqtt/mlflow/edge) |
| `/assets/:id` | Asset 360 | live vibration + bearing-temp charts (3 s poll), predictions, **Run analysis**, stage stepper (WS with poll fallback), event log, Decision Contract panel (SHAP bars, 3-option cost table, window, citations, explanation + source chip, Approve/Reject with approver + note), post-decision lock, what-if form, contract history |
| `/studio` | AutoML Studio | DATA · TASK · FEATURES · MODELS (IMS breakdown) · VALIDATE (leakage, folds, calibration, lead time) · DEPLOY |
| `/modelops` | ModelOps | registry table, pending promotions with champion-vs-challenger delta + Approve/Reject, drift reports + Check drift |
| `/copilot` | Bolt | chat, tool-call chips, source chip, asset context |

Design: dark industrial control-room palette; monospace ids; every screen tolerates null
mid-run fields. One vitest smoke test (Fleet with mocked fetch).

## 7. Demo data (`data/simulator/generate.py`)

24 induction motors at site Ludvika (lines Compressor Hall / Pump House / Conveyor A), 30 days at
10-min steps, three regimes on a shift schedule. Faults (onset→failure as timeline fractions,
repaired afterwards): MTR-042 0.55→1.0 (live demo, caught in time), MTR-017 0.80→1.0,
MTR-003 0.05→0.25, MTR-009 0.12→0.35, MTR-013 0.20→0.42, MTR-023 0.08→0.30 (held-out history),
MTR-021 0.60→0.85 (fails live, held-out). Deterministic under `seed=42`. Outputs
`telemetry.parquet`, `telemetry.csv`, `assets.json`.

Credibility check: `data/benchmark/ai4i_benchmark.py` runs the same regime / IMS / grouped
holdout / leakage / calibration methodology on UCI AI4I 2020 and writes `results.json`.

## 8. Non-functional requirements

- **Governance (hard):** no maintenance action or promotion without a recorded `Approval`; decision contracts and approvals are insert-only; the LLM never computes a metric, cost, SHAP value or citation; the key lives in one file.
- **Validation honesty:** no random shuffle splits; held-out motors the model never saw; only completed failures become labels; metrics reported as measured even when the gate fails.
- **Degradation:** every external dependency is optional at runtime — no LLM key (template mode), no MLflow server (warn), no MQTT broker (HTTP ingest), no embedding model (hashed fallback), no Chroma index (self-build), no ONNX converter for a family (event, approval stands).
- **Performance targets (observed):** conftest fleet end to end ≈ 10 s; real fleet, 6 trials/family ≈ 80–100 s to the approval gate; edge inference sub-millisecond per row.
- **Reproducibility:** simulator, Optuna, KMeans and folds are seeded; replay is deterministic.
- **Quality bar:** `ruff check .` clean; `pytest` green (146 tests: unit per agent, API/orchestrator with fakes, one real end-to-end); web build + smoke test; GitHub Actions runs all of it on push.

## 9. Out of scope (unchanged from CLAUDE.md §8)

Kubernetes · real OPC UA/PLC · CNN/autoencoders · MinIO · Redis · Prometheus/Grafana/OTel ·
Neo4j · blockchain · native mobile · real SAP/ABB integration · full Playwright suite ·
real shadow/canary traffic (lifecycle is a database enum).

## 10. Traceability

| requirement group | code | tests |
|---|---|---|
| 3.x data model | `apps/api/{schemas,db}.py` | `test_schemas.py` |
| 04 | `agents/data_agent/profiling.py` | `test_data_agent.py` (18) |
| 05 | `agents/ml_architect/` | `test_ml_architect.py` (16), `test_edge.py` parity |
| 06 | `agents/validation/checks.py` | `test_validation.py` (14) |
| 07 | `agents/reliability/` | `test_reliability.py` (22) |
| 08 | `agents/mlops/` | `test_mlops.py` (10) |
| 03 / 02 / 09 | `apps/api/` | `test_orchestrator.py` (13), `test_api.py` (8), `test_e2e_pipeline.py` (1) |
| ingest / replay / simulator / edge | `apps/api/routes/telemetry.py`, `streaming/`, `data/simulator/`, `edge/` | `test_telemetry_routes.py` (7), `test_simulator.py` (10), `test_edge.py` (12) |
| benchmark | `data/benchmark/` | `test_benchmark.py` (5) |
