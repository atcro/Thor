# Thor — Interface Spec (build contract for parallel agents)

Every module below is built against `apps/api/schemas.py` (Pydantic contracts) and
`apps/api/db.py` (SQLAlchemy Core, SQLite by default / Timescale in compose). Read both
before writing code. `apps/api/settings.py` holds config (`get_settings()`).

Ground rules (from CLAUDE.md §7): no LLM calls anywhere except `apps/api/orchestrator.py`;
no random shuffle splits; decision contracts + approvals are insert-only; every tool function
has a docstring stating inputs/outputs.

Python: 3.11+ syntax (`X | None`, `StrEnum`), type hints everywhere, ruff clean
(`ruff check .`, line length 100). pandas 3.x is installed — avoid chained assignment and
use `.copy()` before mutating slices.

---

## Data shapes

**Telemetry DataFrame** (from `db.read_telemetry(...)`): columns
`asset_id, ts (tz-aware UTC), vibration_rms, vibration_kurtosis, vibration_crest,
bearing_temp_c, motor_temp_c, current_a, rpm, load_pct, regime, health, failure_within_h`.
Sorted by `(asset_id, ts)`. Sampling: one row per 10 minutes per asset (0.00167 Hz).
`health`/`failure_within_h` are simulator ground truth — models may use `failure_within_h`
ONLY to build the training label, never as a feature.

**Feature frame** (from `build_feature_pipeline`): columns
`asset_id, ts, <feature columns...>, label (int 0/1), rul_h (float, NaN when no failure ahead)`.
Feature columns are exactly `FeatureSpec.features`.

**Fitted model artifact**: joblib pickle at `CandidateModel.artifact_path` of an object with
`.predict_proba(X: pd.DataFrame) -> ndarray[n, 2]` where `X` has exactly
`FeatureSpec.features` columns in that order. Random forest = sklearn, xgboost = `XGBClassifier`,
lightgbm = `LGBMClassifier`. Calibrated champion = sklearn `CalibratedClassifierCV`-like wrapper
that still exposes `.predict_proba`.

**MLflow**: tracking URI from `get_settings().mlflow_tracking_uri` (local `./mlruns` dir by
default). Log params/metrics per candidate; run ids go into `CandidateModel.mlflow_run_id`.
Wrap MLflow calls so that an unreachable server degrades to a warning, never a crash.

---

## Agent A — simulator, replay, ingest, edge

### `data/simulator/generate.py`
```python
ASSET_IDS: list[str]                         # "MTR-001".."MTR-024"
FAULTY_ASSETS: dict[str, float]              # {"MTR-042": <onset fraction>, ...}  NOTE: MTR-042 is
                                             # one of the 24 (replace MTR-004 with MTR-042 so the
                                             # demo asset id matches the architecture doc)
def generate_fleet(days: int = 30, seed: int = 42, step_min: int = 10) -> tuple[pd.DataFrame, list[Asset]]
    """Returns (telemetry DataFrame in the Telemetry shape, asset list).
    Regimes: R1 idle/startup (rpm~600, load~10), R2 nominal (rpm~1480, load~55),
    R3 high-load (rpm~1500, load~85). Each asset cycles regimes on a shift schedule.
    Healthy sensors depend on regime (vibration & temps rise with load).
    Faulty assets get a drive-end bearing degradation: from onset, vibration_rms/kurtosis/crest
    and bearing_temp_c rise with a smooth monotone curve + noise, ending in failure at the
    end of the horizon. `health` decays 1->0, `failure_within_h` counts down; healthy assets
    have health~1 and failure_within_h=None."""
def write_outputs(df, assets, out_dir: Path) -> None   # telemetry.parquet, telemetry.csv, assets.json
def main() -> None                                     # python -m data.simulator.generate [--days 30 --out data/simulator/out]
```
Two faulty assets: `MTR-042` (onset at 55% of horizon — the demo motor) and `MTR-017`
(onset at 80%, subtler). Everything else healthy. Deterministic under `seed`.

### `streaming/replay/replay.py`
```python
def iter_rows(df: pd.DataFrame) -> Iterator[TelemetryRow]      # time-ordered across assets
def publish_mqtt(rows, host, port, speed: float, topic_fmt="telemetry/{asset_id}") -> None
def main() -> None
```
Reads `data/simulator/out/telemetry.parquet` (generates it if missing), sleeps
`(next_ts - ts) / speed` between rows (speed from `REPLAY_SPEED`, default 600). Payload =
`TelemetryRow.model_dump_json()`. If MQTT is unreachable after 10 retries, fall back to
`POST {CONTROL_PLANE_URL}/ingest` with batches of rows. Loop forever from the start when the
sequence ends. Also expose `python -m streaming.replay.replay --once --speed 0` = bulk-load
everything via HTTP as fast as possible (used for local dev and tests).

### `apps/api/routes/telemetry.py`
```python
router = APIRouter()
POST /ingest            body: list[TelemetryRow] | TelemetryRow   -> {"inserted": int}
GET  /assets/{asset_id}/telemetry?hours=24&limit=2000              -> list[TelemetryRow]
GET  /assets/{asset_id}/predictions?limit=500                      -> list[{ts, failure_probability, model_version, source}]
POST /predictions       body: {asset_id, ts, model_version, failure_probability, source}  -> {"ok": true}
def start_mqtt_subscriber(engine) -> None    # background paho client; subscribes telemetry/#,
                                             # inserts rows in batches of 50 or every 2s; never
                                             # raises if broker down (log + retry every 5s)
def stop_mqtt_subscriber() -> None
```

### `edge/main.py` (+ `edge/inference.py`)
Standalone FastAPI app on :8001. **Must not import from `apps/` or `agents/`** — the edge
container only installs fastapi/uvicorn/onnxruntime/paho/numpy/httpx/pydantic.
- Watches `MODELS_DIR` for `*.onnx` + sidecar `*.json`
  (`{"model_name","version","features":[...],"window_rows":int}`); loads newest.
- Subscribes to `telemetry/#`; keeps a per-asset rolling buffer of raw rows; computes the
  SAME feature set as `agents/ml_architect/features.py` using `edge/features.py`, a
  dependency-free copy (numpy only) — keep the two in sync via a shared test in `tests/`.
- Runs onnxruntime, publishes `predictions/{asset_id}` with
  `{asset_id, ts, failure_probability, model_version}` and POSTs the same to
  `{CONTROL_PLANE_URL}/predictions` (best effort).
- `GET /health` -> `{status, model_version, n_assets_seen, uptime_s}`; `GET /predict` body
  `{features: {name: value}}` -> `{failure_probability}`.
- Without a model present: healthy, logs "no model", no predictions.

---

## Agent B — 04 Data Reliability, 05 ML Architect

### `agents/data_agent/profiling.py`
```python
def validate_schema(df: pd.DataFrame) -> list[SchemaIssue]
def detect_missingness(df: pd.DataFrame, step_min: int = 10) -> MissingnessReport
def detect_regime(df: pd.DataFrame, k: int = 3, seed: int = 0) -> RegimeReport
    # KMeans on standardized (rpm, load_pct); label clusters R1/R2/R3 by ascending rpm*load;
    # row_regime aligned to df order
def profile_dataset(df: pd.DataFrame, asset_id: str, step_min: int = 10) -> DataQualityContract
    # quality_score = 100 - penalties (missing, flatline, schema errors, short window);
    # trainable = score >= 60 and n_rows >= 500 and n_assets >= 4
```

### `agents/ml_architect/features.py`
```python
def build_feature_pipeline(df: pd.DataFrame, regimes: RegimeReport | None, horizon_h: float = 48.0, window_rows: int = 12) -> tuple[pd.DataFrame, FeatureSpec]
```
Features (rolling per asset over `window_rows`, regime-normalized = z-score against that
regime's healthy baseline computed from the first 20% of each asset's history):
`vib_rms_mean, vib_rms_slope, vib_kurt_mean, vib_crest_mean, bearing_temp_mean,
bearing_temp_slope, temp_delta (bearing - motor), current_mean, rpm_mean, load_mean,
vib_rms_z, bearing_temp_z, vib_rms_resid (actual - regime-expected)`.
Label = 1 if `failure_within_h <= horizon_h` else 0; `rul_h = failure_within_h`.
Drop the first `window_rows-1` rows per asset. Never include `health`, `failure_within_h`,
`regime` strings, or `ts` as features.

`edge/features.py` must produce the same numbers from a raw row buffer — Agent A owns that
file; Agent B owns a pure-numpy reference `compute_features_np(buffer: np.ndarray, ...)` inside
`features.py` that Agent A copies. Export the feature list as `FEATURE_NAMES: list[str]`.

### `agents/ml_architect/automl.py`
```python
def infer_task(dq: DataQualityContract, df: pd.DataFrame, horizon_h: float = 48.0) -> TaskSpec
def industrial_model_score(metrics: dict[str, float], latency_ms: float) -> IndustrialModelScore
    # inputs: recall, precision, lead_time_h (-> lead_time_score = min(1, lead_time_h/72)),
    # brier (-> calibration_score = 1 - min(1, brier/0.25)), latency_score = 1 - min(1, latency_ms/50)
def launch_trial(features_df: pd.DataFrame, spec: FeatureSpec, task: TaskSpec, family: str, n_trials: int = 6, seed: int = 0, artifacts_dir: Path | None = None) -> CandidateModel
    # Optuna over the family's hyperparams; objective = IMS on an asset-grouped, time-ordered
    # holdout (last 25% of assets by id as holdout, and only rows before each holdout asset's
    # last 20% of time... keep simple: GroupKFold-like single split by asset). Logs to MLflow.
    # Saves best model via joblib to artifacts_dir/<candidate_id>.joblib
def compare_models(candidates: list[CandidateModel]) -> list[str]   # ranked ids by ims.total desc
def train_candidates(features_df, spec, task, n_trials=6, seed=0, artifacts_dir=None, asset_id: str = "") -> CandidateSet
    # runs launch_trial for all three families and fills CandidateSet.ranked
```

---

## Agent C — 06 Validation, 07 Reliability

### `agents/validation/checks.py`
```python
def detect_leakage(features_df: pd.DataFrame, spec: FeatureSpec, train_assets: list[str], test_assets: list[str]) -> LeakageReport
    # temporal: any test ts < max train ts for same asset; feature leakage: |corr(feature, label)| > 0.98
    # or forbidden names (health, failure_within_h); asset_overlap = set intersection non-empty
def run_backtest(features_df, spec, model_factory: Callable[[], Any], n_folds: int = 3) -> list[BacktestFold]
    # asset-level folds: sort assets, rotate held-out asset groups; within each fold train rows
    # are strictly before train_end; metrics: recall, precision, f1, auroc, brier
def calibrate_probabilities(model, X_cal: pd.DataFrame, y_cal: pd.Series) -> tuple[Any, CalibrationReport]
    # isotonic if >= 200 cal rows else sigmoid; returns calibrated wrapper with predict_proba
def compute_lead_time(features_df, spec, model, threshold: float = 0.5) -> LeadTimeReport
    # per failing asset: hours between first sustained (3 consecutive) prob >= threshold and failure
def estimate_rul_interval(features_df, spec, seed: int = 0) -> RULInterval
    # sklearn GradientBoostingRegressor with loss="quantile" at 0.1/0.5/0.9 on rows with rul_h notna
def validate(candidates: CandidateSet, features_df: pd.DataFrame, run_id: str) -> ValidationReport
    # loads each candidate from artifact_path, runs the checks, picks champion by IMS recomputed on
    # backtest metrics; model_version = f"v{int(time.time())%100000}"; passed = leakage ok and
    # champion recall >= 0.6 and lead_time median >= 12h. Saves calibrated champion joblib next to
    # the original as <candidate_id>_calibrated.joblib and stores that path in ValidationReport.notes[0]
    # as "champion_artifact=<path>" (schemas has no field for it; do not edit schemas.py).
```

### `agents/reliability/explain.py`
```python
def explain(model, X_latest: pd.DataFrame, spec: FeatureSpec, asset_id: str, model_version: str, top_k: int = 6) -> Explanation
    # shap.TreeExplainer on the underlying tree model (unwrap CalibratedClassifierCV if needed;
    # fall back to shap.Explainer / permutation if TreeExplainer fails). Structured data only.
def run_whatif(model, X_latest, spec, asset_id, scenario: dict[str, float]) -> WhatIfResult
    # scenario overrides feature values (e.g. {"load_mean": 40}) and re-scores
```

### `agents/reliability/rag.py`
```python
def build_index(manuals_dir: Path, chroma_path: Path, collection: str = "manuals") -> int   # chunks indexed
def retrieve_manual_context(query: str, chroma_path: Path, k: int = 3, collection: str = "manuals") -> list[ManualPassage]
def query_from_explanation(exp: Explanation) -> str    # deterministic query string from top SHAP features
```
Chroma `PersistentClient(path=chroma_path)`; embedding = chromadb's built-in
`DefaultEmbeddingFunction` (all-MiniLM-L6-v2 via onnxruntime, local). If model download fails
(offline), fall back to a deterministic hashed bag-of-words embedding function so the demo never
dies — log a warning. Loads `*.pdf` (pypdf) and `*.md`/`*.txt` from `manuals_dir`; chunk by
heading/section (~600 chars), metadata `{source, section, page}`.
Ship **three** manuals as Markdown in `data/manuals/`:
`motor-maintenance.md` (induction motor: bearings §4.2 drive-end bearing wear signatures —
vibration RMS/kurtosis rise, temp rise, recommended actions & lubrication intervals),
`vibration-analysis-guide.md` (ISO 10816 zones, crest factor, kurtosis meaning),
`plant-maintenance-policy.md` (planned vs unplanned cost policy, approval rules).
Make them realistic, specific, ~150-250 lines each, with numbered sections.

### `agents/reliability/cost.py`
```python
def calculate_failure_cost(p_fail_by: Callable[[float], float] | float, settings: Settings, horizon_h: float = 168.0, now: datetime | None = None) -> CostComparison
    # options: maintain_now (planned cost + planned downtime, p_failure_before ~ p at 0h),
    # maintain_later (next planned window in 72h: planned cost + P(fail before)*unplanned cost),
    # run_to_failure (P(fail in horizon)*(unplanned repair + unplanned downtime cost))
    # expected cost = P(failure)*consequence + planned cost; recommended = argmin
def find_maintenance_window(cost: CostComparison, lead_time: LeadTimeReport, now: datetime | None = None) -> MaintenanceWindow
    # earliest low-load window (default: next 02:00-05:00 UTC) that ends before P(fail) exceeds 0.3,
    # else "now"
```
`p_fail_by(h)` is a hazard curve: use `1 - (1-p_now)**(1 + h/lead_time_median)` or similar
monotone function — document it.

### `agents/reliability/contract.py`
```python
def build_evidence_bundle(explanation, manual_context, cost, window, validation, data_quality_score) -> EvidenceBundle
def create_decision_contract(bundle: EvidenceBundle, run_id: str, explanation_text: str, explanation_source: Literal["llm","template"], engine=None) -> DecisionContract
    # contract_id = f"dc_{asset_id}_{YYYYmmddHHMMSS}"; evidence_hash = sha256 of bundle.model_dump_json();
    # inserts via db.insert_decision_contract; returns model
def template_explanation(bundle: EvidenceBundle) -> str
    # deterministic prose fallback used when no LLM key: cites top features, cost numbers,
    # window, and manual sections by name — this is what the demo shows if ANTHROPIC_API_KEY is blank
```

---

## Agent D — 08 MLOps

### `agents/mlops/lifecycle.py`
```python
def register_model(validation: ValidationReport, artifact_path: Path, name: str = "thor-bearing-classifier", engine=None) -> RegisteredModel
    # inserts model_registry row stage=candidate; also mlflow.register_model best-effort
def compare_champion(challenger: RegisteredModel, engine=None) -> ChampionComparison
    # champion = current stage=production row for same name (or None); delta on ims_total etc.
def request_promotion(comparison: ChampionComparison, to_stage: ModelStage, engine=None) -> PromotionRequest   # inserts promotion_requests row
def promote(promotion_id: str, approval: Approval, engine=None) -> RegisteredModel
    # REQUIRES approval.decision == "approved" and approval.promotion_id == promotion_id else raise PermissionError
    # transitions stage per enum order candidate->validated->shadow->production; demotes previous
    # production to archived; updates promotion_requests.status
def deploy_edge(model: RegisteredModel, artifact_path: Path, features: list[str], models_dir: Path, window_rows: int = 12) -> EdgeDeployment
    # converts sklearn RF via skl2onnx, XGB via onnxmltools.convert_xgboost, LGBM via
    # onnxmltools.convert_lightgbm; unwrap calibration wrapper (export the base estimator and note
    # calibration as a sidecar isotonic table if feasible; otherwise export base and record
    # "calibration": "none"). Writes <name>_<version>.onnx + .json sidecar
    # {"model_name","version","features","window_rows","calibration"}. Verifies with onnxruntime.
def get_production_model(name, engine=None) -> RegisteredModel | None
def list_models(engine=None) -> list[RegisteredModel]
```

### `agents/mlops/drift.py`
```python
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float     # ~30 lines total file
def detect_drift(train_features: pd.DataFrame, recent_features: pd.DataFrame, feature_names: list[str], asset_id: str, model_version: str, threshold: float = 0.2, engine=None) -> DriftReport
    # inserts drift_reports row
```

---

## Agent E — 03 orchestrator, 02 control plane, 09 approval gate

### `apps/api/orchestrator.py` — the ONLY file that reads `settings.anthropic_api_key`
```python
def build_graph() -> CompiledGraph      # LangGraph StateGraph over GraphState (as dict) with nodes:
    # profile -> train -> validate -> explain -> await_approval (interrupt) -> finalize
def route(state: GraphState) -> str     # next node name; uses LLM tool-choice call #1 when key present,
                                        # else deterministic order. LLM may only choose among the fixed
                                        # node names — it cannot skip validate or await_approval.
def draft_explanation(bundle: EvidenceBundle) -> tuple[str, Literal["llm","template"]]
                                        # LLM call #2 with the evidence bundle as JSON; system prompt forbids
                                        # new numbers/citations; falls back to reliability.contract.template_explanation
def start_run(asset_id: str, horizon_h: float, n_trials: int, engine=None) -> GraphState   # creates run_id, persists, kicks off background execution (threading / BackgroundTasks)
def await_approval(state) -> GraphState   # persists stage=awaiting_approval and returns
def resume(run_id: str, approval: Approval, engine=None) -> GraphState   # continues after record_decision
def copilot_reply(req: CopilotRequest, engine=None) -> CopilotResponse
    # tools the LLM may call (all deterministic, read-only): get_fleet, get_asset, get_contract,
    # get_run, list_pending_approvals, search_manuals (Chroma lookup over data/manuals; Bolt
    # may quote only what it returns). Without key: keyword-routed template answers, same tools.
```
Run execution must persist `GraphState` to `pipeline_runs` after every node (db.save_run) with
events appended so the UI can poll progress. Wrap each node in try/except -> stage=failed,
error=str(e).

Pipeline node implementations call:
- `agents.data_agent.profiling.profile_dataset(df, asset_id)` on `db.read_telemetry()` (whole fleet)
- `agents.ml_architect.features.build_feature_pipeline`, `agents.ml_architect.automl.infer_task`, `train_candidates`
- `agents.validation.checks.validate`
- `agents.reliability.explain.explain`, `rag.retrieve_manual_context(rag.query_from_explanation(exp))`, `cost.calculate_failure_cost`, `cost.find_maintenance_window`, `contract.build_evidence_bundle`, `draft_explanation`, `contract.create_decision_contract`
- `agents.mlops.lifecycle.register_model` + `request_promotion` (so ModelOps has something to approve)
- on approval: `agents.mlops.lifecycle.promote` + `deploy_edge`

### `apps/api/main.py`
FastAPI app, CORS open for localhost dev, lifespan: `init_db()`, seed assets from
`data/simulator/out/assets.json` (generate if missing when `SEED_ON_START=1`; also bulk-load the
parquet into telemetry if the table is empty so the fleet screen is never blank), build the RAG
index (best effort), start the MQTT subscriber. Mount routers. `GET /health`.

### Routes
```
apps/api/routes/fleet.py
  GET /fleet                        -> list[FleetAsset]  (health_score = 100*(1 - p_fail) if a prediction exists else derived from latest health/vibration z-score)
  GET /assets/{asset_id}            -> {asset, latest: TelemetryRow|null, prediction: float|null, contracts: list[DecisionContract], runs: list[run summaries]}
  GET /system/health                -> {db: ok, mqtt: bool, mlflow: bool, edge: bool, n_telemetry_rows, last_ingest_ts}
apps/api/routes/pipeline.py
  POST /pipeline/run   body PipelineRunRequest -> PipelineRunResponse
  GET  /pipeline/{run_id}            -> GraphState (as JSON)
  GET  /pipeline?asset_id=           -> list of run summaries
  WS   /ws/runs/{run_id}             -> pushes GraphState JSON every 1s until terminal stage
  POST /whatif  body {run_id, scenario} -> WhatIfResult
apps/api/routes/approvals.py       (09)
  GET  /approvals/pending            -> {contracts: [...], promotions: [...]}
  GET  /approvals/{contract_id}      -> present_evidence(): {contract, evidence: bundle-ish, status}
  POST /approvals/{contract_id}/decision  body DecisionRequest -> Approval   (record_decision(): insert row; then orchestrator.resume)
  POST /promotions/{promotion_id}/decision body DecisionRequest -> Approval  (then lifecycle.promote + deploy_edge)
apps/api/routes/models.py
  GET /models                        -> list[RegisteredModel]
  GET /models/promotions             -> list[PromotionRequest]
  GET /models/drift?asset_id=        -> list[DriftReport]
  POST /models/drift/check  body {asset_id} -> DriftReport
apps/api/routes/copilot.py
  POST /copilot/chat  body CopilotRequest -> CopilotResponse
```

---

## Agent F — React frontend (`apps/web`)

Vite + React 18 + TypeScript, React Router, no UI framework (hand-written CSS with tokens),
`recharts` for charts. `VITE_API_BASE_URL` (default `http://localhost:8000`). Screens:
1. `/` **Fleet** — table/grid of `GET /fleet` sorted by risk; health pill; click → Asset 360.
   Includes a small "System" panel from `GET /system/health` (this is the health panel that
   replaces Grafana).
2. `/assets/:id` **Asset 360** — live vibration/temp chart (poll `/assets/:id/telemetry` every
   3s), regime badge, current failure probability, "Run analysis" button → `POST /pipeline/run`,
   then a stage stepper driven by `WS /ws/runs/:run_id` (fallback poll `GET /pipeline/:run_id`).
   When stage = awaiting_approval, render the Decision Contract evidence: SHAP bars, cost
   comparison (3 options), recommended window, cited manual passages, explanation text, and
   **Approve / Reject** buttons → `POST /approvals/:contract_id/decision`. After decision show the
   immutable contract with approval row.
3. `/studio` **AutoML Studio** — DATA → TASK → FEATURES → MODELS → VALIDATE → DEPLOY stages
   for the latest/selected run: data quality score + regimes, task spec, feature list, 3
   candidates with IMS breakdown, validation report (leakage, backtest folds, calibration bins,
   lead time), deploy status.
4. `/modelops` **ModelOps** — registry table (`GET /models`), pending promotions with
   champion-vs-challenger delta + Approve/Reject (`POST /promotions/:id/decision`), drift
   reports + "Check drift" button.
5. `/copilot` **Bolt** — chat UI over `POST /copilot/chat`, shows tool calls as chips.
Types in `src/api/types.ts` mirror `schemas.py`. One vitest smoke test that renders Fleet with
a mocked fetch. `Dockerfile`: node build → nginx serve on :80 with SPA fallback.
