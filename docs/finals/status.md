# Phase 0 — correctness pass: progress log

Session started 2026-10-11 (Claude Fable 5.1). Branch `finals`, baseline commit `1c4475f`.
One entry per milestone: time (UTC), phase/milestone, files, tests actually run with their
result, and the specific next step. If this session stops, resume from the last "next" line.

Fix order (from `docs/finals/prompt-0-correctness.md`): governance (1, 2) → ground truth (3) →
thresholds (6) → leakage (4) → edge parity (5) → analysis reuse (7) → demo prep → training
substance (model card, AutoML Studio view, AI4I rerun, Bolt eval case).

---

## 2026-10-11 — start: orientation and baseline

- **Phase:** 0 / orientation.
- **Done:** read CLAUDE.md §7 and §15, the brief, and every file named in the seven defects
  (`orchestrator.py`, `routes/approvals.py`, `routes/fleet.py`, `lifecycle.py`, `checks.py`,
  `automl.py`, `features.py`, `profiling.py`, `edge/*`, `replay.py`, `cost.py`, tests).
  All seven defects re-verified against the code at `1c4475f`.
- **Files changed:** `docs/finals/status.md` (this file).
- **Tests:** full `pytest` baseline running in the background (result recorded in the next entry).
- **Next:** fix 1+2 (governance): split maintenance approval from model promotion in
  `orchestrator._finalize`, open promotion requests only for passed validation, gate
  `lifecycle.request_promotion`/`promote` on `validation_passed`, fix `reject_promotion` call in
  `routes/approvals.py`; add tests; commit.

## 2026-10-11 13:45 UTC-5 — fix group 1 done: governance (defects 1, 2)

- **Baseline:** `pytest` at `1c4475f`: 180 passed in 48 s (full suite, e2e included).
- **Commit:** `e797f21` "Separate maintenance approval from model promotion and gate promotion
  on validation".
- **Files:** `apps/api/orchestrator.py` (`_validate`, `_finalize`), `apps/api/routes/approvals.py`
  (`decide_promotion_db`, `feature_spec_for_version`), `agents/mlops/lifecycle.py`
  (`validation_failures`, `PromotionBlocked`, `_assert_promotable`), `CLAUDE.md` §6,
  tests in `test_orchestrator.py`, `test_api.py`, `test_mlops.py`, `test_e2e_pipeline.py`.
- **Tests run:** `pytest tests/test_mlops.py tests/test_orchestrator.py tests/test_api.py
  tests/test_e2e_pipeline.py` → 41 passed (after fixing two new tests); `ruff check .` clean.
- **Next:** fix 3 (ground truth): strip `health`/`failure_within_h`/`regime` at the replay's
  publish boundary, `health_score: None` when no prediction, infer `regime` from rpm/load with
  the production baseline, scrub asset/telemetry/copilot responses, isolation test, web types.

## 2026-10-11 13:56 — fix group 2 done: ground truth isolation (defect 3)

- **Commit:** `6af028b` "Keep simulator ground truth out of every operational path".
- **Files:** `apps/api/schemas.py` (`ObservedTelemetryRow`, `GROUND_TRUTH_COLUMNS`, nullable
  `FleetAsset.health_score` + `prediction_ts`/`model_version`), `apps/api/db.py`
  (`latest_prediction_rows`), `apps/api/routes/fleet.py` (no fallback health, `infer_regimes`),
  `apps/api/routes/telemetry.py`, `apps/api/orchestrator.py` (copilot tools/templates,
  feature spec passed to `register_model`), `agents/mlops/lifecycle.py` (`feature_spec` payload,
  `production_feature_spec`), `agents/ml_architect/features.py` (`assign_regime_ids`),
  `streaming/replay/replay.py` (`public_record`/`public_payload`), web types + Fleet/Asset360/
  HealthPill/MotorSprite, `docs/INTERFACES.md`; new `tests/test_ground_truth_isolation.py`.
- **Tests run:** `pytest` on isolation, api, telemetry_routes, orchestrator, mlops,
  replay_work_orders, ml_architect → 77 passed; `ruff check .` clean;
  `npm run build` ok; `npm run test` 4 passed.
- **Next:** fix 6 (thresholds): named `CLASSIFICATION_THRESHOLD` (0.5) and
  `EARLY_WARNING_THRESHOLD` (0.3) in `apps/api/schemas.py`, one lead-time definition (3 sustained
  rows at 0.3) shared by `automl.py` and `checks.py`, `cost.py` window limit = early warning,
  thresholds reported in `ValidationReport`, Asset 360 chart draws both; test.

## 2026-10-11 14:02 — fix group 3 done: thresholds (defect 6)

- **Commit:** see `git log` ("Give each threshold one meaning shared by AutoML search and
  validation").
- **Files:** `apps/api/schemas.py` (`CLASSIFICATION_THRESHOLD`, `EARLY_WARNING_THRESHOLD`,
  `EARLY_WARNING_SUSTAINED_ROWS`, `Thresholds`, `ValidationReport.thresholds`), new
  `agents/metrics.py` (shared metrics, lead time, IMS), `agents/ml_architect/automl.py`,
  `agents/validation/checks.py`, `agents/reliability/cost.py`, `data/benchmark/ai4i_benchmark.py`,
  `apps/api/routes/fleet.py` (thresholds in responses), web chart/Asset360/Studio/types;
  new `tests/test_thresholds.py`.
- **Tests run:** thresholds, validation, ml_architect, reliability, benchmark, schemas, e2e →
  74 passed; `ruff check .` clean; web build ok, 4 vitest passed.
- **Next:** fix 4 (leakage): `AssetSplit` (seeded, recorded in the run), regimes + baselines
  fitted on train assets only, Optuna on the validation group, calibration on the
  calibration group, final metrics on untouched test assets; right-censor the last
  horizon of unlabelled rows; tests for leak-free + reproducible split.

## 2026-10-11 14:25 — fix group 4 done: leak-free validation (defect 4)

- **Commit:** "Fix the asset-level split first and fit nothing on held-out motors".
- **Files:** new `agents/ml_architect/split.py` (`make_asset_split`, `rows_in`,
  `completed_failure_assets`), `apps/api/schemas.py` (`AssetSplit`, `GateCheck`,
  `FeatureSpec.fit_assets`, `RegimeReport.fit_assets`, `CandidateSet.split`,
  `ValidationReport.split/gate/test_metrics`, `GraphState.split`), `agents/data_agent/profiling.py`
  (`fit_assets`), `agents/ml_architect/features.py` (`fit_assets`, fixed `baseline`, `label_known`
  right-censoring), `agents/ml_architect/automl.py` (split-aware `launch_trial`/`train_candidates`,
  `legacy_split`), `agents/validation/checks.py` (`validate` rewritten), `apps/api/orchestrator.py`
  (`tool_make_split`, split fixed in `_profile`, wired through train/validate/_features_for);
  new `tests/test_asset_split.py`; updated orchestrator/validation/ml_architect tests.
- **Arrangement for 24 motors:** 4 completed failures → train 2, test 1, tune 1; validation and
  calibration share the tune group (documented in `AssetSplit.notes`; four separate groups when
  ≥ 5 completed failures). The analysed asset is always forced into test.
- **Tests run:** full `pytest` → 201 passed in 55 s; `ruff check .` clean.
- **Next:** fix 5 (edge parity): export calibration as a serialisable transform in the sidecar,
  edge applies it; edge `compute_features_np` uses the training baseline format (per-row
  nearest-centroid regime from rpm/load); parity check (central vs edge calibrated probability
  on held-out feature vectors + raw windows) recorded in `ValidationReport.parity`, failing
  parity blocks `promote()`; fix `tests/test_edge.py` to the real format.

## 2026-10-11 14:50 — fix group 5 done: edge parity (defect 5)

- **Commit:** "Make the edge reproduce the centre and gate promotion on parity".
- **Files:** `edge/features.py` (rewritten to the training baseline format, regime inferred from
  rpm/load), `edge/inference.py` (`parse_calibration`, `apply_calibration`, calibrated
  `predict_proba`), `edge/main.py` (no regime label), `agents/validation/checks.py`
  (`calibration_transform`, `apply_calibration`, `CalibrationReport.transform`),
  `agents/mlops/lifecycle.py` (`export_bundle`, `check_edge_parity`, parity in payload and in
  `_assert_promotable`), `apps/api/schemas.py` (`ParityReport`, `ValidationReport.parity`),
  `apps/api/orchestrator.py` (`tool_check_parity`, `_parity_inputs`, gate check `edge_parity`),
  `CLAUDE.md` §3/§6; tests: `test_edge.py` (real format + exported calibration), `test_mlops.py`
  (export, lying model fails, parity blocks promotion), `test_e2e_pipeline.py` (parity passed on
  the real run), `test_orchestrator.py` (fake parity).
- **Tests run:** full `pytest` → 206 passed in 58 s; `ruff check .` clean.
- **Next:** fix 7 (analysis reuse): `PipelineRunRequest.mode` ("analyze" default / "train"),
  analyze = load production model + baseline → features for the asset (no refit) → explain →
  cost → contract → approval gate (seconds); train mode ends at a new `trained` stage after
  registration (no contract); `POST /pipeline/run` 409 without a production model; fleet
  scoring helper with the production model; Asset 360 "Run analysis", Studio/ModelOps "Train
  new model"; drift check uses the production baseline.

## 2026-10-11 15:05 — fix group 6 done: analysis reuses the production model (defect 7)

- **Commit:** `f08cfc4` "Reuse the production model for Run analysis and make training an explicit run".
- **Files:** `apps/api/schemas.py` (`RunMode`, `ModelRef`, `PipelineStage.trained`, `GraphState.mode/model_ref`,
  `PipelineRunRequest.mode`), `apps/api/orchestrator.py` (`next_node_for`, `_load_production`,
  `_finish_training`, `tool_production_bundle`, `tool_score_asset`), new `apps/api/scoring.py`,
  `apps/api/routes/pipeline.py` (`mode`, 409 without a model, `POST /pipeline/score`),
  `apps/api/routes/models.py` (drift uses the production baseline), `agents/mlops/lifecycle.py`
  (`production_bundle`, validation report stored in the registry payload), web (mode-aware stepper,
  Studio "Train new model"); tests rewritten to the real demo flow.
- **Tests run:** full `pytest` → 209 passed in 59 s (after fixing the e2e poll's terminal set, which had
  cost 4 min); `ruff check .` clean; web build + 4 vitest passed.

## 2026-10-11 15:22 — demo prep run (real numbers) and the parity gate catching a bug

- First prep run (`scripts/demo_prep.py --n-trials 6`): the gate REFUSED promotion: edge parity
  max |Δp| 0.90 (isotonic LightGBM). Root cause: scikit-learn calibrates on `decision_function`
  (log-odds) for LightGBM/XGBoost, the edge fed probabilities. Fixed in `74dc94b` (transform
  records `input`, both `apply_calibration` twins take the logit); tests now cover all three
  families × sigmoid/isotonic.
- Second prep run: trained lightgbm v57257 in 55 s, gate all pass (parity 0.0), promoted via
  `POST /promotions/.../decision` as demo-prep@thor, fleet scored (24/24). Record:
  `docs/finals/prep-report.json`. Commit `9a9e548` (prep script, compose `prep` service, reset scripts).
- **Numbers:** test recall 0.865, precision 1.000, F1 0.928, AUROC 0.933, Brier 0.0021, lead time
  41.5 h (1 event), IMS 0.844. Written to `docs/finals/metrics.md` and `model-card.md` (`646e6e0`).

## 2026-10-11 15:30 — training substance done

- `646e6e0`: model card, metrics.md, AutoML Studio split/trials/gate/parity view, AI4I rerun
  (lightgbm, recall 0.784, precision 0.896, AUROC 0.974, 2,502 held-out rows; command in §10),
  Bolt eval cases "Approve MTR-042" (53/53 pass, template mode).
- **Tests run:** full `pytest` → 213 passed in 59 s; `ruff check .` clean; web build + 4 vitest passed.
- **Next:** docs (CLAUDE.md §10/§16, DEMO.md, session-notes.md) commit; `docker compose build`
  (running); fresh-context re-audit (running); fix flags; final report.

## 2026-10-11 15:45 — Docker build

- `docker compose build` from the repo root: images `thor-main-api`, `thor-main-prep`,
  `thor-main-edge`, `thor-main-replay`, `thor-main-web` built (exit 0); `docker compose config
  --quiet` valid. Commit `81bb49d` holds the docs. Fresh-context re-audit in progress.

## 2026-10-11 15:45 — re-audit and fixes

- Fresh-context audit: verified defects 1–7 against code and its own `pytest` run (212 passed +
  1 environmental: `test_env_hygiene` needs git on PATH; 213 with the §16 PATH refresh) and
  `ruff` clean. Flags fixed here: 49 files had been rewritten with CRLF line endings by the
  patch scripts (restored to the baseline LF in this commit; `git diff -w --ignore-cr-at-eol`
  is the real diff), the compose stack now has a one-shot `prep` job beside the seven services
  (CLAUDE.md wording), the reset scripts no longer claim a second parity re-check (parity is
  proven once, at validation, on the same export path `deploy_edge` uses). Noted, not changed:
  a registry row registered without any parity record (`parity_passed: None`) is promotable;
  the pipeline always records one, and a recorded failure is refused. The analyze run's light
  data-quality profile reclusters one motor's recent rows for display only; features use the
  production baseline.
- **Tests run after the fixes:** full `pytest` → 213 passed in 59 s; `ruff check .` clean; web
  build + 4 vitest passed. Images rebuilt with `docker compose build`.

## 2026-10-11 16:00 — demo reset run on the real Docker stack

- `.\scripts\demo_reset.ps1` (project `thor-main`): stack wiped and restarted; `prep` trained
  lightgbm v58611 in 78 s inside Docker with the same metrics as the native run (test recall
  0.865, precision 1.0, lead time 41.5 h, IMS 0.844, parity 0.0), promoted it as demo-prep@thor,
  wrote the ONNX bundle and scored 24/24 motors; edge and replay then started (edge: model
  v58611 loaded, MQTT connected, 6,696 predictions after 100 s). Fleet: MTR-042 rising
  (p 0.29, health 71, regime R2), all others at p 0.0. `GET /system/health`: db ok, mqtt, mlflow,
  edge all up.
- Script fixes found by running it: volume probe via `docker volume ls` (`5ce6b5f`) and
  explicit exit-code checks instead of `ErrorActionPreference = "Stop"` (docker's stderr
  progress lines aborted the script after the stack was already up).
