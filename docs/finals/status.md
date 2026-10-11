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
