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

## 2026-10-11 16:05 — demo reset confirmed clean

- Four reset runs on the real stack surfaced three script bugs, all fixed and pushed: volume
  probe (`5ce6b5f`), docker stderr under `ErrorActionPreference Stop` (`64db1d4`),
  `docker compose logs -f` / `docker compose wait` never returning for the one-shot prep
  container (`681a926`, `3ee6563`, `b4b64ec`: poll `docker compose ps -a` for the exited state).
- Final run: `SCRIPT EXIT: 0`; prep 66.9 s, lightgbm v59706 promoted (IMS 0.844, parity 0.0),
  24/24 scored, edge + replay streaming, `/system/health` db/mqtt/mlflow/edge all up.
  The api wait loop's 2 s request timeout was shorter than the health probes; raised to 15 s.

## 2026-10-11 16:55 — demo rehearsal on the real stack

- Two more replay fixes, both pushed: the simulated CMMS executes an approved work order at the
  approval (`deb00c7`, `REPLAY_WORK_ORDER_EXECUTE=approval`, the window mode kept and tested),
  and later replay passes shift their timestamps by one full timeline so plant time never
  freezes at the end of the history (`db6e253`; previously duplicate rows were silently
  ignored and a repair after the first pass never reached the dashboard).
- Measured rehearsal after a clean reset (exit 0, model v62552): MTR-042 green at 0 s, red
  (p 0.64) at 74 s; Run analysis to the gate in 17 s with no training step; after approval
  "in maintenance" within 10 s and green with the work order completed by 30 s; registry
  unchanged. Steps and lines: `docs/finals/demo-walkthrough.md`.

## 2026-10-11 17:40 — calibration and display rule, final rehearsal

- Platt (sigmoid) calibration when the calibration motors hold < 2 failure events (`f3ef9ca`);
  metrics unchanged (recall 0.865, precision 1.0, lead time 41.5 h, IMS 0.844, parity 3e-5),
  Brier after 0.00044. Displayed / analysed probability = median of the last three predictions
  (`0c7ae68`): the first sigmoid rehearsal had the per-sample probability flapping (0.68, 0.0,
  0.81) and an analysis fired on a 0.0 sample drafted a run-to-failure contract. Full suite 217
  passed, pushed.
- Final rehearsal at REPLAY_SPEED=900 on the reset stack (model v64634): red at 111 s with no
  flicker, analysis 15 s with the same probability as the card (maintain_now), approval →
  in maintenance → green with the work order completed in 14 s, registry untouched.
  `docs/finals/prep-report.json` is the v64634 record.

# Phase A — 3-D Digital Twin + X-Ray (2026-10-11, session 2)

## 2026-10-11 18:00 — contract, plant layout, Dashboard rename

- `f10936a`: the Twin API contract (`apps/api/schemas.py` Twin section, `Asset.role`,
  `assets.role` column with an idempotent ALTER on old databases, real asset upsert) and its
  TypeScript mirror + client calls (`getTwin`, `getTwinHud`, `getTwinRewind`, `getTwinEvents`,
  `twinWsUrl`).
- Plant layout in one place: `data/simulator/layout.py` (`PLANT_LAYOUT`, `duty_profile`,
  `python -m data.simulator.layout` prints the measured duty table). Packaging Lines 1-5 × 4 roles
  (infeed conveyor, filler, capper/labeler, case packer) + Compressor Hall × 4 air compressors.
  Lines 3-4 = the seven fault-history motors + MTR-018; Lines 1-2 = the eight lowest-duty healthy
  motors. Measured: Lines 3-4 bearing temp mean 53.4 °C / max 91.2 °C vs 50.8 / 72.9 on Lines 1-2;
  vibration 2.66 / 8.18 vs 2.07 / 3.13 mm/s; load and starts flat by design. `generate.py` keeps
  `rated_kw`/`criticality` per asset id (`_RATINGS`); regenerated `data/simulator/out` and proved
  the telemetry parquet byte-identical (`DataFrame.equals` → True). Seed now upserts the catalogue
  on every start. `data/simulator/out` added to `.dockerignore` (images generate it; no stale
  catalogue can be baked in).
- Web: `Fleet` → `Dashboard` (`/dashboard`, nav "Dashboard"), `/` → `Twin` placeholder, nav
  "3-D Digital Twin"; three@0.169 + @react-three/fiber@8.18 + drei@9.122 installed (React 18 kept).
- **Tests run:** `pytest tests/test_simulator.py tests/test_replay_work_orders.py` → 25 passed;
  `pytest tests/test_api.py tests/test_ground_truth_isolation.py` → 19 passed; ruff clean on the
  touched files; `tsc --noEmit` clean; vitest 4 passed.
- Running in parallel: backend sub-agent on `apps/api/live.py`, `apps/api/plant.py`,
  `apps/api/routes/twin.py`, edge `inference_ms`, `tests/test_twin.py`.
- **Next:** twin scaffolding (`apps/web/src/twin/{palette,status,sceneTypes}.ts`), then the HUD /
  X-Ray / rewind sub-agent and the 3D scene sub-agent; I write `Twin.tsx`, `useTwinLive.ts` and the
  route smoke test.

## 2026-10-11 18:45 — twin API landed

- Backend sub-agent delivered `apps/api/live.py` (live bus: seq'd events, 60 s per-asset message
  counts, 10 s rates, edge `inference_ms` p50/p95, motor→edge→control-plane path latency from
  one clock, `data_state` with `TWIN_STALE_AFTER_S` 30 s), `apps/api/plant.py` (simulated MES feed,
  PLC tag mapping), `apps/api/routes/twin.py` (`/twin`, `/twin/hud/{id}`, `/twin/rewind`,
  `/twin/events`, `/ws/twin`), hooks in `routes/telemetry.py`, `inference_ms` from `edge/main.py`,
  `tests/test_twin.py` (18 tests incl. the MES-feeds-no-decision rule), isolation test extended
  to the four twin routes, `docs/INTERFACES.md`. I added `record_prediction` to
  `apps/api/scoring.py` so control-plane scores (Run analysis, `/pipeline/score`) pulse too.
- **Tests run (by me):** `pytest tests/test_twin.py tests/test_ground_truth_isolation.py
  tests/test_telemetry_routes.py tests/test_edge.py tests/test_api.py` → 59 passed;
  `pytest tests/test_twin.py tests/test_orchestrator.py` → 35 passed; `ruff check .` clean.
  Sub-agent's own full run: 237 passed. Native smoke on :8010 (reported by the sub-agent): events,
  counters and `data_state` move after `POST /ingest` + `POST /predictions`.
- **Next:** HUD/X-Ray/rewind sub-agent and 3D scene sub-agent are still running; then wire
  `XRayProvider` in `App.tsx`, run web build + vitest + full pytest, native end-to-end with the
  replay, fps measurement, Docker build, fresh-context audit.

## 2026-10-11 19:00 — X-Ray HUD, tooltip, Dashboard X-Ray, rewind bar, Twin screen

- HUD sub-agent delivered `apps/web/src/twin/{TwinThemeRoot,XRayToggle,XRayTooltip,XRayHud,RewindBar}.tsx`,
  `useXRay.ts`, `xray.css`, `testFixtures.ts`, `XRayHud.test.tsx`; Dashboard X-Ray integration
  (hover tooltip with glyph + three facts, click opens the HUD drawer) + one Dashboard test.
  I wrote `screens/Twin.tsx` (toolbar, site tabs, fly-to buttons, provenance badge, measured
  counters badge, legend, hover/selection strip, rewind, HUD drawer), `twin/useTwinLive.ts`
  (WS `/ws/twin` with 500 ms poll fallback, pulse queue), `twin/SiteTabs.tsx`, `twin/twin.css`,
  `twin/{palette,status,sceneTypes,webgl}.ts`, `screens/Twin.test.tsx` (route smoke test + WebGL
  fallback), `XRayProvider` in `App.tsx`, WebGL fallback notice in `Layout.tsx`.
- **Tests run:** `tsc --noEmit` clean; `vitest run` → 4 files, 20 passed (Twin route 2, HUD 2,
  Dashboard 5, scene layout 11). Web build (sub-agent's run): PlantScene chunk 873 kB (233 kB gzip),
  lazy-loaded on the twin route only.
- **Next:** scene sub-agent report → commit `twin/scene/`; native end-to-end (API :8010 + prep +
  replay + web dev server) to view the twin, measure fps, check pulses/counters/rewind; full
  pytest; Docker build; fresh-context audit; docs (session-notes, CLAUDE.md §15 status).

## 2026-10-11 20:15 — scene landed, end-to-end on a live stack, measured

- Scene sub-agent delivered `apps/web/src/twin/scene/` (`5f252be`): procedural lines/machines/
  motors/compressors, edge node + control plane + path lines, instanced pulses driven only by the
  event queue, regime rings, status glyphs, decision chips, camera rig with overview/line/asset
  fly-to, fps reporter, performance mode, fixture + 11 layout tests.
- Native end-to-end (API :8010 + native edge :8011 subscribed read-only to the Docker broker on
  :1883, prep v68365 promoted by demo-prep@thor, parity 3.4e-5): `/twin` live counters
  36 msgs/s, 24 preds/s, edge inference p50 0.45 ms, path latency p50 514 ms, 240 samples,
  24/24 motors live. Headless Chrome (Iris Xe, 1920×1080, D3D11): overview with live pulses
  **56-60 fps**, performance mode 60 fps, HUD open 48 fps; counters/provenance/HUD/PLC/MES/rewind
  all read from the API (console errors: 0). Dashboard X-Ray hover + HUD drawer verified in the
  browser. Synthetic predictions on the scratch stack: MTR-042 critical (median 0.62), HUD trend
  rising, rewind markers early_warning / predicted_failure, label "MTR-042 critical p 0.62".
- **Tests run:** full `pytest` → 237 passed in 189 s; `vitest run` → 20 passed; `npm run build`
  ok (PlantScene chunk 873 kB / 233 kB gzip, lazy). Fixed two overlay overlaps (rewind bar vs
  badges, rewind status line). `docs/finals/session-notes.md` has the Phase A notes.
- **Next:** fresh-context audit of the "done" list; `docker compose build` (python images
  running, web after the CSS fix); CLAUDE.md §15 Phase A status; final report; stop native
  processes (API 8010, edge 8011, vite 5175, chrome 9333).

## 2026-10-11 21:10 — fresh-context audit and fixes

- Audit (fresh-context sub-agent, read-only) re-ran the suites itself: `pytest` 237 passed,
  `ruff` clean, web build ok, vitest 20 passed; verified items 2-10 of the done list against
  code with file:line evidence. Flags fixed in `1b5c0f2`:
  1. control-plane scores (Run analysis, `/pipeline/score`) pulsed as "edge prediction" and fed
     the motor→edge→control-plane latency → `TwinEvent.source` ("edge" | "control_plane"), the
     bus takes a path-latency sample only for edge predictions, the scene skips control-plane
     pulses, counter labels say "(edge-scored)" / "(edge + control plane)";
  2. the layout docstring stated a selection rule the table did not reproduce → the rule is now
     code (`baseline_duty_index`, `derive_groups`), Lines 1-2 take the eight lowest healthy by
     that index (MTR-010/019 in, MTR-007/014 to Line 5 / Compressor Hall), and
     `test_committed_layout_equals_the_derived_groups` asserts the committed layout equals the
     derived one on the default fleet; group table in the docstring regenerated from the CLI;
  3. rewind button label reads `thresholds.classification`;
  4. the contract-pending badge no longer uses the primary yellow (dashed accent instead).
  Noted, not changed: rewind frames carry no offline/no-data state (stored predictions only);
  the scene collapses stale/none into "no data" while the HUD says "stale"; MTR-042 is the
  default rewind focus; the scene's 11 layout-math tests exceed the brief's test budget.
- **Tests run after the fixes:** ruff clean; `pytest tests/test_simulator.py tests/test_twin.py
  tests/test_ground_truth_isolation.py tests/test_orchestrator.py tests/test_telemetry_routes.py`
  → 65 passed; `tsc` clean; vitest 20 passed; web build ok.

## 2026-10-11 21:30 — Docker images and wrap-up

- `docker compose build`: `thor-main-api`, `thor-main-prep` (rebuilt after the audit fixes),
  `thor-main-edge`, `thor-main-replay`, `thor-main-web` all built (exit 0); `docker compose config
  --quiet` valid. The team's running stack still runs the pre-Phase-A images: `docker compose
  up -d --build` (or `.\scripts\demo_reset.ps1`) picks up the new ones.
- Native processes stopped (API 8010, edge 8011, vite 5175, headless Chrome 9333). Scratch state
  lives only in `%LOCALAPPDATA%\Temp\thor-native`.
- **Next (Prompt B session):** read `docs/finals/session-notes.md` Phase A section; Bolt Designer
  adds its own `DecisionState` values, HUD module and `/twin` badge styles. Before the demo:
  `.\scripts\demo_reset.ps1`, open http://localhost:5173 (lands on the 3-D Digital Twin), press
  Overview, then follow `docs/finals/demo-walkthrough.md` from the Dashboard/Asset 360 steps.

## 2026-10-12 02:15 — demo reset on the real Docker stack with the Phase A images

- `.\scripts\demo_reset.ps1` (team asked for it to be run): exit 0, prep trained and promoted
  lightgbm v70478 in 82 s (recall 0.865, precision 1.0, lead time 41.5 h, IMS 0.844, parity
  3.4e-5), 24/24 scored, edge + replay streaming, `/system/health` db/mqtt/mlflow/edge up
  (LLM mode: the .env key is set, so the stack runs in LLM mode; template mode still works).
- `/twin` on :8000 right after the reset: six lines in layout order, 24 motors live, model
  provenance v70478 / demo-prep@thor / parity passed, live counters 36 msgs/s, 16.8 preds/s,
  edge inference p50 0.17 ms, path latency p50 509 ms.
- Headless Chrome against the Docker web on :5173 (Iris Xe, 1080p): overview with live pulses
  49-60 fps, performance mode 60 fps, X-Ray hover + HUD (PLC `Speed_Act ← rpm`, simulated
  `Speed_SP`, MES labelled), rewind, Dashboard X-Ray hover + HUD drawer; console errors 0.
- MTR-042 status trace from `/twin` (plant time): healthy p 0.21 (08-27 21:00) → warning p 0.42
  (22:00) → critical p 0.72 (23:30) → critical p 0.9-1.0 from 08-28 02:00 on, through R1/R2/R3.
  Before the sustained rise one idle-shift window dipped the median back to 0.12 for about a
  minute of wall clock; the decision contract path (Run analysis → approval → work order) was
  not exercised here, the team rehearses it per `demo-walkthrough.md`.

## 2026-10-12 02:30 — approval flow on MTR-042, live Docker stack (reset, v71489)

- Status trace from `/twin` after prep (900×, plant time): the first ~3 min of wall clock flap
  healthy ↔ warning ↔ critical as regimes change (08-26 05:30 critical p 1.0 → 07:00 healthy
  0.0 → … → 08-28 02:50 critical 0.94), then **sustained critical from 08-28 02:50 on**
  (p 0.99 in R3). Start the story once the motor has been red for a minute (about 3.5 min after
  prep finishes), not at the first red flash. This is the production model's regime behaviour
  (Phase 0 calibration + median-of-3 rule), unchanged by Phase A.
- Run analysis (mode analyze, model v71489): awaiting_approval in 18 s, `maintain_now`, p 0.996,
  window 08-28 20:50–23:50, drivers temp_delta / vib_kurt_mean / vib_crest_mean; twin badge
  `contract_pending`; HUD drivers cite the contract id.
- Approval as engineer@plant via `POST /approvals/{id}/decision`: registry unchanged
  (production still v71489, no promotion); twin went `offline · work order` /
  `work_order_in_progress` within 2 s, then healthy p 0.0 with `work_order_completed` at 17 s.
  `/twin/rewind` for MTR-042: early_warning 08-26 05:10, work_order_started 08-29 04:50,
  completed 07:50, lead_time_h 71.7 ("early warning → plant took the motor offline").
  Screenshots taken at contract-pending and offline (scratchpad only).
- The stack is left running after this flow (MTR-042 repaired, work order completed); run
  `.\scripts\demo_reset.ps1` before the real demo.

# Phase A.1 — the hero demo, shown by hand (2026-10-12)

## 2026-10-12 04:30 — backend milestone: re-timed faults, the latch, the replay gate, the sandbox

- **Fault timing** (`data/simulator/generate.py`, commit `dc743dc`): MTR-017 → (0.50, 0.72),
  MTR-021 → (0.40, 0.62), so six completed failures (003/009/013/017/021/023) sit inside the seeded
  80 %; MTR-042 → onset `DEMO_ONSET` = 0.80 + 4 h/720 h, ramp `FAULT_RAMP_H` = 16 h to full
  severity, then held until the end of the timeline (never repaired in the data). Why a ramp and
  hold instead of the plain `progress ** 2`: with the quadratic alone the model's sustained
  critical comes at ~70 % of the onset→failure span and the self-repair at 100 %, so at ANY replay
  speed the motor heals itself in the data only ~0.4× the time-to-red after turning red — never
  enough for analysis + Bolt + approval. The per-sensor signature and `FAULT_AMPLITUDE` are
  unchanged (`check_hashes`: only MTR-017/021/042 rows differ; every healthy motor is
  byte-identical). Layout rule still reproduces `PLANT_LAYOUT`; docstring table regenerated.
- **One status rule** `apps/api/alarms.py`: `ALARM_RAISE_ROWS` = 3 (the validation lead-time rule),
  critical latches until `clear_alarm("work_order_completed")`, `WARNING_CLEAR_ROWS` = 6 (twice
  the on-delay: more evidence to clear than to raise, never chatters). Insert-only `alarm_events`
  (`GET /alarms`), in-memory cursors rebuilt from the DB after a restart. `asset_status` precedence:
  offline > no_prediction > no_data (stale) > critical > warning > healthy; `DataState.none` (the
  stream has not started) never hides a prediction. Used by `/fleet` (new `status`, `alarm`,
  `alarm_since`, `alarm_reason`, `sandbox`), `/assets/{id}`, `/twin`, the HUD and the rewind
  markers (`critical_alarm` = first of 3 sustained ≥ 0.5; `lead_time_label` / `validated_lead_time_label`).
- **Clean start:** compose `prep` moved to profile `prep` (edge and replay no longer wait for it);
  `GET /replay` (`apps/api/replay_control.py`) reports `released` iff a production model exists,
  `POST /replay/pause|resume`; the replay container (`ReplayGate`) holds pass 0 until released and
  honours pause. `decide_promotion_db` now refreshes the edge (`POST /models/refresh`) and scores
  the fleet after a successful deploy, so sign-off → green immediately. Reset scripts rewritten
  (`-Prep` / `--prep` = fallback `docker compose run --rm prep`).
- **Sandbox:** `apps/api/sandbox.py` + routes (`GET /sandbox`, `POST /sandbox/set|fault|random-fault|shake|reset`)
  as insert-only `sandbox_events`; the replay's `SandboxApplier` rewrites its own rows (sliders with
  the simulator's load relations, `bearing_fault_delta` = the same signature, ramp `(t/2.5 h)²`
  then hold) and tags them `sandbox=True`; the edge flushes a motor's buffer when the tag flips so
  no window mixes sandbox and plant rows; `predictions.sandbox`, `DecisionContract.sandbox_scenario`,
  `WorkOrder.sandbox_scenario`, `AlarmEvent.sandbox`. Training (`_telemetry`), drift and the rewind
  read `include_sandbox=False` / `sandbox IS FALSE`; a reset retires sandbox predictions
  (`retired_prediction_max_id`) and clears sandbox-only alarms as sandbox alarm events; a sandbox
  work order starting clears that motor's override.
- **Tests:** `tests/test_alarms.py` (11: on-delay, latch, warning clear, status precedence, latch
  cleared only by a completed work order, restart, start-up has no model/predictions and holds the
  replay, release on production, sandbox rows excluded from training, sandbox contracts/work orders
  labelled, reset clears sandbox-only alarms, shake/random fault skip offline motors);
  `tests/test_twin.py` rewritten for the latch; `tests/test_simulator.py` for the new timing
  (5-day test fleet; state-aware bearing-temperature check). Full `pytest`: 195 passed (65 s) before
  the last two simulator fixes, which pass on their own; `ruff` clean.
- **Line endings:** the patch scripts converted 21 files to CRLF; restored to LF in `12c8eb7`
  (lesson: write with `newline="\n"`).
- **Next:** measure the hero flow natively (broker 1884 / API 8010 / edge 8011 / replay 900×):
  training time per trial count, sign-off → green, MTR-042 green → amber → red, red → analysis →
  approval → green; record the new metrics in `metrics.md`, `model-card.md`, `prep-report.json`;
  web (Fleet rename, nav, replay indicator, sandbox panel, state wording) and Bolt hero answers
  are running as two sub-agents; then the walkthrough, Docker build, rehearsal, judge audit.

## 2026-10-12 05:10 — measured the hero flow natively, three times; web and Bolt landed

- **Web** (`79eab00`, sub-agent): Fleet back on `/`, twin on `/twin` under Bolt, sprites/cards/HUD
  from the API's latched status, replay indicator with pause/resume, sandbox panel + banner, the
  two lead-time figures, "state" wording with `wording.test.tsx`; vitest 6 files / 24 tests, build
  ok. Fleet polls every 2 s now (the amber phase is short).
- **Bolt** (`89db582`, sub-agent): `get_asset` carries status / alarm / trend / recent predictions /
  contract details; template answers for the three shift questions before and after a contract;
  eval 62/62 (100 %) in template mode with 9 new `hero` cases; visible "regime" → "state" in the
  explanation, contract and profiling text; `ae38719` finishes the sweep in the docs and the
  remaining API strings (identifiers, fields, feature names and the sidecar untouched).
- **Measured natively** (broker 1884, API 8010, edge 8011, replay 900×, template mode;
  scratchpad `hero_measure.py`, timelines `hero_timeline_{4,6,4b}.json`):
  - start-up: 82,944 rows seeded, 0 models, 24 motors "no prediction yet", `/replay released=false`.
  - training (`mode=train`, MTR-042 held out): 4 trials 118 s / 148 s, 6 trials 156 s, 2 and 3
    trials 92 s — on a laptop also running the team's Docker stack and, in two runs, the test suite.
    Split 12 / 3 / 3 / 6 (four groups: six completed failures), test positives MTR-003 + MTR-013.
    The model differs run to run (Optuna): random forest each time; 4-trial runs gave recall
    0.919–0.933, precision 0.90–0.96, lead time 53–59 h, IMS 0.888–0.908, parity ≤ 1.7e-6; 2–3
    trials gave precision 0.75. Default set to 4 (`DEMO_PREP_TRIALS`, Studio form). The brief's
    "about 60 s" is not reachable on this laptop: validation + parity alone take ~35 s.
  - sign-off → promotion returns in 5–11 s (edge refresh True, 24 scored) → `released` and
    24 green within 6–12 s of the click.
  - MTR-042: green for 35–72 s after the sign-off (run 1 on the 16 h ramp: amber at 68 s for
    ~2 s, red at 70 s; run 2 on the 10 h ramp: red at 35 s, no amber; run 3 on the 20 h ramp:
    red at 72 s, no amber). The random-forest + Platt probability steps from < 0.01 to 0.81
    within one sample (edge rows 14:40 0.005 → 14:50 0.81), so the warning and the critical fire
    on the same sample; amber is a few seconds at most and cannot be promised. Kept the 20 h ramp
    (onset 0.5 h after the seed point); documented in the walkthrough.
  - red held 60–90 s → Run analysis 3.6–4.9 s (maintain_now, p 0.81, drivers vib_kurt_mean /
    temp_delta / vib_rms); Bolt's three answers < 1 s each (template); approval → offline in
    2–3 s → healthy with `work_order_completed` in 13–16 s; rewind: this run 17–24 h vs validated
    53–59 h; alarm log: critical → healthy(work_order_completed).
  - run 1 exposed a re-latch after the repair (the edge window mixed pre-outage and healthy rows):
    fixed in `4e0b4b3` (window restarts after a gap > 2.5 steps, edge and control plane); runs 2–3
    clean.
  - **false latches over one full pass (no sandbox): 0** — after pass 0 (04:45 → 05:55 UTC) the
    alarm log holds MTR-042's critical/healthy only (plus the sandbox motor I faulted on purpose).
  - sandbox (`sandbox_check.py`): Troubleshoot MTR-001 → sandbox-tagged p 0.44 at 11 s, critical
    (sandbox) at 12 s; sliders on MTR-005 (rpm 650 / load 12, then bearing 95 °C): sandbox-tagged
    rows, no alarm; Shake up: 24 overrides, 23 healthy + the faulted motor, nothing broke; Run
    analysis on the sandbox motor → contract `sandbox_scenario=true`, approval → green in 15.7 s
    with a sandbox work order; Reset → no overrides, no sandbox flags, MTR-005 back on its recorded
    stream. Exposed `clear` missing from `SandboxEvent.kind` (fixed in `d4b3ebb`, tested).
- **Docs:** `demo-walkthrough.md` rewritten as the 10-step click path with these timings;
  `CLAUDE.md` §3/§5/§9/§10/§15/§16, `DEMO.md`, `README.md`, `session-notes.md` updated.
- **Next:** `scripts/demo_prep.py --force --n-trials 4` on the native stack → `prep-report.json`,
  then `metrics.md` and `model-card.md` from it; `docker compose build` (running); Docker rehearsal
  with `.\scripts\demo_reset.ps1` + headless Chrome; fresh-context judge audit; final report.

## 2026-10-12 05:20 — a real gate failure, fixed; the prep report on the re-timed data

- The first `scripts/demo_prep.py --force --n-trials 4` on the native stack **failed the gate**:
  LightGBM champion v81325 with test recall 0.327 and edge parity max |Δp| 0.888. Cause: the
  champion's raw scores reached −1.7e6 on pure leaves (`scale_pos_weight` 91.6 on the new split),
  sklearn's calibrator used `decision_function`, and Platt scaling collapsed to a flat line
  (a = −3e-6) that the edge — which only has the ONNX probability — cannot reproduce. Fix
  (`cf509bf`, `c7e008c`): `_ProbabilityOnly` hides `decision_function` so every family is
  calibrated on `predict_proba` (sidecar `input: predict_proba`), the ONNX export unwraps it, and
  `test_calibration_uses_the_probability_for_every_family` proves parity through
  `check_edge_parity`. The second attempt then hit `MissingShapeCalculator` for the shim in the
  export (fixed in the same commit); the third passed.
- **Prep report** (`docs/finals/prep-report.json`, run on the native stack, template mode):
  random forest v82031, split 12 / 3 / 3 / 6 (positives train 017+021, validation 009,
  calibration 023, test 003+013), test recall 0.919, precision 0.962, F1 0.940, AUROC 0.996,
  Brier 0.0040, lead time 53.1 h (2 events), IMS 0.882, parity 1.6e-6 on 200 vectors, sigmoid on
  the probability; prep 183.8 s end to end (training ≈ 170 s with the Docker stack also running).
  `metrics.md` and `model-card.md` regenerated from it (scratchpad `write_metrics_docs.py`); the
  old 0.865 / 41.5 h / 0.844 appear only as the "previous run" column.
- **Next:** two more 4-trial train runs to confirm the gate passes repeatedly; `docker compose
  build` (api + `--profile prep`); Docker rehearsal; judge audit; final report.

## 2026-10-12 05:35 — Docker rehearsal with the Phase A.1 images (LLM mode)

- `docker compose build` + `--profile prep build prep` ok; `.\scripts\demo_reset.ps1` → seeded
  82,944 rows, 0 models, `released=False`, LLM mode (openai gpt-5 from `.env`). Fleet screenshot:
  24 grey "no prediction yet", "State: unknown", top bar "held: no production model yet".
- `hero_measure.py` against :8000 (`hero_timeline_docker.json`): **train 71.5 s** (4 trials, RF
  v82626, recall 0.919, precision 0.962, lead 53.1 h, IMS 0.892, parity 1.6e-6) → promotion 2.7 s →
  released + 24 green at 2.8 s → MTR-042 critical at 64 s after the sign-off (no amber; p steps
  2.8e-4 → 0.81) → red held 60 s → Run analysis 30.2 s (LLM drafts the paragraph; 4–5 s in
  template) → Bolt (LLM) 15 / 21 / 29 s with good answers (status + latch + readings + contract;
  drivers, costs, citations, precedent; options + window + procedure + Asset 360) → approval →
  offline 1 s → healthy + work order completed 12.8 s; rewind this run 38.5 h vs validated 53.1 h.
  Twin screenshot: 60 fps, 36 msgs/s, path p50 495 ms, MTR-042 red octagon with CONTRACT PENDING.
- Fixed the asset id wrapping beside the "latched since" pill (`380fe8d`); reset script counts the
  fleet correctly (`b96c039`).
- **Next:** fresh-context judge audit on the running stack; fix flags; final report.

## 2026-10-12 06:45 — fresh-context judge audit: 20 findings, the ones that matter fixed

- Audit (sub-agent, read-only, Docker stack, LLM mode): reset 63 s; train 56 s (replay held) /
  90 s + gate FAIL when trained again after sign-off; sign-off 2.7 s, 24 green at 2.8 s; red at
  63 s, no amber; analysis 43 s (LLM); Bolt 13 / 27 / 21 s; approve -> offline 0.4 s -> green
  12.7 s; Troubleshoot -> red 11-14 s; reset clears in 0.5 s; sandbox isolation checks all passed;
  0 false latches on never-failing motors. Verdict: steps 1-8 repeatable; two conditions to remove.
- **Fixed (code):** (1) the hero healed itself when the recording looped (`UnrepairedFault` in
  `streaming/replay/work_orders.py`: a motor the data never repairs keeps its full-severity rows
  across passes until a work order; plus the replay now loops the story tail, not the month, so
  the six historical failures never return "live", `e6fd43c`); (3) "Train new model" after the
  sign-off failed the gate because replayed rows carry no outcome and trained as negatives
  (`orchestrator._telemetry` keeps rows with stored ground truth only); (4) Shake tagged a real
  alarm as sandbox and Reset cleared it (shake now skips motors with a plant-raised alarm);
  (5) "latched since" now dates the FIRST row of the sustained run, the same instant as the
  lead-time rule and the rewind marker; (7) a browser refresh on Asset 360 returned nginx 404
  (Vite `build.assetsDir: "static"`, nginx `location /static/`); (8) near-zero SHAP drivers read
  "little effect" instead of "lowers risk" in the contract prose and the HUD; Bolt's LLM prompt
  quotes tool numbers exactly; (9) "operating regimes detected" -> states; (10) HUD "planned
  window" with its basis; (11) button "Awaiting approval"; (12) contract panel "validated lead
  time (held-out motors)" in hours; (14) reset scripts remove the exited prep container that
  pinned the models volume; (15) "high criticality"; (17) sandbox hue is violet, amber means
  warning only; (18) the hover tooltip hides during a rewind.
- **Accepted, documented, not changed:** no amber phase (the calibrated probability steps within
  one sample); 81.2 % is the calibration ceiling (one failing calibration motor); IMS varies with
  machine speed (slides quote recall/precision/lead time, "IMS about 0.88-0.89"); the plant
  manuals quoted by Bolt and the contract say "regime" (the plant's own documents; rebuilding the
  Chroma index before the freeze is not worth the risk); overlapping 3-D labels when every
  overlay is on at once; prediction gaps of one window when an override starts or stops (the
  edge restarts the window so sandbox and plant rows never mix); drivers from a closed contract
  still shown in the HUD with their contract id.
- **Next:** vitest + build, full pytest, rebuild images, final reset for the team, final report.

## 2026-10-12 — Phase A.3: Thor learns from every repair + API Key Up / Down

Brief: `docs/finals/prompt-A3.txt` (Bolt Designer and the sandbox cut by the team). Built natively
on branch `finals`; the team rebuilds Docker.

- **Built.** (1) Global **API Key Up / Down** in the header of every screen: `orchestrator.key_up /
  key_down / key_state` (the only module that holds a key), `GET /llm/key`, `POST /llm/key/up|down`,
  `llm.key` in `/system/health`, `KeySwitch.tsx`; starts Down (`LLM_KEY_ON_START`). Down blanks the
  key fields on the cached Settings (no client is ever cached); Up rebuilds Settings from the
  process environment / `.env`. Bolt replies and contracts keep their `llm` / `template` chip.
  (2) **"What did the technician find?"**: insert-only `repair_findings` (`finding_id, contract_id,
  asset_id, finding, note, recorded_by, recorded_at, plant_ts, sandbox, supersedes`),
  `POST/GET /findings` (409 unless the work order is completed; a correction supersedes, both rows
  stay), `WorkOrder.finding`, the Asset 360 panel, the Fleet card line, **alarm trust** on Fleet and
  ModelOps (`GET /findings/trust`, counts only below five acted-on alarms). (3) **Bolt remembers**:
  `search_field_history` gains `this_plant` (deterministic failure-mode / asset-family match,
  `apps/api/findings.plant_precedent`) next to the public `cases` ("public work orders, not this
  plant"); template mode answers "Have we seen this before?" / "Has this happened before?" plant
  findings first, "no confirmed finding yet" when none; eval category `plant` (4 cases, 66 total).
  (4) **Training learns from confirmed findings**: `agents/ml_architect/labels.py`; `_telemetry`
  = recorded history + the confirmed motor's rows up to the intervention (`failure_within_h` =
  hours until it; the live replayed rows included; the simulator's values for that motor
  discarded); the split is computed on the history alone and the motor is then moved into
  train with a note; `GraphState.label_sources` → the **label ledger** in AutoML Studio with the
  findings queued for the next run (`GET /findings/ledger`). NFF / Other / sandbox change nothing.
  (5) **Alarm → signed decision** timer on an approved contract (`lib/timer.ts`, wall clock,
  tooltip). (6) Docs: CLAUDE.md §3/§6/§7/§9/§15/§16, model card (label sources, intervention-time
  limit), walkthrough (steps 11–15, "Phase A.3 beats", Scenario 2), positioning skill (spoken
  headline), session notes (what Down means, design notes), INTERFACES.md.
- **Cut / shaped.** The Fleet card shows the finding line (or "What did the technician find? →",
  the card already links to Asset 360) rather than a second copy of the form inside a link. Bolt
  Designer stays on hold (CLAUDE.md §15). Nothing in `apps/api/alarms.py` or the replay changed.
- **Verified (tests).** `tests/test_key_switch.py` (Down: no client, `_llm_enabled` False on every
  path, Settings fields blank, env untouched; Up → Down → Up; `LLM_KEY_ON_START`; the routes and
  `/system/health` never carry the key; only `orchestrator.py` touches the switch);
  `tests/test_findings.py` (insert-only over HTTP, supersede, 409s, sandbox excluded from trust;
  **zero findings → the training frame is the same object as the history and the split is
  identical**; one confirmed finding → MTR-042's 60 pre-repair rows labelled to the event, none
  after it, the simulator's values gone, the other motors byte-identical, MTR-042 moved from test
  to train with the note, test group otherwise unchanged; NFF / Other / sandbox → no training
  change; label window stops at the previous repair; a > 1 h gap before the outage is skipped;
  Bolt cites the finding with source and name, never a superseded, sandbox or non-existent one;
  Bolt refuses to act even when nothing is pending). Web: `lib/timer.test.ts`,
  `DecisionContractPanel.test.tsx` (the timer row), `Layout.test.tsx` (the switch flips through
  `POST /llm/key/up|down`, never shows a key). Bolt eval template mode **66/66** (plant 4/4).
- **Verified (rehearsal, native second stack on 8010/8011/1884 at 900×, Docker stack also
  running, key Down from start):** train 127.6 s (RF v5797, gate passed, test recall 0.919 /
  precision 0.962 / lead 53.1 h / IMS 0.888 — the prep report's numbers) → sign-off 5.3 s → 24
  green 0.3 s later → **MTR-042 critical 58.7 s after the sign-off** (no other motor alarmed) →
  Run analysis 4.1 s, contract `explanation_source=template`, 0 LLM calls → Bolt's three answers
  0.3 / 1.3 / 0.7 s (template) → approval → offline 1.6 s → healthy with the work order completed
  14.1 s; alarm → signed decision 52.3 s (held red 30 s on purpose) → finding `201` (trust 1 acted
  on / 1 confirmed; the Fleet card carries it) → "Have we seen this before?" cites `MTR-042,
  2025-08-26 04:20: Confirmed: bearing damage (work order dc_…, recorded by J. Ortega)` before the
  public cases → ledger queued MTR-042 → **retrain with the finding: 171 s, RF v6099, gate
  PASSED** (test recall 0.915, precision 0.980, F1 0.946, lead time 49.5 h over the five remaining
  test motors, IMS 0.878, parity 4.7e-6; split notes: "MTR-042 … moved from test to train";
  label ledger: MTR-042 confirmed by work order, 3626 rows labelled); its promotion rejected so
  production stayed v5797. Scenario 2: Up (`provider openai`) → "What's happening with MTR-042?"
  9.4 s LLM → Down → 0.2 s template: **same probability (0.000 / 0.00 and the contract's 0.78),
  same status word**; the same check while red: 0.812 / 0.81, 0.78, critical in both. Refusal:
  "Nothing is waiting for approval right now. I cannot approve…" (fixed during the rehearsal: the
  template only listed pending items when nothing was pending). No `sk-` in any response.
  Screenshots (headless Chrome, Vite on 5175 → 8010): header `API Key Down` on every screen,
  Fleet `Alarm trust` line and `technician: Confirmed: bearing damage · J. Ortega` on the card,
  the Asset 360 panel with the stored row, the Studio label ledger with `in this run` after the
  retrain, ModelOps trust line, Copilot header.
- **Final verification (native, 2026-10-12):** `ruff check .` clean; `pytest` **263 passed** (100 s;
  252 before A.3); vitest **8 files / 28 tests** (6 / 24 before); `npm run build` ok; Bolt eval
  template mode **66/66**. New files normalised to LF.
- **Docker rehearsal (the stage stack, rebuilt images, after `demo_reset.ps1`; driver
  `scripts/rehearse_a3.py`, key Down from start):** train **58.6 s** (RF v7981, gate passed, test
  recall 0.919 / precision 0.962 / lead 53.1 h / IMS 0.893) → sign-off 2.7 s → 24 green at once →
  **MTR-042 critical 60.1 s after the sign-off** → Run analysis 6.1 s (template, 0 LLM calls) →
  Bolt 0.1 / 0.5 / 0.3 s (template) → Key Up while red: LLM 12.6 s with the same 0.81 / 0.78 /
  critical as the template → approval → offline 3.1 s → healthy + work order completed 15.7 s;
  alarm → signed decision 49.9 s → finding 201 (trust 1 / 1) → recall cites the finding by work
  order and name → ledger queued → **retrain with the finding 127.2 s, RF v8237, gate PASSED**
  (recall 0.915, precision 0.980, lead 49.5 h on five test motors, IMS 0.880, parity 4.7e-6),
  promotion rejected → Scenario 2: Up (`provider openai`, container env from `.env`) 11.7 s LLM,
  Down 0.1 s template, same 0.000 / 0.78 / healthy → refusal explicit. The stack was reset again
  afterwards for the team.
- **Honest limits (for Q&A).** The intervention time is not a failure time (model card §12); the
  retrained model passing the gate is one run, not evidence that findings improve the model; the
  plant has one failure mode so far, so "same failure mode" matching is bearing vocabulary plus
  the contract's drivers; right after the repair the displayed probability stays at the last
  median (0.81) for a few seconds until three post-outage predictions exist (status is already
  healthy); the LLM and template readings can differ by one 10-minute sample between two questions.
- **Not done.** Docker images not rebuilt here (the team does); LLM mode of the full Bolt eval not
  re-run (template 66/66; the LLM path was exercised live in the rehearsal with gpt-5); the
  Docker rehearsal of the A.3 beats is for the team's reset.

## 2026-10-12 07:00 — final verification on the final images; Phase A.1 closed

- After the audit fixes: `pytest` 252 passed (97 s), `ruff` clean, vitest 6 files / 24 tests, web
  build ok (`/static/` assets; `/assets/MTR-042` returns 200 through nginx), `docker compose
  build` api / replay / web and `--profile prep build prep` ok.
- `demo_reset.ps1` on the final images: all three volumes removed (the exited prep container no
  longer pins `models`), seeded 82,944 rows, 0 models, 24 motors without a prediction, replay held.
- `hero_measure.py` on the final Docker stack (LLM mode): **train 56 s** (RF v86359, recall 0.919,
  precision 0.962, lead 53.1 h, IMS 0.895, parity 1.6e-6, gate passed) → promotion 2.8 s →
  released + 24 green at 2.9 s → MTR-042 critical at 64 s after the sign-off, "latched since
  08-25 14:50" = the rewind's early-warning and critical markers (same instant now) → Run
  analysis 12.7 s (LLM) → Bolt 7 / 16 / 27 s (LLM; the prompt now asks for three-decimal
  probabilities, `13a1ce2`) → approval → offline 2 s → healthy with the work order completed
  15.3 s; rewind this run 27.7 h vs validated 53.1 h; no other motor alarmed.
- The stack is left RESET (seeded, no model, replay held) with the final images; the team opens
  http://localhost:5173 and follows `docs/finals/demo-walkthrough.md`.
- **Not done / handed over:** the plant manuals still say "regime" in quoted passages; no amber
  phase exists with this champion (documented in the walkthrough and the talk track); the 3-D
  overlays overlap when Overview + X-Ray + Sandbox are all on; LLM-mode answers were checked
  with the OpenAI key from `.env` only (not Anthropic). Prompt B (Bolt Designer) remains on hold.
