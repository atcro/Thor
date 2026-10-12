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

## Display rule (added after the rehearsal)

- `FleetAsset.failure_probability` / `health_score` and `/assets/{id}.prediction` are the
  **median of the last three predictions** (`db.latest_prediction_rows`, `DISPLAY_MEDIAN_ROWS`),
  stamped with the newest row's `prediction_ts` / `model_version`; `n_rows` says how many rows
  the median used. The per-sample rows are still in `GET /assets/{id}/predictions`. The twin
  should colour from the median value, not from the raw series.
- "Run analysis" explains the median-probability window of the last three, so the contract's
  probability matches the card. Calibration is Platt (sigmoid) when the calibration motors hold
  fewer than two failure events (the demo case), isotonic otherwise.

## Replay behaviour that matters for the twin (added after the rehearsal)

- The simulated CMMS executes an approved work order at the approval (`REPLAY_WORK_ORDER_EXECUTE`
  defaults to `approval`; `window` is the faithful mode). The `in_progress` work-order event
  note names the scheduled window. A repaired motor streams its healthy baseline forever after.
- After the first pass the replay shifts the STORY (the tail after the seed point) by its own
  length per pass (`streaming/replay/replay.py::shift_pass` on the tail; changed 2026-10-12 so
  the historical failures never replay "live"), so `last_ts` / plant time keeps growing
  continuously and the dashboard never freezes. Anything keyed on absolute plant time must
  tolerate that.
- Demo timings from the rehearsal: MTR-042 red ~75 s after prep finishes; analysis 17 s;
  approval → green ~30 s. See `demo-walkthrough.md`.

# Phase A notes (3-D Digital Twin + X-Ray, 2026-10-11 evening)

## What exists now (for Prompt B / Bolt Designer)

- Screens: `/` is the **3-D Digital Twin** (`apps/web/src/screens/Twin.tsx`), `/dashboard` is the
  old Fleet view renamed **Dashboard**. Nav keys 1-5: Twin, Dashboard, AutoML Studio, ModelOps,
  Copilot. `App.tsx` mounts `XRayProvider`; the X-Ray toggle state is shared across screens
  (localStorage `thor.xray`).
- Plant layout: `data/simulator/layout.py` (`PLANT_LAYOUT`, `line_members`, `duty_profile`,
  `python -m data.simulator.layout`). Lines 3-4 hold the seven fault-history motors + MTR-018.
  `Asset.role` is a DB column (`assets.role`, ALTER on old databases in `db.init_db`), and the seed
  upserts the catalogue on every start, so a Docker volume seeded before Phase A gets the new
  names/lines/roles on restart. MTR-042 = Packaging Line 3 filler. Rated kW/criticality are keyed
  by asset id in `generate.py::_RATINGS`; the telemetry is byte-identical to before (checked).
- Twin API (`apps/api/routes/twin.py`): `GET /twin` (snapshot, polled every 2 s by the page),
  `GET /twin/hud/{id}`, `GET /twin/rewind`, `GET /twin/events?since=`, `WS /ws/twin` (250 ms
  frames). The live bus `apps/api/live.py` is in-process: it starts empty on every API restart
  (all motors show "no data" until the first message arrives; this is honest, not a bug).
  `TWIN_STALE_AFTER_S` (30 s) decides "no data".
- Status precedence (API, `twin_status`): offline (work order in progress) > no_data > critical
  (p ≥ 0.5) > warning (p ≥ 0.3) > healthy > no_prediction. The Dashboard sprite tiers still use
  0.15/0.4; the twin and HUD use the API thresholds. Colours: `apps/web/src/twin/palette.ts`
  (`TWIN_THEME = "neon" | "app"`, one line) and `STATUS_COLOR`; glyph shapes in `status.ts`.
- Decision badges come from `decision_state`; a pending contract created after the newest work
  order's approval wins over that older work order (so a second analysis after a repair shows
  "contract pending"). Prompt B should add its own `DecisionState` value(s) (e.g.
  `design_change_pending`, `upgraded_spec_installed`) in `schemas.py` + `types.ts` +
  `DECISION_WORD` + the scene's badge styles (`scene/PlantScene.tsx` / `scene.css`).
- HUD modules each have a `data-testid` (`hud-probability`, `hud-regime`, `hud-sensors`,
  `hud-drivers`, `hud-decision`, `hud-provenance`, `hud-plc`, `hud-mes`); add a Designer module
  there rather than a new panel. PLC tags: `apps/api/plant.py::plc_tags` (`source` = telemetry
  field, `simulated` + `basis` otherwise). MES: `simulated_mes_context`, imported by the twin
  route only; `tests/test_twin.py` greps `agents/`, orchestrator, scoring, pipeline, approvals and
  `edge/` for it. Keep it that way.
- The simulated-CMMS "upgraded spec installed" step (Prompt B) can be surfaced as a work-order
  event note + a new decision state; the replay's `work_orders.py` already posts insert-only
  events the twin reads through `db.work_order_by_asset`.

## Verified in a real browser (headless Chrome over CDP, this laptop = Iris Xe, 1080p)

- Overview with live pulses at 33.6 msgs/s: 56-60 fps; performance mode 60 fps; HUD open 48 fps
  (drei `Html` labels + 420 px drawer; still above the 45 fps bar).
- Counters are measured at the control plane: msgs/s, preds/s, edge inference p50 0.6 ms,
  motor→edge→control plane p50 ~525 ms (dominated by the edge's 0.5 s POST batching interval
  `EDGE_POST_INTERVAL_S`, which is a real property of the system; say so if asked).
- No Chrome extension / in-app browser is available to Claude Code on this laptop; the
  screenshot route is `chrome.exe --headless=new --remote-debugging-port=9333` + a small CDP
  script (scratchpad `app_cdp.mjs` pattern) against `npx vite --port 5175` with
  `VITE_API_BASE_URL=http://127.0.0.1:8010`.

## Native end-to-end recipe that worked (while the team's Docker stack was up)

- API on :8010 with scratch `DATABASE_URL`/`MODELS_DIR`/`CHROMA_PATH`/`MLFLOW_TRACKING_URI`,
  `THOR_SKIP_BACKGROUND=0`, `MQTT_BROKER_URL=mqtt://127.0.0.1:1883` (the Docker broker, read-only
  subscription), `EDGE_URL=http://127.0.0.1:8011`; native edge `uvicorn edge.main:app --port 8011`
  with the same `MODELS_DIR` and `CONTROL_PLANE_URL=http://127.0.0.1:8010`; `scripts.demo_prep`
  against :8010. The Docker replay's MQTT stream then feeds the native stack without touching the
  Docker one. Do NOT run a second replay against the Docker broker: it would feed the team's stack.
- With the Docker stack down, the simpler recipe in CLAUDE.md §16 (HTTP replay) works but has no
  edge, so only `POST /pipeline/score` / Run analysis produce predictions.

## Gotchas

- Bash heredocs with apostrophes/backticks silently failed on this machine; use the Write tool
  for files, Python scripts for patching.
- `data/simulator/out` is now in `.dockerignore`; images generate the fleet at startup
  (`SEED_ON_START=1`), so a stale local catalogue can never be baked in.
- `edge_up` in `/twin` probes `EDGE_URL`; natively that can hit the Docker edge on :8001 and
  truthfully report "up" for the other stack.
- The rewind bar lays its series on plant time; contract/approval wall-clock times are not on
  the axis (only early_warning, predicted_failure, work_order_started/completed are).

# Phase A.1 notes (the hero demo shown by hand, 2026-10-12)

## What changed underneath (keep these true)

- **Fault timing is re-timed** (team-approved 2026-10-12; CLAUDE.md §9/§15). Six completed
  failures inside the seeded 80 %; MTR-042 healthy until 4 plant hours after the seed point,
  ramps over `FAULT_RAMP_H` = 16 h and then runs on the damaged bearing until the end of the
  timeline. Nothing in the data repairs it: the engineer's work order is the only repair. The
  ramp-and-hold exists because the quadratic-only progression cannot give both a quick red and a
  usable margin (red at ~70 % of the span, self-repair at 100 %, at any speed). `bearing_fault_delta`
  is the same per-sensor signature as a scalar function; the sandbox uses it.
- **The split has four groups now** (`make_asset_split`: ≥ 5 completed failures → validation and
  calibration are separate): 12 train / 3 validation / 3 calibration / 6 test, MTR-042 forced into
  test. Metrics on the re-timed data are in `docs/finals/metrics.md`; never quote the 0.865 / 41.5 h /
  0.844 numbers again.
- **Status is one rule in the API** (`apps/api/alarms.py`): on-delay 3 rows (the same sustained
  rule validation uses for lead time), critical LATCHED until a completed work order, warning clears
  after 6 rows below 0.3. `FleetAsset.status/alarm/alarm_since/alarm_reason/sandbox` and the same on
  `TwinAsset`; the browser never recomputes a status (the rewind scrub maps STORED points with
  `statusForProbability`, that is all). The raw per-sample probabilities stay visible (Asset 360
  chart, HUD sparkline). `GET /alarms` lists the insert-only transitions.
  - Precedence: offline > no_prediction > no_data (stale) > critical > warning > healthy.
    `DataState.none` (nothing received since the API started, e.g. the replay is held) does not
    hide a prediction; only `stale` (messages stopped) does.
  - The cursors are in memory and rebuilt from `alarm_events` after a restart (the rows after the
    newest event are walked again, capped at 5,000 per motor).
- **Nothing happens at start-up.** `docker compose up` seeds and holds: no model, no predictions,
  `GET /replay.released = false`. The replay container (`streaming/replay/sandbox.py::ReplayGate`)
  polls it every second and starts pass 0 only when a production model exists; `POST /replay/pause`
  / `resume` are the only other controls (in-memory, an API restart resumes). The promotion sign-off
  (`decide_promotion_db`) also POSTs `EDGE_URL/models/refresh` and scores the fleet once, so Fleet
  turns green within seconds of the click and the edge scores the live stream from its first window.
- **The `prep` compose service is a profile** (`docker compose run --rm prep`); a plain `up` never
  trains or promotes. `DEMO_PREP_TRIALS` default 4.
- **Sandbox** (`apps/api/sandbox.py`, `streaming/replay/sandbox.py`): insert-only `sandbox_events`
  (set / fault / shake / reset / clear); the active override of a motor is derived from the events
  after the newest reset; the replay rewrites its own rows and tags them `sandbox=True`; the edge
  clears a motor's buffer when the tag flips so a prediction window is never mixed. Everything
  downstream carries the tag (`predictions.sandbox`, `DecisionContract.sandbox_scenario`,
  `WorkOrder.sandbox_scenario`, `AlarmEvent.sandbox`). Excluded: training (`orchestrator._telemetry`
  → `read_telemetry(include_sandbox=False)`), drift, the rewind series and its lead time. "Reset
  sandbox" writes `retired_prediction_max_id` so retired sandbox predictions leave the displayed
  median, and clears sandbox-raised alarms with `reason="sandbox_reset"`. A sandbox work order
  starting writes a per-asset `clear`. Sliders re-derive the load-dependent sensors with the
  simulator's healthy relations (current, housing temperatures through the same lags, vibration
  scaled between state levels); the temperature slider sets the bearing housing (lagged).
- **Ground truth isolation** still holds: `sandbox` is an observable tag on the reading, not ground
  truth; `tests/test_ground_truth_isolation.py` expects it on `latest`.
- **Line endings:** `Path.write_text` on Windows writes CRLF. Patch scripts must open with
  `newline="\n"`; `git ls-files --eol | grep w/crlf` shows offenders.

## Native hero-flow stack (what was used to measure)

- Broker: `C:\Program Files\mosquitto\mosquitto.exe -c <conf>` with `listener 1884`,
  `allow_anonymous true` (the Windows service stays stopped; the Docker broker owns 1883).
- API on 8010 (scratch `DATABASE_URL`/`MODELS_DIR`/`CHROMA_PATH`/`MLFLOW_TRACKING_URI`, keys " ",
  `MQTT_BROKER_URL=mqtt://127.0.0.1:1884`, `EDGE_URL=http://127.0.0.1:8011`, `SEED_ON_START=1`);
  edge on 8011 (`CONTROL_PLANE_URL=http://127.0.0.1:8010`, `EDGE_MODEL_POLL_S=3`); replay with
  `REPLAY_SPEED=900`, `CONTROL_PLANE_URL=http://127.0.0.1:8010` (held by the gate).
- Driver script: scratchpad `hero_measure.py` (train → sign-off → watch MTR-042 → analysis → Bolt
  → approve → green; writes a JSON timeline). Timings are in `docs/finals/demo-walkthrough.md`.

## What the rehearsals taught (2026-10-12, keep in mind for the slides and Q&A)

- **Training on stage takes about 72 s on the Docker stack** with 4 trials per family (replay held,
  nothing else running); the same run took 2.5-3 min natively while the Docker stack was also
  streaming. Validation + parity are ~35 s of it. Do not open anything heavy during step 2.
- **The champion varies run to run** (Optuna): random forest in every Docker/native run on the
  re-timed data (recall 0.92-0.93, precision 0.90-0.96, lead time 53-59 h, IMS 0.88-0.91). One
  native run produced a LightGBM whose raw scores reached -1.7e6 on pure leaves; Platt scaling on
  that input was a flat line (test recall 0.33) and the edge parity gate blocked it — the gate
  did its job, and the fix (calibrate every family on `predict_proba`, `_ProbabilityOnly` in
  `agents/validation/checks.py`, unwrapped by `lifecycle._unwrap_estimator`) removed the failure
  mode. If a train run ever fails the gate on stage: train again (it is a different Optuna
  search) or run the fallback (`.\scripts\demo_reset.ps1 -Prep`).
- **There is no amber phase to promise.** The calibrated probability steps from < 0.001 to 0.81
  within one 10-minute sample, so the warning and the critical alarm fire on the same sample.
  Say "amber then red" only if you see it; the alarm log (`GET /alarms`, the HUD status line) is
  the honest record.
- **LLM mode costs time:** Run analysis 30 s (the provider drafts the paragraph) vs 4-5 s in
  template mode; Bolt 15-30 s per answer vs < 1 s. Ask Bolt the first question as soon as the
  analysis starts. Template mode is the fallback and the demo is identical otherwise.
- **This run's lead time** on the rewind depends on how long the motor sat red before the
  approval (24-39 h in the rehearsals); the validated figure (53.1 h) is the registry's.
- **False latches:** none over a full replay pass without the sandbox (only MTR-042 alarmed).

## Phase A.3 (2026-10-12): what "API Key Down" means, exactly

- **Where the key comes from.** Natively, pydantic-settings reads `ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` from the process environment first and the repo-root `.env` second. In Docker
  the `api` service gets `.env` through `env_file`, so inside the container the key is a process
  environment variable (`.env` itself is not in the image, `.dockerignore`). `get_settings()` is
  an `lru_cache`d `Settings` object: the one in-memory copy of the key.
- **Down** (`orchestrator.key_down()`, `POST /llm/key/down`): sets the switch to Down and blanks
  `anthropic_api_key` / `openai_api_key` on that cached `Settings` object. No client is cached
  anywhere (clients are built per call and discarded), so nothing in Thor's runtime objects holds
  the key. Every path checks the switch first: `_provider()` is `none`, `_api_key()` /
  `_openai_key()` are blank, `_llm_enabled()` is False, `_llm_client()` / `_openai_client()`
  raise, `llm_status()` says template, and Bolt, `draft_explanation` and the routing call use the
  templates. **What Down does not do:** it does not modify or delete `.env`, and it does not
  remove the variable from the process environment (natively or in the container) -- that is
  the source Up re-reads from, and no code path reads it except `key_up()` (CLAUDE.md §7,
  `tests/test_key_switch.py::test_only_orchestrator_touches_the_switch_state`). Say on stage:
  "the API drops the key from memory; nothing in Thor holds it until a person turns it back on",
  not "the key is gone from the machine".
- **Up** (`key_up()`, `POST /llm/key/up`): clears the settings cache and rebuilds `Settings` from
  the same source, then sets the switch Up. With no key at the source it reports
  `provider: none` and stays in template mode. Startup state is Down unless `LLM_KEY_ON_START=up`.
- **Never exposed:** `GET /llm/key` and `/system/health` carry `{key: up|down, provider}` only;
  the tests assert the fake key is absent from every response and from `llm_status()`.
- The §16 trick of setting the keys to a space is no longer needed for a native check: the switch
  starts Down. Leave the real keys in `.env` reachable when a rehearsal needs Up.

## Phase A.3 (2026-10-12): findings, labels, Bolt's memory -- design notes

- `repair_findings` is insert-only like contracts and approvals; the effective finding of a
  contract is its newest non-sandbox row (every later row supersedes the earlier ones; both stay
  listed, the UI greys the superseded one). `POST /findings` fills `supersedes` with the current
  finding when the client omits it and refuses any other value (409).
- `WorkOrder.finding` (derived view) carries the effective finding, so the Fleet card, Asset 360
  and Bolt's `get_asset` see it without a second query.
- Training: `_telemetry` builds the training frame as the recorded history + the labelled rows
  of every confirmed finding (`tool_apply_field_labels`); the split is computed on the history
  alone (`h.cache["df_history"]`) and `tool_adopt_into_train` moves the confirmed motor into
  train. A finding whose last reading sits more than `COMPLETED_FAILURE_MAX_H` (1 h) before the
  outage cannot stand in for a completed failure and is skipped with a warning (the demo's
  simulated CMMS takes the motor offline at the next replayed row, so the gap is one step).
- Bolt: `search_field_history` returns `this_plant` (deterministic keyword / asset-family match,
  `apps/api/findings.plant_precedent`) next to the public `cases`; the template puts the plant
  findings first and says "no confirmed finding yet" when there are none. In LLM mode the tool
  description tells the model to quote a finding with its source and name and never invent one;
  `asset_id` is an optional tool argument so the plant findings are matched to the asset in
  context.

## After the judge audit (2026-10-12, late)

- `streaming/replay/work_orders.py::UnrepairedFault`: in replay passes after the first, a motor
  the data never repairs (MTR-042: `FAULTY_ASSETS` failure at 1.0 and a `FAULT_RAMP_H` entry)
  keeps its own full-severity rows instead of the healthy rows at the start of the looped story,
  until the plant's work order (the RepairSimulator runs after it and wins). The replay loops the
  story tail, not the month, so the historical failures never replay. Net: nothing in the data
  ever heals the hero; the approval can wait as long as the Q&A needs.
- `orchestrator._telemetry` (train runs) keeps only rows with stored ground truth: replayed rows
  carry no outcome and used to train as negatives (a second "Train new model" after the sign-off
  failed the recall gate). Training is always "the recorded history", whatever has streamed.
- `AlarmEvent.plant_ts` of a raise is the FIRST row of the sustained run (= the lead-time rule and
  the rewind marker); a clear is dated at its own row.
- `sandbox.shake` skips motors with a plant-raised alarm so Reset can never clear a real alarm.
- Web: built files live under `/static/` (`vite build.assetsDir`), so the SPA route `/assets/:id`
  survives a refresh behind nginx; the sandbox colour is violet (`--tw-sandbox`), amber is only
  ever "warning"; `|SHAP| < 0.01` drivers read "little effect".
- Known and accepted: the plant manuals (`data/manuals/*.md`) say "regime" and are quoted
  verbatim; the 3-D overlays overlap when Overview + X-Ray + Sandbox are all on; one window of
  predictions is skipped when an override starts or stops.
