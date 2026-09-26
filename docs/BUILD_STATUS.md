# Thor — Build Status (updated 2026-09-25)

Prototype build of the full Theme 1 platform, done on 2026-09-13 by six parallel agents
working against `docs/INTERFACES.md`, then extended on 2026-09-25 (copilot corpus, eval, API-key
wiring -- see the session section below). Everything below is committed on `main`.

## What runs

The P0 chain from CLAUDE.md §13 runs end to end with no faked components, through the HTTP API:

```
ingest / replay → 04 profile (3 operating regimes, data-quality score)
→ 05 three Optuna candidates (RF / XGBoost / LightGBM) ranked by Industrial Model Score
→ 06 asset-level, time-ordered validation (leakage, backtest, isotonic calibration,
     warning lead time, quantile RUL interval) → champion
→ 07 SHAP attribution + cited manual passages (Chroma) + cost comparison
     (maintain now / maintain later / run to failure) → immutable Decision Contract
→ 09 human approval (a second decision on the same contract is refused with 409)
→ 08 promotion to production + ONNX export + baseline sidecar for the edge container
```

Result on the real 24-motor, 30-day synthetic fleet with MTR-042 inside its alarm window
(`SEED_FRACTION=0.95`, `n_trials=6`):

| item | value |
|---|---|
| champion | LightGBM, IMS 0.883 |
| held-out recall | 0.89 |
| warning lead time (held-out failures) | median 100.7 h, p90 114.8 h (2 events) |
| validation gate | passed |
| calibrated P(failure within 48 h) | 0.50 |
| top SHAP drivers | bearing temperature (+0.19), bearing-temp z-score (+0.17), vibration RMS (+0.12) — all raise risk |
| first citation | `motor-maintenance.md` §4.2 Drive-end bearing wear |
| recommendation | maintain now (under the configured plant-cost assumptions) |
| wall time to approval gate | ~85 s |

After approval: `thor-bearing-classifier v… → production`, `models/<name>_<version>.onnx` plus
a JSON sidecar carrying `features`, `window_rows`, `calibration`, and the per-regime
`baseline` the edge container needs to compute the same features.

## Verification (2026-09-13; current numbers in the session section below)

- `ruff check .` — clean
- `pytest` — **146 passed**, including `tests/test_e2e_pipeline.py` (the whole chain, real
  toolboxes, ~10 s on the conftest fleet)
- `npm --prefix apps/web run build` and `npm --prefix apps/web test -- --run` — pass
- AI4I 2020 credibility benchmark (`python -m data.benchmark.ai4i_benchmark`, real UCI data):
  LightGBM champion, recall 0.75 / precision 0.87 / AUROC 0.98 / Brier 0.011 after calibration
  (see `data/benchmark/README.md` for what is and is not comparable)

## Docker verification (2026-09-14)

`docker compose up --build` was run after starting Docker Desktop. All four images built
(`thor-api` 3.4 GB, `thor-replay` 634 MB, `thor-edge` 440 MB, `thor-web` 103 MB) and all
seven services came up healthy. Observed within the first minute:

- `GET /system/health` → db ok, mqtt true, mlflow true, 85,058 telemetry rows (80 % seed +
  live rows arriving through replay → Mosquitto → API subscriber; `last_ingest_ts` advancing).
- `GET /fleet` → 24 assets; MTR-021 (failing live) and MTR-042 (amber) ranked first.
- Edge: MQTT connected, 24 assets seen, 2.8 k rows buffered, no model yet (expected).
- Web (`:5173`) and MLflow (`:5000`) serve; the API downloaded the MiniLM embedding model and
  built the manual index in its background thread.
- Fixed while verifying: the API probed `localhost:8001` for the edge health dot; compose now
  sets `EDGE_URL=http://edge:8001`.

Second run (stack restarted with the `EDGE_URL` fix): the full approval flow was driven
through the containers via the HTTP API —

- run to the approval gate in 82 s (XGBoost champion, held-out recall 0.76, lead time 37.8 h,
  validation passed; at that point the replay was ~4 days before MTR-042's failure, so the
  48 h probability was honestly low at 0.07);
- approval → `thor-bearing-classifier v62033` promoted to production → ONNX + sidecar
  written to the shared `models` volume;
- the **edge container picked the model up on its own** (`model_version v62033`,
  168 predictions within ~30 s, no errors) and published `predictions/{asset_id}`, which the
  API ingested: `/system/health` shows `edge: true`, and `/fleet` switched from the health
  index to model-driven probabilities (MTR-042 p = 0.56 → health 43.7 as the replay advanced).

Note: the edge serves the tree model's *uncalibrated* probability (`calibration: "none"` in
the sidecar), so fleet numbers from the edge and the contract's calibrated probability are
not on the same scale; healthy motors sit at the uncalibrated base rate (~0.11).

## Session 2026-09-25 -- copilot corpus, eval, API-key wiring

Six commits (`784850e` .. `308fabd`). Nothing in the P0 pipeline changed; all work is on the
copilot (Bolt), its retrieval layer, and operator plumbing.

**API key placeholders** (`784850e`). `.env` / `.env.*` are git- and docker-ignored;
`.env.example` documents blank-key template mode. `orchestrator.llm_status()` reports
`mode` / `model` without exposing the key and warns on a non-`sk-ant-` value;
`orchestrator.llm_ping()` is a one-call operator smoke test. The API logs `LLM mode:` at
startup, `GET /system/health` returns it under `llm`, and the Fleet System panel shows an LLM
row. `docs/API_KEY.md` is the runbook for the day the key arrives.

**Bolt tools** (`76929d4`, `29aaa7d`, `affc21e`). Bolt went from five read-only tools to eight:

| tool | backing | notes |
|---|---|---|
| `search_manuals` | `rag.retrieve_manual_context` over the `manuals` collection | the only manual text Bolt may quote |
| `search_field_history` | `rag.retrieve_field_history` over a **second** collection `field_history_v2` | real unplanned work orders as precedent; never a manual citation |
| `field_history_stats` | same retrieval, up to 100 cases | labor-hour / cost quartiles, year range, top components, computed in Python |

Both modes use them: the LLM tool loop declares them; template mode keyword-routes to them.

**Field-history corpus** (`29aaa7d`, `8e1e391`). Source: Facility Management Unified
Classification Database (FMUCD), 3.73 M work orders from 12 North American universities,
2002-2021, CC BY-NC (`data/external/README.md`). `data/field_history/build_fmucd_slice.py`
carves unplanned orders on fans / pumps / motors / compressors / chillers / drives whose text
names a symptom, dedupes, and scrubs e-mail, phone numbers and named contacts (best effort;
the source is published with names in free text). Committed slice: **8,355 cases, 1.2 MB**.
Indexed at API startup next to the manual index (skipped on restart when the count matches).
Retrieval normalizes work-order shorthand (EXH/MTR/BRG/VFD/CHW ...) and re-ranks embedding
candidates with keyword overlap (0.7 sim + 0.3 overlap). The Kaggle "Maintenance Work Orders
Dataset" was evaluated and rejected: synthetic, with resolution notes drawn independently of
the reported issue (every issue pairs with all 24 notes at ~5 % each).

**Chained answers** (`8e1e391`). "Why is MTR-042 at risk?" now runs `get_asset` ->
`get_contract` -> `search_field_history` (query derived from the contract's risk-raising SHAP
drivers) and answers in three parts: contract evidence, the contract's own manual citations,
precedent with labor hours. Asking for a contract id appends precedent too. The LLM system
prompt asks for the same chain, and gained a **voice** section (handover-note tone, lead with
the conclusion, confidence stated in the contract's terms, say what returned nothing, plant
vocabulary, no filler).

**Bolt eval** (`affc21e`). `evals/bolt/`: 51 fixed cases (fleet, asset, why-chain, contract,
pending, manual citations, 12 real FMUCD phrasings, stats, guardrails) graded on route /
content / guardrail, writing `results.jsonl` + traces. First run scored 88 %; every failure
was a real template-routing bug -- notably "approve dc_..." was answered as a lookup instead
of refused. After fixes: **51 / 51**. `--mode llm` runs the same set once the key exists;
`--min-pass` is the intended CI gate.

**Demo script** (`308fabd`). `docs/DEMO.md` numbers replaced with recorded ones (quality
100/100, P(48 h) about 0.5 inside the window, maintain-now on recorded runs, regime shares,
version-id shape, ~30 s edge pickup) plus a quotable-numbers table with ranges.

**Verification at end of session:** `ruff check .` clean; `pytest` **161 passed**;
`npm --prefix apps/web run build` and tests pass; Bolt eval 51/51 (template, hashed embedding).

**Still not verified:** LLM mode. No key has ever been configured, so the drafted explanation,
the Bolt tool loop, the chained three-part answer and the voice are all unexercised. First
steps when it arrives: `docs/API_KEY.md`, then `python evals/bolt/run_bolt_eval.py --mode llm`.

**Housekeeping:** `streaming/__init__.py` had been dragged to the repo root (empty file);
restored. Raw datasets live in `data/external/` (ignored). `evals/bolt/out/` is ignored.

## Not verified in this session (2026-09-13)

- **LLM mode** — no `ANTHROPIC_API_KEY` was configured, so every run used the deterministic
  template explanation and the template Copilot. Both LLM paths exist in
  `apps/api/orchestrator.py` (the only module allowed to read the key) and fall back to the
  template on any failure.

## Decisions made during integration

1. **Seven injected faults instead of "1-2".** MTR-042 is the live demo fault (onset 55 %,
   caught before failure); MTR-021 fails live on a held-out motor; MTR-003/009/013/023 are
   completed historical failures the model learns from. With a single training failure the
   champion was unusable (recall 0.33, lead time 0 h). CLAUDE.md §9 updated.
2. **Only completed failures produce training labels.** The simulator carries ground truth
   for degradations still in progress; labelling those rows would be looking into the future.
   In-progress faults are treated as unknown (`agents/ml_architect/features.py`).
3. **Alarm threshold 0.3 for lead-time measurement** (classification metrics keep 0.5): a
   calibrated 30 % chance of failure within the horizon is actionable when unplanned cost far
   exceeds planned cost. Shared constant in `agents/validation/checks.py` and
   `agents/ml_architect/automl.py`.
4. **Demo timing.** The API seeds the first 80 % of the history (`SEED_FRACTION`); the replay
   streams the remaining six days at 1800× (`REPLAY_SPEED`, `REPLAY_START_FRACTION`), so the
   alarm window for MTR-042 arrives ~3 min after start. `docs/DEMO.md` has the shot list.
5. **12 h hazard floor.** A model that has not demonstrated early warning gets a conservative
   hazard scale instead of a curve that jumps to 100 % at the first planned window.
6. **SHAP robustness.** LightGBM + this shap release returns non-additive attributions from
   `TreeExplainer`; `explain()` checks additivity (base + Σ ≈ probability) and falls back to
   a model-agnostic explainer over the tree model with a stratified background (healthy fleet
   rows + the asset's recent rows).
7. **RAG self-heals** (builds the Chroma index from `data/manuals` if missing) and returns
   one passage per manual section.

## Known soft spots (P1 / P2 candidates)

- Calibrated P(failure) plateaus near 0.50 inside the 48 h window — honest isotonic behaviour
  on rare positives (the model fires ~100 h early, and rows 48–100 h before failure are labelled
  0 by the horizon definition), but less punchy on stage. A 72 h horizon would raise it.
- What-if deltas are 0 through the isotonic step function (`POST /whatif`).
- MLflow model registration logs a warning: params/metrics are tracked but no model artifact
  is logged under `model`, so `mlflow.register_model` is skipped (best effort).
- Fleet health before any prediction exists uses the simulator's ground-truth `health` column;
  once the edge container publishes predictions the score switches to `100·(1−p)`.
- `apps/web` types do not yet expose `FeatureSpec.baseline` (harmless; unused by the UI).

## Environment notes (Windows build box)

- In Git Bash the Microsoft Store `python` stub can shadow the real interpreter and hang; use
  `C:/Users/<you>/AppData/Local/Python/pythoncore-3.14-64/python.exe` or PowerShell. `ruff` is
  not on PATH — use `python -m ruff`.
- `docs/decision-contracts/example.json` is the reference Decision Contract shape and is
  validated against the schema by `tests/test_schemas.py`.

## Commit log

```
308fabd Align the demo script with recorded pipeline results
affc21e Add field_history_stats tool, Bolt eval harness, routing fixes, and Bolt voice
8e1e391 Chain evidence, citations and precedent in Bolt; hybrid field-history retrieval
29aaa7d Add FMUCD field-history corpus and Bolt search_field_history tool
76929d4 Add search_manuals tool to the Bolt copilot
784850e Add API key placeholder wiring, LLM mode visibility, and key runbook
f7f5784 Add Thor-vs-Bolt overview to the architecture doc and CLAUDE.md
8738cae Integrate the P0 chain end to end on real simulator data
136a88e Add AI4I 2020 credibility benchmark
f09a98d Add Validation (06) and Reliability (07) agents with manual corpus
6e67d64 Ship the regime baseline to the edge sidecar
9946fd9 Add LangGraph orchestrator, FastAPI control plane, and human approval gate
e6735a3 Add React + TypeScript frontend with five screens
6bfc24b Add Data Reliability (04) and ML Architect (05) agents
248e9e7 Add synthetic fleet simulator, MQTT replay, telemetry ingest, edge ONNX service
3f12368 Scaffold Thor platform: contracts, DB layer, compose, MLOps agent
```
