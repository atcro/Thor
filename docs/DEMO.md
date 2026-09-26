# Thor — 3-minute demo script

One motor, one emerging fault, one decision. No feature tour.

## Setup (before recording)

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY for LLM-drafted explanations; blank = template mode
docker compose up --build       # api :8000, web :5173, mlflow :5000, edge :8001, mqtt :1883
```

Wait for `api` to finish seeding and the replay container to start publishing. The API seeds the
first 80% of the 30-day history (four completed bearing failures the model can learn from); the
replay then streams the remaining six days at 1800x (30 minutes of plant time per second, so
about 5 minutes of wall-clock). Timeline from the moment the replay starts:

- ~1 min: MTR-021 fails unplanned (no model was watching it yet) — the "old way".
- ~3 min: MTR-042 enters the 48 h pre-failure window; its health index is visibly falling.
  Click **Run analysis** here. Training takes ~60-90 s.
- ~5 min: the replay reaches the end of the history and loops.

`docker compose down -v` resets everything to t=0. Open http://localhost:5173.

## Shot list

| t | Screen | What happens | What to say |
|---|---|---|---|
| 0:00 | Fleet | 24 motors, all green. MTR-042 turns amber as vibration RMS climbs. The System panel shows db / mqtt / mlflow / edge all up. | "Thor watches the whole fleet. This is replayed historical telemetry over MQTT, not random noise, so the story is reproducible." |
| 0:25 | Asset 360 | Click MTR-042. Live vibration + bearing temperature chart; regime badge shows the current operating state (R1 idle/startup, R2 nominal, R3 high-load; the fleet spends roughly 23 / 47 / 30 % of its time in each). | "Before anything trains, Thor clusters operating state. A load change is never mistaken for a fault." |
| 0:40 | Asset 360 | Click **Run analysis**. Stepper advances: profiling → training → validating → explaining. | "One LangGraph orchestrator, five deterministic toolboxes. The LLM decides what runs next; it never computes a number." |
| 1:10 | AutoML Studio | Switch tabs while training finishes. Data quality 100/100 (clean synthetic channels; the score drops for missing or flatlined signals), task = classification, 13 regime-normalized features, three candidates (RF / XGBoost / LightGBM) ranked by **Industrial Model Score** — recall, lead time, calibration, latency — not accuracy. | "Validation is asset-level and time-ordered: the champion is scored on machines it has never seen." |
| 1:40 | Asset 360 | Stage = awaiting approval. The Decision Contract panel: calibrated failure probability of about 0.5 within the 48 h horizon (isotonic calibration on rare positives, so it is honest rather than dramatic), SHAP bars led by bearing temperature and vibration RMS, three cost options, the recommended option and window, cited manual passages (the first is motor-maintenance.md section 4.2 Drive-end bearing wear), explanation text with a "drafted by LLM" or "template" chip. | "Prescriptive, not just predictive. Under configurable plant-cost assumptions Thor compares maintain now, maintain in the next low-load window, and run to failure, and recommends the cheapest expected outcome -- on the recorded runs that was maintain now. Every number here came from Python; the paragraph came from the LLM, restricted to those numbers." |
| 2:15 | Asset 360 | Type an approver name, click **Approve**. Panel locks; approval row appears under the contract. | "Nothing executes without this click. The contract is immutable — the approval is a linked row, and the contract hash is the audit trail. This JSON is what hands off to the plant's CMMS." |
| 2:35 | ModelOps | Registry shows the new version (ids look like v62033) promoted to production; ONNX artifact path; within about 30 s the edge container picks it up and the System panel's edge dot turns on. Drift panel shows PSI per feature. | "Approval also promotes the champion and ships it to the edge container over a mounted volume. MLOps keeps watching for drift after the demo ends." |
| 2:50 | Copilot | Ask Bolt: "why is MTR-042 at risk?" It chains three read-only tools and answers in three parts: the contract's evidence, the contract's own manual citation, and precedent -- real unplanned work orders on motor bearings from a public dataset, with labor hours. Follow with "how long does a bearing job usually take?" (quartiles over ~50 similar cases) and "approve it" (it refuses and points at the UI). | "The copilot reads the same evidence, adds precedent from 8,000 real work orders, and still cannot act." |

## Numbers you can quote (from recorded runs, `docs/BUILD_STATUS.md`)

Results move between runs because Optuna trials and the replay position differ; quote ranges,
not single values, unless you are reading them off the screen.

| item | recorded values |
|---|---|
| training data | 24 motors, 30 days, ~83 k rows seeded (80 %), 4 completed failures as labels |
| data quality score | 100 / 100 |
| operating regimes | 3 (idle/startup, nominal, high-load) |
| features | 13, regime-normalized |
| champion | LightGBM (IMS 0.88, held-out recall 0.89) or XGBoost (recall 0.76) depending on trials |
| warning lead time (held-out failures) | 38 h to 101 h median |
| calibrated P(failure within 48 h) at the gate | 0.07 early in the window, about 0.5 inside it |
| time to the approval gate | 82 to 85 s |
| edge pickup after approval | ~30 s, then predictions flow to the Fleet screen |
| field history behind Bolt | 8,355 real unplanned work orders (FMUCD, CC BY-NC) |

Never say a dollar figure as fact: the plant-cost assumptions are configurable settings.

## Fallbacks

- No API key: explanations and Bolt run in template mode (deterministic prose). The demo is
  identical otherwise. Say so on camera — it demonstrates that the LLM is not load-bearing.
- No Docker: `python -m data.simulator.generate`, `uvicorn apps.api.main:app --reload`,
  `python -m streaming.replay.replay --once --speed 0`, `npm --prefix apps/web run dev`.
- Reset between takes: `docker compose down -v` (drops the volumes; the replay restarts from t=0).

## Never say

- A specific dollar saving as fact — always "under these configurable plant-cost assumptions".
- "MES". Thor is a companion to the plant MES/CMMS; the Decision Contract is the handoff.
