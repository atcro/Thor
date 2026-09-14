# Thor — 3-minute demo script

One motor, one emerging fault, one decision. No feature tour.

## Setup (before recording)

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY for LLM-drafted explanations; blank = template mode
docker compose up --build       # api :8000, web :5173, mlflow :5000, edge :8001, mqtt :1883
```

Wait for `api` to log `seeded 24 assets` and the replay container to start publishing. The replay
runs at 600x (10 minutes of telemetry per second), so MTR-042's bearing degradation is visible
within about a minute of wall-clock time. Open http://localhost:5173.

## Shot list

| t | Screen | What happens | What to say |
|---|---|---|---|
| 0:00 | Fleet | 24 motors, all green. MTR-042 turns amber as vibration RMS climbs. The System panel shows db / mqtt / mlflow / edge all up. | "Thor watches the whole fleet. This is replayed historical telemetry over MQTT, not random noise, so the story is reproducible." |
| 0:25 | Asset 360 | Click MTR-042. Live vibration + bearing temperature chart; regime badge reads R3 (high load). | "Before anything trains, Thor clusters operating state. A load change is never mistaken for a fault." |
| 0:40 | Asset 360 | Click **Run analysis**. Stepper advances: profiling → training → validating → explaining. | "One LangGraph orchestrator, five deterministic toolboxes. The LLM decides what runs next; it never computes a number." |
| 1:10 | AutoML Studio | Switch tabs while training finishes. Data quality 96/100, task = classification, 13 regime-normalized features, three candidates (RF / XGBoost / LightGBM) ranked by **Industrial Model Score** — recall, lead time, calibration, latency — not accuracy. | "Validation is asset-level and time-ordered: the champion is scored on machines it has never seen." |
| 1:40 | Asset 360 | Stage = awaiting approval. The Decision Contract panel: 81% failure probability, SHAP bars, three cost options, recommended window, two cited manual passages, explanation text. | "Prescriptive, not just predictive. Under configurable plant-cost assumptions, maintaining in Friday's low-load window beats stopping now and beats running to failure. Every number here came from Python; the paragraph came from the LLM, restricted to those numbers." |
| 2:15 | Asset 360 | Type an approver name, click **Approve**. Panel locks; approval row appears under the contract. | "Nothing executes without this click. The contract is immutable — the approval is a linked row, and the contract hash is the audit trail. This JSON is what hands off to the plant's CMMS." |
| 2:35 | ModelOps | Registry shows v17 promoted to production; ONNX artifact path; edge container picks it up. Drift panel shows PSI per feature. | "Approval also promotes the champion and ships it to the edge container over a mounted volume. MLOps keeps watching for drift after the demo ends." |
| 2:50 | Copilot | Ask Bolt: "why is MTR-042 at risk?" It answers from the contract via tool calls, and reminds you it cannot approve anything. | "The copilot reads the same evidence; it cannot act." |

## Fallbacks

- No API key: explanations and Bolt run in template mode (deterministic prose). The demo is
  identical otherwise. Say so on camera — it demonstrates that the LLM is not load-bearing.
- No Docker: `python -m data.simulator.generate`, `uvicorn apps.api.main:app --reload`,
  `python -m streaming.replay.replay --once --speed 0`, `npm --prefix apps/web run dev`.
- Reset between takes: `docker compose down -v` (drops the volumes; the replay restarts from t=0).

## Never say

- A specific dollar saving as fact — always "under these configurable plant-cost assumptions".
- "MES". Thor is a companion to the plant MES/CMMS; the Decision Contract is the handoff.
