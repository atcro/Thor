# Thor — submission draft (Theme 1: Agentic Predictive Maintenance Studio)

> Draft for the Prototype Phase form. Every claim below is either the approved language from
> `CLAUDE.md` §2 or a number recorded in `docs/BUILD_STATUS.md`. Fill the bracketed fields, trim
> to the form's length limits, do not add claims. Never state a dollar saving as fact and never
> describe Thor as an MES.

---

## Project name

**Thor** — with **Bolt**, its in-app copilot.

## One-liner

An agentic AutoML + MLOps platform that turns raw industrial sensor data into a validated,
explainable, human-approved predictive-maintenance decision.

## Theme

Theme 1 — Agentic Predictive Maintenance Studio.

## The problem

Most predictive-maintenance demos stop at an anomaly threshold and a dashboard. Plants do not
run on anomalies. They run on decisions: which machine, which fault, when to intervene, at what
cost, approved by whom. A load change looks like a fault to a naive model. Accuracy says nothing
about whether a warning arrives in time to plan a shutdown. And a recommendation nobody can
audit is a recommendation nobody will act on.

## What Thor does

Thor ingests raw sensor telemetry, profiles it, trains and validates a predictive-maintenance
model, explains its prediction with cited evidence, turns that into a cost-based maintenance
recommendation, and stops. Every consequential action requires an explicit human approval,
logged as an immutable **Decision Contract**.

Thor is a companion to the plant's MES/CMMS, not a replacement for one. It does not schedule
production, manage inventory, or dispatch labor. Its output, the Decision Contract, is a
handoff artifact designed to feed the systems a plant already runs.

## What makes it different

1. **Regime-aware fault detection.** Thor clusters operating state (RPM, load, startup) before
   anything trains, so a load change is never mistaken for a failure.
2. **Industrial Model Score.** Champion selection is weighted by recall, warning lead time,
   calibration, and inference latency, not raw accuracy. The model that wins is the one that
   warns early, is honest about its confidence, and runs at the edge.
3. **Decision Contract.** An immutable, evidence-linked audit record for every recommendation:
   provenance, explainability, and auditability in one JSON document.
4. **Prescriptive, not just predictive.** Thor compares the expected cost of maintain now,
   maintain in the next low-load window, and run to failure before recommending a window, under
   configurable plant-cost assumptions.
5. **Governed agent architecture.** The LLM only plans and explains. Every number comes from
   deterministic code. A human approves every action. This is the architecture, not a slogan:
   exactly one file in the codebase can call the model, and a test enforces it.

## How it works

One LangGraph orchestrator drives five deterministic toolboxes:

- **Data Reliability** profiles the telemetry, detects missingness, clusters operating regimes,
  and issues a Data Quality Contract.
- **ML Architect** infers the task, builds regime-normalized features, and runs an Optuna search
  over Random Forest, XGBoost, and LightGBM, ranked by the Industrial Model Score and tracked in
  MLflow.
- **Validation** checks for leakage, backtests with time-ordered, asset-level splits (the
  champion is scored on machines it has never seen), calibrates probabilities, and measures
  warning lead time.
- **Reliability** produces SHAP attributions, retrieves cited passages from the plant manuals,
  runs the cost comparison, and writes the Decision Contract.
- **MLOps** registers the champion, moves it through candidate → validated → shadow →
  production on approval, exports it to ONNX for the edge container, and watches for drift.

The orchestrator makes two LLM calls per run: one to route, one to phrase the explanation from
numbers Python already produced. Without an API key the explanation is a deterministic
template and the demo is otherwise identical, which is the point: the LLM is not load-bearing.

**Bolt**, the copilot, is a conversational surface over the same evidence. It reaches data only
through read-only tools on the control plane: fleet and asset state, contracts, the manual
index, and a field-history index of 8,355 real unplanned work orders on rotating equipment from
a public dataset (FMUCD, CC BY-NC). It can explain and cite; it cannot approve, promote, train,
or invent a citation.

## Tech stack

Python, FastAPI, LangGraph, an LLM behind one provider switch (OpenAI or Anthropic, tool
calling), scikit-learn, XGBoost,
LightGBM, Optuna, SHAP, MLflow, Chroma with local embeddings, PostgreSQL with TimescaleDB,
Mosquitto (MQTT), ONNX Runtime, React + TypeScript, Docker Compose, GitHub Actions.

## What works today

The full chain runs end to end through the HTTP API and in Docker with one command:

replay ingest → profile → three candidates → validation → champion → SHAP explanation with
manual citations → cost comparison → Decision Contract → human approval → promotion → ONNX
export → the edge container picks the model up on its own and starts publishing predictions.

Recorded results on the 24-motor, 30-day synthetic fleet with one emerging drive-end bearing
fault and four completed historical failures as labels (values vary between runs):

| item | recorded |
|---|---|
| champion | LightGBM, Industrial Model Score 0.88 |
| held-out recall | 0.89 |
| warning lead time on held-out failures | median ~100 h |
| calibrated P(failure within 48 h) inside the alarm window | ~0.5 |
| time to the approval gate | ~85 s |

Credibility check on real public data, same pipeline, AI4I 2020 (UCI): LightGBM champion,
recall 0.75, precision 0.87, AUROC 0.98, Brier 0.011 after calibration.

Verification: 161 automated tests, lint clean, frontend build and smoke tests passing, and a
51-case copilot eval that grades tool routing, citations, and refusal behaviour.

## Honest limitations

- The primary demo data is synthetic by design, for a reproducible "one motor, one fault, one
  decision" story. The AI4I benchmark is the evidence the method is not tuned to the story.
- The calibrated probability plateaus near 0.5 inside the 48 h window. That is isotonic
  calibration behaving honestly on rare positives, not a bug, and it is less dramatic on stage.
- The promotion lifecycle is modelled as a database state machine; there is no real production
  traffic to shadow-test against in a demo.
- The field-history corpus behind the copilot is campus facilities maintenance, the closest
  real public data with technician-written text. In a deployment it would be the plant's own
  CMMS history, which is exactly what the Decision Contract hands off to.

## Scalability

Asset → site → enterprise through the Decision Contract as the integration point. Every
recommendation leaves Thor as the same structured, hashed, approval-linked document, so
connecting a site means connecting one handoff to the existing MES/CMMS, not building a bespoke
integration per plant. The control plane is one service; the edge inference container is
independent of it and subscribes to telemetry directly.

## Rubric mapping

- **Innovation:** regime-awareness plus the prescriptive cost layer, not another
  anomaly-threshold demo.
- **Technical excellence:** time- and asset-aware validation, calibration, drift detection,
  MLflow registry, edge deployment.
- **Problem-solution fit:** directly implements every bullet in ABB's Theme 1 feature list.
- **Scalability:** asset → site → enterprise via the Decision Contract as the MES/CMMS
  integration point.
- **UX / presentation:** one motor, one emerging fault, one decision. No feature tour.

## Demo

[Video link] — three minutes, following `docs/DEMO.md`: fleet turns amber, regime badge,
run analysis, candidates ranked by Industrial Model Score, the Decision Contract, the approval
click, promotion to the edge, and Bolt explaining why with evidence, citation, and precedent.

## Repository

[https://github.com/atcro/Thor] — `docker compose up --build` starts everything: API, web,
TimescaleDB, MQTT, MLflow, replay, edge.

## Team

[Names, roles, contact]
