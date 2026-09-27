# Thor Prototype Technical Documentation

## ABB Hacker Earth submission

Thor is a predictive maintenance prototype that turns industrial motor telemetry into a reviewed maintenance recommendation. It profiles the data, trains and validates candidate failure models, compares maintenance choices under configurable cost assumptions, and presents an evidence linked Decision Contract for a person to approve. Bolt is the companion copilot for reading fleet state, contracts, maintenance manuals, and work order precedent. Thor is designed to hand an approved decision to an existing MES or CMMS; the prototype does not dispatch work or control equipment.

## Solution architecture

The Docker Compose deployment contains seven services: TimescaleDB, Mosquitto MQTT, MLflow, the FastAPI control plane, the edge inference service, telemetry replay, and the React web application. The control plane stores telemetry and workflow state in PostgreSQL with TimescaleDB, while the edge service consumes MQTT telemetry and picks up approved ONNX models from a shared model volume. MLflow tracks experiments. Chroma stores local embeddings for maintenance manual passages and a separate work order history corpus.

The user interface exposes Fleet, Asset 360, AutoML Studio, ModelOps, and Copilot screens. Fleet and Asset 360 show monitored motors and the selected motor's telemetry; Studio shows profiling, candidates, and validation; ModelOps shows model lifecycle and drift; Copilot exposes Bolt's evidence lookup.

### Analysis and decision flow

1. **Ingest and profile.** The simulator creates a reproducible 24 motor, 30 day history with injected bearing degradation. The API seeds part of that history; replay publishes the remainder over MQTT. Data Reliability checks missing or flatlined channels and groups operating regimes so load and startup behavior can be accounted for.
2. **Train candidates.** ML Architect builds regime normalized features and uses Optuna to search Random Forest, XGBoost, and LightGBM candidates. Candidates are compared using the Industrial Model Score, which accounts for recall, warning lead time, probability calibration, and inference latency.
3. **Validate.** Validation applies leakage checks and time ordered, asset level backtesting, calibrates probabilities, and estimates warning lead time and remaining usage intervals. The selected model is evaluated on motors withheld from training.
4. **Explain and compare actions.** Reliability computes SHAP attributions, retrieves cited maintenance manual passages, and compares the expected costs of maintaining now, maintaining in a later low load window, and running to failure. Costs are scenario assumptions, not measured savings.
5. **Gate and deploy.** The API writes a hashed Decision Contract containing evidence and the proposed action. A named human approver must explicitly accept or reject it; a second decision on the same contract is refused. Approval can advance the model lifecycle and export ONNX plus a feature and baseline sidecar. The edge service then loads the model and publishes predictions back over MQTT.

A LangGraph orchestrator coordinates these deterministic toolboxes. An optional OpenAI or Anthropic model can route and phrase explanations, but numerical results come from Python. Without an API key, routing and prose fall back to deterministic templates. Bolt uses read only tools to retrieve fleet, asset, contract, manual, and work order evidence. It cannot approve a decision, train a model, or promote a model.

## Technologies and their roles

|Component|Technology|Role|
|-|-|-|
|User interface|React, TypeScript, Vite|Fleet monitoring, analysis, decision review, and copilot screens|
|Control plane|Python, FastAPI, Pydantic, SQLAlchemy|API, telemetry persistence, workflow state, and approval endpoints|
|Workflow|LangGraph|Analysis sequencing and agent tool coordination|
|Data and optimization|pandas, NumPy, scikit learn, Optuna|Feature engineering, operating regimes, model search, and validation|
|Candidate models|Random Forest, XGBoost, LightGBM|Failure classification|
|Explanation and evidence|SHAP, Chroma, local embeddings|Feature attribution and retrieval of cited manuals and work orders|
|Tracking and lifecycle|MLflow, ONNX, ONNX Runtime|Experiment tracking, model export, and edge inference|
|Streaming and storage|Mosquitto MQTT, PostgreSQL with TimescaleDB|Telemetry and prediction messages; persistent application data|
|Packaging|Docker Compose|Reproducible local seven service deployment|
|Optional language model|OpenAI or Anthropic|Explanation drafting and Bolt tool assisted answers|

## Implementation approach and scope

The prototype uses synthetic telemetry to make the motor fault and maintenance decision reproducible. Completed historical failures supply training labels; an emerging fault is treated as unknown rather than labeled using future information. The replay accelerates the final portion of the history so an operator can observe a motor approaching the 48 hour failure horizon during a short demo.

The Decision Contract is the system's integration boundary. It retains the risk estimate, explanation, cited evidence, cost comparison, recommendation, and approval linkage in a structured audit artifact. A production MES or CMMS connection is a future handoff; this archive implements the contract and approval workflow, not a live work order integration. Model shadow and production stages are represented in the prototype lifecycle, without real production traffic for shadow evaluation. The edge currently serves the tree model's uncalibrated probability, so its fleet score can differ from the calibrated probability in a Decision Contract.

The repository also includes an AI4I 2020 benchmark path and a Bolt evaluation harness. Bolt's optional precedent corpus contains 8,355 selected unplanned rotating equipment work orders from the public FMUCD dataset, described in `data/external/README.md`; these are precedent, not site specific plant records or authoritative maintenance instructions.

## Setup and run

### Prerequisites

Install Docker Desktop or Docker Engine with the Compose plugin. Allow Docker enough disk and memory for Python ML images, TimescaleDB, MLflow, and local embedding initialization. Internet access may be needed on the first build to fetch images and dependencies and on the first manual index build to retrieve the embedding model. Run the following commands from the extracted `Thor` directory.

```bash
cp .env.example .env
# Leave API keys empty for deterministic template mode.
docker compose up --build
```

On Windows PowerShell, copy the environment template with `Copy-Item .env.example .env`. No `.env` file is included in the submission, and no API key is provided. Judges who want the language model enabled should place their own OpenAI or Anthropic key in the `.env` they create from `.env.example`; with the keys left blank, the full pipeline, approval gate, and Bolt copilot run in deterministic template mode. Do not commit or share `.env`.

|Address|Purpose|
|-|-|
|http://localhost:5173|Web application|
|http://localhost:8000/docs|Interactive API documentation|
|http://localhost:8000/system/health|Database, messaging, tracking, edge, and language model status|
|http://localhost:5000|MLflow interface|
|http://localhost:8001/health|Edge service status|

Wait for the API to seed telemetry and the replay service to publish MQTT messages. Open Fleet, select `MTR-042` in Asset 360, and use **Run analysis** when its condition worsens. Review the candidates and Decision Contract, then enter an approver name and approve if you want to demonstrate promotion and edge pickup. The demo sequence is expanded in `docs/DEMO.md`. The default seed and replay fractions are 0.80 and replay speed is 1800; the exact timing and model metrics vary with replay position and training runs.

To run detached and inspect services:

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api replay edge
```

To stop while keeping data, use `docker compose down`. To reset the demo database and volumes, use `docker compose down -v` **only when discarding the current demo state is intended**. The latter deletes Compose volumes including stored telemetry, MLflow state, Chroma indices, and exported models.

### Optional language model

The submission ships without an API key. To see language model routed explanations and Bolt answers, set your own `OPENAI\_API\_KEY` and optionally `OPENAI\_MODEL`, or `ANTHROPIC\_API\_KEY` and optionally `ANTHROPIC\_MODEL`, in the private `.env` file. `LLM\_PROVIDER=auto` prefers OpenAI when both keys are present. Restart the API or Compose stack after editing the file. Confirm the selected mode in `/system/health` or the Fleet System panel. The numerical pipeline and approval gate also operate in template mode. See `docs/API\_KEY.md` for a provider smoke test. External model calls may incur charges.

### Developer run and checks

With Python 3.11 or newer and Node.js installed, from the project root:

```bash
python -m venv .venv
# Activate the virtual environment using the command for your shell.
python -m pip install -e '.\[dev]'
python -m data.simulator.generate
uvicorn apps.api.main:app --reload
npm --prefix apps/web install
npm --prefix apps/web run dev
pytest
ruff check .
```

Run the API and web commands in separate terminals. The local API defaults to SQLite, whereas the full Compose deployment provides PostgreSQL, MQTT replay, MLflow, and edge inference as configured in `infra/docker-compose.yml`. For the complete demonstration, use Compose. Additional checks include `npm --prefix apps/web run build`, `npm --prefix apps/web test -- --run`, and `python evals/bolt/run\_bolt\_eval.py` for offline template mode. These commands are provided for reproducibility; the documentation itself does not claim a fresh test run in this environment.

## Submission notes

The recorded build report in `docs/BUILD\_STATUS.md` describes end to end Docker verification, automated checks, and example metrics. They are historical run results and should be presented as examples because search trials and replay timing change outcomes. This prototype does not claim measured production savings or deployment to actual plant assets. For a three minute walkthrough, follow `docs/DEMO.md`: show the changing fleet signal, run analysis, examine the contract and manual evidence, approve it, then show the model appearing at the edge and ask Bolt to explain the decision.

