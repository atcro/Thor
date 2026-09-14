# Thor

An agentic AutoML + MLOps platform that turns raw industrial sensor data into a validated,
explainable, human-approved predictive-maintenance decision.

Governed architecture: **LLM plans, deterministic code computes, engineer decides.**
Thor is a companion to the plant MES/CMMS -- its output, the Decision Contract, is a handoff
artifact designed to feed one, not become one.

## Run everything

    cp .env.example .env        # optionally add ANTHROPIC_API_KEY
    docker compose up --build

- Web UI:     http://localhost:5173
- API docs:   http://localhost:8000/docs
- MLflow:     http://localhost:5000
- Edge:       http://localhost:8001/health

## Run pieces locally

    pip install -e ".[dev]"
    python -m data.simulator.generate            # synthetic fleet -> data/simulator/out/
    uvicorn apps.api.main:app --reload           # API on :8000 (SQLite by default)
    npm --prefix apps/web install && npm --prefix apps/web run dev
    pytest
    ruff check .

See CLAUDE.md for the architecture, agent contracts, and non-negotiable rules.
