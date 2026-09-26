"""P0 end-to-end: ingest -> profile -> train 3 candidates -> validate -> champion -> SHAP ->
cost recommendation -> human approval -> Decision Contract logged -> promote + edge ONNX.

No toolbox is faked. Uses the conftest synthetic fleet (8 motors, 2 degrading) and the real
manuals under data/manuals with the offline hashed embedding so no network is needed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api import db
from apps.api.schemas import Asset, TelemetryRow
from apps.api.settings import get_settings
from tests.conftest import make_synthetic_fleet

REPO = Path(__file__).resolve().parents[1]
TARGET = "MTR-003"  # degrading asset in the training set (onset at 55% of horizon)
TERMINAL = {"awaiting_approval", "approved", "rejected", "failed"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'e2e.db').as_posix()}")
    monkeypatch.setenv("SIMULATOR_OUT", str(tmp_path / "sim_out"))
    monkeypatch.setenv("MANUALS_DIR", str(REPO / "data" / "manuals"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")  # a real key in .env must never reach a test
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("THOR_RAG_EMBEDDING", "hashed")
    monkeypatch.setenv("THOR_SKIP_BACKGROUND", "1")
    monkeypatch.delenv("SEED_ON_START", raising=False)
    get_settings.cache_clear()
    db.get_engine.cache_clear()

    from apps.api.main import app

    with TestClient(app) as c:
        engine = db.get_engine()
        df = make_synthetic_fleet(n_assets=8, days=6, step_min=10, seed=7)
        assets = [
            Asset(asset_id=a, name=f"Motor {a[-3:]}", site="Ludvika", line="Pump House")
            for a in sorted(df["asset_id"].unique())
        ]
        db.upsert_assets(assets, engine=engine)
        rows = [TelemetryRow(**r) for r in _records(df)]
        db.insert_telemetry(rows, engine=engine)

        from agents.reliability import rag

        s = get_settings()
        rag.build_index(Path(s.manuals_dir), Path(s.chroma_path))
        yield c
    get_settings.cache_clear()
    db.get_engine.cache_clear()


def _records(df: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if v != v:  # NaN -> None
                v = None
            clean[k] = v
        out.append(clean)
    return out


def _poll(client: TestClient, run_id: str, timeout: float = 240.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    state: dict[str, Any] = {}
    while time.time() < deadline:
        state = client.get(f"/pipeline/{run_id}").json()
        if state.get("stage") in TERMINAL:
            return state
        time.sleep(1.0)
    return state


@pytest.mark.skipif(os.environ.get("THOR_SKIP_E2E") == "1", reason="slow")
def test_p0_chain_end_to_end(client: TestClient) -> None:
    fleet = client.get("/fleet").json()
    assert len(fleet) == 8

    r = client.post("/pipeline/run", json={"asset_id": TARGET, "horizon_h": 48, "n_trials": 2})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]

    state = _poll(client, run_id)
    events = "\n".join(f"  [{e['stage']}] {e['tool']}: {e['message']}" for e in state["events"])
    assert state["stage"] == "awaiting_approval", f"{state.get('error')}\n{events}"

    # 04 -> 05 -> 06
    assert state["data_quality"]["trainable"] is True
    assert len(state["data_quality"]["regimes"]["regimes"]) == 3
    assert len(state["candidates"]["candidates"]) == 3
    assert state["candidates"]["features"]["baseline"] is not None
    assert state["validation"]["leakage"]["passed"] is True
    assert state["validation"]["champion_family"] in {"random_forest", "xgboost", "lightgbm"}

    # 07
    contract = state["contract"]
    assert contract["asset_id"] == TARGET
    assert 0.0 <= contract["failure_probability"] <= 1.0
    assert len(contract["top_features"]) >= 3
    assert contract["explanation_source"] == "template"
    assert contract["cost_comparison"]["recommended"] == contract["recommendation"]
    assert len(contract["manual_context"]) >= 1, "RAG returned no passages"
    assert "plant-cost assumptions" in contract["explanation_text"]

    # 09: present evidence, then record the decision
    ev = client.get(f"/approvals/{contract['contract_id']}").json()
    assert ev["status"] == "pending"
    d = client.post(
        f"/approvals/{contract['contract_id']}/decision",
        json={"decision": "approved", "approver": "engineer@plant", "note": "e2e"},
    )
    assert d.status_code == 200, d.text
    again = client.post(
        f"/approvals/{contract['contract_id']}/decision",
        json={"decision": "rejected", "approver": "someone", "note": "too late"},
    )
    assert again.status_code == 409

    deadline = time.time() + 120
    while time.time() < deadline:
        state = client.get(f"/pipeline/{run_id}").json()
        if state["stage"] in {"approved", "failed"}:
            break
        time.sleep(1.0)
    events = "\n".join(f"  [{e['stage']}] {e['tool']}: {e['message']}" for e in state["events"])
    assert state["stage"] == "approved", events

    # 08: promotion + edge deployment happened as a consequence of the approval
    models = client.get("/models").json()
    assert any(m["stage"] == "production" for m in models), events
    onnx_files = list(Path(get_settings().models_dir).glob("*.onnx"))
    assert onnx_files, events
    sidecar = onnx_files[0].with_suffix(".json")
    assert sidecar.exists()

    # contract immutability: stored payload hash unchanged after approval
    stored = client.get(f"/approvals/{contract['contract_id']}").json()
    assert stored["status"] == "approved"
    assert stored["contract"]["evidence_hash"] == contract["evidence_hash"]
