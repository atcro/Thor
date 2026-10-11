"""P0 end-to-end, the demo flow: an explicit train run (ingest -> profile -> seeded split ->
3 candidates -> validate -> parity -> register), promotion as its own ModelOps decision ->
edge ONNX, fleet scoring, then Run analysis reusing the production model (SHAP -> cost ->
Decision Contract -> human approval -> work order) in seconds.

No toolbox is faked. Uses the conftest synthetic fleet (8 motors, 2 degrading) and the real
manuals under data/manuals with the offline hashed embedding so no network is needed.
"""

from __future__ import annotations

import json
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
TERMINAL = {"awaiting_approval", "trained", "approved", "rejected", "failed"}


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
    assert all(f["health_score"] is None for f in fleet)  # nothing scored yet, no guessing

    # Run analysis before any model exists is refused; nothing trains behind the scenes.
    r = client.post("/pipeline/run", json={"asset_id": TARGET})
    assert r.status_code == 409, r.text
    assert client.post("/pipeline/score", json={}).status_code == 409

    # --- 1. explicit training run: profile -> split -> AutoML -> validate -> register -------
    r = client.post(
        "/pipeline/run",
        json={"asset_id": TARGET, "horizon_h": 48, "n_trials": 2, "mode": "train"},
    )
    assert r.status_code == 200, r.text
    train_run = r.json()["run_id"]
    state = _poll(client, train_run)
    events = "\n".join(f"  [{e['stage']}] {e['tool']}: {e['message']}" for e in state["events"])
    assert state["stage"] == "trained", f"{state.get('error')}\n{events}"
    assert state["contract"] is None  # a train run drafts no decision

    # 04 -> 05 -> 06
    assert state["data_quality"]["trainable"] is True
    assert len(state["data_quality"]["regimes"]["regimes"]) == 3
    assert len(state["candidates"]["candidates"]) == 3
    assert state["candidates"]["features"]["baseline"] is not None
    assert state["validation"]["leakage"]["passed"] is True
    assert state["validation"]["champion_family"] in {"random_forest", "xgboost", "lightgbm"}
    # the split was fixed first, the analysed asset is held out, metrics come from test assets
    split = state["split"]
    assert TARGET in split["test"] and TARGET not in split["train"]
    assert state["validation"]["split"] == split
    assert state["validation"]["backtest"][0]["test_assets"] == split["test"]
    assert state["validation"]["test_metrics"]["n_assets"] == len(split["test"])
    assert state["data_quality"]["regimes"]["fit_assets"] == split["train"]
    assert state["candidates"]["features"]["fit_assets"] == split["train"]
    # edge-vs-centre parity was checked on real held-out vectors and raw windows, and passed
    parity = state["validation"]["parity"]
    assert parity["passed"] is True and parity["n_rows"] > 0 and parity["n_windows"] > 0, parity
    assert parity["max_abs_diff_probability"] <= parity["tolerance_probability"]
    assert parity["max_abs_diff_features"] <= parity["tolerance_features"]
    gate = {g["name"]: g for g in state["validation"]["gate"]}
    assert set(gate) == {"leakage", "test_failure_events", "recall", "lead_time", "edge_parity"}
    assert state["validation"]["passed"] is True, gate
    assert state["validation"]["thresholds"] == {"classification": 0.5, "early_warning": 0.3, "sustained_rows": 3}
    version = state["validation"]["model_version"]

    # 08: registered as a candidate only; promotion is pending for a human
    models = client.get("/models").json()
    assert [m["stage"] for m in models] == ["candidate"], events
    assert not list(Path(get_settings().models_dir).glob("*.onnx")), events
    pending = client.get("/approvals/pending").json()["promotions"]
    assert len(pending) == 1 and pending[0]["status"] == "pending"
    assert client.post("/pipeline/run", json={"asset_id": TARGET}).status_code == 409

    # --- 2. promotion: its own decision, its own approver, through ModelOps ---------------
    p = client.post(
        f"/promotions/{pending[0]['promotion_id']}/decision",
        json={"decision": "approved", "approver": "mlops@plant", "note": "e2e"},
    )
    assert p.status_code == 200, p.text
    assert p.json()["detail"]["promoted"]["stage"] == "production", p.text
    assert "edge" in p.json()["detail"], p.text
    assert any(m["stage"] == "production" for m in client.get("/models").json())
    onnx_files = list(Path(get_settings().models_dir).glob("*.onnx"))
    assert onnx_files and onnx_files[0].with_suffix(".json").exists()
    sidecar = json.loads(onnx_files[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["calibration"]["method"] in {"sigmoid", "isotonic", "none"}
    assert "_regime_centroids" in sidecar["baseline"]

    # --- 3. fleet scoring with the approved model: every motor gets an honest health -------
    sc = client.post("/pipeline/score", json={})
    assert sc.status_code == 200, sc.text
    assert sc.json()["model_version"] == version and len(sc.json()["scored"]) == 8
    fleet = client.get("/fleet").json()
    assert all(f["health_score"] is not None and f["model_version"] == version for f in fleet)
    assert all(f["regime"] in {"R1", "R2", "R3"} for f in fleet)  # inferred, not the label

    # --- 4. Run analysis: reuse the production model; seconds, not a retrain --------------
    t0 = time.time()
    r = client.post("/pipeline/run", json={"asset_id": TARGET, "horizon_h": 48})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert r.json()["mode"] == "analyze"
    state = _poll(client, run_id)
    took = time.time() - t0
    events = "\n".join(f"  [{e['stage']}] {e['tool']}: {e['message']}" for e in state["events"])
    assert state["stage"] == "awaiting_approval", f"{state.get('error')}\n{events}"
    assert took < 60, f"analysis took {took:.0f}s; it must not retrain\n{events}"
    assert state["mode"] == "analyze" and state["model_ref"]["version"] == version
    assert state["candidates"] is None and state["split"] is None
    tools = [e["tool"] for e in state["events"]]
    assert "train_candidates" not in tools and "load_production_model" in tools
    assert len(client.get("/models").json()) == 1  # no new registry rows

    # 07
    contract = state["contract"]
    assert contract["asset_id"] == TARGET and contract["model_version"] == version
    assert 0.0 <= contract["failure_probability"] <= 1.0
    assert len(contract["top_features"]) >= 3
    assert contract["explanation_source"] == "template"
    assert contract["cost_comparison"]["recommended"] == contract["recommendation"]
    assert len(contract["manual_context"]) >= 1, "RAG returned no passages"
    assert "plant-cost assumptions" in contract["explanation_text"]
    preds = client.get(f"/assets/{TARGET}/predictions").json()
    assert any(p["source"] == "api" and p["model_version"] == version for p in preds)

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

    # the maintenance approval changed nothing in the registry
    assert [m["stage"] for m in client.get("/models").json()] == ["production"]
    assert client.get("/work-orders").json()[0]["contract_id"] == contract["contract_id"]

    # contract immutability: stored payload hash unchanged after approval
    stored = client.get(f"/approvals/{contract['contract_id']}").json()
    assert stored["status"] == "approved"
    assert stored["contract"]["evidence_hash"] == contract["evidence_hash"]
