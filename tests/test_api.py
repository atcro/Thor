"""HTTP tests for the FastAPI control plane (apps/api/main.py + routes).

Runs against a temp SQLite file with every toolbox call monkeypatched (see
tests/test_orchestrator.py::install_fake_toolbox) -- no agent modules required.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api import db
from tests.test_orchestrator import configure_env, install_fake_toolbox, seed_small_fleet


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    configure_env(tmp_path, monkeypatch)
    from apps.api.main import app

    with TestClient(app) as c:
        seed_small_fleet(db.get_engine())
        yield c


def _poll(
    client: TestClient, run_id: str, stages: set[str], timeout: float = 15.0
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        r = client.get(f"/pipeline/{run_id}")
        assert r.status_code == 200
        last = r.json()
        if last["stage"] in stages:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} stuck at {last.get('stage')}: {last.get('error')}")


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    r = client.get("/system/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db"] == "ok" and body["n_telemetry_rows"] == 120
    assert body["last_ingest_ts"] is not None
    assert body["llm"]["mode"] == "template" and body["llm"]["model"]


def test_fleet_and_asset(client: TestClient) -> None:
    r = client.get("/fleet")
    assert r.status_code == 200
    fleet = r.json()
    assert len(fleet) == 4
    assert fleet[0]["asset"]["asset_id"] == "MTR-042"  # lowest health first
    assert fleet[0]["health_score"] < fleet[-1]["health_score"]
    assert fleet[-1]["health_score"] == 100.0
    assert fleet[0]["regime"] == "R2" and fleet[0]["failure_probability"] is None
    assert fleet[0]["stage"] is None and fleet[0]["open_contract_id"] is None

    r = client.get("/assets/MTR-042")
    assert r.status_code == 200
    body = r.json()
    assert body["asset"]["asset_id"] == "MTR-042"
    assert body["latest"]["asset_id"] == "MTR-042"
    assert body["prediction"] is None and body["contracts"] == [] and body["runs"] == []
    assert client.get("/assets/MTR-999").status_code == 404


def test_fleet_uses_prediction_when_present(client: TestClient) -> None:
    from datetime import UTC, datetime

    db.insert_prediction("MTR-001", datetime.now(UTC), "v42", 0.25, engine=db.get_engine())
    fleet = {f["asset"]["asset_id"]: f for f in client.get("/fleet").json()}
    assert fleet["MTR-001"]["failure_probability"] == 0.25
    assert fleet["MTR-001"]["health_score"] == 75.0
    assert client.get("/assets/MTR-001").json()["prediction"] == 0.25


def test_pipeline_run_to_approval(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    calls = install_fake_toolbox(monkeypatch, orchestrator)
    assert client.post("/pipeline/run", json={"asset_id": "MTR-999"}).status_code == 404

    r = client.post("/pipeline/run", json={"asset_id": "MTR-042", "horizon_h": 48, "n_trials": 2})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert r.json()["stage"] == "queued"

    state = _poll(client, run_id, {"awaiting_approval", "failed"})
    assert state["stage"] == "awaiting_approval", state["error"]
    contract_id = state["contract"]["contract_id"]
    assert state["evidence"]["cost"]["recommended"] == "maintain_now"
    assert state["contract"]["explanation_source"] == "template"

    runs = client.get("/pipeline", params={"asset_id": "MTR-042"}).json()
    assert runs and runs[0]["run_id"] == run_id and runs[0]["stage"] == "awaiting_approval"
    assert client.get("/pipeline/does-not-exist").status_code == 404

    fleet = {f["asset"]["asset_id"]: f for f in client.get("/fleet").json()}
    assert fleet["MTR-042"]["open_contract_id"] == contract_id
    pending = client.get("/approvals/pending").json()
    assert [c["contract_id"] for c in pending["contracts"]] == [contract_id]

    r = client.get(f"/approvals/{contract_id}")
    assert r.status_code == 200
    ev = r.json()
    assert ev["status"] == "pending"
    assert ev["contract"]["contract_id"] == contract_id
    assert ev["evidence"]["explanation"]["failure_probability"] == 0.78
    assert ev["evidence"]["manual_context"][0]["section"] == "4.2"
    assert client.get("/approvals/dc_nope").status_code == 404

    # what-if sandbox is available once a champion exists
    r = client.post("/whatif", json={"run_id": run_id, "scenario": {"load_mean": 40}})
    assert r.status_code == 200 and r.json()["delta"] == -0.28

    decision = {"decision": "approved", "approver": "jane@plant", "note": "go"}
    r = client.post(f"/approvals/{contract_id}/decision", json=decision)
    assert r.status_code == 200
    approval = r.json()
    assert approval["contract_id"] == contract_id and approval["decision"] == "approved"
    assert len(approval["approval_id"]) == 32

    final = _poll(client, run_id, {"approved", "rejected", "failed"})
    assert final["stage"] == "approved"
    assert final["approval"]["approver"] == "jane@plant"
    assert calls["promote"] == ["promo_test_1"] and len(calls["deploy_edge"]) == 1

    r = client.post(f"/approvals/{contract_id}/decision", json=decision)
    assert r.status_code == 409
    assert client.get(f"/approvals/{contract_id}").json()["status"] == "approved"
    assert client.get("/approvals/pending").json()["contracts"] == []
    assert client.post("/approvals/dc_nope/decision", json=decision).status_code == 404

    asset = client.get("/assets/MTR-042").json()
    assert asset["contracts"][0]["contract_id"] == contract_id
    assert asset["runs"][0]["stage"] == "approved"


def test_rejection_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    calls = install_fake_toolbox(monkeypatch, orchestrator)
    run_id = client.post("/pipeline/run", json={"asset_id": "MTR-042"}).json()["run_id"]
    state = _poll(client, run_id, {"awaiting_approval", "failed"})
    contract_id = state["contract"]["contract_id"]
    r = client.post(
        f"/approvals/{contract_id}/decision",
        json={"decision": "rejected", "approver": "bob", "note": "not now"},
    )
    assert r.status_code == 200
    final = _poll(client, run_id, {"approved", "rejected", "failed"})
    assert final["stage"] == "rejected"
    assert calls["promote"] == [] and calls["deploy_edge"] == []


def test_models_routes_empty_and_promotion_404(client: TestClient) -> None:
    assert client.get("/models").json() == []
    assert client.get("/models/promotions").json() == []
    assert client.get("/models/drift", params={"asset_id": "MTR-042"}).json() == []
    r = client.post("/promotions/nope/decision", json={"decision": "approved", "approver": "x"})
    assert r.status_code == 404


def test_copilot_template_mode_over_http(client: TestClient) -> None:
    r = client.post(
        "/copilot/chat",
        json={"messages": [{"role": "user", "content": "Which assets are at risk in the fleet?"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "template"
    assert body["tool_calls"][0]["name"] == "get_fleet"
    assert "MTR-042" in body["reply"]
    r = client.post(
        "/copilot/chat",
        json={
            "messages": [{"role": "user", "content": "status of MTR-003?"}],
            "asset_id": "MTR-003",
        },
    )
    assert r.json()["tool_calls"][0]["args"] == {"asset_id": "MTR-003"}
    assert client.post("/copilot/chat", json={"messages": []}).status_code == 422
    # Regression: a second turn re-sends Bolt's own long reply as history. The 2000-char
    # cap applies to user input only, so this must not 422.
    r = client.post(
        "/copilot/chat",
        json={
            "messages": [
                {"role": "user", "content": "Why is MTR-042 flagged?"},
                {"role": "assistant", "content": "Actionable conclusion " * 200},
                {"role": "user", "content": "How can we remediate this step by step?"},
            ]
        },
    )
    assert r.status_code == 200
    too_long = {"messages": [{"role": "user", "content": "z" * 2001}]}
    assert client.post("/copilot/chat", json=too_long).status_code == 422


def test_ws_run_stream(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    install_fake_toolbox(monkeypatch, orchestrator)
    run_id = client.post("/pipeline/run", json={"asset_id": "MTR-042"}).json()["run_id"]
    _poll(client, run_id, {"awaiting_approval", "failed"})
    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        msg = ws.receive_json()
    assert msg["run_id"] == run_id and msg["stage"] == "awaiting_approval"


def test_public_state_strips_row_regime() -> None:
    """GET /pipeline/{run} and the WS push must not ship one label per telemetry row."""
    from apps.api.graph_state import public_state

    raw = {
        "run_id": "r1",
        "stage": "training",
        "data_quality": {
            "quality_score": 100.0,
            "regimes": {
                "regimes": [{"regime_id": "R1"}],
                "row_regime": ["R1"] * 5000,
                "method": "kmeans",
            },
        },
    }
    slim = public_state(raw)
    assert slim["data_quality"]["regimes"]["row_regime"] == []
    assert slim["data_quality"]["regimes"]["row_regime_count"] == 5000
    assert slim["data_quality"]["regimes"]["regimes"] == [{"regime_id": "R1"}]
    assert slim["data_quality"]["quality_score"] == 100.0
    assert len(raw["data_quality"]["regimes"]["row_regime"]) == 5000  # stored state untouched
    assert public_state({"run_id": "r2", "stage": "queued", "data_quality": None}) == {
        "run_id": "r2",
        "stage": "queued",
        "data_quality": None,
    }
