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
from apps.api.settings import get_settings
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
    # No approved-model prediction yet -> no health score and no regime (the seeded rows carry
    # simulator ground truth, which must never be used as a fallback).
    assert [f["asset"]["asset_id"] for f in fleet] == ["MTR-001", "MTR-002", "MTR-003", "MTR-042"]
    assert all(f["health_score"] is None and f["failure_probability"] is None for f in fleet)
    assert all(f["regime"] is None for f in fleet)
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
    fleet_list = client.get("/fleet").json()
    fleet = {f["asset"]["asset_id"]: f for f in fleet_list}
    assert fleet["MTR-001"]["failure_probability"] == 0.25
    assert fleet["MTR-001"]["health_score"] == 75.0
    assert fleet["MTR-001"]["model_version"] == "v42"
    assert fleet_list[0]["asset"]["asset_id"] == "MTR-001"  # scored assets sort first
    detail = client.get("/assets/MTR-001").json()
    assert detail["prediction"] == 0.25 and detail["health_score"] == 75.0


def test_pipeline_run_to_approval(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    calls = install_fake_toolbox(monkeypatch, orchestrator)
    assert client.post("/pipeline/run", json={"asset_id": "MTR-999"}).status_code == 404

    r = client.post("/pipeline/run", json={"asset_id": "MTR-042", "horizon_h": 48, "n_trials": 2})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert r.json()["stage"] == "queued" and r.json()["mode"] == "analyze"

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
    # the maintenance approval recorded the decision and nothing else
    assert calls["request_promotion"] == []
    assert calls["promote"] == [] and calls["deploy_edge"] == []
    # the analysis stored the production model's prediction: the fleet shows it at once
    fleet = {f["asset"]["asset_id"]: f for f in client.get("/fleet").json()}
    assert fleet["MTR-042"]["failure_probability"] == 0.78
    assert fleet["MTR-042"]["model_version"] == "v42"

    r = client.post(f"/approvals/{contract_id}/decision", json=decision)
    assert r.status_code == 409
    assert client.get(f"/approvals/{contract_id}").json()["status"] == "approved"
    assert client.get("/approvals/pending").json()["contracts"] == []
    assert client.post("/approvals/dc_nope/decision", json=decision).status_code == 404

    asset = client.get("/assets/MTR-042").json()
    assert asset["contracts"][0]["contract_id"] == contract_id
    assert asset["runs"][0]["stage"] == "approved"


def test_train_mode_and_analyze_without_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.api import orchestrator

    calls = install_fake_toolbox(monkeypatch, orchestrator, production=False)
    # no production model -> Run analysis is refused, nothing is trained behind the scenes
    r = client.post("/pipeline/run", json={"asset_id": "MTR-042"})
    assert r.status_code == 409 and "no production model" in r.json()["detail"]
    # an explicit train run registers a champion and stops at `trained`
    r = client.post("/pipeline/run", json={"asset_id": "MTR-042", "mode": "train", "n_trials": 2})
    assert r.status_code == 200 and r.json()["mode"] == "train"
    state = _poll(client, r.json()["run_id"], {"trained", "failed"})
    assert state["stage"] == "trained", state["error"]
    assert state["contract"] is None and state["evidence"] is None
    assert calls["request_promotion"] == ["v42"] and calls["promote"] == []
    runs = client.get("/pipeline", params={"asset_id": "MTR-042"}).json()
    assert runs[0]["mode"] == "train" and runs[0]["stage"] == "trained"
    assert client.get("/approvals/pending").json()["contracts"] == []


def _register_pending_promotion(engine: Any, version: str, passed: bool = True) -> str:
    """Register a tiny real model and open a promotion request through lifecycle."""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    from agents.mlops import lifecycle
    from apps.api.schemas import ModelStage
    from tests.test_orchestrator import make_validation

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    artifact = Path(get_settings().models_dir) / f"{version}.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y), artifact)
    val = make_validation("MTR-042").model_copy(update={"model_version": version, "passed": passed})
    rm = lifecycle.register_model(val, artifact, engine=engine)
    if not passed:
        return rm.version
    req = lifecycle.request_promotion(
        lifecycle.compare_champion(rm, engine=engine), ModelStage.production, engine=engine
    )
    return req.promotion_id


def test_promotion_decision_is_its_own_approval(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 1: model promotion is a separate ModelOps decision with a named approver, and
    both approve and reject work (reject used to crash on a missing argument)."""
    from apps.api import orchestrator
    from apps.api.schemas import EdgeDeployment

    deployed: list[str] = []

    def fake_deploy(model: Any, artifact_path: Path, features: list[str], models_dir: Path,
                    window_rows: int, baseline: dict[str, Any] | None = None) -> EdgeDeployment:
        deployed.append(model.version)
        from datetime import UTC, datetime

        return EdgeDeployment(model_name=model.name, version=model.version,
                              onnx_path=str(models_dir / "m.onnx"), deployed_at=datetime.now(UTC),
                              input_features=features)

    # real lifecycle.promote through the route; only the ONNX export is faked
    monkeypatch.setattr(orchestrator, "tool_deploy_edge", fake_deploy)
    engine = db.get_engine()
    pid_reject = _register_pending_promotion(engine, "v70001")
    pid_approve = _register_pending_promotion(engine, "v70002")
    pending = client.get("/approvals/pending").json()
    assert {p["promotion_id"] for p in pending["promotions"]} == {pid_reject, pid_approve}

    r = client.post(
        f"/promotions/{pid_reject}/decision",
        json={"decision": "rejected", "approver": "mlops@plant", "note": "not yet"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["promotion_id"] == pid_reject and body["approver"] == "mlops@plant"
    assert body["detail"] == {"rejected": True}
    promos = {p["promotion_id"]: p for p in client.get("/models/promotions").json()}
    assert promos[pid_reject]["status"] == "rejected"
    assert all(m["stage"] != "production" for m in client.get("/models").json())

    r = client.post(
        f"/promotions/{pid_approve}/decision",
        json={"decision": "approved", "approver": "mlops@plant", "note": "ship it"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["detail"]["promoted"]["stage"] == "production"
    assert "edge" in body["detail"] and deployed == ["v70002"]
    stages = {m["version"]: m["stage"] for m in client.get("/models").json()}
    assert stages["v70002"] == "production" and stages["v70001"] == "candidate"
    # second decision on either request is refused; the approval rows are insert-only
    assert client.post(
        f"/promotions/{pid_approve}/decision",
        json={"decision": "rejected", "approver": "x", "note": ""},
    ).status_code == 409
    assert client.post(
        "/promotions/pr_nope/decision", json={"decision": "approved", "approver": "x"}
    ).status_code == 404
    assert len(db.approvals_for(promotion_id=pid_approve, engine=engine)) == 1
    assert db.approvals_for(promotion_id=pid_approve, engine=engine)[0].contract_id is None


def test_failed_validation_cannot_be_promoted(client: TestClient) -> None:
    """Defect 2: a model whose validation failed gets no promotion request, and the reasons
    are recorded in the registry payload."""
    from agents.mlops import lifecycle
    from apps.api.routes.models import registry_row_payload
    from apps.api.schemas import ModelStage

    engine = db.get_engine()
    version = _register_pending_promotion(engine, "v70003", passed=False)
    rm = lifecycle.get_registered_model(lifecycle.DEFAULT_MODEL_NAME, version, engine=engine)
    assert rm is not None and rm.stage == ModelStage.candidate
    payload = registry_row_payload(rm.name, rm.version, engine)
    assert payload["validation_passed"] is False and payload["validation_failures"]
    with pytest.raises(ValueError, match="failed validation"):
        lifecycle.request_promotion(
            lifecycle.compare_champion(rm, engine=engine), ModelStage.production, engine=engine
        )
    assert client.get("/approvals/pending").json()["promotions"] == []


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
    from datetime import UTC, datetime

    db.insert_prediction("MTR-042", datetime.now(UTC), "v42", 0.6, engine=db.get_engine())
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


def _decided_contract(engine: Any, asset_id: str, recommendation: str, decision: str = "approved") -> str:
    """Insert one contract and its decision row directly; the gate route itself is covered above."""
    from datetime import UTC, datetime

    from apps.api.schemas import Approval
    from tests.test_orchestrator import make_bundle, make_contract

    bundle = make_bundle(asset_id)
    bundle = bundle.model_copy(
        update={"cost": bundle.cost.model_copy(update={"recommended": recommendation})}
    )
    dc = make_contract(bundle, run_id="run_wo", text="t", source="template")
    db.insert_decision_contract(dc, engine=engine)
    db.insert_approval(
        Approval(
            approval_id=f"ap_{dc.contract_id}",
            contract_id=dc.contract_id,
            decision=decision,  # type: ignore[arg-type]
            approver="engineer@plant",
            decided_at=datetime.now(UTC),
        ),
        engine=engine,
    )
    return dc.contract_id


def test_work_orders_follow_approvals_and_plant_reports(client: TestClient) -> None:
    """Approved maintain_* contracts become work orders; the plant reports progress, insert-only."""
    engine = db.get_engine()
    assert client.get("/work-orders").json() == []
    wo_id = _decided_contract(engine, "MTR-042", "maintain_later")
    _decided_contract(engine, "MTR-001", "run_to_failure")  # nothing to hand to the plant
    _decided_contract(engine, "MTR-002", "maintain_now", decision="rejected")  # not approved
    orders = client.get("/work-orders").json()
    assert [o["contract_id"] for o in orders] == [wo_id]
    assert orders[0]["status"] == "scheduled" and orders[0]["planned_downtime_h"] == 3.0
    assert orders[0]["window_start"] is not None and orders[0]["approver"] == "engineer@plant"

    fleet = {a["asset"]["asset_id"]: a for a in client.get("/fleet").json()}
    assert fleet["MTR-042"]["work_order"]["status"] == "scheduled"
    assert fleet["MTR-001"]["work_order"] is None and fleet["MTR-002"]["work_order"] is None

    ev = {"status": "in_progress", "plant_ts": "2025-08-27T02:00:00Z"}
    assert client.post("/work-orders/dc_nope/events", json=ev).status_code == 404
    r = client.post(f"/work-orders/{wo_id}/events", json=ev)
    assert r.status_code == 200 and r.json()["status"] == "in_progress"
    assert r.json()["started_ts"].startswith("2025-08-27T02:00:00")
    # repeats and backwards moves are refused: the reporter can replay idempotently
    ev["plant_ts"] = "2025-08-27T02:10:00Z"
    assert client.post(f"/work-orders/{wo_id}/events", json=ev).status_code == 409
    done = {"status": "completed", "plant_ts": "2025-08-27T05:00:00Z"}
    r = client.post(f"/work-orders/{wo_id}/events", json=done)
    assert r.status_code == 200 and r.json()["status"] == "completed"
    assert r.json()["completed_ts"].startswith("2025-08-27T05:00:00")
    assert client.post(f"/work-orders/{wo_id}/events", json=done).status_code == 409

    assert client.get("/work-orders?active=true").json() == []
    detail = client.get("/assets/MTR-042").json()
    assert detail["work_orders"][0]["status"] == "completed"
    assert fleet_status(client, "MTR-042") == "completed"
    assert len(db.list_work_order_events(wo_id, engine=engine)) == 2  # both reports kept


def fleet_status(client: TestClient, asset_id: str) -> str | None:
    for a in client.get("/fleet").json():
        if a["asset"]["asset_id"] == asset_id:
            return a["work_order"]["status"] if a["work_order"] else None
    return None
