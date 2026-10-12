"""Ground truth stays behind the curtain (CLAUDE.md section 15).

The simulator's `health` and `failure_within_h` exist for training labels and offline
evaluation only. These tests fail if they (or the simulator's `regime` label) reach any
operational surface: the fleet table, Asset 360, the telemetry series, Bolt's tools, or the
replay's published rows. They also pin the rule "no prediction -> no health score".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api import db
from apps.api.schemas import GROUND_TRUTH_COLUMNS, SIMULATOR_ONLY_COLUMNS
from tests.test_orchestrator import configure_env, seed_small_fleet


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    configure_env(tmp_path, monkeypatch)
    from apps.api.main import app

    with TestClient(app) as c:
        seed_small_fleet(db.get_engine())  # rows DO carry health / failure_within_h / regime
        yield c


def _keys(obj: Any) -> Iterator[str]:
    """Every dict key anywhere inside a JSON-like structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def _assert_no_ground_truth(payload: Any, where: str) -> None:
    leaked = sorted(set(_keys(payload)) & set(GROUND_TRUTH_COLUMNS))
    assert not leaked, f"{where} carries simulator ground truth: {leaked}"


def test_db_rows_carry_ground_truth_but_responses_do_not(client: TestClient) -> None:
    """The stored rows have the labels (training needs them); no operational response does."""
    engine = db.get_engine()
    stored = db.read_telemetry(asset_id="MTR-042", limit=1, engine=engine).iloc[-1]
    assert stored["health"] < 1.0 and stored["failure_within_h"] is not None
    assert stored["regime"] == "R2"

    fleet = client.get("/fleet").json()
    _assert_no_ground_truth(fleet, "GET /fleet")
    asset = client.get("/assets/MTR-042").json()
    _assert_no_ground_truth(asset, "GET /assets/MTR-042")
    assert set(asset["latest"]) == {
        "asset_id",
        "ts",
        "vibration_rms",
        "vibration_kurtosis",
        "vibration_crest",
        "bearing_temp_c",
        "motor_temp_c",
        "current_a",
        "rpm",
        "load_pct",
    }
    series = client.get("/assets/MTR-042/telemetry", params={"hours": 24}).json()
    assert series
    _assert_no_ground_truth(series, "GET /assets/MTR-042/telemetry")
    assert not (set(_keys(series)) & set(SIMULATOR_ONLY_COLUMNS))
    health = client.get("/system/health").json()
    _assert_no_ground_truth(health, "GET /system/health")


def test_bolt_tools_never_see_ground_truth(client: TestClient) -> None:
    from apps.api import orchestrator

    engine = db.get_engine()
    for name, args in (
        ("get_fleet", {}),
        ("get_asset", {"asset_id": "MTR-042"}),
        ("list_pending_approvals", {}),
    ):
        res = orchestrator._run_copilot_tool(name, args, engine)
        _assert_no_ground_truth(json.loads(json.dumps(res, default=str)), f"Bolt tool {name}")
    asset = orchestrator.copilot_get_asset("MTR-042", engine)
    assert "regime" not in asset["latest_telemetry"]
    assert asset["asset"]["health_score"] is None  # nothing scored it yet
    reply = orchestrator.copilot_reply(
        orchestrator.CopilotRequest(
            messages=[orchestrator.CopilotMessage(role="user", content="How is MTR-042 doing?")]
        ),
        engine=engine,
    )
    assert "no prediction yet" in reply.reply
    assert "health 2" not in reply.reply  # the simulator says 0.2; Bolt must not know


def test_no_prediction_means_no_health_score(client: TestClient) -> None:
    """A motor nobody has scored shows no score -- never 100*health from the simulator."""
    fleet = {f["asset"]["asset_id"]: f for f in client.get("/fleet").json()}
    for f in fleet.values():
        assert f["health_score"] is None and f["failure_probability"] is None
        assert f["prediction_ts"] is None and f["model_version"] is None
        assert f["regime"] is None  # no production model -> no inferred regime either
    asset = client.get("/assets/MTR-042").json()
    assert asset["prediction"] is None and asset["health_score"] is None

    db.insert_prediction("MTR-042", datetime.now(UTC), "v9", 0.6, engine=db.get_engine())
    fleet = client.get("/fleet").json()
    assert fleet[0]["asset"]["asset_id"] == "MTR-042"  # scored assets sort first, worst first
    assert fleet[0]["health_score"] == 40.0 and fleet[0]["model_version"] == "v9"
    assert fleet[0]["prediction_ts"] is not None
    assert all(f["health_score"] is None for f in fleet[1:])
    asset = client.get("/assets/MTR-042").json()
    assert asset["health_score"] == 40.0 and asset["model_version"] == "v9"


def test_fleet_probability_is_the_median_of_the_last_three_samples(client: TestClient) -> None:
    """One noisy sample never flips a colour: the displayed probability is the median of the
    asset's last three predictions (the alarm's own sustained-rows rule), stamped with the
    newest row's time and model."""
    from datetime import timedelta

    engine = db.get_engine()
    base = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    for i, p in enumerate([0.9, 0.0, 0.8, 0.05]):  # newest last
        db.insert_prediction("MTR-003", base + timedelta(minutes=10 * i), "v9", p, engine=engine)
    row = db.latest_prediction_rows(engine)["MTR-003"]
    assert row["failure_probability"] == 0.05  # median of [0.0, 0.8, 0.05] -> 0.05
    assert row["n_rows"] == 3 and row["ts"] == base + timedelta(minutes=30)
    db.insert_prediction("MTR-003", base + timedelta(minutes=40), "v9", 0.95, engine=engine)
    row = db.latest_prediction_rows(engine)["MTR-003"]
    assert row["failure_probability"] == 0.8  # median of [0.8, 0.05, 0.95]
    fleet = {f["asset"]["asset_id"]: f for f in client.get("/fleet").json()}
    assert fleet["MTR-003"]["failure_probability"] == 0.8
    assert fleet["MTR-003"]["health_score"] == pytest.approx(20.0)
    # a single prediction is its own median
    db.insert_prediction("MTR-001", base, "v9", 0.4, engine=engine)
    assert db.latest_prediction_rows(engine)["MTR-001"]["failure_probability"] == 0.4


def test_fleet_regime_is_inferred_from_rpm_and_load(client: TestClient) -> None:
    """With a production model the regime comes from rpm/load vs the model's centroids, not
    from the simulator's `regime` column (which says R2 for every seeded row)."""
    from apps.api.routes import fleet as fleet_routes
    from apps.api.schemas import FeatureSpec

    baseline = {
        "_regime_centroids": {
            "R1_rpm": 600.0,
            "R1_load_pct": 10.0,
            "R2_rpm": 1480.0,
            "R2_load_pct": 55.0,
            "R3_rpm": 1500.0,
            "R3_load_pct": 85.0,
            "rpm_scale": 400.0,
            "load_pct_scale": 30.0,
        }
    }
    spec = FeatureSpec(features=[], window_rows=12, description="t", baseline=baseline)
    rows = [
        {"rpm": 1480.0, "load_pct": 55.0},
        {"rpm": 620.0, "load_pct": 12.0},
        {"rpm": 1500.0, "load_pct": 90.0},
        {"rpm": None, "load_pct": 50.0},
    ]
    assert fleet_routes.infer_regimes(rows, spec) == ["R2", "R1", "R3", None]
    assert fleet_routes.infer_regimes(rows, None) == [None] * 4


def test_replay_publishes_observable_fields_only() -> None:
    """The replay strips simulator-only fields at the publish boundary (MQTT and HTTP)."""
    from streaming.replay.replay import PUBLIC_EXCLUDE, iter_rows, publish_http, publish_mqtt
    from tests.conftest import make_synthetic_fleet

    df = make_synthetic_fleet(n_assets=2, days=1)
    assert df["health"].min() < 1.0  # the source really carries ground truth
    rows = list(iter_rows(df))[:40]

    class FakeMqtt:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def publish(self, topic: str, payload: str, qos: int = 0) -> None:
            assert topic.startswith("telemetry/")
            self.payloads.append(json.loads(payload))

    mq = FakeMqtt()
    assert publish_mqtt(rows, "localhost", 1883, speed=0.0, client=mq) == 40  # type: ignore[arg-type]
    assert len(mq.payloads) == 40
    for p in mq.payloads:
        assert not (set(p) & PUBLIC_EXCLUDE), p
        assert {"asset_id", "ts", "vibration_rms", "rpm", "load_pct"} <= set(p)

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

    class FakeHttp:
        def __init__(self) -> None:
            self.batches: list[list[dict[str, Any]]] = []

        def post(self, url: str, json: list[dict[str, Any]]) -> FakeResp:
            assert url.endswith("/ingest")
            self.batches.append(json)
            return FakeResp()

    http = FakeHttp()
    assert publish_http(rows, "http://cp", speed=0.0, batch_size=16, client=http) == 40  # type: ignore[arg-type]
    sent = [r for b in http.batches for r in b]
    assert len(sent) == 40
    for r in sent:
        assert not (set(r) & PUBLIC_EXCLUDE), r
