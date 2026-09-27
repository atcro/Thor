"""Replay-side simulated CMMS (streaming/replay/work_orders.py).

An approved work order takes the motor offline for the planned downtime at the approved window
and then streams its own healthy baseline; other assets, rows before the window, rejected or
absent orders change nothing. Runs on the dependency-free synthetic fleet from conftest.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
import pytest

from streaming.replay.replay import iter_rows
from streaming.replay.work_orders import (
    HealthyBaseline,
    RepairSimulator,
    WorkOrder,
    WorkOrderClient,
)
from tests.conftest import make_synthetic_fleet

ASSET = "MTR-003"  # degrades from 55 % of the horizon in make_synthetic_fleet
STEP = timedelta(minutes=10)


@pytest.fixture(scope="module")
def fleet() -> pd.DataFrame:
    return make_synthetic_fleet(n_assets=4, days=4, degrading={ASSET: 0.55})


def _tail(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    t0, t1 = df["ts"].min(), df["ts"].max()
    return df[df["ts"] >= t0 + (t1 - t0) * frac]


def _window(play: pd.DataFrame, frac: float) -> datetime:
    t0, t1 = play["ts"].min(), play["ts"].max()
    return (t0 + (t1 - t0) * frac).to_pydatetime()


def test_passthrough_without_orders(fleet: pd.DataFrame) -> None:
    play = _tail(fleet, 0.8)
    sim = RepairSimulator(fleet)
    out = list(sim.apply(iter_rows(play)))
    assert len(out) == len(play)
    assert sim.events == []


def test_repair_at_window_then_healthy_baseline(fleet: pd.DataFrame) -> None:
    play = _tail(fleet, 0.6)
    window = _window(play, 0.5)
    sim = RepairSimulator(fleet)
    sim.load([WorkOrder("dc_1", ASSET, window_start=window, planned_downtime_h=3.0)])
    out = list(sim.apply(iter_rows(play)))

    others = [r for r in out if r.asset_id != ASSET]
    assert len(others) == int((play["asset_id"] != ASSET).sum())  # untouched

    src = play[play["asset_id"] == ASSET].sort_values("ts")
    mine = [r for r in out if r.asset_id == ASSET]
    before = [r for r in mine if r.ts < window]
    assert len(before) == int((src["ts"] < window).sum())
    assert all(r.health is not None and r.health < 1.0 for r in before[-5:])  # history intact

    after = [r for r in mine if r.ts >= window]
    assert len(after) == int((src["ts"] >= window).sum()) - 18  # 3 h outage at 10-min step
    assert after[0].ts >= window + timedelta(hours=3)
    assert all(r.health == 1.0 and r.failure_within_h is None for r in after)
    # sensor values come from the asset's own healthy history, not from the degraded rows
    assert src["vibration_rms"].max() > 3.0
    assert max(r.vibration_rms for r in after) < 3.0

    assert [s for _, s, _ in sim.events] == ["in_progress", "completed"]
    (_, _, started), (_, _, completed) = sim.events
    assert window <= started < window + STEP
    assert completed - started >= timedelta(hours=3)


def test_replay_loop_replays_the_same_story(fleet: pd.DataFrame) -> None:
    play = _tail(fleet, 0.6)
    sim = RepairSimulator(fleet)
    sim.load([WorkOrder("dc_1", ASSET, window_start=_window(play, 0.5), planned_downtime_h=3.0)])
    first = list(sim.apply(iter_rows(play)))
    second = list(sim.apply(iter_rows(play)))
    assert len(first) == len(second)
    assert [r.ts for r in first] == [r.ts for r in second]
    assert all(r.health == 1.0 for r in second if r.asset_id == ASSET and r.ts > sim.events[-1][2])
    assert len(sim.events) == 2  # no re-reporting on the second pass


def test_maintain_now_without_window_starts_at_the_next_row(fleet: pd.DataFrame) -> None:
    play = _tail(fleet, 0.9)
    sim = RepairSimulator(fleet)
    sim.load([WorkOrder("dc_now", ASSET, window_start=None, planned_downtime_h=1.0)])
    out = list(sim.apply(iter_rows(play)))
    first_ts = play.loc[play["asset_id"] == ASSET, "ts"].min().to_pydatetime()
    assert sim.events[0][1:] == ("in_progress", first_ts)
    mine = [r for r in out if r.asset_id == ASSET]
    assert mine[0].ts >= first_ts + timedelta(hours=1)
    assert all(r.health == 1.0 for r in mine)


def test_load_merges_without_rolling_back_progress(fleet: pd.DataFrame) -> None:
    sim = RepairSimulator(fleet)
    t = datetime(2025, 1, 3, tzinfo=UTC)
    sim.load([WorkOrder("dc", ASSET, None, 3.0, status="in_progress", started_ts=t)])
    sim.load([WorkOrder("dc", ASSET, None, 3.0)])  # a stale poll result
    assert sim._by_asset[ASSET].status == "in_progress"
    sim.load([WorkOrder("dc", ASSET, None, 3.0, status="completed", started_ts=t, completed_ts=t)])
    assert sim._by_asset[ASSET].status == "completed"


def test_healthy_baseline_samples_same_regime(fleet: pd.DataFrame) -> None:
    hb = HealthyBaseline(fleet[fleet["asset_id"] == ASSET])
    assert len(hb) > 0
    ts = datetime(2025, 1, 2, 10, 0, tzinfo=UTC)
    a, b = hb.next_row("R2", ts), hb.next_row("R2", ts)
    assert a is not None and b is not None and a != b  # walks the pool, no repeats
    assert 40.0 < a["load_pct"] < 70.0  # R2 nominal load in the synthetic fleet
    assert hb.next_row("R9", ts) is not None  # unknown regime falls back to any healthy row


def test_client_never_raises_into_the_stream() -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(f"{req.method} {req.url.path}")
        if req.method == "GET":
            return httpx.Response(500)
        return httpx.Response(409, json={"detail": "already recorded"})

    client = WorkOrderClient(
        "http://api.test", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert client.fetch() is None
    assert (
        client.report("dc_1", "completed", datetime(2025, 1, 1, tzinfo=UTC)) is True
    )  # 409 == done
    assert calls == ["GET /work-orders", "POST /work-orders/dc_1/events"]


def test_from_json_parses_api_shape() -> None:
    wo = WorkOrder.from_json(
        {
            "contract_id": "dc_MTR-042_1",
            "asset_id": "MTR-042",
            "recommendation": "maintain_later",
            "window_start": "2025-08-27T02:00:00Z",
            "window_end": "2025-08-27T05:00:00+00:00",
            "planned_downtime_h": 3,
            "approved_at": "2026-09-27T04:50:00+00:00",
            "approver": "engineer@plant",
            "status": "scheduled",
            "started_ts": None,
            "completed_ts": None,
        }
    )
    assert wo.window_start == datetime(2025, 8, 27, 2, tzinfo=UTC)
    assert wo.planned_downtime_h == 3.0 and wo.rank == 0
