"""Tests for data/simulator/generate.py (Agent A)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apps.api.schemas import SENSOR_COLUMNS, Asset, TelemetryRow
from data.simulator.generate import (
    ASSET_IDS,
    DEMO_ASSET,
    FAULTY_ASSETS,
    TELEMETRY_COLUMNS,
    build_assets,
    df_to_rows,
    generate_fleet,
    write_outputs,
)

DAYS = 3
STEP = 10
ROWS_PER_ASSET = DAYS * 24 * 60 // STEP


@pytest.fixture(scope="module")
def fleet() -> tuple[pd.DataFrame, list[Asset]]:
    return generate_fleet(days=DAYS, seed=7, step_min=STEP)


def test_asset_ids_replace_mtr_004_with_demo_asset() -> None:
    assert len(ASSET_IDS) == 24
    assert len(set(ASSET_IDS)) == 24
    assert DEMO_ASSET == "MTR-042"
    assert "MTR-042" in ASSET_IDS
    assert "MTR-004" not in ASSET_IDS
    assert ASSET_IDS[0] == "MTR-001" and ASSET_IDS[-1] == "MTR-024"
    assert {"MTR-042", "MTR-017"} <= set(FAULTY_ASSETS)
    assert FAULTY_ASSETS["MTR-042"] == (pytest.approx(0.55), pytest.approx(1.0))
    assert FAULTY_ASSETS["MTR-017"] == (pytest.approx(0.80), pytest.approx(1.0))
    for onset, fail in FAULTY_ASSETS.values():
        assert 0.0 <= onset < fail <= 1.0
    # at least three failures complete before the seed cutoff (0.45): training history
    assert sum(1 for _, fail in FAULTY_ASSETS.values() if fail <= 0.45) >= 3


def test_assets_catalogue_matches_contract() -> None:
    assets = build_assets()
    assert [a.asset_id for a in assets] == ASSET_IDS
    assert {a.site for a in assets} == {"Ludvika"}
    assert {a.line for a in assets} == {"Compressor Hall", "Pump House", "Conveyor A"}
    demo = next(a for a in assets if a.asset_id == DEMO_ASSET)
    assert demo.criticality == "high"
    for a in assets:
        Asset.model_validate(a.model_dump())  # round-trips through the shared contract


def test_shape_and_columns(fleet: tuple[pd.DataFrame, list[Asset]]) -> None:
    df, assets = fleet
    assert list(df.columns) == TELEMETRY_COLUMNS
    assert list(df.columns[2:10]) == list(SENSOR_COLUMNS)
    assert len(df) == 24 * ROWS_PER_ASSET
    assert df["asset_id"].nunique() == 24
    assert len(assets) == 24
    assert str(df["ts"].dt.tz) == "UTC"
    # sorted by (asset_id, ts), 10-minute cadence per asset
    assert df.equals(df.sort_values(["asset_id", "ts"], kind="mergesort").reset_index(drop=True))
    steps = df.groupby("asset_id")["ts"].diff().dropna().dt.total_seconds().unique()
    assert set(steps) == {STEP * 60}
    assert not df[list(SENSOR_COLUMNS)].isna().any().any()


def test_deterministic_under_seed() -> None:
    a, _ = generate_fleet(days=1, seed=3)
    b, _ = generate_fleet(days=1, seed=3)
    c, _ = generate_fleet(days=1, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a["vibration_rms"].equals(c["vibration_rms"])


def test_regimes_are_load_dependent(fleet: tuple[pd.DataFrame, list[Asset]]) -> None:
    df, _ = fleet
    assert set(df["regime"].unique()) == {"R1", "R2", "R3"}
    healthy = df[df["asset_id"] == "MTR-001"]
    by = healthy.groupby("regime")[["rpm", "load_pct", "vibration_rms", "motor_temp_c"]].mean()
    assert by.loc["R1", "rpm"] < 800 < by.loc["R2", "rpm"] < by.loc["R3", "rpm"]
    assert by.loc["R1", "load_pct"] < by.loc["R2", "load_pct"] < by.loc["R3", "load_pct"]
    assert (
        by.loc["R1", "vibration_rms"]
        < by.loc["R2", "vibration_rms"]
        < by.loc["R3", "vibration_rms"]
    )
    assert by.loc["R1", "motor_temp_c"] < by.loc["R3", "motor_temp_c"]
    # every asset spends time in every regime
    assert (df.groupby("asset_id")["regime"].nunique() == 3).all()


def test_healthy_assets_have_no_fault(fleet: tuple[pd.DataFrame, list[Asset]]) -> None:
    df, _ = fleet
    healthy = df[~df["asset_id"].isin(FAULTY_ASSETS)]
    assert (healthy["health"] == 1.0).all()
    assert healthy["failure_within_h"].isna().all()
    assert healthy["vibration_rms"].max() < 4.5  # ISO 10816 zone B-ish for healthy motors
    assert healthy["bearing_temp_c"].max() < 90.0


@pytest.mark.parametrize("asset_id", ["MTR-042", "MTR-017", "MTR-003", "MTR-021"])
def test_bearing_degradation_signature(
    fleet: tuple[pd.DataFrame, list[Asset]], asset_id: str
) -> None:
    df, _ = fleet
    d = df[df["asset_id"] == asset_id].reset_index(drop=True)
    onset_f, fail_f = FAULTY_ASSETS[asset_id]
    onset = int(round(onset_f * (len(d) - 1)))
    fail_idx = int(round(fail_f * (len(d) - 1)))
    pre, active, after = d.iloc[:onset], d.iloc[onset : fail_idx + 1], d.iloc[fail_idx + 1 :]
    # ground truth
    assert (pre["health"] == 1.0).all()
    assert pre["failure_within_h"].isna().all()
    assert active["failure_within_h"].notna().all()
    assert active["failure_within_h"].iloc[-1] == 0.0
    assert (np.diff(active["failure_within_h"].to_numpy()) < 0).all()  # counts down
    assert (np.diff(active["health"].to_numpy()) <= 0).all()  # monotone decay 1 -> 0
    assert active["health"].iloc[-1] == 0.0
    # after the failure the bearing is replaced: healthy again
    assert (after["health"] == 1.0).all()
    assert after["failure_within_h"].isna().all()
    # sensor signature: the last ~10 % before failure (>= 4 h, so it spans regimes) is
    # clearly worse than the pre-onset period
    tail = active.iloc[-max(len(active) // 10, 24) :]
    assert tail["vibration_rms"].mean() > 2.0 * pre["vibration_rms"].mean()
    assert tail["vibration_kurtosis"].mean() > pre["vibration_kurtosis"].mean() + 1.0
    if len(after) > 50:
        assert after["vibration_rms"].mean() < 0.6 * tail["vibration_rms"].mean()
    assert tail["vibration_crest"].mean() > pre["vibration_crest"].mean() + 0.5
    assert tail["bearing_temp_c"].mean() > pre["bearing_temp_c"].mean() + 10.0


def test_demo_asset_is_more_severe_than_mtr_017(fleet: tuple[pd.DataFrame, list[Asset]]) -> None:
    df, _ = fleet
    v42 = df.loc[df["asset_id"] == "MTR-042", "vibration_rms"].max()
    v17 = df.loc[df["asset_id"] == "MTR-017", "vibration_rms"].max()
    v01 = df.loc[df["asset_id"] == "MTR-001", "vibration_rms"].max()
    assert v42 > v17 > v01


def test_rows_validate_against_telemetry_row(fleet: tuple[pd.DataFrame, list[Asset]]) -> None:
    df, _ = fleet
    sample = pd.concat([df[df["asset_id"] == DEMO_ASSET].tail(3), df.head(3)])
    rows = df_to_rows(sample)
    assert all(isinstance(r, TelemetryRow) for r in rows)
    assert rows[2].failure_within_h == 0.0 and rows[2].health == 0.0  # last row of MTR-042
    assert rows[0].failure_within_h == pytest.approx(2 * STEP / 60, abs=1e-3)
    assert rows[-1].failure_within_h is None and rows[-1].health == 1.0  # MTR-001 healthy
    TelemetryRow.model_validate_json(rows[2].model_dump_json())


def test_write_outputs(tmp_path: Path) -> None:
    df, assets = generate_fleet(days=1, seed=1)
    out = tmp_path / "out"
    write_outputs(df, assets, out)
    assert (out / "telemetry.parquet").is_file()
    assert (out / "telemetry.csv").is_file()
    assert (out / "assets.json").is_file()
    back = pd.read_parquet(out / "telemetry.parquet")
    assert len(back) == len(df)
    assert list(back.columns) == TELEMETRY_COLUMNS
    assert str(pd.to_datetime(back["ts"], utc=True).dt.tz) == "UTC"
    csv = pd.read_csv(out / "telemetry.csv", nrows=5)
    assert csv["ts"].iloc[0].endswith("Z")
    catalogue = json.loads((out / "assets.json").read_text(encoding="utf-8"))
    assert [Asset.model_validate(a).asset_id for a in catalogue] == ASSET_IDS
