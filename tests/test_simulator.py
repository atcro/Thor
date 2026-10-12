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
    DEMO_ONSET,
    FAULT_RAMP_H,
    FAULTY_ASSETS,
    SEED_FRACTION_DEFAULT,
    TELEMETRY_COLUMNS,
    build_assets,
    df_to_rows,
    generate_fleet,
    write_outputs,
)
from data.simulator.layout import (
    LINES,
    PACKAGING_ROLES,
    PLANT_LAYOUT,
    derive_groups,
    duty_profile,
    line_members,
)

DAYS = 5
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
    # Only the demo motor goes wrong live: its onset lies after the seed point and it is
    # never repaired in the data; every other failure completes inside the seeded history.
    assert FAULTY_ASSETS["MTR-042"] == (pytest.approx(DEMO_ONSET), pytest.approx(1.0))
    assert SEED_FRACTION_DEFAULT < DEMO_ONSET < SEED_FRACTION_DEFAULT + 0.01
    assert set(FAULT_RAMP_H) == {DEMO_ASSET}
    for asset_id, (onset, fail) in FAULTY_ASSETS.items():
        assert 0.0 <= onset < fail <= 1.0
        if asset_id != DEMO_ASSET:
            assert fail < SEED_FRACTION_DEFAULT, asset_id
    assert sum(1 for _, fail in FAULTY_ASSETS.values() if fail < SEED_FRACTION_DEFAULT) == 6


def test_assets_catalogue_matches_contract() -> None:
    assets = build_assets()
    assert [a.asset_id for a in assets] == ASSET_IDS
    assert {a.site for a in assets} == {"Ludvika"}
    assert {a.line for a in assets} == set(LINES)
    assert {a.line for a in assets} == {*(f"Packaging Line {i}" for i in range(1, 6)), "Compressor Hall"}
    demo = next(a for a in assets if a.asset_id == DEMO_ASSET)
    assert demo.criticality == "high" and demo.rated_kw == 160.0
    assert demo.line == "Packaging Line 3" and demo.role == "filler"
    for a in assets:
        Asset.model_validate(a.model_dump())  # round-trips through the shared contract


def test_plant_layout_groups_and_roles() -> None:
    """Five packaging lines of four roles each plus a four-motor Compressor Hall; the
    fault-history motors sit on Lines 3-4 and ratings never moved with the layout."""
    assets = {a.asset_id: a for a in build_assets()}
    assert set(PLANT_LAYOUT) == set(ASSET_IDS)
    for line in LINES:
        members = line_members(line)
        assert len(members) == 4
        roles = [assets[a].role for a in members]
        if line == "Compressor Hall":
            assert roles == ["air compressor"] * 4
        else:
            assert roles == list(PACKAGING_ROLES)
    lines_3_4 = {a for a, s in PLANT_LAYOUT.items() if s.line in ("Packaging Line 3", "Packaging Line 4")}
    assert set(FAULTY_ASSETS) <= lines_3_4
    # ratings are keyed by asset id (the generator's current draw depends on them)
    assert assets["MTR-042"].rated_kw == 160.0 and assets["MTR-016"].rated_kw == 11.0
    assert assets["MTR-003"].criticality == "high" and assets["MTR-007"].criticality == "low"


def test_committed_layout_equals_the_derived_groups() -> None:
    """The layout rule in `derive_groups` reproduces `PLANT_LAYOUT` on the default fleet
    (30 days, seed 42): the line assignment follows the measurement, not the other way round."""
    df, assets = generate_fleet()
    groups = derive_groups(df, {a.asset_id: a.rated_kw for a in assets})
    by_line = {
        "lines_1_2": {a for a, s in PLANT_LAYOUT.items() if s.line in ("Packaging Line 1", "Packaging Line 2")},
        "lines_3_4": {a for a, s in PLANT_LAYOUT.items() if s.line in ("Packaging Line 3", "Packaging Line 4")},
        "line_5": {a for a, s in PLANT_LAYOUT.items() if s.line == "Packaging Line 5"},
        "compressor_hall": {a for a, s in PLANT_LAYOUT.items() if s.line == "Compressor Hall"},
    }
    assert groups == by_line
    assert "MTR-018" in groups["lines_3_4"] and set(FAULTY_ASSETS) < groups["lines_3_4"]


def test_lines_3_4_measure_harder_duty_than_lines_1_2(fleet: tuple[pd.DataFrame, list[Asset]]) -> None:
    """The layout's premise is true in the data: Lines 3-4 run hotter and vibrate more than
    Lines 1-2 over the generated history (observable sensors only)."""
    df, _ = fleet
    prof = duty_profile(df)
    hard = prof[prof["line"].isin(["Packaging Line 3", "Packaging Line 4"])]
    easy = prof[prof["line"].isin(["Packaging Line 1", "Packaging Line 2"])]
    assert len(hard) == 8 and len(easy) == 8
    assert hard["bearing_temp_max"].mean() > easy["bearing_temp_max"].mean() + 5.0
    assert hard["vibration_max"].mean() > 1.5 * easy["vibration_max"].mean()
    assert hard["vibration_mean"].mean() > easy["vibration_mean"].mean()


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
    if asset_id in FAULT_RAMP_H:
        # the demo motor ramps over FAULT_RAMP_H hours and then holds at full severity
        ramp_rows = FAULT_RAMP_H[asset_id] * 60 / STEP
        expected = 1.0 - min(1.0, (len(active) - 1) / ramp_rows) ** 2
        assert active["health"].iloc[-1] == pytest.approx(expected, abs=1e-4)
    else:
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
    # bearing temperature depends on the operating state (load), so compare like with like:
    # within every state the tail has enough rows for, the damaged bearing runs >= 10 C hotter
    pre_by_state = pre.groupby("regime")["bearing_temp_c"].mean()
    compared = 0
    for state, rows in tail.groupby("regime"):
        if len(rows) >= 6 and state in pre_by_state.index:
            assert rows["bearing_temp_c"].mean() > pre_by_state[state] + 10.0, state
            compared += 1
    assert compared >= 1


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
    assert rows[2].failure_within_h == 0.0 and rows[2].health < 1.0  # last row of MTR-042
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
