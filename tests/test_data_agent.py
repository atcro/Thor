"""Tests for agents/data_agent/profiling.py (04 Data Reliability)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from agents.data_agent.profiling import (
    detect_missingness,
    detect_regime,
    profile_dataset,
    validate_schema,
)
from apps.api.schemas import SENSOR_COLUMNS, DataQualityContract, RegimeReport
from tests.conftest import TELEMETRY_COLUMNS, make_synthetic_fleet

# --------------------------------------------------------------------------------------
# fixture sanity (other agents rely on this shape)
# --------------------------------------------------------------------------------------


def test_synthetic_fleet_shape(synthetic_fleet: pd.DataFrame, degrading_assets: list[str]) -> None:
    df = synthetic_fleet
    assert list(df.columns) == TELEMETRY_COLUMNS
    assert df["asset_id"].nunique() == 8
    assert len(df) == 8 * 6 * 144
    assert str(df["ts"].dt.tz) == "UTC"
    assert df.equals(df.sort_values(["asset_id", "ts"]).reset_index(drop=True))
    healthy = df[~df["asset_id"].isin(degrading_assets)]
    assert healthy["failure_within_h"].isna().all()
    assert (healthy["health"] == 1.0).all()
    for a in degrading_assets:
        g = df[df["asset_id"] == a]
        assert g["failure_within_h"].notna().all()
        assert g["failure_within_h"].iloc[-1] == 0.0
        assert g["failure_within_h"].is_monotonic_decreasing
        assert g["health"].is_monotonic_decreasing
        # monotone rise: last 10 % clearly above the pre-onset level for every fault sensor,
        # compared within the same operating regime so load effects cannot mask it
        head, tail = g.iloc[: len(g) // 2], g.iloc[-len(g) // 10 :]
        for regime in tail["regime"].unique():
            h, t = head[head["regime"] == regime], tail[tail["regime"] == regime]
            for col in ("vibration_rms", "vibration_kurtosis", "vibration_crest", "bearing_temp_c"):
                assert t[col].mean() > h[col].mean() + 3 * h[col].std()


def test_synthetic_fleet_is_deterministic() -> None:
    a = make_synthetic_fleet(n_assets=3, days=1)
    b = make_synthetic_fleet(n_assets=3, days=1)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------------------
# validate_schema
# --------------------------------------------------------------------------------------


def test_validate_schema_clean(synthetic_fleet: pd.DataFrame) -> None:
    assert validate_schema(synthetic_fleet) == []


def test_validate_schema_missing_required_column(synthetic_fleet: pd.DataFrame) -> None:
    issues = validate_schema(synthetic_fleet.drop(columns=["vibration_rms"]))
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 1 and errors[0].column == "vibration_rms"


def test_validate_schema_missing_optional_is_info(synthetic_fleet: pd.DataFrame) -> None:
    issues = validate_schema(synthetic_fleet.drop(columns=["health"]))
    assert [(i.column, i.severity) for i in issues] == [("health", "info")]


def test_validate_schema_non_numeric_and_ranges(synthetic_fleet: pd.DataFrame) -> None:
    df = synthetic_fleet
    df["rpm"] = df["rpm"].astype(str)
    df.loc[df.index[:5], "load_pct"] = 999.0
    issues = {(i.column, i.severity) for i in validate_schema(df)}
    assert ("rpm", "error") in issues
    assert ("load_pct", "warning") in issues


def test_validate_schema_duplicates_and_naive_ts(synthetic_fleet: pd.DataFrame) -> None:
    df = pd.concat([synthetic_fleet, synthetic_fleet.head(3)], ignore_index=True)
    df["ts"] = df["ts"].dt.tz_localize(None)
    issues = validate_schema(df)
    msgs = {(i.column, i.issue.split(" ")[-1], i.severity) for i in issues}
    assert ("ts", "timestamps", "warning") in msgs  # timezone-naive
    assert any(i.column == "ts" and "duplicate" in i.issue for i in issues)


def test_validate_schema_empty() -> None:
    issues = validate_schema(pd.DataFrame(columns=TELEMETRY_COLUMNS))
    assert issues and issues[0].severity == "error"


# --------------------------------------------------------------------------------------
# detect_missingness
# --------------------------------------------------------------------------------------


def test_detect_missingness_clean(synthetic_fleet: pd.DataFrame) -> None:
    rep = detect_missingness(synthetic_fleet)
    assert set(rep.missing_fraction) == set(SENSOR_COLUMNS)
    assert all(v == 0.0 for v in rep.missing_fraction.values())
    assert rep.gap_windows == []
    assert rep.flatlined_sensors == []


def test_detect_missingness_nans_gaps_flatline(synthetic_fleet: pd.DataFrame) -> None:
    df = synthetic_fleet
    # 30-row hole (5 h) in one asset
    g = df.index[df["asset_id"] == "MTR-002"]
    df = df.drop(index=g[100:130]).reset_index(drop=True)
    n = len(df)
    df.loc[df.index[: n // 4], "motor_temp_c"] = np.nan
    df["current_a"] = 42.0  # fleet-wide flatline

    rep = detect_missingness(df, step_min=10)
    assert abs(rep.missing_fraction["motor_temp_c"] - (n // 4) / n) < 1e-9
    assert "current_a" in rep.flatlined_sensors
    assert len(rep.gap_windows) == 1
    start, end = rep.gap_windows[0]
    assert (end - start) == pd.Timedelta(minutes=310)


def test_detect_missingness_per_asset_flatline(synthetic_fleet: pd.DataFrame) -> None:
    df = synthetic_fleet
    # stuck sensor on most assets -> flatlined; healthy noise elsewhere stays untouched
    stuck = df["asset_id"].isin([f"MTR-{i:03d}" for i in range(1, 7)])
    df.loc[stuck, "vibration_crest"] = 3.7
    rep = detect_missingness(df)
    assert rep.flatlined_sensors == ["vibration_crest"]


# --------------------------------------------------------------------------------------
# detect_regime
# --------------------------------------------------------------------------------------


def test_detect_regime_recovers_three_regimes(synthetic_fleet: pd.DataFrame) -> None:
    df = synthetic_fleet
    rep = detect_regime(df, k=3, seed=0)
    assert isinstance(rep, RegimeReport)
    assert rep.method == "kmeans"
    assert [r.regime_id for r in rep.regimes] == ["R1", "R2", "R3"]
    assert [r.label for r in rep.regimes] == ["idle/startup", "nominal", "high-load"]
    assert len(rep.row_regime) == len(df)
    assert set(rep.row_regime) == {"R1", "R2", "R3"}
    products = [r.rpm_mean * r.load_mean for r in rep.regimes]
    assert products == sorted(products)
    assert abs(sum(r.share for r in rep.regimes) - 1.0) < 1e-9
    assert sum(r.n_rows for r in rep.regimes) == len(df)
    assert rep.silhouette is not None and 0.5 < rep.silhouette <= 1.0
    # clusters should coincide with the simulator's ground-truth shift schedule
    assert adjusted_rand_score(df["regime"], rep.row_regime) > 0.95


def test_detect_regime_is_deterministic(synthetic_fleet: pd.DataFrame) -> None:
    a = detect_regime(synthetic_fleet, seed=3)
    b = detect_regime(synthetic_fleet, seed=3)
    assert a.row_regime == b.row_regime
    assert a.silhouette == b.silhouette


def test_detect_regime_handles_nan_and_degenerate_input(synthetic_fleet: pd.DataFrame) -> None:
    df = synthetic_fleet
    df.loc[df.index[:50], "rpm"] = np.nan
    rep = detect_regime(df)
    assert len(rep.row_regime) == len(df)

    flat = df.head(200).copy()
    flat["rpm"] = 1480.0
    flat["load_pct"] = 55.0
    rep1 = detect_regime(flat)
    assert len(rep1.regimes) == 1 and rep1.silhouette is None
    assert rep1.row_regime == ["R1"] * 200

    empty = detect_regime(df.head(0))
    assert empty.regimes == [] and empty.row_regime == []


# --------------------------------------------------------------------------------------
# profile_dataset
# --------------------------------------------------------------------------------------


def test_profile_dataset_trainable(synthetic_fleet: pd.DataFrame) -> None:
    dq = profile_dataset(synthetic_fleet, "MTR-008", step_min=10)
    assert isinstance(dq, DataQualityContract)
    assert dq.asset_id == "MTR-008"
    assert dq.quality_score >= 90.0
    assert dq.trainable is True
    assert dq.n_rows == len(synthetic_fleet)
    assert dq.n_assets == 8
    assert abs(dq.sampling_rate_hz - 1.0 / 600.0) < 1e-12
    assert dq.columns == TELEMETRY_COLUMNS
    assert dq.window_start == synthetic_fleet["ts"].min()
    assert dq.window_end == synthetic_fleet["ts"].max()
    assert dq.schema_issues == []
    assert len(dq.regimes.regimes) == 3
    assert any("MTR-008" in n for n in dq.notes)


def test_profile_dataset_penalises_bad_data(synthetic_fleet: pd.DataFrame) -> None:
    clean = profile_dataset(synthetic_fleet, "MTR-001").quality_score
    df = synthetic_fleet
    df.loc[df.sample(frac=0.5, random_state=0).index, "vibration_rms"] = np.nan
    df["current_a"] = 1.0
    df = df.drop(columns=["bearing_temp_c"])
    dq = profile_dataset(df, "MTR-001")
    assert dq.quality_score < clean
    # flatline (10) + missing required column (15) + 50 % NaN in one of 7 sensors (~2.9)
    assert dq.quality_score == pytest.approx(100 - 10 - 15 - 40 * 0.5 / 7, abs=0.05)
    assert dq.trainable is True  # degraded but still above the 60 threshold
    assert any(n.startswith("penalty") for n in dq.notes)

    # three stuck sensors on top -> flatline cap (30) pushes it below the training threshold
    df["motor_temp_c"] = 40.0
    df["vibration_crest"] = 3.5
    dq2 = profile_dataset(df, "MTR-001")
    assert dq2.quality_score < 60.0
    assert dq2.trainable is False
    assert any("not trainable" in n for n in dq2.notes)


def test_profile_dataset_small_frames_not_trainable(synthetic_fleet: pd.DataFrame) -> None:
    few_assets = synthetic_fleet[synthetic_fleet["asset_id"].isin(["MTR-001", "MTR-002"])]
    dq = profile_dataset(few_assets, "MTR-001")
    assert dq.n_assets == 2 and dq.trainable is False

    short = make_synthetic_fleet(n_assets=8, days=1).groupby("asset_id").head(40)
    dq2 = profile_dataset(short, "MTR-001")
    assert dq2.n_rows < 500 and dq2.trainable is False
    assert dq2.quality_score < 100.0  # short-window penalty

    absent = profile_dataset(synthetic_fleet, "MTR-999")
    assert absent.quality_score <= 80.0
    assert any("no rows" in n for n in absent.notes)


def test_profile_dataset_empty_frame() -> None:
    dq = profile_dataset(pd.DataFrame(columns=TELEMETRY_COLUMNS), "MTR-001")
    assert dq.quality_score == 0.0 and dq.trainable is False and dq.n_rows == 0
