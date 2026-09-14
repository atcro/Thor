"""Tests for data/benchmark/ai4i_benchmark.py - no network, synthetic AI4I-shaped frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.benchmark.ai4i_benchmark import (
    FAILURE_MODE_FLAGS,
    RAW_FEATURES,
    build_features,
    detect_regime,
    format_markdown,
    grouped_split,
    load_ai4i,
    product_group,
    run_benchmark,
)

UCI_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def make_ai4i_like(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    """AI4I-shaped frame with the original UCI headers; ~4 % failures driven by torque + wear."""
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = 300.0 + rng.normal(0.0, 2.0, n)
    process = air + 10.0 + rng.normal(0.0, 1.0, n)
    rpm = np.clip(rng.normal(1540.0, 180.0, n), 1170.0, 2900.0)
    torque = np.clip(40.0 - 0.02 * (rpm - 1540.0) + rng.normal(0.0, 8.0, n), 4.0, 80.0)
    wear = rng.uniform(0.0, 250.0, n)
    logit = -3.7 + 0.09 * (torque - 40.0) + 0.012 * (wear - 125.0)
    p = 1.0 / (1.0 + np.exp(-logit))
    failure = (rng.uniform(size=n) < p).astype(int)
    flags = np.zeros((n, 5), dtype=int)
    for i in np.flatnonzero(failure):
        flags[i, rng.integers(0, 5)] = 1
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"{t}{14860 + i}" for i, t in enumerate(types)],
            "Type": types,
            "Air temperature [K]": air.round(1),
            "Process temperature [K]": process.round(1),
            "Rotational speed [rpm]": rpm.round(0).astype(int),
            "Torque [Nm]": torque.round(1),
            "Tool wear [min]": wear.round(0).astype(int),
            "Machine failure": failure,
            "TWF": flags[:, 0],
            "HDF": flags[:, 1],
            "PWF": flags[:, 2],
            "OSF": flags[:, 3],
            "RNF": flags[:, 4],
        }
    )[UCI_COLUMNS]


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ai4i") / "ai4i2020.csv"
    make_ai4i_like().to_csv(path, index=False)
    return path


def test_load_ai4i_normalises_columns(csv_path: Path) -> None:
    df = load_ai4i(csv_path)
    expected = {
        "udi",
        "product_id",
        "type",
        "air_temp_k",
        "process_temp_k",
        "rpm",
        "torque_nm",
        "tool_wear_min",
        "machine_failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
        "power_w",
        "temp_delta_k",
    }
    assert expected <= set(df.columns)
    assert len(df) == 1500
    assert 0.02 < df["machine_failure"].mean() < 0.08
    row = df.iloc[0]
    assert row["power_w"] == pytest.approx(row["torque_nm"] * row["rpm"] * 2 * np.pi / 60)
    assert row["temp_delta_k"] == pytest.approx(row["process_temp_k"] - row["air_temp_k"])


def test_detect_regime_labels(csv_path: Path) -> None:
    df = load_ai4i(csv_path)
    regime = detect_regime(df, k=3, seed=0)
    assert len(regime) == len(df)
    assert set(regime.unique()) <= {"R1", "R2", "R3"}
    assert regime.nunique() == 3


def test_build_features_excludes_failure_mode_flags(csv_path: Path) -> None:
    df = load_ai4i(csv_path)
    frame, names = build_features(df, detect_regime(df))
    assert not set(names) & set(FAILURE_MODE_FLAGS)
    assert "machine_failure" not in names and "product_id" not in names
    for col in RAW_FEATURES:
        assert col in names and f"{col}_z" in names
    assert set(frame["label"].unique()) <= {0, 1}
    assert frame[names].notna().all().all()


def test_grouped_split_has_zero_group_overlap(csv_path: Path) -> None:
    df = load_ai4i(csv_path)
    train_idx, test_idx = grouped_split(df, seed=0)
    groups = product_group(df).to_numpy()
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(train_idx) + len(test_idx) == len(df)
    assert not set(groups[train_idx]) & set(groups[test_idx])
    assert 0.15 < len(test_idx) / len(df) < 0.35
    # deterministic under seed
    again = grouped_split(df, seed=0)
    assert np.array_equal(again[0], train_idx) and np.array_equal(again[1], test_idx)


def test_run_benchmark_reports_metrics_and_writes_json(csv_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    result = run_benchmark(csv_path, n_trials=2, seed=0, out_path=out)

    for key in (
        "recall",
        "precision",
        "f1",
        "auroc",
        "brier_before",
        "brier_after",
        "positive_rate",
        "n_train",
        "n_test",
        "families",
        "leakage_checks",
    ):
        assert key in result, key
    assert 0.0 <= result["brier_before"] <= 1.0
    assert 0.0 <= result["brier_after"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0 and 0.0 <= result["auroc"] <= 1.0
    assert result["n_train"] + result["n_test"] == 1500
    assert result["champion_family"] in {"random_forest", "xgboost", "lightgbm"}
    assert {f["family"] for f in result["families"] if "ims" in f}
    for fam in result["families"]:
        if "ims" in fam:
            assert fam["ims"]["lead_time_score"] == 0.0

    lc = result["leakage_checks"]
    assert lc["failure_mode_flags_in_features"] == []
    assert lc["group_overlap_count"] == 0 and lc["row_overlap_count"] == 0
    assert lc["detect_leakage_passed"] is True

    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["n_test"] == result["n_test"]
    md = format_markdown(result)
    assert "recall" in md and result["champion_family"] in md
