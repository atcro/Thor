"""Tests for agents/ml_architect (05): feature pipeline + Optuna AutoML + IMS."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from agents.data_agent.profiling import detect_regime, profile_dataset
from agents.ml_architect.automl import (
    FAMILIES,
    compare_models,
    industrial_model_score,
    infer_task,
    launch_trial,
    split_by_asset,
    train_candidates,
)
from agents.ml_architect.features import (
    BASELINE_META_KEY,
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_COLUMNS,
    baseline_for,
    build_feature_pipeline,
    compute_features_np,
)
from apps.api.schemas import (
    SENSOR_COLUMNS,
    CandidateModel,
    CandidateSet,
    FeatureSpec,
    IndustrialModelScore,
    RegimeReport,
    TaskSpec,
)

EXPECTED_FEATURES = [
    "vib_rms_mean",
    "vib_rms_slope",
    "vib_kurt_mean",
    "vib_crest_mean",
    "bearing_temp_mean",
    "bearing_temp_slope",
    "temp_delta",
    "current_mean",
    "rpm_mean",
    "load_mean",
    "vib_rms_z",
    "bearing_temp_z",
    "vib_rms_resid",
]
WINDOW = 12
HORIZON = 48.0


@pytest.fixture
def regimes(synthetic_fleet: pd.DataFrame) -> RegimeReport:
    return detect_regime(synthetic_fleet)


@pytest.fixture
def features(synthetic_fleet: pd.DataFrame, regimes: RegimeReport) -> tuple[pd.DataFrame, FeatureSpec]:
    return build_feature_pipeline(synthetic_fleet, regimes, horizon_h=HORIZON, window_rows=WINDOW)


@pytest.fixture
def task(synthetic_fleet: pd.DataFrame) -> TaskSpec:
    return infer_task(profile_dataset(synthetic_fleet, "MTR-008"), synthetic_fleet, HORIZON)


# --------------------------------------------------------------------------------------
# features.py
# --------------------------------------------------------------------------------------


def test_feature_names_match_interface_spec() -> None:
    assert FEATURE_NAMES == EXPECTED_FEATURES
    assert not set(FEATURE_NAMES) & set(FORBIDDEN_FEATURE_COLUMNS)


def test_build_feature_pipeline_shape_and_labels(
    synthetic_fleet: pd.DataFrame, features: tuple[pd.DataFrame, FeatureSpec]
) -> None:
    fdf, spec = features
    assert list(fdf.columns) == ["asset_id", "ts", *FEATURE_NAMES, "label", "rul_h"]
    assert spec.features == FEATURE_NAMES
    assert spec.window_rows == WINDOW and spec.regime_normalized is True
    per_asset_raw = synthetic_fleet.groupby("asset_id").size()
    per_asset_feat = fdf.groupby("asset_id").size()
    assert (per_asset_feat == per_asset_raw - (WINDOW - 1)).all()
    assert not fdf[FEATURE_NAMES].isna().any().any()
    assert np.isfinite(fdf[FEATURE_NAMES].to_numpy()).all()
    assert set(fdf["label"].unique()) == {0, 1}
    assert fdf["label"].dtype.kind == "i"
    # label is exactly failure_within_h <= horizon; rul_h carries the raw hours
    expected = (fdf["rul_h"] <= HORIZON).fillna(False).astype(int)
    assert (fdf["label"] == expected).all()
    healthy = fdf[~fdf["asset_id"].isin(["MTR-003", "MTR-008"])]
    assert healthy["rul_h"].isna().all() and (healthy["label"] == 0).all()
    # ts of a feature row is the last row of its window
    g = synthetic_fleet[synthetic_fleet["asset_id"] == "MTR-001"]
    assert fdf[fdf["asset_id"] == "MTR-001"]["ts"].iloc[0] == g["ts"].iloc[WINDOW - 1]
    assert isinstance(fdf.attrs["baseline"], dict)


def test_feature_pipeline_never_leaks(features: tuple[pd.DataFrame, FeatureSpec]) -> None:
    fdf, spec = features
    for forbidden in ("health", "failure_within_h", "regime", "ts"):
        assert forbidden not in spec.features
    assert "regime" not in fdf.columns and "health" not in fdf.columns
    # no feature is a near-copy of the label
    corr = fdf[FEATURE_NAMES].corrwith(fdf["label"]).abs()
    assert (corr < 0.98).all()


def test_feature_pipeline_handles_unsorted_and_nan_input(
    synthetic_fleet: pd.DataFrame, regimes: RegimeReport, features: tuple[pd.DataFrame, FeatureSpec]
) -> None:
    ref, _ = features
    shuffled = synthetic_fleet.sample(frac=1.0, random_state=1)
    shuffled_regimes = regimes.model_copy(
        update={"row_regime": [regimes.row_regime[i] for i in shuffled.index]}
    )
    fdf, _ = build_feature_pipeline(shuffled, shuffled_regimes, HORIZON, WINDOW)
    pd.testing.assert_frame_equal(fdf, ref, check_like=True)

    df = synthetic_fleet
    df.loc[df.index[200:230], "vibration_rms"] = np.nan
    fdf2, _ = build_feature_pipeline(df, regimes, HORIZON, WINDOW)
    assert not fdf2[FEATURE_NAMES].isna().any().any()

    # no regimes at all -> falls back to df['regime'] column; still identical shape
    fdf3, spec3 = build_feature_pipeline(synthetic_fleet.drop(columns=["regime"]), None, HORIZON, WINDOW)
    assert fdf3.shape == ref.shape and spec3.features == FEATURE_NAMES


def test_baseline_for_structure_is_json_and_per_regime(
    synthetic_fleet: pd.DataFrame, regimes: RegimeReport
) -> None:
    bl = baseline_for(synthetic_fleet, regimes)
    json.dumps(bl)  # sidecar-serialisable
    assert set(bl) == set(SENSOR_COLUMNS) | {BASELINE_META_KEY}
    for sensor in SENSOR_COLUMNS:
        entry = bl[sensor]
        assert {"mean", "std", "R1_mean", "R1_std", "R2_mean", "R2_std", "R3_mean", "R3_std"} <= set(entry)
        assert entry["std"] > 0 and entry["R1_std"] > 0
    meta = bl[BASELINE_META_KEY]
    assert {"R1_rpm", "R1_load_pct", "R3_rpm", "R3_load_pct", "rpm_scale", "load_pct_scale"} <= set(meta)
    # healthy baseline: vibration rises with regime load
    v = bl["vibration_rms"]
    assert v["R1_mean"] < v["R2_mean"] < v["R3_mean"]
    # the healthy baseline of the degrading asset is unaffected by its later fault
    assert v["R3_mean"] < 2.6


def test_compute_features_np_matches_pipeline_exactly(
    synthetic_fleet: pd.DataFrame, features: tuple[pd.DataFrame, FeatureSpec]
) -> None:
    fdf, _ = features
    bl = fdf.attrs["baseline"]
    for asset in ("MTR-001", "MTR-008"):
        g = synthetic_fleet[synthetic_fleet["asset_id"] == asset].reset_index(drop=True)
        f = fdf[fdf["asset_id"] == asset].reset_index(drop=True)
        for end in (WINDOW - 1, 57, 300, len(g) - 1):
            buffer = g.loc[end - WINDOW + 1 : end, list(SENSOR_COLUMNS)].to_numpy(dtype=float)
            vec = compute_features_np(buffer, bl)
            assert vec.shape == (len(FEATURE_NAMES),)
            np.testing.assert_allclose(vec, f.loc[end - WINDOW + 1, FEATURE_NAMES].to_numpy(dtype=float), atol=1e-9)


def test_compute_features_np_formulas() -> None:
    """Hand-checked formulas on a tiny window so the edge port can be verified line by line."""
    w = 4
    buf = np.zeros((w, len(SENSOR_COLUMNS)))
    buf[:, 0] = [1.0, 2.0, 3.0, 4.0]  # vibration_rms
    buf[:, 3] = [50.0, 50.0, 52.0, 52.0]  # bearing_temp_c
    buf[:, 4] = [40.0, 40.0, 40.0, 40.0]  # motor_temp_c
    buf[:, 6] = 1480.0  # rpm
    buf[:, 7] = 55.0  # load
    bl = {
        s: {"mean": 0.0, "std": 1.0, "R1_mean": 0.0, "R1_std": 1.0, "R2_mean": 2.0, "R2_std": 0.5}
        for s in SENSOR_COLUMNS
    }
    bl["bearing_temp_c"] = {"mean": 0.0, "std": 1.0, "R1_mean": 30.0, "R1_std": 1.0, "R2_mean": 50.0, "R2_std": 2.0}
    bl[BASELINE_META_KEY] = {
        "R1_rpm": 600.0, "R1_load_pct": 10.0, "R2_rpm": 1480.0, "R2_load_pct": 55.0,
        "rpm_scale": 400.0, "load_pct_scale": 30.0,
    }
    vec = dict(zip(FEATURE_NAMES, compute_features_np(buf, bl)))
    assert vec["vib_rms_mean"] == 2.5
    assert vec["vib_rms_slope"] == pytest.approx(1.0)  # OLS slope per row
    assert vec["bearing_temp_mean"] == 51.0
    assert vec["bearing_temp_slope"] == pytest.approx(0.8)  # sum(ic*x)/sum(ic^2) = 4/5
    assert vec["temp_delta"] == 11.0
    assert vec["rpm_mean"] == 1480.0 and vec["load_mean"] == 55.0
    # window sits on the R2 centroid -> R2 baseline used
    assert vec["vib_rms_z"] == pytest.approx((2.5 - 2.0) / 0.5)
    assert vec["bearing_temp_z"] == pytest.approx((51.0 - 50.0) / 2.0)
    assert vec["vib_rms_resid"] == pytest.approx(0.5)

    with pytest.raises(ValueError):
        compute_features_np(buf[:, :5], bl)


def test_regime_normalisation_separates_fault_from_load(
    synthetic_fleet: pd.DataFrame, features: tuple[pd.DataFrame, FeatureSpec]
) -> None:
    fdf, _ = features
    healthy = fdf[fdf["asset_id"] == "MTR-001"]
    # a load change on a healthy motor must not look like a fault
    assert healthy["vib_rms_z"].abs().quantile(0.99) < 4.0
    sick = fdf[fdf["asset_id"] == "MTR-008"]
    assert sick["vib_rms_z"].iloc[-1] > 10.0
    assert sick["vib_rms_z"].iloc[-1] > sick["vib_rms_z"].iloc[:50].mean() + 10.0
    assert sick["bearing_temp_z"].iloc[-1] > 5.0


# --------------------------------------------------------------------------------------
# automl.py
# --------------------------------------------------------------------------------------


def test_infer_task(task: TaskSpec) -> None:
    assert task.task == "classification"
    assert task.target == "failure_within_h"
    assert task.horizon_h == HORIZON
    assert "2 asset(s)" in task.rationale and "trainable" in task.rationale


def test_industrial_model_score_formula() -> None:
    ims = industrial_model_score(
        {"recall": 0.8, "precision": 0.5, "lead_time_h": 36.0, "brier": 0.05}, latency_ms=10.0
    )
    assert isinstance(ims, IndustrialModelScore)
    assert ims.lead_time_score == pytest.approx(0.5)
    assert ims.calibration_score == pytest.approx(0.8)
    assert ims.latency_score == pytest.approx(0.8)
    expected = 0.35 * 0.8 + 0.25 * 0.5 + 0.15 * 0.5 + 0.15 * 0.8 + 0.10 * 0.8
    assert ims.total == pytest.approx(expected)
    # saturation / worst cases
    top = industrial_model_score({"recall": 1, "precision": 1, "lead_time_h": 500, "brier": 0}, 0.0)
    assert top.total == pytest.approx(1.0)
    worst = industrial_model_score({}, latency_ms=1e6)
    assert worst.total == 0.0
    assert industrial_model_score({"recall": 1.0, "brier": float("nan")}, 0.0).calibration_score == 0.0


def test_split_by_asset_is_grouped_and_ordered(features: tuple[pd.DataFrame, FeatureSpec]) -> None:
    fdf, _ = features
    train, hold = split_by_asset(fdf)
    assert hold == ["MTR-007", "MTR-008"]
    assert train == [f"MTR-{i:03d}" for i in range(1, 7)]
    assert not set(train) & set(hold)
    assert split_by_asset(fdf[fdf["asset_id"] == "MTR-001"]) == (["MTR-001"], [])
    two = fdf[fdf["asset_id"].isin(["MTR-001", "MTR-002"])]
    assert split_by_asset(two) == (["MTR-001"], ["MTR-002"])


def test_launch_trial_lightgbm(
    features: tuple[pd.DataFrame, FeatureSpec], task: TaskSpec, tmp_artifacts_dir: Path
) -> None:
    fdf, spec = features
    cand = launch_trial(fdf, spec, task, "lightgbm", n_trials=2, seed=0, artifacts_dir=tmp_artifacts_dir)
    assert isinstance(cand, CandidateModel)
    assert cand.family == "lightgbm"
    assert cand.candidate_id.startswith("lightgbm-")
    assert set(cand.metrics) >= {"recall", "precision", "f1", "auroc", "brier", "lead_time_h"}
    assert 0.0 <= cand.metrics["brier"] <= 1.0 and cand.metrics["lead_time_h"] >= 0.0
    assert cand.inference_latency_ms > 0.0
    assert cand.ims is not None and 0.0 <= cand.ims.total <= 1.0
    assert cand.params["n_estimators"] >= 50 and cand.params["scale_pos_weight"] > 1.0
    assert cand.mlflow_run_id is not None  # file store in the temp dir

    path = Path(cand.artifact_path)
    assert path.exists() and path.parent == tmp_artifacts_dir
    model = joblib.load(path)
    proba = model.predict_proba(fdf[spec.features].head(20))
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_launch_trial_random_forest_and_xgboost_are_reproducible(
    features: tuple[pd.DataFrame, FeatureSpec], task: TaskSpec, tmp_artifacts_dir: Path
) -> None:
    fdf, spec = features
    for family in ("random_forest", "xgboost"):
        a = launch_trial(fdf, spec, task, family, n_trials=2, seed=1, artifacts_dir=tmp_artifacts_dir)
        b = launch_trial(fdf, spec, task, family, n_trials=2, seed=1, artifacts_dir=tmp_artifacts_dir)
        assert a.params == b.params
        assert a.metrics == b.metrics
        assert a.candidate_id != b.candidate_id


def test_launch_trial_rejects_bad_input(
    features: tuple[pd.DataFrame, FeatureSpec], task: TaskSpec, tmp_artifacts_dir: Path
) -> None:
    fdf, spec = features
    with pytest.raises(ValueError):
        launch_trial(fdf, spec, task, "svm", n_trials=1, artifacts_dir=tmp_artifacts_dir)
    # no failing asset among the training assets -> single class -> explicit error
    only_healthy_train = fdf[~fdf["asset_id"].isin(["MTR-003"])]
    with pytest.raises(ValueError, match="single class"):
        launch_trial(only_healthy_train, spec, task, "xgboost", n_trials=1, artifacts_dir=tmp_artifacts_dir)


def test_compare_models_ranks_by_ims() -> None:
    def cand(cid: str, total: float | None, lat: float = 1.0) -> CandidateModel:
        ims = None if total is None else IndustrialModelScore(
            recall=0, precision=0, lead_time_score=0, calibration_score=0, latency_score=0, total=total
        )
        return CandidateModel(candidate_id=cid, family="xgboost", params={}, metrics={}, ims=ims, inference_latency_ms=lat)

    ranked = compare_models([cand("a", 0.5), cand("b", None), cand("c", 0.9), cand("d", 0.5, lat=0.1)])
    assert ranked == ["c", "d", "a", "b"]


def test_train_candidates_end_to_end(
    features: tuple[pd.DataFrame, FeatureSpec], task: TaskSpec, tmp_artifacts_dir: Path
) -> None:
    fdf, spec = features
    cs = train_candidates(fdf, spec, task, n_trials=2, seed=0, artifacts_dir=tmp_artifacts_dir, asset_id="MTR-008")
    assert isinstance(cs, CandidateSet)
    assert cs.asset_id == "MTR-008"
    assert cs.mlflow_experiment == "thor-automl-MTR-008"
    assert [c.family for c in cs.candidates] == list(FAMILIES)
    assert cs.features == spec and cs.task == task
    totals = [c.ims.total for c in cs.candidates]
    by_id = {c.candidate_id: c for c in cs.candidates}
    assert cs.ranked == compare_models(cs.candidates)
    assert [by_id[i].ims.total for i in cs.ranked] == sorted(totals, reverse=True)
    for c in cs.candidates:
        assert Path(c.artifact_path).exists()
    # the synthetic fault is clear: the held-out failing motor must be caught with lead time
    best = by_id[cs.ranked[0]]
    assert best.metrics["auroc"] > 0.9
    assert best.metrics["recall"] > 0.5
    assert best.metrics["lead_time_h"] >= 12.0
