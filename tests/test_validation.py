"""Tests for agents/validation/checks.py (06 Validation).

Self-contained: builds a tiny synthetic feature frame (8 motors, 3 of them degrading) and
fits small tree models. Does not depend on tests/conftest.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from agents.validation import checks
from apps.api.schemas import CandidateModel, CandidateSet, FeatureSpec, TaskSpec

FEATURES = ["vib_rms_z", "bearing_temp_z", "vib_kurt_mean", "load_mean"]
FAULTY = ("MTR-003", "MTR-006", "MTR-008")
HORIZON_H = 48.0


def make_frame(n_assets: int = 8, n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Feature frame per docs/INTERFACES.md: asset_id, ts, features, label, rul_h."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-09-01 00:00", tz="UTC")
    frames = []
    for i in range(1, n_assets + 1):
        aid = f"MTR-{i:03d}"
        ts = t0 + pd.to_timedelta(np.arange(n_rows) * 10, unit="m")
        vib = rng.normal(0.0, 1.0, n_rows)
        temp = rng.normal(0.0, 1.0, n_rows)
        kurt = rng.normal(3.0, 0.2, n_rows)
        load = rng.normal(55.0, 5.0, n_rows)
        rul = np.full(n_rows, np.nan)
        label = np.zeros(n_rows, dtype=int)
        if aid in FAULTY:
            onset = int(n_rows * 0.45)
            ramp = np.clip((np.arange(n_rows) - onset) / (n_rows - onset), 0.0, 1.0)
            vib = vib + 5.0 * ramp**1.5
            temp = temp + 3.0 * ramp
            kurt = kurt + 3.0 * ramp
            rul = (n_rows - 1 - np.arange(n_rows)) * 10.0 / 60.0
            label = (rul <= HORIZON_H).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "asset_id": aid,
                    "ts": ts,
                    "vib_rms_z": vib,
                    "bearing_temp_z": temp,
                    "vib_kurt_mean": kurt,
                    "load_mean": load,
                    "label": label,
                    "rul_h": rul,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["asset_id", "ts"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return FeatureSpec(features=FEATURES, window_rows=12, description="test features")


@pytest.fixture(scope="module")
def fitted_rf(frame: pd.DataFrame) -> RandomForestClassifier:
    train = frame.loc[frame["asset_id"] <= "MTR-006"]
    rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0)
    rf.fit(train[FEATURES], train["label"])
    return rf


# --------------------------------------------------------------------------------------
# detect_leakage
# --------------------------------------------------------------------------------------


def test_detect_leakage_clean_split_passes(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    assets = sorted(frame["asset_id"].unique())
    rep = checks.detect_leakage(frame, spec, assets[:6], assets[6:])
    assert rep.passed
    assert not rep.asset_overlap
    assert not rep.temporal_leakage
    assert rep.feature_leakage == []


def test_detect_leakage_flags_overlap_forbidden_and_correlated(frame: pd.DataFrame) -> None:
    df = frame.copy()
    df["leaky"] = df["label"].astype(float) + 0.001 * np.arange(len(df)) / len(df)
    bad_spec = FeatureSpec(
        features=FEATURES + ["leaky", "failure_within_h"], window_rows=12, description="bad"
    )
    assets = sorted(df["asset_id"].unique())
    rep = checks.detect_leakage(df, bad_spec, assets[:6], assets[5:])
    assert not rep.passed
    assert rep.asset_overlap
    assert rep.temporal_leakage
    assert "failure_within_h" in rep.feature_leakage
    assert "leaky" in rep.feature_leakage
    assert "vib_rms_z" not in rep.feature_leakage


# --------------------------------------------------------------------------------------
# run_backtest
# --------------------------------------------------------------------------------------


def test_run_backtest_folds_are_asset_level_and_time_ordered(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    folds = checks.run_backtest(
        frame, spec, lambda: RandomForestClassifier(n_estimators=20, random_state=0), n_folds=3
    )
    assert len(folds) == 3
    all_assets = set(frame["asset_id"].unique())
    seen_test: set[str] = set()
    for f in folds:
        assert not (set(f.train_assets) & set(f.test_assets)), "held-out assets leaked into train"
        assert set(f.train_assets) | set(f.test_assets) == all_assets
        assert f.train_end.tzinfo is not None
        for key in ("recall", "precision", "f1", "brier", "n_test", "n_pos_test"):
            assert key in f.metrics
        # auroc / lead_time_h are omitted (not NaN) when undefined so the report is JSON-safe
        assert all(np.isfinite(v) for v in f.metrics.values())
        assert 0.0 <= f.metrics["brier"] <= 1.0
        seen_test |= set(f.test_assets)
    assert seen_test == all_assets, "every asset is held out exactly once across folds"
    # a fold whose held-out assets include a failing motor must show recall on it
    scored = [f for f in folds if f.metrics["n_pos_test"] > 0]
    assert scored and max(f.metrics["recall"] for f in scored) > 0.5


def test_run_backtest_handles_single_class_holdout(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    folds = checks.run_backtest(
        frame, spec, lambda: RandomForestClassifier(n_estimators=10, random_state=0), n_folds=3
    )
    single = [f for f in folds if f.metrics["n_pos_test"] == 0]
    assert single, "fixture should produce at least one fold with only healthy held-out motors"
    assert "auroc" not in single[0].metrics
    assert "lead_time_h" not in single[0].metrics
    scored = [f for f in folds if f.metrics["n_pos_test"] > 0]
    assert scored and "auroc" in scored[0].metrics


# --------------------------------------------------------------------------------------
# calibrate_probabilities
# --------------------------------------------------------------------------------------


def test_calibrate_isotonic_when_enough_rows(frame: pd.DataFrame, fitted_rf: RandomForestClassifier) -> None:
    hold = frame.loc[frame["asset_id"] > "MTR-006"]
    assert len(hold) >= 200
    wrapper, rep = checks.calibrate_probabilities(fitted_rf, hold[FEATURES], hold["label"])
    assert rep.method == "isotonic"
    proba = wrapper.predict_proba(hold[FEATURES])
    assert proba.shape == (len(hold), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert rep.brier_after <= rep.brier_before + 0.02
    assert rep.reliability_bins and all(c > 0 for _, _, c in rep.reliability_bins)


def test_calibrate_sigmoid_when_few_rows(frame: pd.DataFrame, fitted_rf: RandomForestClassifier) -> None:
    hold = frame.loc[frame["asset_id"] == "MTR-008"].iloc[::6].head(100)
    assert hold["label"].nunique() == 2 and len(hold) < 200
    wrapper, rep = checks.calibrate_probabilities(fitted_rf, hold[FEATURES], hold["label"])
    assert rep.method == "sigmoid"
    assert wrapper.predict_proba(hold[FEATURES]).shape == (len(hold), 2)


def test_calibrate_none_when_single_class(frame: pd.DataFrame, fitted_rf: RandomForestClassifier) -> None:
    hold = frame.loc[frame["asset_id"] == "MTR-007"]
    wrapper, rep = checks.calibrate_probabilities(fitted_rf, hold[FEATURES], hold["label"])
    assert rep.method == "none"
    assert wrapper.predict_proba(hold[FEATURES]).shape == (len(hold), 2)


# --------------------------------------------------------------------------------------
# compute_lead_time / estimate_rul_interval
# --------------------------------------------------------------------------------------


def test_compute_lead_time_counts_failing_assets(frame: pd.DataFrame, spec: FeatureSpec, fitted_rf: RandomForestClassifier) -> None:
    rep = checks.compute_lead_time(frame, spec, fitted_rf, threshold=0.5)
    assert rep.n_events == len(FAULTY)
    assert rep.threshold == 0.5
    assert rep.p10_h <= rep.median_h <= rep.p90_h
    assert rep.median_h >= 12.0


def test_compute_lead_time_no_events(frame: pd.DataFrame, spec: FeatureSpec, fitted_rf: RandomForestClassifier) -> None:
    healthy = frame.loc[~frame["asset_id"].isin(FAULTY)]
    rep = checks.compute_lead_time(healthy, spec, fitted_rf)
    assert rep.n_events == 0 and rep.median_h == 0.0


def test_estimate_rul_interval_ordered(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    rul = checks.estimate_rul_interval(frame, spec, seed=0, asset_id="MTR-008")
    assert rul.method == "quantile_gbm"
    assert 0.0 <= rul.p10_h <= rul.p50_h <= rul.p90_h
    # latest row of a motor about to fail: median RUL should be short
    assert rul.p50_h < 30.0


def test_estimate_rul_interval_requires_labels(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    healthy = frame.loc[~frame["asset_id"].isin(FAULTY)]
    with pytest.raises(ValueError):
        checks.estimate_rul_interval(healthy, spec)


# --------------------------------------------------------------------------------------
# validate (end to end)
# --------------------------------------------------------------------------------------


def _candidate_set(frame: pd.DataFrame, spec: FeatureSpec, tmp_path: Path) -> CandidateSet:
    train = frame.loc[frame["asset_id"] <= "MTR-006"]
    rf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0)
    rf.fit(train[FEATURES], train["label"])
    lgbm = LGBMClassifier(n_estimators=60, num_leaves=8, random_state=0, verbose=-1)
    lgbm.fit(train[FEATURES], train["label"])
    cands = []
    for cid, family, model, latency in (
        ("cand_rf", "random_forest", rf, 4.0),
        ("cand_lgbm", "lightgbm", lgbm, 1.5),
    ):
        path = tmp_path / f"{cid}.joblib"
        joblib.dump(model, path)
        cands.append(
            CandidateModel(
                candidate_id=cid,
                family=family,
                params={},
                metrics={},
                artifact_path=str(path),
                inference_latency_ms=latency,
            )
        )
    cands.append(
        CandidateModel(
            candidate_id="cand_missing",
            family="xgboost",
            params={},
            metrics={},
            artifact_path=str(tmp_path / "does_not_exist.joblib"),
            inference_latency_ms=2.0,
        )
    )
    return CandidateSet(
        asset_id="MTR-008",
        task=TaskSpec(task="classification", target="label", horizon_h=HORIZON_H, rationale="t"),
        features=spec,
        candidates=cands,
        mlflow_experiment="test",
    )


def test_validate_end_to_end(frame: pd.DataFrame, spec: FeatureSpec, tmp_path: Path) -> None:
    cs = _candidate_set(frame, spec, tmp_path)
    report = checks.validate(cs, frame, run_id="run_test")

    assert report.asset_id == "MTR-008"
    assert report.champion_id in {"cand_rf", "cand_lgbm"}
    assert report.champion_family in {"random_forest", "lightgbm"}
    assert report.model_version.startswith("v")
    assert report.leakage.passed
    assert len(report.backtest) == 3
    assert report.calibration.method in {"isotonic", "sigmoid"}
    assert 0.0 <= report.ims.total <= 1.0
    assert abs(sum(report.ims.weights.values()) - 1.0) < 1e-9
    assert report.lead_time.n_events >= 1
    assert report.rul is not None and report.rul.p10_h <= report.rul.p90_h
    assert report.passed, report.notes

    # the calibrated champion is communicated through notes[0]
    assert report.notes[0].startswith("champion_artifact=")
    cal_path = checks.champion_artifact_path(report)
    assert cal_path is not None and cal_path.exists()
    assert cal_path.name == f"{report.champion_id}_calibrated.joblib"
    assert cal_path.parent == tmp_path
    model = joblib.load(cal_path)
    proba = model.predict_proba(frame[FEATURES].tail(5))
    assert proba.shape == (5, 2)


def test_validate_raises_without_loadable_candidates(frame: pd.DataFrame, spec: FeatureSpec, tmp_path: Path) -> None:
    cs = CandidateSet(
        asset_id="MTR-008",
        task=TaskSpec(task="classification", target="label", rationale="t"),
        features=spec,
        candidates=[
            CandidateModel(
                candidate_id="nope",
                family="xgboost",
                params={},
                metrics={},
                artifact_path=str(tmp_path / "missing.joblib"),
                inference_latency_ms=1.0,
            )
        ],
        mlflow_experiment="test",
    )
    with pytest.raises(ValueError):
        checks.validate(cs, frame, run_id="r")


def test_industrial_model_score_formula() -> None:
    ims = checks._industrial_model_score(
        {"recall": 0.8, "precision": 0.5, "lead_time_h": 36.0, "brier": 0.05}, latency_ms=10.0
    )
    assert ims.lead_time_score == pytest.approx(0.5)
    assert ims.calibration_score == pytest.approx(0.8)
    assert ims.latency_score == pytest.approx(0.8)
    expected = 0.35 * 0.8 + 0.25 * 0.5 + 0.15 * 0.5 + 0.15 * 0.8 + 0.10 * 0.8
    assert ims.total == pytest.approx(expected)
