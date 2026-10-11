"""Leak-free, reproducible asset-level split (defect 4).

The split is fixed first and recorded; regimes and baselines are fitted on training assets
only; the Optuna objective sees validation assets; calibration uses calibration assets; the
final metrics come from untouched test assets; right-censored rows carry no label.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from agents.data_agent.profiling import detect_regime
from agents.ml_architect import automl
from agents.ml_architect.features import FEATURE_NAMES, baseline_for, build_feature_pipeline
from agents.ml_architect.split import completed_failure_assets, make_asset_split, rows_in
from agents.validation import checks
from apps.api.schemas import CandidateModel, CandidateSet, TaskSpec
from tests.conftest import make_synthetic_fleet


def _fleet(n_assets: int = 12, degrading: dict[str, float] | None = None) -> pd.DataFrame:
    if degrading is None:
        degrading = {"MTR-003": 0.5, "MTR-005": 0.55, "MTR-008": 0.6, "MTR-011": 0.5}
    return make_synthetic_fleet(n_assets=n_assets, days=5, degrading=degrading)


def test_split_is_seeded_disjoint_and_complete() -> None:
    df = _fleet()
    a = make_asset_split(df, seed=0, forced_test=["MTR-001"])
    b = make_asset_split(df, seed=0, forced_test=["MTR-001"])
    assert a == b  # reproducible
    c = make_asset_split(df, seed=1, forced_test=["MTR-001"])
    assert c.train != a.train  # the seed matters
    groups = [a.train, a.validation, a.calibration, a.test]
    assert sorted(set(a.train) | set(a.validation) | set(a.calibration) | set(a.test)) == sorted(
        df["asset_id"].unique()
    )
    assert not set(a.train) & set(a.test)
    assert not set(a.calibration) & set(a.test)
    assert not set(a.validation) & set(a.test)
    assert all(g for g in groups)
    assert "MTR-001" in a.test and a.forced_test == ["MTR-001"]
    assert completed_failure_assets(df) == ["MTR-003", "MTR-005", "MTR-008", "MTR-011"]


def test_positives_are_dealt_train_test_tune_first() -> None:
    df = _fleet()  # 4 completed failures -> train 2, test 1, tune 1 (validation == calibration)
    s = make_asset_split(df, seed=0)
    assert len(s.positives["train"]) == 2
    assert len(s.positives["test"]) == 1
    assert len(s.positives["validation"]) == 1
    assert s.validation == s.calibration
    assert any("share the tune group" in n for n in s.notes)
    # with five or more completed failures validation and calibration become separate groups
    six = _fleet(
        n_assets=16,
        degrading={f"MTR-{i:03d}": 0.5 for i in (2, 4, 6, 8, 10, 12)},
    )
    s6 = make_asset_split(six, seed=0)
    assert s6.validation != s6.calibration
    assert not set(s6.validation) & set(s6.calibration)
    assert all(s6.positives[g] for g in ("train", "test"))


def test_regimes_and_baseline_depend_on_training_assets_only() -> None:
    df = _fleet()
    split = make_asset_split(df, seed=0)
    regimes = detect_regime(df, fit_assets=split.train)
    assert regimes.fit_assets == split.train
    base = baseline_for(df, regimes, fit_assets=split.train)
    # perturb every non-training asset wildly: the fitted centroids / stats must not move
    tampered = df.copy()
    held = ~tampered["asset_id"].isin(split.train)
    tampered.loc[held, "rpm"] = tampered.loc[held, "rpm"] * 3.0 + 900.0
    tampered.loc[held, "vibration_rms"] = tampered.loc[held, "vibration_rms"] * 10.0
    tampered.loc[held, "bearing_temp_c"] = tampered.loc[held, "bearing_temp_c"] + 50.0
    regimes_t = detect_regime(tampered, fit_assets=split.train)
    base_t = baseline_for(tampered, regimes_t, fit_assets=split.train)
    assert [r.regime_id for r in regimes_t.regimes] == [r.regime_id for r in regimes.regimes]
    train_mask = df["asset_id"].isin(split.train).to_numpy()
    assert (
        np.asarray(regimes.row_regime)[train_mask] == np.asarray(regimes_t.row_regime)[train_mask]
    ).all()
    for sensor in ("vibration_rms", "bearing_temp_c", "rpm"):
        assert base[sensor] == pytest.approx(base_t[sensor])
    assert base["_regime_centroids"] == pytest.approx(base_t["_regime_centroids"])
    # and the all-assets fit DOES move when the held-out rows are tampered (so the test bites)
    assert baseline_for(tampered, detect_regime(tampered))["vibration_rms"]["mean"] != pytest.approx(
        baseline_for(df, detect_regime(df))["vibration_rms"]["mean"]
    )


def test_feature_pipeline_right_censors_unknown_tail_and_records_fit_assets() -> None:
    df = _fleet()
    split = make_asset_split(df, seed=0)
    regimes = detect_regime(df, fit_assets=split.train)
    fdf, spec = build_feature_pipeline(df, regimes, horizon_h=48.0, fit_assets=split.train)
    assert list(fdf.columns) == ["asset_id", "ts", *FEATURE_NAMES, "label", "rul_h", "label_known"]
    assert spec.fit_assets == split.train and spec.baseline is not None
    healthy = fdf[fdf["asset_id"] == "MTR-001"]
    end = healthy["ts"].max()
    tail = healthy[healthy["ts"] > end - pd.Timedelta(hours=48)]
    head = healthy[healthy["ts"] <= end - pd.Timedelta(hours=48)]
    assert not tail["label_known"].any() and head["label_known"].all()
    failing = fdf[fdf["asset_id"] == "MTR-003"]
    assert failing["label_known"].all()  # a completed failure has a known future everywhere
    assert rows_in(fdf, ["MTR-001"])["label_known"].all()
    assert len(rows_in(fdf, ["MTR-001"], known_only=False)) == len(healthy)
    # a fixed baseline is reused verbatim (nothing refitted) -- what Run analysis relies on
    fdf2, spec2 = build_feature_pipeline(df, None, horizon_h=48.0, baseline=spec.baseline)
    assert spec2.baseline == spec.baseline
    pd.testing.assert_frame_equal(fdf2[FEATURE_NAMES], fdf[FEATURE_NAMES])


def test_validate_scores_test_assets_only_and_records_the_split(tmp_path: Path) -> None:
    df = _fleet()
    split = make_asset_split(df, seed=0)
    regimes = detect_regime(df, fit_assets=split.train)
    fdf, spec = build_feature_pipeline(df, regimes, horizon_h=48.0, fit_assets=split.train)
    train = rows_in(fdf, split.train)
    cands = []
    for cid, n in (("rf_a", 30), ("rf_b", 10)):
        rf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=0)
        rf.fit(train[spec.features], train["label"])
        path = tmp_path / f"{cid}.joblib"
        joblib.dump(rf, path)
        cands.append(
            CandidateModel(
                candidate_id=cid,
                family="random_forest",
                params={},
                metrics={},
                artifact_path=str(path),
                inference_latency_ms=2.0,
            )
        )
    cs = CandidateSet(
        asset_id="MTR-003",
        task=TaskSpec(task="classification", target="label", horizon_h=48.0, rationale="t"),
        features=spec,
        candidates=cands,
        mlflow_experiment="t",
        split=split,
    )
    report = checks.validate(cs, fdf, run_id="r")
    assert report.split == split
    assert len(report.backtest) == 1
    assert report.backtest[0].test_assets == split.test
    assert report.backtest[0].train_assets == split.train
    assert set(report.test_metrics) >= {"recall", "precision", "brier", "lead_time_h", "n_assets"}
    assert report.test_metrics["n_assets"] == len(split.test)
    assert report.lead_time.n_events == len(split.positives["test"]) == 1
    assert {g.name for g in report.gate} == {"leakage", "test_failure_events", "recall", "lead_time"}
    assert report.passed == all(g.passed for g in report.gate)
    cal_note = next(n for n in report.notes if n.startswith("calibration ("))
    for a in split.test:
        assert a not in cal_note  # calibration never touched a test asset
    assert report.leakage.passed


def test_launch_trial_trains_and_selects_on_the_split() -> None:
    df = _fleet()
    split = make_asset_split(df, seed=0)
    regimes = detect_regime(df, fit_assets=split.train)
    fdf, spec = build_feature_pipeline(df, regimes, horizon_h=48.0, fit_assets=split.train)
    task = TaskSpec(task="classification", target="label", horizon_h=48.0, rationale="t")
    requested: list[set[str]] = []
    real_rows_in = automl.rows_in

    def spy_rows_in(frame: pd.DataFrame, assets, known_only: bool = True) -> pd.DataFrame:
        requested.append(set(assets))
        return real_rows_in(frame, assets, known_only)

    automl.rows_in = spy_rows_in  # type: ignore[assignment]
    try:
        cand = automl.launch_trial(fdf, spec, task, "random_forest", n_trials=1, split=split)
    finally:
        automl.rows_in = real_rows_in  # type: ignore[assignment]
    assert cand.artifact_path and Path(cand.artifact_path).exists()
    assert requested == [set(split.train), set(split.validation)]
    assert not any(r & set(split.test) for r in requested)  # test assets never touched
    # the candidate's recorded metrics are validation-group metrics at the shared thresholds
    assert set(cand.metrics) >= {"recall", "precision", "lead_time_h", "brier"}
