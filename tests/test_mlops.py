"""Tests for 08 MLOps: registry lifecycle, governed promotion, ONNX edge export, PSI drift.

Self-contained: in-memory SQLite engine and fabricated ValidationReports -- no conftest.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from agents.mlops import drift, lifecycle
from apps.api import db
from apps.api.schemas import (
    Approval,
    BacktestFold,
    CalibrationReport,
    IndustrialModelScore,
    LeadTimeReport,
    LeakageReport,
    ModelStage,
    ValidationReport,
)

warnings.filterwarnings("ignore")

FEATURES = ["vib_rms_mean", "vib_rms_slope", "bearing_temp_mean", "temp_delta", "load_mean"]
ASSETS = [f"MTR-{i:03d}" for i in range(1, 9)]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mlflow_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture
def engine():
    eng = db.get_engine("sqlite://")
    db.init_db(eng)
    yield eng
    db.metadata.drop_all(eng)


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
    y = pd.Series(((X["vib_rms_mean"] + 0.5 * X["bearing_temp_mean"]) > 0.2).astype(int))
    X_test = pd.DataFrame(rng.normal(size=(20, len(FEATURES))), columns=FEATURES)
    return X, y, X_test


@pytest.fixture(scope="module")
def models(data) -> dict[str, object]:
    X, y, _ = data
    return {
        "random_forest": RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y),
        "xgboost": XGBClassifier(n_estimators=30, max_depth=3, random_state=0).fit(X, y),
        "lightgbm": LGBMClassifier(
            n_estimators=30, max_depth=3, random_state=0, verbose=-1
        ).fit(X, y),
    }


def make_validation(
    version: str, family: str = "random_forest", ims_total: float = 0.8, asset_id: str = "MTR-042"
) -> ValidationReport:
    return ValidationReport(
        asset_id=asset_id,
        champion_id=f"cand_{family}_{version}",
        champion_family=family,
        champion_mlflow_run_id=None,
        model_version=version,
        leakage=LeakageReport(temporal_leakage=False, asset_overlap=False, passed=True),
        backtest=[
            BacktestFold(
                fold=0,
                train_assets=ASSETS[:6],
                test_assets=ASSETS[6:],
                train_end=datetime(2026, 9, 1, tzinfo=UTC),
                metrics={"recall": 0.9, "precision": 0.8, "f1": 0.85, "auroc": 0.95, "brier": 0.08},
            )
        ],
        calibration=CalibrationReport(method="isotonic", brier_before=0.12, brier_after=0.08),
        lead_time=LeadTimeReport(median_h=36.0, p90_h=60.0, p10_h=18.0, n_events=2, threshold=0.5),
        rul=None,
        ims=IndustrialModelScore(
            recall=0.9,
            precision=0.8,
            lead_time_score=0.5,
            calibration_score=0.68,
            latency_score=0.9,
            total=ims_total,
        ),
        passed=True,
        notes=[f"champion_artifact=/tmp/{version}.joblib"],
    )


def approval_for(promotion_id: str, decision: str = "approved") -> Approval:
    return Approval(
        approval_id=f"apr_{promotion_id}",
        promotion_id=promotion_id,
        decision=decision,  # type: ignore[arg-type]
        approver="engineer@plant",
        note="test",
        decided_at=datetime.now(UTC),
    )


# --------------------------------------------------------------------------------------
# Registry + comparison
# --------------------------------------------------------------------------------------


def test_register_model_inserts_candidate(engine, tmp_path: Path) -> None:
    val = make_validation("v10001", family="xgboost", ims_total=0.81)
    reg = lifecycle.register_model(val, tmp_path / "m.joblib", engine=engine)
    assert reg.name == lifecycle.DEFAULT_MODEL_NAME
    assert reg.version == "v10001"
    assert reg.stage == ModelStage.candidate
    assert reg.family == "xgboost"
    assert reg.ims_total == pytest.approx(0.81)
    assert reg.dataset_version is not None and len(reg.dataset_version) == 16
    assert reg.onnx_path is None
    assert lifecycle.get_artifact_path(reg.name, reg.version, engine=engine) == tmp_path / "m.joblib"
    # idempotent re-registration
    again = lifecycle.register_model(val, tmp_path / "other.joblib", engine=engine)
    assert again.version == reg.version
    assert len(lifecycle.list_models(engine=engine)) == 1


def test_compare_champion_without_production(engine, tmp_path: Path) -> None:
    reg = lifecycle.register_model(make_validation("v10002"), tmp_path / "m.joblib", engine=engine)
    cmp = lifecycle.compare_champion(reg, engine=engine)
    assert cmp.champion is None
    assert cmp.recommend_promote is True
    assert cmp.delta["ims_total"] == pytest.approx(reg.ims_total)
    assert lifecycle.get_production_model(engine=engine) is None


# --------------------------------------------------------------------------------------
# Governed promotion
# --------------------------------------------------------------------------------------


def test_promote_requires_matching_approval(engine, tmp_path: Path) -> None:
    reg = lifecycle.register_model(make_validation("v10003"), tmp_path / "m.joblib", engine=engine)
    cmp = lifecycle.compare_champion(reg, engine=engine)
    req = lifecycle.request_promotion(cmp, ModelStage.validated, engine=engine)
    assert req.status == "pending"
    assert req.from_stage == ModelStage.candidate

    with pytest.raises(PermissionError):
        lifecycle.promote(req.promotion_id, approval_for(req.promotion_id, "rejected"), engine=engine)
    with pytest.raises(PermissionError):
        lifecycle.promote(req.promotion_id, approval_for("pr_someone_else"), engine=engine)
    # nothing moved
    assert lifecycle.get_registered_model(reg.name, reg.version, engine=engine).stage == (
        ModelStage.candidate
    )
    assert lifecycle.get_promotion(req.promotion_id, engine=engine).status == "pending"

    promoted = lifecycle.promote(req.promotion_id, approval_for(req.promotion_id), engine=engine)
    assert promoted.stage == ModelStage.validated
    assert lifecycle.get_promotion(req.promotion_id, engine=engine).status == "approved"
    # a consumed request cannot be replayed
    with pytest.raises(ValueError):
        lifecycle.promote(req.promotion_id, approval_for(req.promotion_id), engine=engine)


def test_failed_validation_blocks_request_and_promote(engine, tmp_path: Path) -> None:
    """Defect 2: `validation_passed = False` in the registry payload refuses both the request
    and (if a request slipped through earlier) the promotion itself."""
    from sqlalchemy import update

    failed = make_validation("v10009").model_copy(
        update={"passed": False, "notes": ["gate: leakage_ok=True recall=0.41 lead_time_median_h=36.0"]}
    )
    reg = lifecycle.register_model(failed, tmp_path / "m.joblib", engine=engine)
    assert lifecycle.get_artifact_path(reg.name, reg.version, engine=engine) is not None
    cmp = lifecycle.compare_champion(reg, engine=engine)
    with pytest.raises(lifecycle.PromotionBlocked, match="failed validation"):
        lifecycle.request_promotion(cmp, ModelStage.production, engine=engine)
    assert lifecycle.list_promotions(engine=engine) == []

    # a request opened while validation passed, then the row is marked failed -> promote refuses
    ok = lifecycle.register_model(make_validation("v10010"), tmp_path / "n.joblib", engine=engine)
    req = lifecycle.request_promotion(
        lifecycle.compare_champion(ok, engine=engine), ModelStage.production, engine=engine
    )
    with engine.begin() as conn:
        row = conn.execute(
            db.select(db.model_registry).where(db.model_registry.c.version == "v10010")
        ).mappings().first()
        payload = dict(row["payload"])
        payload["validation_passed"] = False
        payload["validation_failures"] = ["recall 0.41 < 0.6"]
        conn.execute(
            update(db.model_registry)
            .where(db.model_registry.c.version == "v10010")
            .values(payload=payload)
        )
    with pytest.raises(ValueError, match="recall 0.41"):
        lifecycle.promote(req.promotion_id, approval_for(req.promotion_id), engine=engine)
    assert lifecycle.get_registered_model(ok.name, ok.version, engine=engine).stage == (
        ModelStage.candidate
    )
    assert lifecycle.get_production_model(engine=engine) is None


def test_request_promotion_rejects_backward_transition(engine, tmp_path: Path) -> None:
    reg = lifecycle.register_model(make_validation("v10004"), tmp_path / "m.joblib", engine=engine)
    cmp = lifecycle.compare_champion(reg, engine=engine)
    with pytest.raises(ValueError):
        lifecycle.request_promotion(cmp, ModelStage.candidate, engine=engine)
    with pytest.raises(ValueError):
        lifecycle.request_promotion(cmp, ModelStage.archived, engine=engine)


def test_reject_promotion_marks_status(engine, tmp_path: Path) -> None:
    reg = lifecycle.register_model(make_validation("v10005"), tmp_path / "m.joblib", engine=engine)
    req = lifecycle.request_promotion(
        lifecycle.compare_champion(reg, engine=engine), ModelStage.validated, engine=engine
    )
    with pytest.raises(PermissionError):
        lifecycle.reject_promotion(req.promotion_id, approval_for(req.promotion_id), engine=engine)
    out = lifecycle.reject_promotion(
        req.promotion_id, approval_for(req.promotion_id, "rejected"), engine=engine
    )
    assert out.status == "rejected"
    assert lifecycle.list_promotions(status="rejected", engine=engine)[0].promotion_id == (
        req.promotion_id
    )
    assert lifecycle.get_registered_model(reg.name, reg.version, engine=engine).stage == (
        ModelStage.candidate
    )


def test_production_promotion_archives_previous(engine, tmp_path: Path) -> None:
    first = lifecycle.register_model(
        make_validation("v10006", ims_total=0.70), tmp_path / "a.joblib", engine=engine
    )
    req1 = lifecycle.request_promotion(
        lifecycle.compare_champion(first, engine=engine), ModelStage.production, engine=engine
    )
    prod1 = lifecycle.promote(req1.promotion_id, approval_for(req1.promotion_id), engine=engine)
    assert prod1.stage == ModelStage.production
    assert lifecycle.get_production_model(engine=engine).version == "v10006"

    second = lifecycle.register_model(
        make_validation("v10007", family="lightgbm", ims_total=0.85), tmp_path / "b.joblib", engine=engine
    )
    cmp = lifecycle.compare_champion(second, engine=engine)
    assert cmp.champion is not None and cmp.champion.version == "v10006"
    assert cmp.delta["ims_total"] == pytest.approx(0.15)
    assert cmp.recommend_promote is True
    req2 = lifecycle.request_promotion(cmp, ModelStage.production, engine=engine)
    prod2 = lifecycle.promote(req2.promotion_id, approval_for(req2.promotion_id), engine=engine)
    assert prod2.stage == ModelStage.production
    assert lifecycle.get_production_model(engine=engine).version == "v10007"
    stages = {m.version: m.stage for m in lifecycle.list_models(engine=engine)}
    assert stages == {"v10006": ModelStage.archived, "v10007": ModelStage.production}

    # a weaker challenger is not recommended
    third = lifecycle.register_model(
        make_validation("v10008", ims_total=0.60), tmp_path / "c.joblib", engine=engine
    )
    assert lifecycle.compare_champion(third, engine=engine).recommend_promote is False


# --------------------------------------------------------------------------------------
# ONNX edge export
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["random_forest", "xgboost", "lightgbm"])
def test_deploy_edge_exports_onnx_matching_predict_proba(
    engine, tmp_path: Path, models, data, family: str
) -> None:
    import onnxruntime as ort

    _, _, X_test = data
    fitted = models[family]
    artifact = tmp_path / f"{family}.joblib"
    joblib.dump(fitted, artifact)
    reg = lifecycle.register_model(
        make_validation(f"v2{family[:3]}", family=family), artifact, engine=engine
    )
    models_dir = tmp_path / "models"
    dep = lifecycle.deploy_edge(
        reg,
        artifact,
        FEATURES,
        models_dir,
        window_rows=12,
        baseline={"vib_rms_mean": {"mean": 0.0, "std": 1.0}},
        extra={"asset_id": "MTR-042"},
        engine=engine,
    )
    onnx_path = Path(dep.onnx_path)
    assert onnx_path.exists() and onnx_path.suffix == ".onnx"
    sidecar = json.loads(onnx_path.with_suffix(".json").read_text())
    assert sidecar["model_name"] == reg.name
    assert sidecar["version"] == reg.version
    assert sidecar["features"] == FEATURES
    assert sidecar["window_rows"] == 12
    assert sidecar["calibration"] == {"method": "none"}
    assert sidecar["family"] == family
    assert sidecar["baseline"]["vib_rms_mean"]["std"] == 1.0
    assert sidecar["asset_id"] == "MTR-042"
    assert dep.input_features == FEATURES
    assert lifecycle.get_registered_model(reg.name, reg.version, engine=engine).onnx_path == (
        str(onnx_path)
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: X_test.to_numpy(np.float32)})
    probs = next(o for o in outs if getattr(o, "ndim", 0) == 2 and o.shape[-1] == 2)
    assert probs.shape == (20, 2)
    np.testing.assert_allclose(probs[:, 1], fitted.predict_proba(X_test)[:, 1], atol=1e-3)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("family", ["random_forest", "xgboost", "lightgbm"])
def test_deploy_edge_exports_calibration_and_edge_reproduces_it(
    engine, tmp_path: Path, data, method: str, family: str
) -> None:
    """The ONNX graph is the raw tree; the sidecar carries the calibration transform (and
    whether sklearn fitted it on log-odds or on probabilities); the edge (edge.inference)
    reproduces the centre's calibrated probability for every family."""
    from sklearn.frozen import FrozenEstimator

    from agents.validation.checks import apply_calibration
    from edge.inference import load_model, predict_proba

    X, y, X_test = data
    base = {
        "random_forest": lambda: RandomForestClassifier(n_estimators=15, random_state=0),
        "xgboost": lambda: XGBClassifier(n_estimators=30, max_depth=3, random_state=0),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=30, max_depth=3, random_state=0, verbose=-1
        ),
    }[family]().fit(X[:300], y[:300])
    cal = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X[300:], y[300:])
    artifact = tmp_path / f"cal_{family}_{method}.joblib"
    joblib.dump(cal, artifact)
    reg = lifecycle.register_model(
        make_validation(f"v3{family[:2]}{method[:3]}", family=family), artifact, engine=engine
    )
    dep = lifecycle.deploy_edge(reg, artifact, FEATURES, tmp_path / "models", engine=engine)
    sidecar = json.loads(Path(dep.onnx_path).with_suffix(".json").read_text())
    assert sidecar["calibration"]["method"] == method
    expected_input = "decision_function" if hasattr(base, "decision_function") else "predict_proba"
    assert sidecar["calibration"]["input"] == expected_input
    model = load_model(Path(dep.onnx_path), Path(dep.onnx_path).with_suffix(".json"))
    p_edge = predict_proba(model, X_test.to_numpy())
    expected = cal.predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(p_edge, expected, atol=1e-3)
    # the sidecar alone is enough: the reference transform agrees with sklearn exactly
    raw = base.predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(apply_calibration(raw, sidecar["calibration"]), expected, atol=1e-6)


def test_deploy_edge_refuses_multi_fold_calibrator(engine, tmp_path: Path, data) -> None:
    X, y, _ = data
    cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=15, random_state=0), cv=3)
    cal.fit(X, y)
    artifact = tmp_path / "cv3.joblib"
    joblib.dump(cal, artifact)
    reg = lifecycle.register_model(make_validation("v30009"), artifact, engine=engine)
    with pytest.raises(ValueError, match="cannot be exported"):
        lifecycle.deploy_edge(reg, artifact, FEATURES, tmp_path / "models", engine=engine)


class _LyingWrapper:
    """A central model whose predict_proba disagrees with the tree it exposes for export."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def predict_proba(self, X):
        p = self.estimator.predict_proba(X)
        return np.column_stack([p[:, 1], p[:, 0]])  # swapped on purpose


def test_check_edge_parity_passes_and_fails_honestly(tmp_path: Path, data) -> None:
    from sklearn.frozen import FrozenEstimator

    X, y, X_test = data
    rf = RandomForestClassifier(n_estimators=15, random_state=0).fit(X[:300], y[:300])
    cal = CalibratedClassifierCV(FrozenEstimator(rf), method="isotonic").fit(X[300:], y[300:])
    good = tmp_path / "good.joblib"
    joblib.dump(cal, good)
    report = lifecycle.check_edge_parity(good, FEATURES, X_test, window_rows=12, baseline=None)
    assert report.passed and report.n_rows == 20
    lgbm = LGBMClassifier(n_estimators=30, max_depth=3, random_state=0, verbose=-1)
    lgbm.fit(X[:300], y[:300])
    cal_l = CalibratedClassifierCV(FrozenEstimator(lgbm), method="isotonic").fit(X[300:], y[300:])
    good_l = tmp_path / "good_lgbm.joblib"
    joblib.dump(cal_l, good_l)
    report = lifecycle.check_edge_parity(good_l, FEATURES, X_test, window_rows=12, baseline=None)
    assert report.passed, report.notes
    assert report.max_abs_diff_probability <= report.tolerance_probability
    assert report.calibration_method == "isotonic"

    bad = tmp_path / "bad.joblib"
    joblib.dump(_LyingWrapper(rf), bad)
    report = lifecycle.check_edge_parity(bad, FEATURES, X_test, window_rows=12, baseline=None)
    assert not report.passed and report.max_abs_diff_probability > 0.1
    assert any("PARITY FAILED" in n for n in report.notes)

    broken = tmp_path / "missing.joblib"
    report = lifecycle.check_edge_parity(broken, FEATURES, X_test)
    assert not report.passed and any("error" in n for n in report.notes)


def test_failed_parity_blocks_promotion(engine, tmp_path: Path) -> None:
    from apps.api.schemas import ParityReport

    failed = ParityReport(
        passed=False,
        n_rows=20,
        max_abs_diff_probability=0.4,
        tolerance_probability=1e-3,
        notes=["PARITY FAILED: the edge would not reproduce the centre's probability"],
    )
    val = make_validation("v10020").model_copy(update={"parity": failed})
    reg = lifecycle.register_model(val, tmp_path / "m.joblib", engine=engine)
    with pytest.raises(lifecycle.PromotionBlocked, match="parity"):
        lifecycle.request_promotion(
            lifecycle.compare_champion(reg, engine=engine), ModelStage.production, engine=engine
        )
    ok = make_validation("v10021").model_copy(update={"parity": failed.model_copy(update={"passed": True})})
    reg2 = lifecycle.register_model(ok, tmp_path / "n.joblib", engine=engine)
    req = lifecycle.request_promotion(
        lifecycle.compare_champion(reg2, engine=engine), ModelStage.production, engine=engine
    )
    assert lifecycle.promote(req.promotion_id, approval_for(req.promotion_id), engine=engine).stage == (
        ModelStage.production
    )


# --------------------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------------------


def test_psi_identical_and_shifted() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=5000)
    assert drift.psi(base, base) == pytest.approx(0.0, abs=1e-6)
    assert drift.psi(base, rng.normal(size=5000)) < 0.05
    assert drift.psi(base, rng.normal(loc=1.5, size=5000)) > 0.2
    assert drift.psi(np.ones(100), np.ones(100)) == 0.0
    assert drift.psi(np.array([]), base) == 0.0


def test_detect_drift_inserts_report(engine) -> None:
    rng = np.random.default_rng(2)
    train = pd.DataFrame(rng.normal(size=(2000, len(FEATURES))), columns=FEATURES)
    recent = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
    recent["vib_rms_mean"] = recent["vib_rms_mean"] + 2.0
    recent["ts"] = pd.date_range("2026-09-10", periods=500, freq="10min", tz="UTC")
    report = drift.detect_drift(train, recent, FEATURES, "MTR-042", "v10001", engine=engine)
    assert report.drift_detected is True
    assert report.drifted_features == ["vib_rms_mean"]
    assert set(report.psi) == set(FEATURES)
    assert report.psi["load_mean"] < 0.2
    assert report.window_start < report.window_end
    with engine.connect() as conn:
        rows = conn.execute(db.select(db.drift_reports)).mappings().all()
    assert len(rows) == 1 and rows[0]["drift_detected"] is True
    assert rows[0]["payload"]["drifted_features"] == ["vib_rms_mean"]

    calm = drift.detect_drift(train, train.tail(500), FEATURES, "MTR-001", "v10001", engine=engine)
    assert calm.drift_detected is False and calm.drifted_features == []
