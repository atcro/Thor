"""Tests for agents/reliability (07): explain, rag, cost, contract.

Self-contained: builds its own tiny feature frame and models. Forces the offline hashed
embedding (THOR_RAG_EMBEDDING=hashed) so no model download ever happens in CI.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from agents.reliability import contract, cost, explain, rag
from apps.api import db
from apps.api.schemas import (
    BacktestFold,
    CalibrationReport,
    Explanation,
    FeatureSpec,
    IndustrialModelScore,
    LeadTimeReport,
    LeakageReport,
    ManualPassage,
    ShapFeature,
    ValidationReport,
)
from apps.api.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUALS_DIR = REPO_ROOT / "data" / "manuals"
FEATURES = ["vib_rms_z", "bearing_temp_z", "vib_kurt_mean", "load_mean"]
FAULTY = ("MTR-003", "MTR-006", "MTR-008")

os.environ.setdefault("THOR_RAG_EMBEDDING", "hashed")


def make_frame(n_assets: int = 8, n_rows: int = 400, seed: int = 1) -> pd.DataFrame:
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
            onset = int(n_rows * 0.4)
            ramp = np.clip((np.arange(n_rows) - onset) / (n_rows - onset), 0.0, 1.0)
            vib = vib + 5.0 * ramp**1.5
            temp = temp + 3.0 * ramp
            kurt = kurt + 3.0 * ramp
            rul = (n_rows - 1 - np.arange(n_rows)) * 10.0 / 60.0
            label = (rul <= 48.0).astype(int)
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
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="module")
def spec() -> FeatureSpec:
    return FeatureSpec(features=FEATURES, window_rows=12, description="test")


@pytest.fixture(scope="module")
def rf(frame: pd.DataFrame) -> RandomForestClassifier:
    m = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0)
    m.fit(frame[FEATURES], frame["label"])
    return m


@pytest.fixture(scope="module")
def latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Recent rows of the failing demo motor (latest last)."""
    return frame.loc[frame["asset_id"] == "MTR-008"].tail(40).reset_index(drop=True)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="module")
def engine():
    eng = db.get_engine("sqlite://")
    db.init_db(eng)
    return eng


# --------------------------------------------------------------------------------------
# explain / run_whatif
# --------------------------------------------------------------------------------------


def _assert_explanation(exp: Explanation, top_k: int) -> None:
    assert 0.0 <= exp.failure_probability <= 1.0
    assert 0 < len(exp.top_features) <= top_k
    mags = [abs(f.shap_value) for f in exp.top_features]
    assert mags == sorted(mags, reverse=True)
    for f in exp.top_features:
        assert f.feature in FEATURES
        assert f.direction == ("raises_risk" if f.shap_value > 0 else "lowers_risk")


def test_explain_random_forest(
    rf: RandomForestClassifier, latest_rows: pd.DataFrame, spec: FeatureSpec
) -> None:
    exp = explain.explain(rf, latest_rows, spec, "MTR-008", "v1", top_k=3)
    _assert_explanation(exp, 3)
    assert exp.asset_id == "MTR-008" and exp.model_version == "v1"
    assert exp.failure_probability > 0.5
    assert exp.top_features[0].feature in {"vib_rms_z", "bearing_temp_z", "vib_kurt_mean"}
    assert exp.top_features[0].direction == "raises_risk"


def test_explain_unwraps_calibrated_wrapper(
    rf: RandomForestClassifier, frame: pd.DataFrame, latest_rows: pd.DataFrame, spec: FeatureSpec
) -> None:
    hold = frame.loc[frame["asset_id"].isin(["MTR-007", "MTR-008"])]
    cal = CalibratedClassifierCV(FrozenEstimator(rf), method="isotonic")
    cal.fit(hold[FEATURES], hold["label"])
    assert explain.unwrap_tree_model(cal) is rf
    exp = explain.explain(cal, latest_rows, spec, "MTR-008", "v2")
    _assert_explanation(exp, 6)
    assert exp.failure_probability == pytest.approx(
        cal.predict_proba(latest_rows[FEATURES].tail(1))[0, 1]
    )


@pytest.mark.parametrize("family", ["lightgbm", "xgboost"])
def test_explain_boosted_families(
    family: str, frame: pd.DataFrame, latest_rows: pd.DataFrame, spec: FeatureSpec
) -> None:
    if family == "lightgbm":
        m = LGBMClassifier(n_estimators=40, num_leaves=8, random_state=0, verbose=-1)
    else:
        m = XGBClassifier(n_estimators=40, max_depth=3, random_state=0, verbosity=0)
    m.fit(frame[FEATURES], frame["label"])
    exp = explain.explain(m, latest_rows, spec, "MTR-008", "v3")
    _assert_explanation(exp, 6)
    assert exp.failure_probability > 0.5


def test_explain_model_agnostic_fallback(
    rf: RandomForestClassifier, latest_rows: pd.DataFrame, spec: FeatureSpec
) -> None:
    class Opaque:
        """Not a tree model: forces the shap.Explainer fallback path."""

        def __init__(self, inner: RandomForestClassifier) -> None:
            self._inner = inner

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return self._inner.predict_proba(X)

    exp = explain.explain(Opaque(rf), latest_rows, spec, "MTR-008", "v4")
    _assert_explanation(exp, 6)


def test_run_whatif_lowering_vibration_lowers_risk(
    rf: RandomForestClassifier, latest_rows: pd.DataFrame, spec: FeatureSpec
) -> None:
    res = explain.run_whatif(
        rf,
        latest_rows,
        spec,
        "MTR-008",
        {"vib_rms_z": 0.0, "bearing_temp_z": 0.0, "vib_kurt_mean": 3.0},
    )
    assert res.asset_id == "MTR-008"
    assert res.scenario == {"vib_rms_z": 0.0, "bearing_temp_z": 0.0, "vib_kurt_mean": 3.0}
    assert res.delta == pytest.approx(res.scenario_probability - res.baseline_probability)
    assert res.scenario_probability < res.baseline_probability


def test_run_whatif_rejects_unknown_feature(
    rf: RandomForestClassifier, latest_rows: pd.DataFrame, spec: FeatureSpec
) -> None:
    with pytest.raises(ValueError):
        explain.run_whatif(rf, latest_rows, spec, "MTR-008", {"not_a_feature": 1.0})


# --------------------------------------------------------------------------------------
# rag
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chroma_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("chroma")
    n = rag.build_index(MANUALS_DIR, path)
    assert n > 30
    return path


def test_manuals_exist_with_required_section() -> None:
    names = {p.name for p in MANUALS_DIR.glob("*.md")}
    assert {
        "motor-maintenance.md",
        "vibration-analysis-guide.md",
        "plant-maintenance-policy.md",
    } <= names
    motor = (MANUALS_DIR / "motor-maintenance.md").read_text(encoding="utf-8")
    assert "### 4.2 Drive-end bearing wear" in motor
    for name in names:
        n_lines = len((MANUALS_DIR / name).read_text(encoding="utf-8").splitlines())
        assert 150 <= n_lines <= 250, f"{name} has {n_lines} lines"


def test_hashed_embedding_is_forced_and_deterministic() -> None:
    assert os.environ["THOR_RAG_EMBEDDING"] == "hashed"
    ef = rag._embedding_function()
    assert isinstance(ef, rag.HashedEmbeddingFunction)
    a = ef(["drive end bearing wear"])[0]
    b = ef(["drive end bearing wear"])[0]
    assert np.allclose(a, b) and len(a) == rag.HASHED_DIM
    assert np.linalg.norm(a) == pytest.approx(1.0)


def test_chunk_markdown_keeps_numbered_section_labels() -> None:
    chunks = rag.chunk_markdown(
        (MANUALS_DIR / "motor-maintenance.md").read_text(encoding="utf-8"), "motor-maintenance.md"
    )
    sections = {c["metadata"]["section"] for c in chunks}
    assert "4.2 Drive-end bearing wear" in sections
    assert all(len(c["text"]) <= 900 for c in chunks)
    assert all(c["metadata"]["source"] == "motor-maintenance.md" for c in chunks)


def test_retrieve_manual_context_hits_bearing_section(chroma_path: Path) -> None:
    query = "drive-end bearing wear vibration RMS kurtosis crest factor bearing temperature signature recommended actions"
    passages = rag.retrieve_manual_context(query, chroma_path, k=3)
    assert len(passages) == 3
    assert all(isinstance(p, ManualPassage) for p in passages)
    assert all(0.0 <= p.score <= 1.0 for p in passages)
    assert passages[0].score >= passages[-1].score
    assert passages[0].source == "motor-maintenance.md"
    assert passages[0].section.startswith("4.2")
    assert passages[0].page is None


def test_retrieve_missing_index_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Retrieval self-heals by indexing the configured manuals dir; with no manuals there is
    # nothing to index and the result must be empty rather than an error.
    from apps.api.settings import get_settings

    empty = tmp_path / "no-manuals"
    empty.mkdir()
    monkeypatch.setenv("MANUALS_DIR", str(empty))
    get_settings.cache_clear()
    try:
        assert rag.retrieve_manual_context("anything", tmp_path / "nowhere") == []
    finally:
        get_settings.cache_clear()


def test_query_from_explanation_is_deterministic() -> None:
    exp = Explanation(
        asset_id="MTR-042",
        failure_probability=0.7,
        top_features=[
            ShapFeature(
                feature="vib_rms_z", shap_value=0.3, feature_value=3.1, direction="raises_risk"
            ),
            ShapFeature(
                feature="load_mean", shap_value=-0.1, feature_value=40.0, direction="lowers_risk"
            ),
            ShapFeature(
                feature="bearing_temp_z",
                shap_value=0.05,
                feature_value=1.2,
                direction="raises_risk",
            ),
        ],
        base_value=0.1,
        model_version="v1",
    )
    q1 = rag.query_from_explanation(exp)
    q2 = rag.query_from_explanation(exp)
    assert q1 == q2
    assert "vibration RMS above regime baseline" in q1
    assert "bearing temperature above regime baseline" in q1
    assert "motor load" not in q1


# --------------------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------------------


def test_hazard_curve_is_monotone_and_anchored() -> None:
    p = cost.hazard_curve(0.4, 48.0)
    assert p(0) == pytest.approx(0.4)
    vals = [p(h) for h in range(0, 200, 10)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert p(48.0) == pytest.approx(1 - 0.6**2)
    assert 0.0 <= p(10_000) <= 1.0


def test_calculate_failure_cost_high_risk_recommends_now(settings: Settings) -> None:
    now = datetime(2025, 9, 10, 12, 0, tzinfo=UTC)
    cmp_ = cost.calculate_failure_cost(
        0.62, settings, now=now, asset_id="MTR-042", lead_time_median_h=48.0
    )
    assert cmp_.asset_id == "MTR-042"
    assert [o.option for o in cmp_.options] == ["maintain_now", "maintain_later", "run_to_failure"]
    assert cmp_.recommended == "maintain_now"
    by = {o.option: o for o in cmp_.options}
    planned = (
        settings.cost_planned_maintenance
        + settings.downtime_planned_h * settings.cost_downtime_per_hour
    )
    assert by["maintain_now"].expected_cost == pytest.approx(planned)
    assert by["maintain_now"].when == now
    assert by["maintain_later"].when == now + timedelta(hours=72)
    assert by["run_to_failure"].when is None
    assert (
        by["maintain_now"].p_failure_before
        <= by["maintain_later"].p_failure_before
        <= by["run_to_failure"].p_failure_before
    )
    for key in (
        "cost_planned_maintenance",
        "cost_unplanned_repair",
        "cost_downtime_per_hour",
        "downtime_planned_h",
        "downtime_unplanned_h",
    ):
        assert cmp_.assumptions[key] == getattr(settings, key)
    assert cmp_.assumptions["p_now"] == pytest.approx(0.62)


def test_calculate_failure_cost_low_risk_does_not_recommend_now(settings: Settings) -> None:
    cmp_ = cost.calculate_failure_cost(0.05, settings, asset_id="MTR-001")
    assert cmp_.recommended != "maintain_now"
    assert all(o.when is None or o.when.tzinfo is not None for o in cmp_.options)


def test_calculate_failure_cost_accepts_callable(settings: Settings) -> None:
    cmp_ = cost.calculate_failure_cost(lambda h: min(1.0, 0.1 + 0.01 * h), settings, asset_id="X")
    assert cmp_.assumptions["p_now"] == pytest.approx(0.1)
    by = {o.option: o for o in cmp_.options}
    assert by["maintain_later"].p_failure_before == pytest.approx(0.82)


def test_find_maintenance_window_low_risk_picks_low_load_slot(settings: Settings) -> None:
    now = datetime(2025, 9, 10, 12, 0, tzinfo=UTC)
    cmp_ = cost.calculate_failure_cost(
        0.05, settings, now=now, asset_id="MTR-001", lead_time_median_h=48.0
    )
    lt = LeadTimeReport(median_h=48.0, p90_h=60.0, p10_h=20.0, n_events=2, threshold=0.5)
    win = cost.find_maintenance_window(cmp_, lt, now=now)
    assert win.start == datetime(2025, 9, 11, 2, 0, tzinfo=UTC)
    assert win.end == datetime(2025, 9, 11, 5, 0, tzinfo=UTC)
    assert win.p_failure_before_window < 0.3
    assert "low-load" in win.reason


def test_find_maintenance_window_high_risk_is_now(settings: Settings) -> None:
    now = datetime(2025, 9, 10, 12, 0, tzinfo=UTC)
    cmp_ = cost.calculate_failure_cost(0.62, settings, now=now, asset_id="MTR-042")
    lt = LeadTimeReport(median_h=30.0, p90_h=40.0, p10_h=10.0, n_events=1, threshold=0.5)
    win = cost.find_maintenance_window(cmp_, lt, now=now)
    assert win.start == now
    assert win.end == now + timedelta(hours=settings.downtime_planned_h)
    assert win.reason.startswith("now")
    assert win.p_failure_before_window == pytest.approx(0.62)


# --------------------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------------------


def _validation_report() -> ValidationReport:
    return ValidationReport(
        asset_id="MTR-042",
        champion_id="cand_lgbm",
        champion_family="lightgbm",
        champion_mlflow_run_id="run123",
        model_version="v777",
        leakage=LeakageReport(
            temporal_leakage=False, feature_leakage=[], asset_overlap=False, passed=True
        ),
        backtest=[
            BacktestFold(
                fold=0,
                train_assets=["MTR-001"],
                test_assets=["MTR-002"],
                train_end=datetime(2025, 9, 9, tzinfo=UTC),
                metrics={"recall": 0.9},
            )
        ],
        calibration=CalibrationReport(method="isotonic", brier_before=0.12, brier_after=0.08),
        lead_time=LeadTimeReport(median_h=36.0, p90_h=50.0, p10_h=20.0, n_events=2, threshold=0.5),
        rul=None,
        ims=IndustrialModelScore(
            recall=0.9,
            precision=0.7,
            lead_time_score=0.5,
            calibration_score=0.68,
            latency_score=0.9,
            total=0.75,
        ),
        passed=True,
        notes=["champion_artifact=/tmp/cand_lgbm_calibrated.joblib"],
    )


@pytest.fixture
def bundle(settings: Settings, chroma_path: Path):
    exp = Explanation(
        asset_id="MTR-042",
        failure_probability=0.62,
        top_features=[
            ShapFeature(
                feature="vib_rms_z", shap_value=0.31, feature_value=3.4, direction="raises_risk"
            ),
            ShapFeature(
                feature="bearing_temp_z",
                shap_value=0.12,
                feature_value=2.1,
                direction="raises_risk",
            ),
            ShapFeature(
                feature="load_mean", shap_value=-0.04, feature_value=52.0, direction="lowers_risk"
            ),
        ],
        base_value=0.08,
        model_version="v777",
    )
    now = datetime(2025, 9, 10, 12, 0, tzinfo=UTC)
    cmp_ = cost.calculate_failure_cost(
        exp.failure_probability, settings, now=now, asset_id="MTR-042", lead_time_median_h=36.0
    )
    val = _validation_report()
    win = cost.find_maintenance_window(cmp_, val.lead_time, now=now)
    passages = rag.retrieve_manual_context(rag.query_from_explanation(exp), chroma_path, k=3)
    return contract.build_evidence_bundle(exp, passages, cmp_, win, val, 91.5)


def test_build_evidence_bundle_and_hash(bundle) -> None:
    assert bundle.data_quality_score == 91.5
    assert bundle.manual_context and bundle.manual_context[0].source == "motor-maintenance.md"
    h = contract.evidence_hash(bundle)
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    assert h == contract.evidence_hash(bundle)


def test_create_decision_contract_inserts_immutable_row(bundle, engine) -> None:
    text = contract.template_explanation(bundle)
    dc = contract.create_decision_contract(bundle, "run_abc", text, "template", engine=engine)
    assert re.fullmatch(r"dc_MTR-042_\d{14}", dc.contract_id)
    assert dc.created_at.tzinfo is not None
    assert dc.recommendation == bundle.cost.recommended == "maintain_now"
    assert dc.expected_cost == pytest.approx(bundle.cost.options[0].expected_cost)
    assert dc.window_start == bundle.window.start and dc.window_end == bundle.window.end
    assert dc.lead_time_h == 36.0 and dc.calibration_brier == 0.08
    assert dc.champion_mlflow_run_id == "run123" and dc.model_version == "v777"
    assert dc.evidence_hash == contract.evidence_hash(bundle)
    assert dc.explanation_source == "template" and dc.explanation_text == text

    stored = db.get_decision_contract(dc.contract_id, engine=engine)
    assert stored is not None
    assert stored.model_dump_json() == dc.model_dump_json()
    assert db.contract_status(dc.contract_id, engine=engine) == "pending"

    # a second contract in the same second gets a suffix rather than colliding
    dc2 = contract.create_decision_contract(bundle, "run_abc", text, "template", engine=engine)
    assert dc2.contract_id != dc.contract_id
    assert dc2.contract_id.startswith(dc.contract_id[:15])
    assert len(db.list_decision_contracts("MTR-042", engine=engine)) >= 2


def test_template_explanation_cites_only_bundle_numbers(bundle) -> None:
    text = contract.template_explanation(bundle)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s]
    assert 4 <= len(sentences) <= 6, sentences
    assert "62%" in text
    assert "vibration RMS is 3.4 standard deviations above its regime baseline" in text
    assert "bearing temperature is 2.1 standard deviations above its regime baseline" in text
    assert "Under the configured plant-cost assumptions" in text
    by = {o.option: o for o in bundle.cost.options}
    for o in by.values():
        assert f"${o.expected_cost:,.0f}" in text
    assert "maintain now" in text
    assert "2025-09-10 12:00 UTC" in text
    assert "motor-maintenance.md section 4.2" in text
    assert "36 hours" in text
    assert "approves" in text
    assert text.isascii()


def test_template_explanation_without_manual_context(bundle) -> None:
    bare = contract.build_evidence_bundle(
        bundle.explanation,
        [],
        bundle.cost,
        bundle.window,
        bundle.validation,
        bundle.data_quality_score,
    )
    text = contract.template_explanation(bare)
    assert "section" not in text
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s]
    assert 4 <= len(sentences) <= 6
