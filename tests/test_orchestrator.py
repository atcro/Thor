"""Unit tests for apps/api/orchestrator.py -- run WITHOUT the agent toolboxes present.

Also exports schema factories (`make_*`) and `install_fake_toolbox()` reused by tests/test_api.py.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from apps.api import db
from apps.api.schemas import (
    Approval,
    BacktestFold,
    CalibrationReport,
    CandidateModel,
    CandidateSet,
    ChampionComparison,
    CopilotMessage,
    CopilotRequest,
    CostComparison,
    CostOption,
    DataQualityContract,
    DecisionContract,
    EdgeDeployment,
    EvidenceBundle,
    Explanation,
    FeatureSpec,
    GraphState,
    IndustrialModelScore,
    LeadTimeReport,
    LeakageReport,
    MaintenanceWindow,
    ManualPassage,
    MissingnessReport,
    ModelStage,
    PipelineStage,
    PromotionRequest,
    Regime,
    RegimeReport,
    RegisteredModel,
    RULInterval,
    ShapFeature,
    TaskSpec,
    TelemetryRow,
    ValidationReport,
    WhatIfResult,
)
from apps.api.settings import get_settings

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
FEATURES = ["vib_rms_mean", "vib_rms_slope", "bearing_temp_mean", "load_mean"]
ASSETS = ["MTR-001", "MTR-002", "MTR-003", "MTR-042"]


# --------------------------------------------------------------------------------------
# Environment / DB fixtures
# --------------------------------------------------------------------------------------


def configure_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point settings at a temp SQLite file + empty dirs, and clear the lru caches."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'thor_test.db').as_posix()}")
    monkeypatch.setenv("SIMULATOR_OUT", str(tmp_path / "sim_out"))
    monkeypatch.setenv("MANUALS_DIR", str(tmp_path / "manuals"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("THOR_SKIP_BACKGROUND", "1")
    monkeypatch.delenv("SEED_ON_START", raising=False)
    get_settings.cache_clear()
    db.get_engine.cache_clear()


def seed_small_fleet(engine: Any, n_rows: int = 30) -> None:
    """Four assets, `n_rows` 10-minute samples each; MTR-042 degrading."""
    from apps.api.schemas import Asset

    db.upsert_assets(
        [
            Asset(asset_id=a, name=f"Motor {a[-3:]}", site="Plant A", line=f"L{i % 2 + 1}")
            for i, a in enumerate(ASSETS)
        ],
        engine=engine,
    )
    rows: list[TelemetryRow] = []
    for a in ASSETS:
        faulty = a == "MTR-042"
        for i in range(n_rows):
            frac = i / max(n_rows - 1, 1)
            rows.append(
                TelemetryRow(
                    asset_id=a,
                    ts=NOW - timedelta(minutes=10 * (n_rows - i)),
                    vibration_rms=1.0 + (2.5 * frac if faulty else 0.0),
                    vibration_kurtosis=3.0 + (2.0 * frac if faulty else 0.0),
                    vibration_crest=4.0,
                    bearing_temp_c=60.0 + (20 * frac if faulty else 0.0),
                    motor_temp_c=55.0,
                    current_a=40.0,
                    rpm=1480.0,
                    load_pct=55.0,
                    regime="R2",
                    health=(1.0 - 0.8 * frac) if faulty else 1.0,
                    failure_within_h=(48.0 * (1 - frac) + 1.0) if faulty else None,
                )
            )
    db.insert_telemetry(rows, engine=engine)


@pytest.fixture()
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    configure_env(tmp_path, monkeypatch)
    eng = db.init_db()
    seed_small_fleet(eng)
    return eng


# --------------------------------------------------------------------------------------
# Schema factories
# --------------------------------------------------------------------------------------


def make_dq(asset_id: str = "MTR-042", trainable: bool = True) -> DataQualityContract:
    return DataQualityContract(
        asset_id=asset_id,
        window_start=NOW - timedelta(days=30),
        window_end=NOW,
        n_rows=4320,
        n_assets=4,
        sampling_rate_hz=1 / 600,
        columns=["asset_id", "ts", "vibration_rms", "rpm", "load_pct"],
        quality_score=92.0 if trainable else 40.0,
        missingness=MissingnessReport(missing_fraction={"vibration_rms": 0.0}),
        regimes=RegimeReport(
            regimes=[
                Regime(
                    regime_id="R1", label="idle", n_rows=1000, rpm_mean=600, load_mean=10, share=0.2
                ),
                Regime(
                    regime_id="R2",
                    label="nominal",
                    n_rows=3000,
                    rpm_mean=1480,
                    load_mean=55,
                    share=0.7,
                ),
                Regime(
                    regime_id="R3", label="high", n_rows=320, rpm_mean=1500, load_mean=85, share=0.1
                ),
            ],
            silhouette=0.6,
        ),
        trainable=trainable,
        notes=["synthetic"],
    )


def make_spec() -> FeatureSpec:
    return FeatureSpec(features=list(FEATURES), window_rows=12, description="test features")


def make_task() -> TaskSpec:
    return TaskSpec(task="classification", target="label", horizon_h=48.0, rationale="binary label")


def make_ims(total: float = 0.8) -> IndustrialModelScore:
    return IndustrialModelScore(
        recall=0.9,
        precision=0.7,
        lead_time_score=0.8,
        calibration_score=0.7,
        latency_score=0.9,
        total=total,
    )


def make_candidates(asset_id: str = "MTR-042", artifacts_dir: Path | None = None) -> CandidateSet:
    base = str(artifacts_dir or Path("/tmp/thor-test"))
    cands = [
        CandidateModel(
            candidate_id=f"cand_{fam}",
            family=fam,
            params={"n_estimators": 100},
            metrics={"recall": 0.9, "precision": 0.7, "brier": 0.1},
            ims=make_ims(0.8 - 0.05 * i),
            artifact_path=f"{base}/cand_{fam}.joblib",
            inference_latency_ms=2.0,
        )
        for i, fam in enumerate(("lightgbm", "xgboost", "random_forest"))
    ]
    return CandidateSet(
        asset_id=asset_id,
        task=make_task(),
        features=make_spec(),
        candidates=cands,
        mlflow_experiment="thor-test",
        ranked=[c.candidate_id for c in cands],
    )


def make_validation(asset_id: str = "MTR-042") -> ValidationReport:
    return ValidationReport(
        asset_id=asset_id,
        champion_id="cand_lightgbm",
        champion_family="lightgbm",
        champion_mlflow_run_id="run123",
        model_version="v42",
        leakage=LeakageReport(temporal_leakage=False, asset_overlap=False, passed=True),
        backtest=[
            BacktestFold(
                fold=0,
                train_assets=["MTR-001", "MTR-002"],
                test_assets=["MTR-003", "MTR-042"],
                train_end=NOW - timedelta(days=7),
                metrics={"recall": 0.9, "precision": 0.7, "brier": 0.1, "auroc": 0.95, "f1": 0.79},
            )
        ],
        calibration=CalibrationReport(method="isotonic", brier_before=0.15, brier_after=0.1),
        lead_time=LeadTimeReport(median_h=36.0, p90_h=60.0, p10_h=20.0, n_events=2, threshold=0.5),
        rul=RULInterval(p10_h=20.0, p50_h=40.0, p90_h=70.0),
        ims=make_ims(0.8),
        passed=True,
        notes=["champion_artifact=/tmp/thor-test/cand_lightgbm_calibrated.joblib"],
    )


def make_explanation(asset_id: str = "MTR-042") -> Explanation:
    return Explanation(
        asset_id=asset_id,
        failure_probability=0.78,
        top_features=[
            ShapFeature(
                feature="vib_rms_mean", shap_value=0.31, feature_value=3.2, direction="raises_risk"
            ),
            ShapFeature(
                feature="bearing_temp_mean",
                shap_value=0.18,
                feature_value=78.0,
                direction="raises_risk",
            ),
            ShapFeature(
                feature="load_mean", shap_value=-0.05, feature_value=55.0, direction="lowers_risk"
            ),
        ],
        base_value=0.12,
        model_version="v42",
    )


def make_cost(asset_id: str = "MTR-042") -> CostComparison:
    return CostComparison(
        asset_id=asset_id,
        options=[
            CostOption(
                option="maintain_now",
                when=NOW,
                expected_cost=13500.0,
                p_failure_before=0.0,
                downtime_h=3.0,
                breakdown={"planned": 4200.0, "downtime": 9300.0},
            ),
            CostOption(
                option="maintain_later",
                when=NOW + timedelta(hours=72),
                expected_cost=21000.0,
                p_failure_before=0.45,
                downtime_h=3.0,
                breakdown={"planned": 4200.0},
            ),
            CostOption(
                option="run_to_failure",
                when=None,
                expected_cost=48000.0,
                p_failure_before=0.9,
                downtime_h=14.0,
                breakdown={"unplanned": 18500.0},
            ),
        ],
        recommended="maintain_now",
        assumptions={"cost_planned_maintenance": 4200.0, "cost_unplanned_repair": 18500.0},
    )


def make_window() -> MaintenanceWindow:
    return MaintenanceWindow(
        start=NOW + timedelta(hours=14),
        end=NOW + timedelta(hours=17),
        reason="next low-load window before P(fail) exceeds 0.3",
        p_failure_before_window=0.21,
    )


def make_passages() -> list[ManualPassage]:
    return [
        ManualPassage(
            source="motor-maintenance.md",
            section="4.2",
            page=None,
            text="Drive-end bearing wear raises vibration RMS and kurtosis.",
            score=0.9,
        )
    ]


def make_bundle(asset_id: str = "MTR-042") -> EvidenceBundle:
    return EvidenceBundle(
        explanation=make_explanation(asset_id),
        manual_context=make_passages(),
        cost=make_cost(asset_id),
        window=make_window(),
        validation=make_validation(asset_id),
        data_quality_score=92.0,
    )


def make_contract(bundle: EvidenceBundle, run_id: str, text: str, source: str) -> DecisionContract:
    exp = bundle.explanation
    rec = next(o for o in bundle.cost.options if o.option == bundle.cost.recommended)
    return DecisionContract(
        contract_id=f"dc_{exp.asset_id}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}",
        created_at=datetime.now(UTC),
        asset_id=exp.asset_id,
        run_id=run_id,
        model_version=bundle.validation.model_version,
        champion_mlflow_run_id=bundle.validation.champion_mlflow_run_id,
        failure_probability=exp.failure_probability,
        recommendation=bundle.cost.recommended,
        window_start=bundle.window.start,
        window_end=bundle.window.end,
        expected_cost=rec.expected_cost,
        cost_comparison=bundle.cost,
        top_features=exp.top_features,
        manual_context=bundle.manual_context,
        lead_time_h=bundle.validation.lead_time.median_h,
        calibration_brier=bundle.validation.calibration.brier_after,
        data_quality_score=bundle.data_quality_score,
        explanation_text=text,
        explanation_source=source,  # type: ignore[arg-type]
        evidence_hash="0" * 64,
    )


def make_registered(
    version: str = "v42", stage: ModelStage = ModelStage.candidate
) -> RegisteredModel:
    return RegisteredModel(
        name="thor-bearing-classifier",
        version=version,
        mlflow_run_id="run123",
        stage=stage,
        family="lightgbm",
        ims_total=0.8,
        registered_at=datetime.now(UTC),
    )


def make_features_df() -> pd.DataFrame:
    rows = []
    for a in ASSETS:
        for i in range(10):
            rows.append(
                {
                    "asset_id": a,
                    "ts": NOW - timedelta(minutes=10 * (10 - i)),
                    "vib_rms_mean": 1.0 + (0.3 * i if a == "MTR-042" else 0.0),
                    "vib_rms_slope": 0.01,
                    "bearing_temp_mean": 62.0,
                    "load_mean": 55.0,
                    "label": int(a == "MTR-042" and i > 6),
                    "rul_h": float("nan"),
                }
            )
    return pd.DataFrame(rows)


class FakeModel:
    """Minimal `.predict_proba` object."""

    def predict_proba(self, X: pd.DataFrame) -> Any:
        import numpy as np

        p = np.clip(0.2 + 0.15 * X["vib_rms_mean"].to_numpy(), 0, 1)
        return np.column_stack([1 - p, p])


def install_fake_toolbox(monkeypatch: pytest.MonkeyPatch, orchestrator: Any) -> dict[str, Any]:
    """Monkeypatch every `orchestrator.tool_*` seam with fast fakes; returns a call log."""
    calls: dict[str, Any] = {"promote": [], "deploy_edge": [], "contracts": []}

    monkeypatch.setattr(
        orchestrator, "tool_profile_dataset", lambda df, asset_id: make_dq(asset_id)
    )
    monkeypatch.setattr(
        orchestrator,
        "tool_build_features",
        lambda df, regimes, horizon_h, window_rows=12: (make_features_df(), make_spec()),
    )
    monkeypatch.setattr(orchestrator, "tool_infer_task", lambda dq, df, horizon_h: make_task())
    monkeypatch.setattr(
        orchestrator,
        "tool_train_candidates",
        lambda fdf, spec, task, n_trials, seed, artifacts_dir, asset_id: make_candidates(
            asset_id, artifacts_dir
        ),
    )
    monkeypatch.setattr(
        orchestrator, "tool_validate", lambda cs, fdf, run_id: make_validation(cs.asset_id)
    )
    monkeypatch.setattr(orchestrator, "tool_load_model", lambda path: FakeModel())
    monkeypatch.setattr(
        orchestrator,
        "tool_explain",
        lambda model, x, spec, asset_id, version, background=None: make_explanation(asset_id),
    )
    monkeypatch.setattr(orchestrator, "tool_retrieve_manual_context", lambda exp: make_passages())
    monkeypatch.setattr(
        orchestrator,
        "tool_calculate_failure_cost",
        lambda p_fail_by, settings, horizon_h, now: make_cost(),
    )
    monkeypatch.setattr(
        orchestrator, "tool_find_maintenance_window", lambda cost, lt, now: make_window()
    )
    monkeypatch.setattr(
        orchestrator,
        "tool_build_evidence_bundle",
        lambda exp, mc, cost, window, val, dq: EvidenceBundle(
            explanation=exp,
            manual_context=mc,
            cost=cost,
            window=window,
            validation=val,
            data_quality_score=dq,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "tool_template_explanation",
        lambda bundle: (
            f"TEMPLATE: p={bundle.explanation.failure_probability:.2f}, recommend {bundle.cost.recommended}"
        ),
    )

    def fake_create_contract(
        bundle: EvidenceBundle, run_id: str, text: str, source: str, engine: Any
    ) -> DecisionContract:
        dc = make_contract(bundle, run_id, text, source)
        db.insert_decision_contract(dc, engine=engine)
        calls["contracts"].append(dc.contract_id)
        return dc

    monkeypatch.setattr(orchestrator, "tool_create_decision_contract", fake_create_contract)
    monkeypatch.setattr(
        orchestrator,
        "tool_register_model",
        lambda v, path, engine: make_registered(v.model_version),
    )
    monkeypatch.setattr(
        orchestrator,
        "tool_compare_champion",
        lambda rm, engine: ChampionComparison(
            challenger=rm,
            champion=None,
            delta={"ims_total": 0.8},
            recommend_promote=True,
            rationale="no champion",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "tool_request_promotion",
        lambda comp, to_stage, engine: PromotionRequest(
            promotion_id="promo_test_1",
            model_name=comp.challenger.name,
            version=comp.challenger.version,
            from_stage=comp.challenger.stage,
            to_stage=to_stage,
            comparison=comp,
            created_at=datetime.now(UTC),
        ),
    )

    def fake_promote(promotion_id: str, approval: Approval, engine: Any) -> RegisteredModel:
        assert approval.decision == "approved"
        assert approval.promotion_id == promotion_id
        calls["promote"].append(promotion_id)
        return make_registered(stage=ModelStage.production)

    def fake_deploy(
        model: RegisteredModel,
        artifact_path: Path,
        features: list[str],
        models_dir: Path,
        window_rows: int,
        baseline: dict[str, Any] | None = None,
    ) -> EdgeDeployment:
        calls["deploy_edge"].append(str(artifact_path))
        return EdgeDeployment(
            model_name=model.name,
            version=model.version,
            onnx_path=str(models_dir / "m.onnx"),
            deployed_at=datetime.now(UTC),
            input_features=features,
        )

    monkeypatch.setattr(orchestrator, "tool_promote", fake_promote)
    monkeypatch.setattr(orchestrator, "tool_deploy_edge", fake_deploy)
    monkeypatch.setattr(
        orchestrator,
        "tool_run_whatif",
        lambda model, x, spec, asset_id, scenario: WhatIfResult(
            asset_id=asset_id,
            scenario=scenario,
            baseline_probability=0.78,
            scenario_probability=0.5,
            delta=-0.28,
        ),
    )
    return calls


def wait_for_stage(
    orchestrator: Any, run_id: str, stages: set[str], engine: Any, timeout: float = 15.0
) -> GraphState:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        gs = orchestrator.get_run(run_id, engine)
        if gs is not None and gs.stage.value in stages:
            return gs
        time.sleep(0.05)
    gs = orchestrator.get_run(run_id, engine)
    raise AssertionError(
        f"run {run_id} did not reach {stages}; last = {gs.stage if gs else None} / {gs.error if gs else None}"
    )


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def test_only_orchestrator_reads_api_key() -> None:
    """CLAUDE.md section 7: `anthropic_api_key` may only be read in orchestrator.py."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for p in list((root / "apps").rglob("*.py")) + list((root / "agents").rglob("*.py")):
        if p.name in ("orchestrator.py", "settings.py"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        reads_attr = re.search(r"\.anthropic_api_key\b", text)
        reads_env = re.search(r"(environ|getenv)\s*[\[(]\s*['\"]ANTHROPIC_API_KEY", text)
        if reads_attr or reads_env:
            offenders.append(str(p))
    assert offenders == []


def test_route_is_deterministic_without_key(engine: Any) -> None:
    from apps.api import orchestrator

    gs = GraphState(run_id="r1", asset_id="MTR-042")
    order = []
    for stage in (
        PipelineStage.queued,
        PipelineStage.profiling,
        PipelineStage.training,
        PipelineStage.validating,
        PipelineStage.explaining,
        PipelineStage.awaiting_approval,
    ):
        gs.stage = stage
        order.append(orchestrator.route(gs))
    assert order == ["profile", "train", "validate", "explain", "await_approval", "finalize"]
    gs.stage = PipelineStage.approved
    assert orchestrator.route(gs) == orchestrator.END
    assert gs.llm_calls == 0


def test_route_ignores_llm_choice_that_skips_validate(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.api import orchestrator

    monkeypatch.setattr(orchestrator, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(orchestrator, "_llm_choose_next_node", lambda state, allowed: "finalize")
    gs = GraphState(run_id="r2", asset_id="MTR-042", stage=PipelineStage.training)
    assert orchestrator.route(gs) == "validate"
    assert gs.llm_calls == 1
    ev = [e for e in gs.events if e.tool == "route_llm"]
    assert len(ev) == 1 and ev[0].payload["accepted"] is False
    # only ONE planning call per run: the second routing decision is deterministic
    gs.stage = PipelineStage.validating
    assert orchestrator.route(gs) == "explain"
    assert gs.llm_calls == 1


def test_route_survives_llm_exception(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    def boom(state: GraphState, allowed: list[str]) -> str:
        raise RuntimeError("network down")

    monkeypatch.setattr(orchestrator, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(orchestrator, "_llm_choose_next_node", boom)
    gs = GraphState(run_id="r3", asset_id="MTR-042", stage=PipelineStage.profiling)
    assert orchestrator.route(gs) == "train"
    assert gs.events[-1].payload["error"].startswith("RuntimeError")


def test_draft_explanation_template_without_key(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.api import orchestrator

    monkeypatch.setattr(orchestrator, "tool_template_explanation", lambda b: "template text")
    text, source = orchestrator.draft_explanation(make_bundle())
    assert (text, source) == ("template text", "template")


def test_draft_explanation_falls_back_on_llm_failure(
    engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.api import orchestrator

    monkeypatch.setattr(orchestrator, "_api_key", lambda: "sk-test")
    monkeypatch.setattr(orchestrator, "tool_template_explanation", lambda b: "template text")

    def boom(bundle: EvidenceBundle) -> str:
        raise RuntimeError("429")

    monkeypatch.setattr(orchestrator, "_llm_draft", boom)
    assert orchestrator.draft_explanation(make_bundle()) == ("template text", "template")
    monkeypatch.setattr(orchestrator, "_llm_draft", lambda b: "   ")
    assert orchestrator.draft_explanation(make_bundle()) == ("template text", "template")
    monkeypatch.setattr(orchestrator, "_llm_draft", lambda b: "LLM prose")
    assert orchestrator.draft_explanation(make_bundle()) == ("LLM prose", "llm")


def test_full_run_pauses_at_gate_and_resumes(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    calls = install_fake_toolbox(monkeypatch, orchestrator)
    gs0 = orchestrator.start_run("MTR-042", 48.0, 2, engine=engine)
    assert gs0.stage == PipelineStage.queued
    gs = wait_for_stage(orchestrator, gs0.run_id, {"awaiting_approval", "failed"}, engine)
    assert gs.stage == PipelineStage.awaiting_approval, gs.error
    assert gs.data_quality and gs.candidates and gs.validation and gs.evidence and gs.contract
    assert gs.contract.explanation_source == "template"
    assert gs.llm_calls == 0
    assert db.contract_status(gs.contract.contract_id, engine=engine) == "pending"
    tools = [e.tool for e in gs.events]
    for t in ("profile", "train", "validate", "explain", "await_approval", "request_promotion"):
        assert t in tools

    approval = Approval(
        approval_id="a1",
        contract_id=gs.contract.contract_id,
        decision="approved",
        approver="jane",
        note="ok",
        decided_at=datetime.now(UTC),
    )
    out = orchestrator.resume(gs0.run_id, approval, engine=engine)
    assert out.stage == PipelineStage.approved
    assert calls["promote"] == ["promo_test_1"]
    assert len(calls["deploy_edge"]) == 1
    persisted = orchestrator.get_run(gs0.run_id, engine)
    assert persisted is not None and persisted.stage == PipelineStage.approved
    assert persisted.approval is not None and persisted.approval.approver == "jane"
    with pytest.raises(ValueError):
        orchestrator.resume(gs0.run_id, approval, engine=engine)
    with pytest.raises(KeyError):
        orchestrator.resume("unknown-run", approval, engine=engine)


def test_finalize_refuses_without_approval(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate: finalize can never run without a recorded human decision."""
    from apps.api import orchestrator
    from apps.api.graph_state import state_to_json

    install_fake_toolbox(monkeypatch, orchestrator)
    gs = GraphState(run_id="gate1", asset_id="MTR-042", stage=PipelineStage.awaiting_approval)
    db.save_run(gs.run_id, gs.asset_id, gs.stage.value, state_to_json(gs), engine=engine)
    out = orchestrator._finalize_node({"state": state_to_json(gs), "next": "finalize"})
    assert out["state"]["stage"] == "failed"
    assert "requires a recorded human decision" in out["state"]["error"]


def test_rejected_run_does_not_promote(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    calls = install_fake_toolbox(monkeypatch, orchestrator)
    gs0 = orchestrator.start_run("MTR-042", engine=engine)
    gs = wait_for_stage(orchestrator, gs0.run_id, {"awaiting_approval", "failed"}, engine)
    assert gs.stage == PipelineStage.awaiting_approval, gs.error
    approval = Approval(
        approval_id="a2",
        contract_id=gs.contract.contract_id,
        decision="rejected",
        approver="jane",
        decided_at=datetime.now(UTC),
    )
    out = orchestrator.resume(gs0.run_id, approval, engine=engine)
    assert out.stage == PipelineStage.rejected
    assert calls["promote"] == [] and calls["deploy_edge"] == []


def test_node_failure_marks_run_failed(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    install_fake_toolbox(monkeypatch, orchestrator)
    monkeypatch.setattr(
        orchestrator,
        "tool_profile_dataset",
        lambda df, asset_id: make_dq(asset_id, trainable=False),
    )
    gs0 = orchestrator.start_run("MTR-042", engine=engine)
    gs = wait_for_stage(orchestrator, gs0.run_id, {"awaiting_approval", "failed"}, engine)
    assert gs.stage == PipelineStage.failed
    assert gs.error and "blocks training" in gs.error
    assert gs.candidates is None


def test_missing_toolbox_fails_run_gracefully(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the agent modules the run must end as `failed`, never hang or crash the app."""
    from apps.api import orchestrator

    monkeypatch.setattr(orchestrator, "tool_profile_dataset", lambda df, a: make_dq(a))

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("No module named 'agents.ml_architect.features'")

    monkeypatch.setattr(orchestrator, "tool_build_features", missing)
    gs0 = orchestrator.start_run("MTR-042", engine=engine)
    gs = wait_for_stage(orchestrator, gs0.run_id, {"awaiting_approval", "failed"}, engine)
    assert gs.stage == PipelineStage.failed and "train:" in (gs.error or "")


def test_copilot_template_mode(engine: Any) -> None:
    from apps.api import orchestrator

    def ask(text: str, asset_id: str | None = None) -> Any:
        return orchestrator.copilot_reply(
            CopilotRequest(messages=[CopilotMessage(role="user", content=text)], asset_id=asset_id),
            engine=engine,
        )

    r = ask("Which motors in the fleet are at risk?")
    assert r.source == "template"
    assert r.tool_calls and r.tool_calls[0]["name"] == "get_fleet"
    assert "MTR-042" in r.reply and "4 assets" in r.reply
    r = ask("How is MTR-042 doing?")
    assert r.tool_calls[0]["name"] == "get_asset" and r.tool_calls[0]["args"] == {
        "asset_id": "MTR-042"
    }
    assert r.reply.startswith("MTR-042")
    r = ask("What is pending approval?")
    assert r.tool_calls[0]["name"] == "list_pending_approvals"
    assert "Nothing is waiting" in r.reply
    r = ask("hello there")
    assert r.tool_calls == [] and "Bolt" in r.reply
    assert os.environ.get("ANTHROPIC_API_KEY", "") == ""


def test_graph_state_helpers() -> None:
    from apps.api import graph_state

    rid = graph_state.new_run_id()
    assert len(rid) == 16
    gs = GraphState(run_id=rid, asset_id="MTR-001")
    ev = graph_state.append_event(gs, "profiling", "hi", tool="profile", payload={"a": 1})
    assert ev.stage == PipelineStage.profiling and gs.events == [ev]
    rt = graph_state.state_from_json(graph_state.state_to_json(gs))
    assert rt == gs
    assert graph_state.run_summary(gs)["n_events"] == 1
