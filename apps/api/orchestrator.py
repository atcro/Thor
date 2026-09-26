"""Thor orchestrator (03) -- the only file in the repo that talks to the LLM.

Governed architecture: the LLM plans (one routing call) and explains (one drafting call);
every number comes from the deterministic toolboxes under `agents/`; a human approves every
consequential action through the Human Approval Gate (09).

Graph: profile -> train -> validate -> explain -> await_approval -(interrupt)-> finalize.
`GraphState` is persisted to `pipeline_runs` after every node so the UI can poll progress.

All toolbox calls go through the module-level `tool_*` wrappers below. They import the agent
modules lazily (the modules may not exist yet) and are the intended monkeypatch points for tests
and for the integrator.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.graph_state import (
    append_event,
    new_run_id,
    run_summary,
    state_from_json,
    state_to_json,
)
from apps.api.schemas import (
    Approval,
    CandidateSet,
    ChampionComparison,
    CopilotRequest,
    CopilotResponse,
    CostComparison,
    DataQualityContract,
    DecisionContract,
    EdgeDeployment,
    EvidenceBundle,
    Explanation,
    FeatureSpec,
    GraphState,
    LeadTimeReport,
    MaintenanceWindow,
    ManualPassage,
    ModelStage,
    PipelineStage,
    PromotionRequest,
    RegimeReport,
    RegisteredModel,
    TaskSpec,
    ValidationReport,
    WhatIfResult,
)
from apps.api.settings import Settings, get_settings

log = logging.getLogger("thor.orchestrator")

# --------------------------------------------------------------------------------------
# Node names, deterministic transitions
# --------------------------------------------------------------------------------------

NODE_NAMES: tuple[str, ...] = (
    "profile",
    "train",
    "validate",
    "explain",
    "await_approval",
    "finalize",
)

# stage after a node finished -> the only allowed next node
STAGE_TO_NEXT: dict[PipelineStage, str] = {
    PipelineStage.queued: "profile",
    PipelineStage.profiling: "train",
    PipelineStage.training: "validate",
    PipelineStage.validating: "explain",
    PipelineStage.explaining: "await_approval",
    PipelineStage.awaiting_approval: "finalize",
    PipelineStage.approved: END,
    PipelineStage.rejected: END,
    PipelineStage.failed: END,
}

TERMINAL_STAGES: frozenset[PipelineStage] = frozenset(
    {
        PipelineStage.awaiting_approval,
        PipelineStage.approved,
        PipelineStage.rejected,
        PipelineStage.failed,
    }
)

ROUTE_SYSTEM_PROMPT = (
    "You are the planner of Thor, a governed predictive-maintenance pipeline. "
    "Given the current pipeline stage, call the tool `choose_next_node` exactly once with the "
    "next node to run. The only legal order is profile -> train -> validate -> explain -> "
    "await_approval -> finalize; validate and await_approval can never be skipped."
)

EXPLANATION_SYSTEM_PROMPT = (
    "You are drafting a maintenance explanation for a plant engineer. Use ONLY the numbers, "
    "feature names, cost figures, window, and manual citations present in the JSON. Do not "
    "introduce any number, percentage, date, or citation not present. 4-6 sentences. Refer to "
    "manual passages as '<source> section <section>'."
)

COPILOT_SYSTEM_PROMPT = (
    "You are Bolt, Thor's copilot for plant engineers. Thor is a predictive-maintenance "
    "companion to the plant's MES/CMMS. You may only report numbers, ids and facts returned by "
    "your tools -- never invent or estimate a value. You cannot approve or reject anything: when "
    "a decision contract or promotion is pending, tell the user to approve or reject it in the "
    "Thor UI. For questions about failure mechanisms, inspection or maintenance procedures, "
    "thresholds, or plant policy, call search_manuals and quote only the passages it returns, "
    "citing each as '<source> section <section>'. Never cite a manual passage that was not "
    "returned by search_manuals. For 'has this been seen before / what did it take to fix' "
    "questions, call search_field_history: its cases are real unplanned work orders from a "
    "public dataset of university facilities (FMUCD), not from this plant and not manual "
    "guidance -- present them as precedent ('field history: <component>, <year>') and use "
    "only the labor hours, costs and medians it returns. For 'how long / how much does this "
    "usually take' questions call field_history_stats (quartiles over up to 100 similar cases) "
    "rather than reading numbers off a few cases. When asked why an asset is at risk "
    "or about a decision contract, chain the tools: get_asset or get_contract first, then "
    "search_field_history with the contract's risk-raising drivers as the symptom, and answer "
    "in three parts -- the contract's evidence (probability, top drivers), its manual "
    "citations (already in the contract; do not invent others), and the precedent.\n\n"
    "Voice: you sound like a calm senior reliability engineer writing a shift-handover note, "
    "not an assistant. Lead with the actionable conclusion, then the evidence, then the "
    "precedent. State confidence honestly and in the contract's terms ('the contract puts it "
    "at 0.50 within 48 h', never 'it will fail'). Say plainly what is unknown or what returned "
    "nothing instead of filling the gap. Use the plant's vocabulary (drive-end bearing, regime, "
    "alarm window, lead time), not generic phrases like 'anomaly detected'; name SHAP drivers by "
    "their plain names from top_features_plain, not by feature codes. Mention that "
    "approval happens in the Thor UI only when the user asks you to act, not on every turn. No "
    "exclamation marks, no apologies, no 'great question', no first-person feelings. Short "
    "sentences; numbers on their own line or in a short list when there are more than two."
)

CHOOSE_NEXT_NODE_TOOL: dict[str, Any] = {
    "name": "choose_next_node",
    "description": "Pick the next pipeline node to execute.",
    "input_schema": {
        "type": "object",
        "properties": {"node": {"type": "string", "enum": list(NODE_NAMES)}},
        "required": ["node"],
        "additionalProperties": False,
    },
    "strict": True,
}


# --------------------------------------------------------------------------------------
# LLM access (the only place ANTHROPIC_API_KEY is read)
# --------------------------------------------------------------------------------------


def _openai_key() -> str:
    """Return the configured OpenAI API key ('' when absent). Read nowhere else."""
    return (get_settings().openai_api_key or "").strip()


def _provider() -> Literal["anthropic", "openai", "none"]:
    """Resolve the active LLM provider from settings.llm_provider and which keys are set."""
    s = get_settings()
    anthropic_key = (s.anthropic_api_key or "").strip()
    openai_key = _openai_key()
    if s.llm_provider == "anthropic":
        return "anthropic" if anthropic_key else "none"
    if s.llm_provider == "openai":
        return "openai" if openai_key else "none"
    if openai_key:
        return "openai"
    return "anthropic" if anthropic_key else "none"


def _api_key() -> str:
    """Return the Anthropic key when Anthropic is the active provider, else ''.

    Every LLM path in this module (route, draft, copilot, ping) currently speaks the Anthropic
    Messages API, so this is the single switch that turns those paths on. An OpenAI key alone
    leaves them in template mode until an OpenAI client is implemented here.
    """
    if _provider() != "anthropic":
        return ""
    return (get_settings().anthropic_api_key or "").strip()


def _llm_enabled() -> bool:
    """True when a usable provider key is configured (anthropic or openai).

    Checks `_api_key()` first so a test (or operator) that patches it still enables the
    Anthropic path; otherwise falls back to provider resolution for OpenAI.
    """
    return bool(_api_key()) or _provider() == "openai"


def _llm_client() -> Any:
    """Build an Anthropic client. Raises if the SDK is missing or the key is blank."""
    key = _api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is blank")
    import anthropic

    return anthropic.Anthropic(api_key=key, max_retries=1, timeout=60.0)


def _openai_client() -> Any:
    """Build an OpenAI client. Raises if the SDK is missing or the key is blank."""
    key = _openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is blank")
    import openai

    # max_retries=3: the SDK honours retry-after on 429s, which low-tier accounts hit on a
    # chained Bolt answer (3-4 requests, ~10k tokens) before the per-minute window rolls.
    return openai.OpenAI(api_key=key, max_retries=3, timeout=90.0)


def _openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate this module's tool specs (Anthropic shape) to OpenAI function-calling shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _openai_tool_calls(message: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """(call_id, name, parsed args) for every tool call on an OpenAI chat message."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        try:
            args = json.loads(fn.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        out.append((tc.id, fn.name, args if isinstance(args, dict) else {}))
    return out


def _text_of(message: Any) -> str:
    """Concatenate the text blocks of a Messages API response."""
    parts = [b.text for b in getattr(message, "content", []) if getattr(b, "type", "") == "text"]
    return "\n".join(p.strip() for p in parts if p and p.strip()).strip()


def llm_status() -> dict[str, Any]:
    """Report whether the LLM paths are active, without exposing the key.

    Returns {"mode": "llm" | "template", "provider": "anthropic" | "openai" | "none",
    "model": <configured model id>, "key_configured": bool, "warning": str | None}. `warning`
    flags a key with an unexpected prefix (paste error) or a provider whose client is not
    implemented yet. Safe to log and to return from /system/health.
    """
    s = get_settings()
    provider = _provider()
    warning: str | None = None
    if provider == "openai":
        key = _openai_key()
        model = s.openai_model
        if not key.startswith("sk-"):
            warning = "OPENAI_API_KEY is set but does not start with 'sk-'; check the value"
        mode = "llm"
    elif provider == "anthropic":
        key = _api_key()
        model = s.anthropic_model
        if not key.startswith("sk-ant-"):
            warning = "ANTHROPIC_API_KEY is set but does not start with 'sk-ant-'; check the value"
        mode = "llm"
    else:
        key, model, mode = "", s.anthropic_model, "template"
    return {
        "mode": mode,
        "provider": provider,
        "model": model,
        "key_configured": bool(key),
        "warning": warning,
    }


def llm_ping() -> dict[str, Any]:
    """One minimal Messages API call to prove the configured key and model work.

    Never called by the pipeline -- this is an operator smoke test:
    `python -c "from apps.api.orchestrator import llm_ping; print(llm_ping())"`.
    Returns {"ok": True, "model", "reply", "request_id"} or {"ok": False, "error"}.
    """
    status = llm_status()
    if not status["key_configured"]:
        return {
            "ok": False,
            "error": "ANTHROPIC_API_KEY and OPENAI_API_KEY are both blank (template mode)",
        }
    prompt = "Reply with the single word: ready"
    try:
        if status["provider"] == "openai":
            resp = _openai_client().chat.completions.create(
                model=status["model"],
                max_completion_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = (resp.choices[0].message.content or "").strip()
            request_id = getattr(resp, "_request_id", None) or getattr(resp, "id", None)
        else:
            resp = _llm_client().messages.create(
                model=status["model"],
                max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = _text_of(resp)
            request_id = getattr(resp, "_request_id", None)
    except Exception as e:  # noqa: BLE001 - surface every failure class to the operator
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": True,
        "provider": status["provider"],
        "model": status["model"],
        "reply": reply,
        "request_id": request_id,
    }


# --------------------------------------------------------------------------------------
# Toolbox wrappers -- lazy imports; monkeypatch these in tests
# --------------------------------------------------------------------------------------


def tool_profile_dataset(df: pd.DataFrame, asset_id: str) -> DataQualityContract:
    """04: `agents.data_agent.profiling.profile_dataset(df, asset_id)`."""
    from agents.data_agent.profiling import profile_dataset

    return profile_dataset(df, asset_id)


def tool_build_features(
    df: pd.DataFrame, regimes: RegimeReport | None, horizon_h: float, window_rows: int = 12
) -> tuple[pd.DataFrame, FeatureSpec]:
    """05: `agents.ml_architect.features.build_feature_pipeline(df, regimes, horizon_h, window_rows)`."""
    from agents.ml_architect.features import build_feature_pipeline

    return build_feature_pipeline(df, regimes, horizon_h=horizon_h, window_rows=window_rows)


def tool_infer_task(dq: DataQualityContract, df: pd.DataFrame, horizon_h: float) -> TaskSpec:
    """05: `agents.ml_architect.automl.infer_task(dq, df, horizon_h)`."""
    from agents.ml_architect.automl import infer_task

    return infer_task(dq, df, horizon_h=horizon_h)


def tool_train_candidates(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    task: TaskSpec,
    n_trials: int,
    seed: int,
    artifacts_dir: Path,
    asset_id: str,
) -> CandidateSet:
    """05: `agents.ml_architect.automl.train_candidates(...)` -> CandidateSet."""
    from agents.ml_architect.automl import train_candidates

    return train_candidates(
        features_df,
        spec,
        task,
        n_trials=n_trials,
        seed=seed,
        artifacts_dir=artifacts_dir,
        asset_id=asset_id,
    )


def tool_validate(
    candidates: CandidateSet, features_df: pd.DataFrame, run_id: str
) -> ValidationReport:
    """06: `agents.validation.checks.validate(candidates, features_df, run_id)`."""
    from agents.validation.checks import validate

    return validate(candidates, features_df, run_id)


def tool_load_model(path: str) -> Any:
    """Load a fitted model artifact (joblib pickle exposing `.predict_proba`)."""
    import joblib

    return joblib.load(path)


def tool_explain(
    model: Any,
    x_latest: pd.DataFrame,
    spec: FeatureSpec,
    asset_id: str,
    model_version: str,
    background: pd.DataFrame | None = None,
) -> Explanation:
    """07: `agents.reliability.explain.explain(model, X_latest, spec, asset_id, model_version,
    background=...)`. `background` = healthy fleet rows + the asset's recent rows so SHAP has
    real variation to attribute against."""
    from agents.reliability.explain import explain

    return explain(model, x_latest, spec, asset_id, model_version, background=background)


def _explain_background(
    features_df: pd.DataFrame, spec: FeatureSpec, asset_id: str, n_healthy: int = 30
) -> pd.DataFrame:
    """Stratified SHAP background: up to `n_healthy` label-0 rows from OTHER assets (seeded
    sample) plus the target asset's last 20 rows. Feature columns only."""
    cols = list(spec.features)
    parts: list[pd.DataFrame] = []
    if "asset_id" in features_df.columns and "label" in features_df.columns:
        others = features_df[(features_df["asset_id"] != asset_id) & (features_df["label"] == 0)]
        if len(others):
            parts.append(others.sample(n=min(n_healthy, len(others)), random_state=0)[cols])
        own = features_df[features_df["asset_id"] == asset_id]
        if "ts" in own.columns:
            own = own.sort_values("ts")
        parts.append(own.tail(20)[cols])
    if not parts:
        return features_df[cols].tail(50)
    return pd.concat(parts, ignore_index=True).astype(float)


def tool_run_whatif(
    model: Any,
    x_latest: pd.DataFrame,
    spec: FeatureSpec,
    asset_id: str,
    scenario: dict[str, float],
) -> WhatIfResult:
    """07: `agents.reliability.explain.run_whatif(model, X_latest, spec, asset_id, scenario)`."""
    from agents.reliability.explain import run_whatif

    return run_whatif(model, x_latest, spec, asset_id, scenario)


def tool_retrieve_manual_context(exp: Explanation) -> list[ManualPassage]:
    """07: `rag.retrieve_manual_context(rag.query_from_explanation(exp), chroma_path)`."""
    from agents.reliability import rag

    settings = get_settings()
    query = rag.query_from_explanation(exp)
    return rag.retrieve_manual_context(query, Path(settings.chroma_path))


def tool_calculate_failure_cost(
    p_fail_by: Callable[[float], float] | float,
    settings: Settings,
    horizon_h: float,
    now: datetime,
) -> CostComparison:
    """07: `agents.reliability.cost.calculate_failure_cost(p_fail_by, settings, horizon_h, now)`."""
    from agents.reliability.cost import calculate_failure_cost

    return calculate_failure_cost(p_fail_by, settings, horizon_h=horizon_h, now=now)


def tool_find_maintenance_window(
    cost: CostComparison, lead_time: LeadTimeReport, now: datetime
) -> MaintenanceWindow:
    """07: `agents.reliability.cost.find_maintenance_window(cost, lead_time, now)`."""
    from agents.reliability.cost import find_maintenance_window

    return find_maintenance_window(cost, lead_time, now=now)


def tool_build_evidence_bundle(
    explanation: Explanation,
    manual_context: list[ManualPassage],
    cost: CostComparison,
    window: MaintenanceWindow,
    validation: ValidationReport,
    data_quality_score: float,
) -> EvidenceBundle:
    """07: `agents.reliability.contract.build_evidence_bundle(...)`."""
    from agents.reliability.contract import build_evidence_bundle

    return build_evidence_bundle(
        explanation, manual_context, cost, window, validation, data_quality_score
    )


def tool_template_explanation(bundle: EvidenceBundle) -> str:
    """07: `agents.reliability.contract.template_explanation(bundle)` -- deterministic prose."""
    from agents.reliability.contract import template_explanation

    return template_explanation(bundle)


def tool_create_decision_contract(
    bundle: EvidenceBundle,
    run_id: str,
    explanation_text: str,
    explanation_source: Literal["llm", "template"],
    engine: Engine | None,
) -> DecisionContract:
    """07: `agents.reliability.contract.create_decision_contract(...)` (inserts the row)."""
    from agents.reliability.contract import create_decision_contract

    return create_decision_contract(
        bundle, run_id, explanation_text, explanation_source, engine=engine
    )


def tool_register_model(
    validation: ValidationReport, artifact_path: Path, engine: Engine | None
) -> RegisteredModel:
    """08: `agents.mlops.lifecycle.register_model(validation, artifact_path, engine=engine)`."""
    from agents.mlops.lifecycle import register_model

    return register_model(validation, artifact_path, engine=engine)


def tool_compare_champion(challenger: RegisteredModel, engine: Engine | None) -> ChampionComparison:
    """08: `agents.mlops.lifecycle.compare_champion(challenger, engine=engine)`."""
    from agents.mlops.lifecycle import compare_champion

    return compare_champion(challenger, engine=engine)


def tool_request_promotion(
    comparison: ChampionComparison, to_stage: ModelStage, engine: Engine | None
) -> PromotionRequest:
    """08: `agents.mlops.lifecycle.request_promotion(comparison, to_stage, engine=engine)`."""
    from agents.mlops.lifecycle import request_promotion

    return request_promotion(comparison, to_stage, engine=engine)


def tool_promote(promotion_id: str, approval: Approval, engine: Engine | None) -> RegisteredModel:
    """08: `agents.mlops.lifecycle.promote(promotion_id, approval, engine=engine)`."""
    from agents.mlops.lifecycle import promote

    return promote(promotion_id, approval, engine=engine)


def tool_deploy_edge(
    model: RegisteredModel,
    artifact_path: Path,
    features: list[str],
    models_dir: Path,
    window_rows: int,
    baseline: dict[str, Any] | None = None,
) -> EdgeDeployment:
    """08: `agents.mlops.lifecycle.deploy_edge(model, artifact_path, features, models_dir, window_rows, baseline)`."""
    from agents.mlops.lifecycle import deploy_edge

    return deploy_edge(
        model, artifact_path, features, models_dir, window_rows=window_rows, baseline=baseline
    )


# --------------------------------------------------------------------------------------
# Run registry (in-process): run_id -> thread, status, heavy-object cache
# --------------------------------------------------------------------------------------


class RunHandle:
    """Per-run bookkeeping that does not belong in the persisted GraphState."""

    def __init__(
        self,
        run_id: str,
        asset_id: str,
        horizon_h: float = 48.0,
        n_trials: int = 6,
        engine: Engine | None = None,
    ) -> None:
        self.run_id = run_id
        self.asset_id = asset_id
        self.horizon_h = horizon_h
        self.n_trials = n_trials
        self.engine = engine
        self.thread: threading.Thread | None = None
        self.status: str = "queued"
        self.cache: dict[str, Any] = {}


_RUNS: dict[str, RunHandle] = {}
_RUNS_LOCK = threading.Lock()
_GRAPH: Any = None
_GRAPH_LOCK = threading.Lock()


def get_run_handle(run_id: str) -> RunHandle | None:
    """Return the in-process handle for a run (None if this process never saw it)."""
    with _RUNS_LOCK:
        return _RUNS.get(run_id)


def _handle_for(run_id: str, asset_id: str = "", engine: Engine | None = None) -> RunHandle:
    with _RUNS_LOCK:
        h = _RUNS.get(run_id)
        if h is None:
            h = RunHandle(run_id, asset_id, engine=engine)
            _RUNS[run_id] = h
        if engine is not None and h.engine is None:
            h.engine = engine
        return h


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _persist(gs: GraphState, engine: Engine | None) -> None:
    db.save_run(gs.run_id, gs.asset_id, gs.stage.value, state_to_json(gs), engine=engine)


# --------------------------------------------------------------------------------------
# route() -- LLM planning call #1
# --------------------------------------------------------------------------------------


def _llm_choose_next_node(state: GraphState, allowed: list[str]) -> str | None:
    """Ask the LLM (one tool-choice call) which node to run next. Returns the raw choice."""
    summary = {
        "run_id": state.run_id,
        "asset_id": state.asset_id,
        "current_stage": state.stage.value,
        "completed": [e.tool for e in state.events if e.tool in NODE_NAMES],
        "allowed_next": allowed,
        "has_data_quality": state.data_quality is not None,
        "has_candidates": state.candidates is not None,
        "has_validation": state.validation is not None,
        "has_contract": state.contract is not None,
    }
    user_msg = "Pipeline state:\n" + json.dumps(summary) + "\nCall choose_next_node."
    if _provider() == "openai":
        resp = _openai_client().chat.completions.create(
            model=get_settings().openai_model,
            max_completion_tokens=256,
            messages=[
                {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=_openai_tools([CHOOSE_NEXT_NODE_TOOL]),
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        for _cid, name, args in _openai_tool_calls(resp.choices[0].message):
            if name == "choose_next_node":
                return str(args.get("node", "")) or None
        return None
    resp = _llm_client().messages.create(
        model=get_settings().anthropic_model,
        max_tokens=256,
        system=ROUTE_SYSTEM_PROMPT,
        tools=[CHOOSE_NEXT_NODE_TOOL],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in resp.content:
        if getattr(block, "type", "") == "tool_use" and block.name == "choose_next_node":
            raw = block.input if isinstance(block.input, dict) else json.loads(str(block.input))
            return str(raw.get("node", "")) or None
    return None


def _llm_route_done(state: GraphState) -> bool:
    return any(e.tool == "route_llm" for e in state.events)


def route(state: GraphState) -> str:
    """Return the next node name (or END) for `state`.

    Deterministic transitions: profile -> train -> validate -> explain -> await_approval ->
    finalize -> END. With an API key configured, ONE Claude tool-choice call per run is made at
    the first routing decision; its choice is accepted only if it equals the single legal
    transition (it can never skip validate or await_approval), otherwise it is ignored. The
    call is recorded as a `route_llm` event and counted in `state.llm_calls` (mutated in place).
    """
    nxt = STAGE_TO_NEXT.get(state.stage, END)
    if nxt == END or not _llm_enabled() or _llm_route_done(state):
        return nxt
    choice: str | None = None
    error: str | None = None
    try:
        choice = _llm_choose_next_node(state, [nxt])
    except Exception as e:  # never let an LLM failure fail the run
        error = f"{type(e).__name__}: {e}"
        log.warning("route(): LLM call failed, using deterministic order: %s", error)
    state.llm_calls += 1
    accepted = choice == nxt
    append_event(
        state,
        state.stage,
        f"planner chose '{choice}' -> {'accepted' if accepted else 'ignored'}; next = {nxt}",
        tool="route_llm",
        payload={"llm_choice": choice, "next": nxt, "accepted": accepted, "error": error},
    )
    return nxt


# --------------------------------------------------------------------------------------
# draft_explanation() -- LLM call #2
# --------------------------------------------------------------------------------------


def _llm_draft(bundle: EvidenceBundle) -> str:
    """One LLM call with the evidence bundle JSON; returns the drafted text ('' on refusal)."""
    payload = bundle.model_dump_json(indent=2)
    if _provider() == "openai":
        resp = _openai_client().chat.completions.create(
            model=get_settings().openai_model,
            max_completion_tokens=1024,
            messages=[
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            return ""
        return (msg.content or "").strip()
    resp = _llm_client().messages.create(
        model=get_settings().anthropic_model,
        max_tokens=1024,
        system=EXPLANATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": payload}],
    )
    if getattr(resp, "stop_reason", "") == "refusal":
        return ""
    return _text_of(resp)


def draft_explanation(bundle: EvidenceBundle) -> tuple[str, Literal["llm", "template"]]:
    """Draft the engineer-facing explanation for an `EvidenceBundle`.

    Input: the bundle produced by 07. Output: (text, source). Uses ONE LLM call when a provider key
    is configured; on any exception, a blank key, or a blank/refused reply it falls back to
    `agents.reliability.contract.template_explanation(bundle)` with source "template".
    """
    if _llm_enabled():
        try:
            text = (_llm_draft(bundle) or "").strip()
            if text:
                return text, "llm"
            log.warning("draft_explanation(): blank LLM reply, using template")
        except Exception as e:
            log.warning("draft_explanation(): LLM failed (%s), using template", e)
    return tool_template_explanation(bundle), "template"


# --------------------------------------------------------------------------------------
# Node implementations
# --------------------------------------------------------------------------------------


class RunState(TypedDict, total=False):
    """LangGraph channel schema: the serialized GraphState plus the routed next node."""

    state: dict[str, Any]
    next: str


def _champion_artifact(gs: GraphState) -> str | None:
    """Champion artifact path: `notes[i] == 'champion_artifact=<path>'`, else candidate path."""
    v = gs.validation
    if v is None:
        return None
    for note in v.notes:
        if note.startswith("champion_artifact="):
            return note.split("=", 1)[1].strip()
    if gs.candidates is not None:
        for c in gs.candidates.candidates:
            if c.candidate_id == v.champion_id and c.artifact_path:
                return c.artifact_path
    return None


def _promotion_id_of(gs: GraphState) -> str | None:
    for e in reversed(gs.events):
        if e.tool == "request_promotion" and e.payload and e.payload.get("promotion_id"):
            return str(e.payload["promotion_id"])
    return None


def _telemetry(h: RunHandle, engine: Engine | None) -> pd.DataFrame:
    df = h.cache.get("df")
    if df is None:
        df = db.read_telemetry(engine=engine)
        h.cache["df"] = df
    return df


def _features_for(
    h: RunHandle, gs: GraphState, engine: Engine | None
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Return (features_df, spec) from the run cache, recomputing after a restart."""
    fdf = h.cache.get("features_df")
    spec = h.cache.get("spec")
    if fdf is not None and spec is not None:
        return fdf, spec
    regimes = gs.data_quality.regimes if gs.data_quality else None
    fdf, spec = tool_build_features(_telemetry(h, engine), regimes, h.horizon_h)
    if gs.candidates is not None:
        spec = gs.candidates.features
    h.cache["features_df"], h.cache["spec"] = fdf, spec
    return fdf, spec


def _latest_features(features_df: pd.DataFrame, spec: FeatureSpec, asset_id: str) -> pd.DataFrame:
    sub = features_df
    if "asset_id" in features_df.columns:
        sub = features_df[features_df["asset_id"] == asset_id]
    if sub.empty:
        sub = features_df
    if "ts" in sub.columns:
        sub = sub.sort_values("ts")
    return sub.tail(1)[spec.features].reset_index(drop=True).copy()


def _profile(gs: GraphState, h: RunHandle, engine: Engine | None) -> None:
    df = _telemetry(h, engine)
    if df.empty:
        raise RuntimeError("no telemetry rows in the database")
    dq = tool_profile_dataset(df, gs.asset_id)
    gs.data_quality = dq
    append_event(
        gs,
        gs.stage,
        f"data quality {dq.quality_score:.1f}/100, {dq.n_rows} rows, {dq.n_assets} assets, "
        f"{len(dq.regimes.regimes)} regimes, trainable={dq.trainable}",
        tool="profile_dataset",
        payload={"quality_score": dq.quality_score, "trainable": dq.trainable},
    )
    if not dq.trainable:
        raise RuntimeError(
            f"Data Quality Contract blocks training (score={dq.quality_score:.1f}, "
            f"n_rows={dq.n_rows}, n_assets={dq.n_assets})"
        )


def _train(gs: GraphState, h: RunHandle, engine: Engine | None) -> None:
    if gs.data_quality is None:
        raise RuntimeError("train called without a Data Quality Contract")
    df = _telemetry(h, engine)
    features_df, spec = tool_build_features(df, gs.data_quality.regimes, h.horizon_h)
    h.cache["features_df"], h.cache["spec"] = features_df, spec
    append_event(
        gs,
        gs.stage,
        f"{len(spec.features)} features over {len(features_df)} rows",
        tool="build_feature_pipeline",
        payload={"features": spec.features},
    )
    task = tool_infer_task(gs.data_quality, df, h.horizon_h)
    append_event(gs, gs.stage, f"task: {task.task} ({task.rationale})", tool="infer_task")
    artifacts_dir = Path(get_settings().models_dir) / "artifacts" / gs.run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cs = tool_train_candidates(features_df, spec, task, h.n_trials, 0, artifacts_dir, gs.asset_id)
    # The regime baseline travels with the FeatureSpec so deploy_edge can ship it to the edge.
    baseline = features_df.attrs.get("baseline") if hasattr(features_df, "attrs") else None
    if baseline and cs.features.baseline is None:
        cs.features.baseline = baseline
    gs.candidates = cs
    h.cache["spec"] = cs.features
    ranked = ", ".join(cs.ranked) if cs.ranked else "unranked"
    append_event(
        gs,
        gs.stage,
        f"{len(cs.candidates)} candidates trained; ranking: {ranked}",
        tool="train_candidates",
        payload={"ranked": cs.ranked},
    )


def _validate(gs: GraphState, h: RunHandle, engine: Engine | None) -> None:
    if gs.candidates is None:
        raise RuntimeError("validate called without candidates")
    features_df, _ = _features_for(h, gs, engine)
    v = tool_validate(gs.candidates, features_df, gs.run_id)
    gs.validation = v
    append_event(
        gs,
        gs.stage,
        f"champion {v.champion_family} {v.model_version} (IMS {v.ims.total:.3f}); "
        f"leakage passed={v.leakage.passed}; lead time median {v.lead_time.median_h:.1f}h; "
        f"validation passed={v.passed}",
        tool="validate",
        payload={"champion_id": v.champion_id, "passed": v.passed, "ims_total": v.ims.total},
    )
    if not v.passed:
        append_event(
            gs, gs.stage, "validation thresholds NOT met -- human review required", tool="validate"
        )
    # Register the champion so ModelOps has something to approve (best effort, P1).
    artifact = _champion_artifact(gs)
    try:
        if artifact is None:
            raise RuntimeError("no champion artifact path")
        rm = tool_register_model(v, Path(artifact), engine)
        comparison = tool_compare_champion(rm, engine)
        pr = tool_request_promotion(comparison, ModelStage.production, engine)
        h.cache["promotion_id"] = pr.promotion_id
        append_event(
            gs,
            gs.stage,
            f"registered {rm.name} {rm.version} (stage {rm.stage.value}); "
            f"promotion {pr.promotion_id} pending",
            tool="request_promotion",
            payload={"promotion_id": pr.promotion_id, "version": rm.version, "name": rm.name},
        )
    except Exception as e:
        log.warning("model registration skipped: %s", e)
        append_event(gs, gs.stage, f"model registration skipped: {e}", tool="register_model")


def _explain(gs: GraphState, h: RunHandle, engine: Engine | None) -> None:
    if gs.validation is None or gs.data_quality is None:
        raise RuntimeError("explain called without validation / data quality")
    v = gs.validation
    features_df, spec = _features_for(h, gs, engine)
    model = h.cache.get("model")
    if model is None:
        artifact = _champion_artifact(gs)
        if artifact is None:
            raise RuntimeError("no champion artifact to explain")
        model = tool_load_model(artifact)
        h.cache["model"] = model
    x_latest = _latest_features(features_df, spec, gs.asset_id)
    h.cache["x_latest"] = x_latest
    background = _explain_background(features_df, spec, gs.asset_id)
    exp = tool_explain(model, x_latest, spec, gs.asset_id, v.model_version, background)
    top = exp.top_features[0].feature if exp.top_features else "n/a"
    append_event(
        gs,
        gs.stage,
        f"failure probability {exp.failure_probability:.3f}; top feature {top}",
        tool="explain",
        payload={"failure_probability": exp.failure_probability},
    )
    passages: list[ManualPassage] = []
    try:
        passages = tool_retrieve_manual_context(exp)
        append_event(
            gs,
            gs.stage,
            f"{len(passages)} manual passages retrieved",
            tool="retrieve_manual_context",
        )
    except Exception as e:  # retrieval enriches the explanation, never load-bearing
        append_event(
            gs, gs.stage, f"manual retrieval unavailable: {e}", tool="retrieve_manual_context"
        )
    settings = get_settings()
    now = datetime.now(UTC)
    p_now = float(exp.failure_probability)
    # A model that has not demonstrated early warning gets a conservative 12 h hazard scale
    # rather than a degenerate curve that jumps to 100% at the first planned window.
    lead = max(float(v.lead_time.median_h), 12.0)

    def p_fail_by(hours: float) -> float:
        """Hazard curve: P(fail within h) = 1 - (1 - p_now) ** (1 + h / lead_time_median)."""
        val = 1.0 - (1.0 - p_now) ** (1.0 + max(hours, 0.0) / lead)
        return float(min(1.0, max(0.0, val)))

    cost = tool_calculate_failure_cost(p_fail_by, settings, 168.0, now)
    window = tool_find_maintenance_window(cost, v.lead_time, now)
    append_event(
        gs,
        gs.stage,
        f"recommended {cost.recommended}; window {window.start.isoformat()} -> "
        f"{window.end.isoformat()}",
        tool="calculate_failure_cost",
        payload={"recommended": cost.recommended},
    )
    bundle = tool_build_evidence_bundle(
        exp, passages, cost, window, v, gs.data_quality.quality_score
    )
    gs.evidence = bundle
    text, source = draft_explanation(bundle)
    if _llm_enabled():
        gs.llm_calls += 1
    append_event(gs, gs.stage, f"explanation drafted ({source})", tool="draft_explanation")
    contract = tool_create_decision_contract(bundle, gs.run_id, text, source, engine)
    gs.contract = contract
    append_event(
        gs,
        gs.stage,
        f"decision contract {contract.contract_id} created",
        tool="create_decision_contract",
        payload={"contract_id": contract.contract_id},
    )


def await_approval(state: GraphState) -> GraphState:
    """Human Approval Gate (09): persist stage=awaiting_approval and return the state.

    The graph interrupts before `finalize`; `resume()` continues once a decision is recorded.
    """
    state.stage = PipelineStage.awaiting_approval
    append_event(
        state,
        state.stage,
        "awaiting human approval of decision contract "
        + (state.contract.contract_id if state.contract else "(none)"),
        tool="await_approval",
    )
    h = get_run_handle(state.run_id)
    _persist(state, h.engine if h else None)
    return state


def _finalize(gs: GraphState, h: RunHandle, engine: Engine | None) -> None:
    approval = gs.approval
    if approval is None:
        raise PermissionError("finalize requires a recorded human decision (record_decision)")
    if approval.decision != "approved":
        gs.stage = PipelineStage.rejected
        append_event(
            gs,
            gs.stage,
            f"rejected by {approval.approver}: {approval.note or '-'}",
            tool="finalize",
        )
        return
    gs.stage = PipelineStage.approved
    append_event(
        gs, gs.stage, f"approved by {approval.approver}: {approval.note or '-'}", tool="finalize"
    )
    promotion_id = h.cache.get("promotion_id") or _promotion_id_of(gs)
    if not promotion_id:
        append_event(
            gs, gs.stage, "no promotion request linked to this run; deploy skipped", tool="promote"
        )
        return
    try:
        promo_approval = approval.model_copy(update={"promotion_id": promotion_id})
        rm = tool_promote(promotion_id, promo_approval, engine)
        append_event(
            gs, gs.stage, f"promoted {rm.name} {rm.version} -> {rm.stage.value}", tool="promote"
        )
        artifact = _champion_artifact(gs)
        features = gs.candidates.features.features if gs.candidates else []
        window_rows = gs.candidates.features.window_rows if gs.candidates else 12
        baseline = gs.candidates.features.baseline if gs.candidates else None
        if artifact:
            dep = tool_deploy_edge(
                rm,
                Path(artifact),
                features,
                Path(get_settings().models_dir),
                window_rows,
                baseline=baseline,
            )
            append_event(
                gs,
                gs.stage,
                f"edge deployment written: {dep.onnx_path}",
                tool="deploy_edge",
                payload={"onnx_path": dep.onnx_path},
            )
    except Exception as e:  # promotion/edge are P1; the approval itself stands
        log.warning("promote/deploy_edge skipped: %s", e)
        append_event(gs, gs.stage, f"promote/deploy skipped: {e}", tool="promote")


NodeFn = Callable[[GraphState, RunHandle, Engine | None], None]


def _make_node(name: str, stage: PipelineStage, fn: NodeFn) -> Callable[[RunState], RunState]:
    """Wrap a node body: set stage, persist, run, catch -> failed, route, persist."""

    def node(s: RunState) -> RunState:
        gs = state_from_json(s["state"])
        h = _handle_for(gs.run_id, gs.asset_id)
        engine = h.engine
        gs.stage = stage
        append_event(gs, stage, f"{name} started", tool=name)
        _persist(gs, engine)
        try:
            fn(gs, h, engine)
        except Exception as e:  # any node failure ends the run as `failed`
            log.exception("node %s failed for run %s", name, gs.run_id)
            gs.stage = PipelineStage.failed
            gs.error = f"{name}: {e}"
            append_event(gs, gs.stage, gs.error, tool=name)
            _persist(gs, engine)
            return {"state": state_to_json(gs), "next": END}
        nxt = route(gs)
        _persist(gs, engine)
        return {"state": state_to_json(gs), "next": nxt}

    return node


def _await_node(s: RunState) -> RunState:
    gs = state_from_json(s["state"])
    _handle_for(gs.run_id, gs.asset_id)
    gs = await_approval(gs)
    return {"state": state_to_json(gs), "next": route(gs)}


_profile_node = _make_node("profile", PipelineStage.profiling, _profile)
_train_node = _make_node("train", PipelineStage.training, _train)
_validate_node = _make_node("validate", PipelineStage.validating, _validate)
_explain_node = _make_node("explain", PipelineStage.explaining, _explain)
_finalize_node = _make_node("finalize", PipelineStage.awaiting_approval, _finalize)


def _pick_next(s: RunState) -> str:
    return s.get("next") or END


def build_graph() -> Any:
    """Compile the LangGraph: profile -> train -> validate -> explain -> await_approval -> finalize.

    Every edge is conditional on `route()`. A `MemorySaver` checkpointer plus
    `interrupt_before=["finalize"]` pauses the graph at the Human Approval Gate.
    """
    g: StateGraph = StateGraph(RunState)
    g.add_node("profile", _profile_node)
    g.add_node("train", _train_node)
    g.add_node("validate", _validate_node)
    g.add_node("explain", _explain_node)
    g.add_node("await_approval", _await_node)
    g.add_node("finalize", _finalize_node)
    g.add_edge(START, "profile")
    path_map = {n: n for n in NODE_NAMES}
    path_map[END] = END
    for n in NODE_NAMES:
        g.add_conditional_edges(n, _pick_next, path_map)
    return g.compile(checkpointer=MemorySaver(), interrupt_before=["finalize"])


def _graph() -> Any:
    global _GRAPH
    with _GRAPH_LOCK:
        if _GRAPH is None:
            _GRAPH = build_graph()
        return _GRAPH


# --------------------------------------------------------------------------------------
# Public run control
# --------------------------------------------------------------------------------------


def _execute(run_id: str) -> None:
    h = _handle_for(run_id)
    h.status = "running"
    try:
        raw = db.load_run(run_id, engine=h.engine)
        if raw is None:
            raise RuntimeError(f"run {run_id} not persisted")
        _graph().invoke({"state": raw, "next": "profile"}, _config(run_id))
        h.status = "paused"
    except Exception as e:
        log.exception("run %s crashed", run_id)
        h.status = "failed"
        try:
            raw = db.load_run(run_id, engine=h.engine)
            gs = state_from_json(raw) if raw else GraphState(run_id=run_id, asset_id=h.asset_id)
            if gs.stage not in TERMINAL_STAGES:
                gs.stage = PipelineStage.failed
                gs.error = f"orchestrator: {e}"
                append_event(gs, gs.stage, gs.error, tool="orchestrator")
                _persist(gs, h.engine)
        except Exception:
            log.exception("could not persist failure for run %s", run_id)


def start_run(
    asset_id: str, horizon_h: float = 48.0, n_trials: int = 6, engine: Engine | None = None
) -> GraphState:
    """Create a run for `asset_id`, persist stage=queued and execute the graph in a daemon thread.

    Returns immediately with the queued GraphState (training takes ~1 minute of CPU).
    """
    run_id = new_run_id()
    gs = GraphState(run_id=run_id, asset_id=asset_id)
    append_event(
        gs,
        gs.stage,
        f"run queued (horizon {horizon_h}h, {n_trials} trials/family)",
        tool="start_run",
    )
    _persist(gs, engine)
    with _RUNS_LOCK:
        h = RunHandle(run_id, asset_id, horizon_h, n_trials, engine)
        _RUNS[run_id] = h
    t = threading.Thread(target=_execute, args=(run_id,), name=f"thor-run-{run_id}", daemon=True)
    h.thread = t
    t.start()
    return gs


def resume(run_id: str, approval: Approval, engine: Engine | None = None) -> GraphState:
    """Continue a run paused at the approval gate once `record_decision()` stored `approval`.

    Runs `finalize`: approved -> promote + deploy_edge (best effort) and stage=approved;
    rejected -> stage=rejected. Raises KeyError for unknown runs and ValueError if the run is
    not awaiting approval.
    """
    raw = db.load_run(run_id, engine=engine)
    if raw is None:
        raise KeyError(run_id)
    gs = state_from_json(raw)
    if gs.stage != PipelineStage.awaiting_approval:
        raise ValueError(f"run {run_id} is in stage {gs.stage.value}, not awaiting_approval")
    gs.approval = approval
    append_event(
        gs,
        gs.stage,
        f"decision recorded: {approval.decision} by {approval.approver}",
        tool="record_decision",
        payload={"approval_id": approval.approval_id},
    )
    _persist(gs, engine)
    h = _handle_for(run_id, gs.asset_id, engine)
    h.status = "resuming"
    graph = _graph()
    cfg = _config(run_id)
    try:
        snap = graph.get_state(cfg)
        if snap.next and "finalize" in snap.next:
            graph.update_state(cfg, {"state": state_to_json(gs), "next": "finalize"})
            out = graph.invoke(None, cfg)
            h.status = "done"
            return state_from_json(out["state"])
    except Exception as e:  # checkpoint lost (process restart) -> run finalize directly
        log.warning("graph resume failed for %s (%s); running finalize directly", run_id, e)
    out = _finalize_node({"state": state_to_json(gs), "next": "finalize"})
    h.status = "done"
    return state_from_json(out["state"])


def get_run(run_id: str, engine: Engine | None = None) -> GraphState | None:
    """Load a persisted run as GraphState (None if unknown)."""
    raw = db.load_run(run_id, engine=engine)
    return state_from_json(raw) if raw else None


def whatif(run_id: str, scenario: dict[str, float], engine: Engine | None = None) -> WhatIfResult:
    """Re-score the run's champion with feature overrides (P2 sandbox).

    Needs the run's cached model + latest feature row (rebuilt from the artifact if the process
    restarted). Raises KeyError for unknown runs, RuntimeError when no champion exists yet.
    """
    gs = get_run(run_id, engine)
    if gs is None:
        raise KeyError(run_id)
    if gs.validation is None:
        raise RuntimeError("run has no validated champion yet")
    h = _handle_for(run_id, gs.asset_id, engine)
    model = h.cache.get("model")
    if model is None:
        artifact = _champion_artifact(gs)
        if artifact is None:
            raise RuntimeError("no champion artifact for this run")
        model = tool_load_model(artifact)
        h.cache["model"] = model
    x_latest = h.cache.get("x_latest")
    if x_latest is None:
        features_df, spec = _features_for(h, gs, engine)
        x_latest = _latest_features(features_df, spec, gs.asset_id)
        h.cache["x_latest"] = x_latest
    spec = h.cache.get("spec") or (gs.candidates.features if gs.candidates else None)
    if spec is None:
        raise RuntimeError("no feature spec for this run")
    return tool_run_whatif(model, x_latest, spec, gs.asset_id, scenario)


# --------------------------------------------------------------------------------------
# Copilot (Bolt) -- read-only deterministic tools + optional LLM tool loop
# --------------------------------------------------------------------------------------

ASSET_ID_RE = re.compile(r"\bMTR-\d{3}\b", re.IGNORECASE)
CONTRACT_ID_RE = re.compile(r"\bdc_[A-Za-z0-9_-]+\b")


def _fleet_rows(engine: Engine | None) -> list[dict[str, Any]]:
    from apps.api.routes.fleet import compute_fleet

    rows = []
    for fa in compute_fleet(engine):
        rows.append(
            {
                "asset_id": fa.asset.asset_id,
                "name": fa.asset.name,
                "site": fa.asset.site,
                "line": fa.asset.line,
                "criticality": fa.asset.criticality,
                "health_score": round(fa.health_score, 1),
                "failure_probability": fa.failure_probability,
                "regime": fa.regime,
                "open_contract_id": fa.open_contract_id,
                "model_stage": fa.stage.value if fa.stage else None,
            }
        )
    rows.sort(key=lambda r: r["health_score"])
    return rows


def copilot_get_fleet(engine: Engine | None = None) -> dict[str, Any]:
    """Tool: fleet overview sorted by risk (lowest health first)."""
    rows = _fleet_rows(engine)
    return {"n_assets": len(rows), "assets": rows, "highest_risk": rows[:3]}


def copilot_get_asset(asset_id: str, engine: Engine | None = None) -> dict[str, Any]:
    """Tool: one asset's fleet row, latest telemetry, contracts and runs."""
    asset_id = asset_id.upper()
    row = next((r for r in _fleet_rows(engine) if r["asset_id"] == asset_id), None)
    if row is None:
        return {"error": f"unknown asset {asset_id}"}
    latest = db.read_telemetry(asset_id=asset_id, limit=1, engine=engine)
    latest_row = None
    if not latest.empty:
        rec = latest.iloc[-1].to_dict()
        latest_row = {
            k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in rec.items()
        }
    contracts = [
        {
            "contract_id": c.contract_id,
            "recommendation": c.recommendation,
            "failure_probability": c.failure_probability,
            "expected_cost": c.expected_cost,
            "status": db.contract_status(c.contract_id, engine=engine),
        }
        for c in db.list_decision_contracts(asset_id=asset_id, engine=engine)[:5]
    ]
    runs = [run_summary(r["state"]) for r in db.list_runs(asset_id=asset_id, engine=engine)[:5]]
    return {"asset": row, "latest_telemetry": latest_row, "contracts": contracts, "runs": runs}


def copilot_get_contract(contract_id: str, engine: Engine | None = None) -> dict[str, Any]:
    """Tool: a decision contract (full payload) plus its derived status."""
    dc = db.get_decision_contract(contract_id, engine=engine)
    if dc is None:
        return {"error": f"unknown contract {contract_id}"}
    out = db.dump_model(dc)
    out["status"] = db.contract_status(contract_id, engine=engine)
    try:
        from agents.reliability.explain import FEATURE_PHRASES

        out["top_features_plain"] = [
            {
                "driver": FEATURE_PHRASES.get(f["feature"], f["feature"].replace("_", " ")),
                "feature": f["feature"],
                "direction": f["direction"],
                "shap_value": f["shap_value"],
            }
            for f in out.get("top_features", [])
        ]
    except Exception:  # noqa: BLE001 - plain names are a courtesy, never required
        pass
    return out


def copilot_get_run(run_id: str, engine: Engine | None = None) -> dict[str, Any]:
    """Tool: run summary plus the last five events."""
    gs = get_run(run_id, engine)
    if gs is None:
        return {"error": f"unknown run {run_id}"}
    out = run_summary(gs)
    out["events"] = [
        {"ts": e.ts.isoformat(), "stage": e.stage.value, "message": e.message, "tool": e.tool}
        for e in gs.events[-5:]
    ]
    return out


def copilot_list_pending_approvals(engine: Engine | None = None) -> dict[str, Any]:
    """Tool: pending decision contracts and pending promotion requests."""
    contracts = [
        {
            "contract_id": c.contract_id,
            "asset_id": c.asset_id,
            "recommendation": c.recommendation,
            "failure_probability": c.failure_probability,
            "expected_cost": c.expected_cost,
            "window_start": c.window_start.isoformat() if c.window_start else None,
        }
        for c in db.list_decision_contracts(engine=engine)
        if db.contract_status(c.contract_id, engine=engine) == "pending"
    ]
    engine_ = engine or db.get_engine()
    with engine_.connect() as conn:
        rows = (
            conn.execute(
                select(db.promotion_requests).where(db.promotion_requests.c.status == "pending")
            )
            .mappings()
            .all()
        )
    promotions = [
        {
            "promotion_id": r["promotion_id"],
            "model_name": r["model_name"],
            "version": r["version"],
            "from_stage": r["from_stage"],
            "to_stage": r["to_stage"],
        }
        for r in rows
    ]
    return {"contracts": contracts, "promotions": promotions}


_QUESTION_WORDS = frozenset(
    "what is are in the a an of does do did how should i we you it this that to for on at by "
    "and or with about tell me explain mean means say says please can could would".split()
)


def copilot_search_manuals(
    query: str, k: int = 3, engine: Engine | None = None, chroma_path: Path | None = None
) -> dict[str, Any]:
    """Tool: the k most relevant plant-manual passages for a free-text query.

    Wraps `agents.reliability.rag.retrieve_manual_context` (Chroma similarity lookup, no LLM).
    Returns {"query", "passages": [{source, section, page, text, score}]}; the passages are
    the ONLY manual text Bolt may quote (CLAUDE.md section 7). Empty list when the index is
    missing or nothing matches -- never raises.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}
    k = max(1, min(int(k or 3), 5))
    # Drop question scaffolding ("what is in the ...?") so the content words drive retrieval;
    # matters most for the offline hashed embedding, harmless for MiniLM.
    content_words = [
        w for w in re.findall(r"[A-Za-z0-9-]+", query) if w.lower() not in _QUESTION_WORDS
    ]
    search_query = " ".join(content_words) or query
    try:
        from agents.reliability import rag

        path = chroma_path or Path(get_settings().chroma_path)
        passages = rag.retrieve_manual_context(search_query, path, k=k)
    except Exception as e:  # noqa: BLE001 - retrieval enriches an answer, never blocks it
        log.warning("copilot search_manuals failed: %s", e)
        passages = []
    return {
        "query": query,
        "passages": [
            {
                "source": p.source,
                "section": p.section,
                "page": p.page,
                "text": p.text,
                "score": round(p.score, 3),
            }
            for p in passages
        ],
    }


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return float(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)


def copilot_search_field_history(
    query: str, k: int = 5, engine: Engine | None = None, chroma_path: Path | None = None
) -> dict[str, Any]:
    """Tool: real unplanned work orders similar to a symptom, from the FMUCD field-history slice.

    Wraps `agents.reliability.rag.retrieve_field_history` (Chroma similarity, no LLM).
    Returns {"query", "source", "cases": [{case_id, component, description, university,
    start_date, labor_hours, total_cost, score}], "n_cases", "median_labor_hours",
    "median_total_cost"}. The medians are computed here so Bolt never has to. Empty cases
    when the collection is missing -- never raises.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}
    k = max(1, min(int(k or 5), 10))
    try:
        from agents.reliability import rag

        path = chroma_path or Path(get_settings().chroma_path)
        cases = rag.retrieve_field_history(query, path, k=k)
    except Exception as e:  # noqa: BLE001 - precedent enriches an answer, never blocks it
        log.warning("copilot search_field_history failed: %s", e)
        cases = []
    return {
        "query": query,
        "source": "FMUCD (12 North American universities, 2002-2021, CC BY-NC)",
        "cases": [
            {
                "case_id": c.case_id,
                "component": c.component,
                "description": c.description,
                "university": c.university,
                "start_date": c.start_date,
                "labor_hours": c.labor_hours,
                "total_cost": c.total_cost,
                "score": round(c.score, 3),
            }
            for c in cases
        ],
        "n_cases": len(cases),
        "median_labor_hours": _median([c.labor_hours for c in cases]),
        "median_total_cost": _median([c.total_cost for c in cases]),
    }


def _quartiles(values: list[float | None]) -> dict[str, float | int] | None:
    """{n, p25, p50, p75, min, max} over the non-null values (inclusive method), or None."""
    import statistics

    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        q = [vals[0]] * 3
    else:
        q = statistics.quantiles(vals, n=4, method="inclusive")
    return {
        "n": len(vals),
        "p25": round(q[0], 2),
        "p50": round(q[1], 2),
        "p75": round(q[2], 2),
        "min": round(vals[0], 2),
        "max": round(vals[-1], 2),
    }


def copilot_field_history_stats(
    query: str, n: int = 50, engine: Engine | None = None, chroma_path: Path | None = None
) -> dict[str, Any]:
    """Tool: labor-hour and cost quartiles over the n most similar field-history cases.

    Same retrieval as search_field_history but wider (10-100 cases, default 50), then
    deterministic aggregates: {"query", "source", "n_cases", "labor_hours": {n, p25, p50,
    p75, min, max} | None, "total_cost": {...} | None, "years": [first, last] | None,
    "components": [{component, n}] top 5}. Built so Bolt can say "these jobs usually take
    1-4 h" from a distribution rather than three anecdotes. Never raises.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}
    n = max(10, min(int(n or 50), 100))
    try:
        from agents.reliability import rag

        path = chroma_path or Path(get_settings().chroma_path)
        cases = rag.retrieve_field_history(query, path, k=n)
    except Exception as e:  # noqa: BLE001 - stats enrich an answer, never block it
        log.warning("copilot field_history_stats failed: %s", e)
        cases = []
    years = sorted(int(c.start_date[:4]) for c in cases if c.start_date)
    comp_counts: dict[str, int] = {}
    for c in cases:
        comp_counts[c.component] = comp_counts.get(c.component, 0) + 1
    top_components = sorted(comp_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {
        "query": query,
        "source": "FMUCD (12 North American universities, 2002-2021, CC BY-NC)",
        "n_cases": len(cases),
        "labor_hours": _quartiles([c.labor_hours for c in cases]),
        "total_cost": _quartiles([c.total_cost for c in cases]),
        "years": [years[0], years[-1]] if years else None,
        "components": [{"component": k, "n": v} for k, v in top_components],
    }


COPILOT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_fleet",
        "description": (
            "Fleet overview: every asset with health score, failure probability, regime and "
            "open contract, sorted by risk."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_asset",
        "description": "Details for one asset id (e.g. MTR-042): latest telemetry, contracts, runs.",
        "input_schema": {
            "type": "object",
            "properties": {"asset_id": {"type": "string"}},
            "required": ["asset_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_contract",
        "description": "A decision contract by id: recommendation, costs, window, evidence, status.",
        "input_schema": {
            "type": "object",
            "properties": {"contract_id": {"type": "string"}},
            "required": ["contract_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_run",
        "description": "A pipeline run by id: stage, error, recent events.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_pending_approvals",
        "description": "Decision contracts and model promotions still waiting for a human decision.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "search_manuals",
        "description": (
            "Search the plant manuals (motor maintenance manual, vibration analysis guide, "
            "maintenance policy) for passages about a failure mode, symptom, threshold, "
            "inspection or repair procedure, or policy. Returns cited passages with source and "
            "section; quote only what it returns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up, in plain words."},
                "k": {"type": "integer", "description": "Passages to return (1-5, default 3)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_field_history",
        "description": (
            "Find real unplanned work orders similar to a symptom (e.g. 'noisy fan motor "
            "bearing', 'pump seized') from a public dataset of university facilities. Returns "
            "cases with component, description, labor hours, cost, and medians. Precedent, "
            "not guidance; not from this plant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The symptom, in plain words."},
                "k": {"type": "integer", "description": "Cases to return (1-10, default 5)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "field_history_stats",
        "description": (
            "Quartiles of labor hours and cost over the most similar real unplanned work orders "
            "for a symptom (default 50 cases, max 100), plus year range and top components. Use "
            "for 'how long / how much does this usually take'. Precedent from a public campus "
            "dataset, not this plant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The symptom or job, in plain words."},
                "n": {"type": "integer", "description": "Cases to aggregate (10-100, default 50)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


def _run_copilot_tool(name: str, args: dict[str, Any], engine: Engine | None) -> dict[str, Any]:
    if name == "get_fleet":
        return copilot_get_fleet(engine)
    if name == "get_asset":
        return copilot_get_asset(str(args.get("asset_id", "")), engine)
    if name == "get_contract":
        return copilot_get_contract(str(args.get("contract_id", "")), engine)
    if name == "get_run":
        return copilot_get_run(str(args.get("run_id", "")), engine)
    if name == "list_pending_approvals":
        return copilot_list_pending_approvals(engine)
    if name == "search_manuals":
        return copilot_search_manuals(str(args.get("query", "")), int(args.get("k", 3)), engine)
    if name == "search_field_history":
        return copilot_search_field_history(
            str(args.get("query", "")), int(args.get("k", 5)), engine
        )
    if name == "field_history_stats":
        return copilot_field_history_stats(
            str(args.get("query", "")), int(args.get("n", 50)), engine
        )
    return {"error": f"unknown tool {name}"}


def _result_summary(result: dict[str, Any]) -> str:
    if "error" in result:
        return str(result["error"])
    if "passages" in result:
        ps = result["passages"]
        top = f"; top {ps[0]['source']} section {ps[0]['section']}" if ps else ""
        return f"{len(ps)} manual passages{top}"
    if "cases" in result:
        med = result.get("median_labor_hours")
        tail = f"; median {med:g} labor h" if med is not None else ""
        return f"{result['n_cases']} field-history cases{tail}"
    if "labor_hours" in result and "components" in result:
        lh = result["labor_hours"]
        tail = f"; labor p25/p50/p75 {lh['p25']:g}/{lh['p50']:g}/{lh['p75']:g} h" if lh else ""
        return f"stats over {result['n_cases']} field-history cases{tail}"
    if "n_assets" in result:
        worst = result["highest_risk"][0] if result["highest_risk"] else None
        tail = (
            f"; highest risk {worst['asset_id']} (health {worst['health_score']})" if worst else ""
        )
        return f"{result['n_assets']} assets{tail}"
    if "asset" in result:
        a = result["asset"]
        return f"{a['asset_id']} health {a['health_score']}, p_fail {a['failure_probability']}"
    if "contracts" in result and "promotions" in result:
        return (
            f"{len(result['contracts'])} contracts, {len(result['promotions'])} promotions pending"
        )
    if "contract_id" in result:
        return (
            f"{result['contract_id']} {result.get('recommendation')} status {result.get('status')}"
        )
    if "run_id" in result:
        return f"{result['run_id']} stage {result.get('stage')}"
    return json.dumps(result)[:120]


def _template_fleet_text(fleet: dict[str, Any]) -> str:
    if not fleet["assets"]:
        return "The fleet has no assets yet -- seed the simulator output first."
    lines = [f"Fleet: {fleet['n_assets']} assets. Highest risk right now:"]
    for r in fleet["highest_risk"]:
        p = ""
        if r["failure_probability"] is not None:
            p = f", failure probability {r['failure_probability']:.2f}"
        oc = f", open contract {r['open_contract_id']}" if r["open_contract_id"] else ""
        lines.append(f"- {r['asset_id']} ({r['name']}): health {r['health_score']}{p}{oc}")
    return "\n".join(lines)


def _template_asset_text(res: dict[str, Any]) -> str:
    if "error" in res:
        return str(res["error"])
    a = res["asset"]
    p = ""
    if a["failure_probability"] is not None:
        p = f", failure probability {a['failure_probability']:.2f}"
    text = (
        f"{a['asset_id']} ({a['name']}, {a['site']}/{a['line']}): health {a['health_score']}{p}, "
        f"regime {a['regime'] or 'unknown'}."
    )
    if res["contracts"]:
        c = res["contracts"][0]
        text += (
            f" Latest decision contract {c['contract_id']}: {c['recommendation']} "
            f"(p_fail {c['failure_probability']:.2f}, expected cost {c['expected_cost']:.0f}), "
            f"status {c['status']}."
        )
    if res["runs"]:
        r = res["runs"][0]
        text += f" Latest run {r['run_id']} is {r['stage']}."
    return text


def _template_pending_text(res: dict[str, Any]) -> str:
    if not res["contracts"] and not res["promotions"]:
        return "Nothing is waiting for approval right now."
    lines = ["Pending human decisions (approve or reject them in the Thor UI -- I cannot):"]
    for c in res["contracts"]:
        lines.append(
            f"- contract {c['contract_id']} on {c['asset_id']}: {c['recommendation']} "
            f"(p_fail {c['failure_probability']:.2f}, expected cost {c['expected_cost']:.0f})"
        )
    for p in res["promotions"]:
        lines.append(
            f"- promotion {p['promotion_id']}: {p['model_name']} {p['version']} "
            f"{p['from_stage']} -> {p['to_stage']}"
        )
    return "\n".join(lines)


def _template_manuals_text(res: dict[str, Any]) -> str:
    if "error" in res:
        return str(res["error"])
    if not res["passages"]:
        return f"No manual passages matched '{res['query']}'."
    lines = [f"From the plant manuals (query: {res['query']}):"]
    for p in res["passages"]:
        where = f"{p['source']} section {p['section']}"
        if p["page"] is not None:
            where += f", p. {p['page']}"
        body = " ".join(p["text"].split())
        if len(body) > 400:
            body = body[:400].rsplit(" ", 1)[0] + " ..."
        lines.append(f"- [{where}] {body}")
    return "\n".join(lines)


def _template_history_text(res: dict[str, Any]) -> str:
    if "error" in res:
        return str(res["error"])
    if not res["cases"]:
        return f"No field-history cases matched '{res['query']}'."
    lines = [
        f"Field history for '{res['query']}' -- {res['n_cases']} similar unplanned work orders "
        f"from {res['source']}; precedent, not from this plant:"
    ]
    for c in res["cases"]:
        year = c["start_date"][:4] if c["start_date"] else "n/a"
        bits = []
        if c["labor_hours"] is not None:
            bits.append(f"{c['labor_hours']:g} labor h")
        if c["total_cost"] is not None:
            bits.append(f"cost {c['total_cost']:,.0f}")
        tail = f" ({', '.join(bits)})" if bits else ""
        lines.append(f"- [{c['component']}, {year}] {c['description']}{tail}")
    med_h, med_c = res["median_labor_hours"], res["median_total_cost"]
    if med_h is not None or med_c is not None:
        parts = []
        if med_h is not None:
            parts.append(f"labor {med_h:g} h")
        if med_c is not None:
            parts.append(f"cost {med_c:,.0f}")
        lines.append("Median across these cases: " + ", ".join(parts) + ".")
    return "\n".join(lines)


WHY_WORDS = ("why", "risk", "wrong", "caus", "explain", "happening", "going on", "evidence")
ACTION_WORDS = ("pending", "approv", "reject", "promot", "deploy", "waiting", "decision")


def _history_query_for_contract(dc: dict[str, Any]) -> str:
    """Deterministic symptom query from a contract payload's risk-raising SHAP drivers."""
    from agents.reliability import rag

    feats = [
        f["feature"] for f in dc.get("top_features", []) if f.get("direction") == "raises_risk"
    ]
    feats = feats or [f["feature"] for f in dc.get("top_features", [])]
    return rag.history_query_from_features(feats)


def _template_contract_evidence(dc: dict[str, Any]) -> str:
    """Evidence + manual citations already inside a contract (no new retrieval)."""
    from agents.reliability.explain import FEATURE_PHRASES

    drivers = [
        FEATURE_PHRASES.get(f["feature"], f["feature"].replace("_", " "))
        for f in dc.get("top_features", [])
        if f.get("direction") == "raises_risk"
    ][:3]
    lines = [
        f"Evidence from contract {dc['contract_id']}: failure probability "
        f"{dc['failure_probability']:.2f}, recommendation {dc['recommendation']}, "
        f"expected cost {dc['expected_cost']:,.0f}."
    ]
    if drivers:
        lines.append("Top risk-raising drivers: " + ", ".join(drivers) + ".")
    cites = [f"{p['source']} section {p['section']}" for p in dc.get("manual_context", [])][:3]
    if cites:
        lines.append("Manual citations in the contract: " + "; ".join(cites) + ".")
    return " ".join(lines)


def _template_stats_text(res: dict[str, Any]) -> str:
    if "error" in res:
        return str(res["error"])
    if not res["n_cases"]:
        return f"No field-history cases matched '{res['query']}'."
    lines = [
        f"Field-history statistics for '{res['query']}' over the {res['n_cases']} most similar "
        f"unplanned work orders from {res['source']}; precedent, not from this plant:"
    ]
    lh, tc = res["labor_hours"], res["total_cost"]
    if lh:
        lines.append(
            f"- labor hours: typically {lh['p25']:g}-{lh['p75']:g} h (median {lh['p50']:g} h, "
            f"range {lh['min']:g}-{lh['max']:g}, n={lh['n']})"
        )
    if tc:
        lines.append(
            f"- total cost: typically {tc['p25']:,.0f}-{tc['p75']:,.0f} (median {tc['p50']:,.0f}, "
            f"n={tc['n']}; campus facilities cost basis)"
        )
    if res["years"]:
        lines.append(f"- years: {res['years'][0]}-{res['years'][1]}")
    if res["components"]:
        comps = ", ".join(f"{c['component']} ({c['n']})" for c in res["components"])
        lines.append(f"- components: {comps}")
    return "\n".join(lines)


STATS_WORDS = (
    "how long",
    "how much",
    "usually",
    "typically",
    "on average",
    "average",
    "distribution",
    "how many hours",
    "labor hours",
)

HISTORY_WORDS = (
    "seen before",
    "seen this",
    "history",
    "similar",
    "past",
    "previous",
    "precedent",
    "how often",
    "other plants",
    "elsewhere",
    "cases",
)

MANUAL_WORDS = (
    "manual",
    "guide",
    "policy",
    "procedure",
    "section",
    "how do i",
    "how to",
    "how should",
    "why",
    "cause",
    "mean",
    "indicate",
    "signif",
    "define",
    "inspect",
    "replace",
    "lubric",
    "grease",
    "threshold",
    "iso",
)

HELP_TEXT = (
    "I am Bolt, Thor's copilot. Ask me about the fleet ('which motors are at risk?'), an asset "
    "('how is MTR-042?'), a decision contract or run id, what is pending approval, the plant "
    "manuals ('what does rising kurtosis mean?'), or field history ('has a noisy fan bearing "
    "been seen before?'). I only report numbers produced by Thor's deterministic tools and "
    "quote only retrieved manual passages or field cases; approvals happen in the UI."
)


def _copilot_template(req: CopilotRequest, engine: Engine | None) -> CopilotResponse:
    last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    text = last.lower()
    calls: list[dict[str, Any]] = []

    def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        res = _run_copilot_tool(name, args, engine)
        calls.append({"name": name, "args": args, "result_summary": _result_summary(res)})
        return res

    m_asset = ASSET_ID_RE.search(last)
    m_contract = CONTRACT_ID_RE.search(last)
    fleet_words = ("fleet", "risk", "health", "status", "worst", "motors", "assets")
    wants_stats = any(k in text for k in STATS_WORDS)
    wants_history = not wants_stats and any(k in text for k in HISTORY_WORDS)
    wants_manual = not (wants_stats or wants_history) and any(k in text for k in MANUAL_WORDS)
    wants_why = any(k in text for k in WHY_WORDS)

    def precedent_for(dc: dict[str, Any]) -> str:
        query = _history_query_for_contract(dc)
        return _template_history_text(call("search_field_history", {"query": query, "k": 3}))

    wants_action = any(k in text for k in ACTION_WORDS)
    if wants_action:
        # Checked before the contract-id branch on purpose: "approve dc_..." must be refused,
        # never answered as a lookup. The template can only list what is pending.
        reply = _template_pending_text(call("list_pending_approvals", {}))
    elif m_contract:
        res = call("get_contract", {"contract_id": m_contract.group(0)})
        if "error" in res:
            reply = str(res["error"])
        else:
            reply = (
                f"Contract {res['contract_id']} for {res['asset_id']}: {res['recommendation']}, "
                f"failure probability {res['failure_probability']:.2f}, expected cost "
                f"{res['expected_cost']:.0f}, status {res['status']}. {res['explanation_text']}"
            )
            reply += "\n\n" + precedent_for(res)
    elif m_asset or (req.asset_id and ("asset" in text or "motor" in text or "this" in text)):
        aid = m_asset.group(0).upper() if m_asset else str(req.asset_id)
        asset_res = call("get_asset", {"asset_id": aid})
        reply = _template_asset_text(asset_res)
        query = ASSET_ID_RE.sub("", last).strip() or last
        contracts = asset_res.get("contracts") or []
        if wants_why and contracts:
            # "Why is MTR-042 at risk?": evidence + the contract's own citations + precedent,
            # without the user having to ask for each.
            dc = call("get_contract", {"contract_id": contracts[0]["contract_id"]})
            if "error" not in dc:
                reply += "\n\n" + _template_contract_evidence(dc) + "\n\n" + precedent_for(dc)
        elif wants_stats:
            reply += "\n\n" + _template_stats_text(call("field_history_stats", {"query": query}))
        elif wants_history:
            reply += "\n\n" + _template_history_text(call("search_field_history", {"query": query}))
        elif wants_manual:
            reply += "\n\n" + _template_manuals_text(call("search_manuals", {"query": query}))
    elif wants_stats:
        reply = _template_stats_text(call("field_history_stats", {"query": last}))
    elif wants_history:
        reply = _template_history_text(call("search_field_history", {"query": last}))
    elif any(k in text for k in fleet_words):
        reply = _template_fleet_text(call("get_fleet", {}))
    elif wants_manual:
        reply = _template_manuals_text(call("search_manuals", {"query": last}))
    else:
        reply = HELP_TEXT
    return CopilotResponse(reply=reply, tool_calls=calls, source="template")


def _copilot_system(req: CopilotRequest) -> str:
    system = COPILOT_SYSTEM_PROMPT
    if req.asset_id:
        system += f" The user is currently looking at asset {req.asset_id}."
    return system


def _copilot_llm_openai(req: CopilotRequest, engine: Engine | None) -> CopilotResponse:
    """OpenAI chat-completions tool loop (max 4 tool rounds) over the same read-only tools."""
    client = _openai_client()
    model = get_settings().openai_model
    messages: list[dict[str, Any]] = [{"role": "system", "content": _copilot_system(req)}]
    for m in req.messages:
        role = "assistant" if m.role == "assistant" else "user"
        messages.append({"role": role, "content": m.content})
    if len(messages) == 1 or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "(context)"})
    tools = _openai_tools(COPILOT_TOOLS)
    calls: list[dict[str, Any]] = []
    reply = ""
    for _round in range(4):
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=1024,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        reply = (msg.content or "").strip() or reply
        tool_calls = _openai_tool_calls(msg)
        if not tool_calls:
            break
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for cid, name, args in tool_calls
                ],
            }
        )
        for cid, name, args in tool_calls:
            res = _run_copilot_tool(name, args, engine)
            calls.append({"name": name, "args": args, "result_summary": _result_summary(res)})
            messages.append(
                {"role": "tool", "tool_call_id": cid, "content": json.dumps(res, default=str)}
            )
    else:
        # tool budget exhausted: one last call without tools to get the final text
        resp = client.chat.completions.create(
            model=model, max_completion_tokens=1024, messages=messages
        )
        reply = (resp.choices[0].message.content or "").strip() or reply
    if not reply:
        reply = "I could not produce an answer from the tool results; please try rephrasing."
    return CopilotResponse(reply=reply, tool_calls=calls, source="llm")


def _copilot_llm(req: CopilotRequest, engine: Engine | None) -> CopilotResponse:
    if _provider() == "openai":
        return _copilot_llm_openai(req, engine)
    client = _llm_client()
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        role = "assistant" if m.role == "assistant" else "user"
        if messages and messages[-1]["role"] == role and isinstance(messages[-1]["content"], str):
            messages[-1]["content"] += "\n" + m.content
        else:
            messages.append({"role": role, "content": m.content})
    if not messages or messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "(context)"})
    system = _copilot_system(req)
    calls: list[dict[str, Any]] = []
    reply = ""
    model = get_settings().anthropic_model
    for _round in range(4):
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system, tools=COPILOT_TOOLS, messages=messages
        )
        tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        reply = _text_of(resp) or reply
        if resp.stop_reason != "tool_use" or not tool_uses:
            break
        messages.append({"role": "assistant", "content": resp.content})
        results: list[dict[str, Any]] = []
        for tu in tool_uses:
            args = tu.input if isinstance(tu.input, dict) else json.loads(str(tu.input))
            res = _run_copilot_tool(tu.name, args, engine)
            calls.append({"name": tu.name, "args": args, "result_summary": _result_summary(res)})
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(res, default=str),
                }
            )
        messages.append({"role": "user", "content": results})
    else:
        # tool budget exhausted: one last call without tools to get the final text
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system, messages=messages
        )
        reply = _text_of(resp) or reply
    if not reply:
        reply = "I could not produce an answer from the tool results; please try rephrasing."
    return CopilotResponse(reply=reply, tool_calls=calls, source="llm")


def copilot_reply(req: CopilotRequest, engine: Engine | None = None) -> CopilotResponse:
    """Answer a copilot chat request.

    With an API key: a Claude tool loop (max 4 tool rounds) over the read-only tools get_fleet,
    get_asset, get_contract, get_run, list_pending_approvals, search_manuals,
    search_field_history, field_history_stats. Without a key, or when the LLM fails:
    keyword-routed template answers over the same tools (source "template").
    """
    if _llm_enabled():
        try:
            return _copilot_llm(req, engine)
        except Exception as e:
            log.warning("copilot LLM failed (%s); using template mode", e)
    return _copilot_template(req, engine)
