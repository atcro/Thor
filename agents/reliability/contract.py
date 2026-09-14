"""07 Reliability: evidence bundle, immutable Decision Contract, and template explanation.

`create_decision_contract` is insert-only through `apps.api.db.insert_decision_contract`.
`template_explanation` is the deterministic prose shown when no LLM key is configured; it
only restates numbers already present in the EvidenceBundle and never invents any.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from typing import Any, Literal

from apps.api import db
from apps.api.schemas import (
    CostComparison,
    DecisionContract,
    EvidenceBundle,
    Explanation,
    MaintenanceWindow,
    ManualPassage,
    ShapFeature,
    ValidationReport,
)

RECOMMENDATION_PHRASES: dict[str, str] = {
    "maintain_now": "maintain now",
    "maintain_later": "maintain at the next planned window",
    "run_to_failure": "run to failure and repair on breakdown",
}


def build_evidence_bundle(
    explanation: Explanation,
    manual_context: list[ManualPassage],
    cost: CostComparison,
    window: MaintenanceWindow,
    validation: ValidationReport,
    data_quality_score: float,
) -> EvidenceBundle:
    """Assemble everything the approval gate and the explanation drafter need.

    Inputs: the SHAP Explanation, retrieved ManualPassages (may be empty), CostComparison,
    MaintenanceWindow, ValidationReport, and the 04 data-quality score. Output: an
    EvidenceBundle (pure aggregation; the bundle's JSON is what gets hashed into the
    Decision Contract).
    """
    return EvidenceBundle(
        explanation=explanation,
        manual_context=list(manual_context),
        cost=cost,
        window=window,
        validation=validation,
        data_quality_score=float(data_quality_score),
    )


def evidence_hash(bundle: EvidenceBundle) -> str:
    """sha256 hex digest of bundle.model_dump_json() (the contract's tamper-evidence field)."""
    return hashlib.sha256(bundle.model_dump_json().encode("utf-8")).hexdigest()


def _recommended_option(cost: CostComparison) -> Any:
    for o in cost.options:
        if o.option == cost.recommended:
            return o
    return min(cost.options, key=lambda o: o.expected_cost)


def create_decision_contract(
    bundle: EvidenceBundle,
    run_id: str,
    explanation_text: str,
    explanation_source: Literal["llm", "template"],
    engine: Any = None,
) -> DecisionContract:
    """Create and persist the immutable Decision Contract for one recommendation.

    Inputs: EvidenceBundle, pipeline run id, the explanation prose and whether it came
    from the LLM or the template, optional SQLAlchemy engine (default: process engine).
    contract_id = dc_<asset_id>_<YYYYmmddHHMMSS> UTC (a 4-hex suffix is added if that id
    already exists); evidence_hash = sha256 of bundle.model_dump_json(). The row is
    inserted via db.insert_decision_contract (insert-only; approval is a separate row).
    Output: the DecisionContract model exactly as stored.
    """
    now = datetime.now(UTC)
    asset_id = bundle.explanation.asset_id
    contract_id = f"dc_{asset_id}_{now:%Y%m%d%H%M%S}"
    if db.get_decision_contract(contract_id, engine=engine) is not None:
        contract_id = f"{contract_id}_{secrets.token_hex(2)}"
    option = _recommended_option(bundle.cost)
    dc = DecisionContract(
        contract_id=contract_id,
        created_at=now,
        asset_id=asset_id,
        run_id=run_id,
        model_version=bundle.explanation.model_version,
        champion_mlflow_run_id=bundle.validation.champion_mlflow_run_id,
        failure_probability=bundle.explanation.failure_probability,
        recommendation=bundle.cost.recommended,
        window_start=bundle.window.start,
        window_end=bundle.window.end,
        expected_cost=float(option.expected_cost),
        cost_comparison=bundle.cost,
        top_features=list(bundle.explanation.top_features),
        manual_context=list(bundle.manual_context),
        lead_time_h=float(bundle.validation.lead_time.median_h),
        calibration_brier=float(bundle.validation.calibration.brier_after),
        data_quality_score=float(bundle.data_quality_score),
        explanation_text=explanation_text,
        explanation_source=explanation_source,
        evidence_hash=evidence_hash(bundle),
    )
    db.insert_decision_contract(dc, engine=engine)
    return dc


# --------------------------------------------------------------------------------------
# Template explanation
# --------------------------------------------------------------------------------------


def _trend(v: float) -> str:
    return "up" if v >= 0 else "down"


def _above_below(v: float) -> str:
    return "above" if v >= 0 else "below"


def describe_feature(f: ShapFeature) -> str:
    """Plain-language phrase for one SHAP feature and its observed value (no new numbers)."""
    v = f.feature_value
    name = f.feature
    if name == "vib_rms_z":
        phrase = f"vibration RMS is {abs(v):.1f} standard deviations {_above_below(v)} its regime baseline"
    elif name == "bearing_temp_z":
        phrase = f"bearing temperature is {abs(v):.1f} standard deviations {_above_below(v)} its regime baseline"
    elif name == "vib_rms_mean":
        phrase = f"mean vibration RMS is {v:.2f} mm/s"
    elif name == "vib_rms_slope":
        phrase = f"vibration RMS is trending {_trend(v)} at {abs(v):.3f} per sample"
    elif name == "vib_kurt_mean":
        phrase = f"vibration kurtosis is {v:.1f} (a healthy bearing sits near 3)"
    elif name == "vib_crest_mean":
        phrase = f"vibration crest factor is {v:.1f}"
    elif name == "bearing_temp_mean":
        phrase = f"bearing temperature is {v:.1f} C"
    elif name == "bearing_temp_slope":
        phrase = f"bearing temperature is trending {_trend(v)} at {abs(v):.3f} C per sample"
    elif name == "temp_delta":
        phrase = f"the bearing-minus-motor temperature delta is {v:.1f} K"
    elif name == "current_mean":
        phrase = f"motor current is {v:.1f} A"
    elif name == "rpm_mean":
        phrase = f"motor speed is {v:.0f} rpm"
    elif name == "load_mean":
        phrase = f"motor load is {v:.0f} percent"
    elif name == "vib_rms_resid":
        phrase = f"vibration RMS is {abs(v):.2f} mm/s {_above_below(v)} the regime-expected value"
    else:
        phrase = f"{name.replace('_', ' ')} is {v:.3g}"
    tag = "raising risk" if f.direction == "raises_risk" else "lowering risk"
    return f"{phrase} ({tag})"


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _fmt_ts(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def template_explanation(bundle: EvidenceBundle) -> str:
    """Deterministic 4-6 sentence explanation built only from numbers in the bundle.

    Input: EvidenceBundle. Output: prose that states the failure probability as a
    percentage, describes the top 2-3 SHAP features in plain language, lists the three
    cost options with their dollar figures prefixed by "under the configured plant-cost
    assumptions", names the recommended action and window, cites retrieved manual
    passages as "<source> section <section>", and closes with the validation lead time
    and the approval requirement. Used when ANTHROPIC_API_KEY is blank or the LLM fails.
    """
    exp = bundle.explanation
    cost = bundle.cost
    val = bundle.validation
    win = bundle.window
    by_option = {o.option: o for o in cost.options}

    sentences: list[str] = []
    sentences.append(
        f"Asset {exp.asset_id} has a calibrated failure probability of "
        f"{exp.failure_probability:.0%} within the model horizon (model {exp.model_version}, "
        f"champion {val.champion_family}, held-out recall {val.ims.recall:.2f})."
    )

    top = exp.top_features[:3]
    if top:
        phrases = [describe_feature(f) for f in top]
        if len(phrases) == 1:
            drivers = phrases[0]
        else:
            drivers = "; ".join(phrases[:-1]) + "; and " + phrases[-1]
        sentences.append(f"The main drivers are that {drivers}.")
    else:
        sentences.append("No per-feature attribution was available for this prediction.")

    now_o = by_option.get("maintain_now")
    later_o = by_option.get("maintain_later")
    rtf_o = by_option.get("run_to_failure")
    parts: list[str] = []
    if now_o is not None:
        parts.append(f"maintaining now is expected to cost {_money(now_o.expected_cost)}")
    if later_o is not None:
        parts.append(
            f"maintaining at the next planned window {_money(later_o.expected_cost)} "
            f"(with a {later_o.p_failure_before:.0%} chance of failing first)"
        )
    if rtf_o is not None:
        parts.append(
            f"running to failure {_money(rtf_o.expected_cost)} "
            f"(with a {rtf_o.p_failure_before:.0%} chance of failure within the horizon)"
        )
    sentences.append("Under the configured plant-cost assumptions, " + ", ".join(parts) + ".")

    sentences.append(
        f"The recommended action is to {RECOMMENDATION_PHRASES.get(cost.recommended, cost.recommended)}, "
        f"with the proposed window {_fmt_ts(win.start)} to {_fmt_ts(win.end)} "
        f"({win.p_failure_before_window:.0%} chance of failure before the window starts)."
    )

    seen: list[str] = []
    for p in bundle.manual_context:
        cite = f"{p.source} section {p.section}"
        if cite not in seen:
            seen.append(cite)
    if seen:
        sentences.append("This signature matches " + " and ".join(seen[:3]) + ".")

    sentences.append(
        f"Validation measured a median warning lead time of {val.lead_time.median_h:.0f} hours "
        f"(Brier score {val.calibration.brier_after:.3f}, data quality {bundle.data_quality_score:.0f}/100); "
        "no action is taken until an engineer approves this contract."
    )
    return " ".join(sentences)
