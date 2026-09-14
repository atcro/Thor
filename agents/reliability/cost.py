"""07 Reliability: prescriptive cost comparison and maintenance-window selection.

All cost figures come from `Settings` (configurable plant-cost assumptions) and are echoed
into `CostComparison.assumptions` so no report can present them as fact. Deterministic.

Hazard curve
------------
The champion emits p_now = calibrated P(failure within the model horizon) for the latest
sample. To compare options at different future times we need P(failure before hour h). We
use the monotone curve

    p_fail_by(h) = 1 - (1 - p_now) ** (1 + h / max(lead_time_median_h, 1))

which equals p_now at h = 0 and rises toward 1 as h grows, with the rate set by the
model's demonstrated warning lead time: after one lead-time's worth of hours the survival
probability (1 - p_now) has been squared. It is a documented modelling assumption, not a
fitted survival model; the parameters are recorded in the assumptions dict.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

from apps.api.schemas import CostComparison, CostOption, LeadTimeReport, MaintenanceWindow
from apps.api.settings import Settings

DEFAULT_LEAD_TIME_H = 48.0
NEXT_PLANNED_WINDOW_H = 72.0
WINDOW_P_FAIL_LIMIT = 0.30
LOW_LOAD_WINDOW_START_HOUR = 2
LOW_LOAD_WINDOW_END_HOUR = 5
WINDOW_SEARCH_DAYS = 14


def hazard_curve(p_now: float, lead_time_median_h: float = DEFAULT_LEAD_TIME_H) -> Callable[[float], float]:
    """Build the documented monotone hazard curve.

    Inputs: p_now (calibrated failure probability at h = 0, clamped to [0, 1]) and the
    median warning lead time in hours (floored at 1). Output: p_fail_by(h) in [0, 1],
    non-decreasing in h, with p_fail_by(0) == p_now.
    """
    p0 = min(1.0, max(0.0, float(p_now)))
    scale = max(1.0, float(lead_time_median_h))

    def p_fail_by(h: float) -> float:
        h = max(0.0, float(h))
        return float(min(1.0, max(0.0, 1.0 - (1.0 - p0) ** (1.0 + h / scale))))

    return p_fail_by


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    return now if now.tzinfo is not None else now.replace(tzinfo=UTC)


def calculate_failure_cost(
    p_fail_by: Callable[[float], float] | float,
    settings: Settings,
    horizon_h: float = 168.0,
    now: datetime | None = None,
    asset_id: str = "",
    lead_time_median_h: float = DEFAULT_LEAD_TIME_H,
    next_window_h: float = NEXT_PLANNED_WINDOW_H,
) -> CostComparison:
    """Expected cost of maintain-now vs maintain-later vs run-to-failure.

    Inputs: p_fail_by (a callable hour -> P(failure before hour), or a float p_now from
    which the module's hazard curve is built with lead_time_median_h), Settings for the
    plant-cost assumptions, the run-to-failure horizon in hours, `now` (tz-aware UTC;
    default current time), asset id, and the offset of the next planned window (72 h).
    Formulas (unplanned consequence = cost_unplanned_repair + downtime_unplanned_h *
    cost_downtime_per_hour; planned cost = cost_planned_maintenance + downtime_planned_h *
    cost_downtime_per_hour):
      maintain_now   = planned cost; p_failure_before = p_fail_by(0)
      maintain_later = planned cost + p_fail_by(next_window_h) * unplanned consequence
      run_to_failure = p_fail_by(horizon_h) * unplanned consequence
    Output: CostComparison with recommended = lowest expected cost (ties go to the
    earlier option) and assumptions echoing every number used.
    """
    now = _utc(now)
    if callable(p_fail_by):
        curve = p_fail_by
    else:
        curve = hazard_curve(float(p_fail_by), lead_time_median_h)
    p_now = curve(0.0)

    planned_downtime_cost = settings.downtime_planned_h * settings.cost_downtime_per_hour
    planned_total = settings.cost_planned_maintenance + planned_downtime_cost
    unplanned_downtime_cost = settings.downtime_unplanned_h * settings.cost_downtime_per_hour
    unplanned_consequence = settings.cost_unplanned_repair + unplanned_downtime_cost

    p_later = curve(next_window_h)
    p_rtf = curve(horizon_h)

    options = [
        CostOption(
            option="maintain_now",
            when=now,
            expected_cost=float(planned_total),
            p_failure_before=float(p_now),
            downtime_h=float(settings.downtime_planned_h),
            breakdown={
                "planned_maintenance": float(settings.cost_planned_maintenance),
                "planned_downtime": float(planned_downtime_cost),
                "expected_unplanned": 0.0,
            },
        ),
        CostOption(
            option="maintain_later",
            when=now + timedelta(hours=next_window_h),
            expected_cost=float(planned_total + p_later * unplanned_consequence),
            p_failure_before=float(p_later),
            downtime_h=float(settings.downtime_planned_h),
            breakdown={
                "planned_maintenance": float(settings.cost_planned_maintenance),
                "planned_downtime": float(planned_downtime_cost),
                "expected_unplanned": float(p_later * unplanned_consequence),
            },
        ),
        CostOption(
            option="run_to_failure",
            when=None,
            expected_cost=float(p_rtf * unplanned_consequence),
            p_failure_before=float(p_rtf),
            downtime_h=float(settings.downtime_unplanned_h),
            breakdown={
                "planned_maintenance": 0.0,
                "planned_downtime": 0.0,
                "expected_unplanned": float(p_rtf * unplanned_consequence),
            },
        ),
    ]
    recommended = min(options, key=lambda o: o.expected_cost).option
    assumptions = {
        "cost_planned_maintenance": float(settings.cost_planned_maintenance),
        "cost_unplanned_repair": float(settings.cost_unplanned_repair),
        "cost_downtime_per_hour": float(settings.cost_downtime_per_hour),
        "downtime_planned_h": float(settings.downtime_planned_h),
        "downtime_unplanned_h": float(settings.downtime_unplanned_h),
        "horizon_h": float(horizon_h),
        "next_window_h": float(next_window_h),
        "p_now": float(p_now),
        "lead_time_median_h": float(lead_time_median_h),
    }
    return CostComparison(
        asset_id=asset_id, options=options, recommended=recommended, assumptions=assumptions
    )


def _low_load_windows(
    now: datetime, days: int = WINDOW_SEARCH_DAYS
) -> Iterator[tuple[datetime, datetime]]:
    """Yield (start, end) of the standard 02:00-05:00 UTC low-load windows from `now` on.
    If `now` falls inside a window, the first window starts at `now`."""
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(days + 1):
        start = day + timedelta(days=d, hours=LOW_LOAD_WINDOW_START_HOUR)
        end = day + timedelta(days=d, hours=LOW_LOAD_WINDOW_END_HOUR)
        if end <= now:
            continue
        yield max(start, now), end


def find_maintenance_window(
    cost: CostComparison, lead_time: LeadTimeReport, now: datetime | None = None
) -> MaintenanceWindow:
    """Earliest low-load window that ends before P(failure) exceeds 0.30.

    Inputs: the CostComparison (its assumptions carry p_now and the curve parameters),
    the LeadTimeReport (median lead time drives the hazard curve; the assumptions'
    lead_time_median_h is used when the report has no events) and `now`. Windows are the
    plant's standard 02:00-05:00 UTC low-load slots over the next 14 days. Output:
    MaintenanceWindow with p_failure_before_window = P(failure before the window starts).
    If no window qualifies (or the recommendation is maintain_now and today's window has
    passed), the window is "now" and lasts the planned downtime.
    """
    now = _utc(now)
    p_now = float(cost.assumptions.get("p_now", 0.0))
    lead_h = lead_time.median_h if lead_time.n_events > 0 and lead_time.median_h > 0 else float(
        cost.assumptions.get("lead_time_median_h", DEFAULT_LEAD_TIME_H)
    )
    curve = hazard_curve(p_now, lead_h)
    planned_h = float(cost.assumptions.get("downtime_planned_h", 3.0))

    for start, end in _low_load_windows(now):
        h_end = (end - now).total_seconds() / 3600.0
        if curve(h_end) <= WINDOW_P_FAIL_LIMIT:
            h_start = (start - now).total_seconds() / 3600.0
            return MaintenanceWindow(
                start=start,
                end=end,
                reason=(
                    f"earliest 02:00-05:00 UTC low-load window ending before P(failure) "
                    f"exceeds {WINDOW_P_FAIL_LIMIT:.2f} (P at window end = {curve(h_end):.2f})"
                ),
                p_failure_before_window=float(curve(h_start)),
            )
    return MaintenanceWindow(
        start=now,
        end=now + timedelta(hours=planned_h),
        reason=(
            f"now: no low-load window ends before P(failure) exceeds {WINDOW_P_FAIL_LIMIT:.2f} "
            f"(P now = {p_now:.2f}, median lead time {lead_h:.0f} h)"
        ),
        p_failure_before_window=float(p_now),
    )
