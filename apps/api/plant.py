"""Plant-data integration panels for the X-Ray HUD (Phase A, CLAUDE.md section 15).

Two deterministic, side-effect-free views of "the kind of data a plant already captures":

* `plc_tags()` maps the eight observed telemetry fields one-to-one onto PLC-style tags, each
  carrying the `source` field it was read from. Tags the telemetry cannot provide (a speed
  setpoint) are marked `simulated=True` with a `basis` that says how they were derived.
* `simulated_mes_context()` is a *simulated MES feed*: SKU, container, quantity for a packaging
  line at a plant time. It is context for the HUD only. It feeds no model, score,
  recommendation or decision, and `apps/api/routes/twin.py` is its only importer
  (tests/test_twin.py asserts both).

Pure functions: no database, no LLM, no clock (plant time is an input). Same inputs -> same
output, so a judge can re-derive every number on the panel.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from apps.api.schemas import MesContext, ObservedTelemetryRow, PlcTag

#: Eight-hour shifts on the plant clock (UTC): [start_hour, end_hour) -> name.
SHIFTS: tuple[tuple[int, int, str], ...] = ((0, 8, "night"), (8, 16, "day"), (16, 24, "evening"))
SHIFT_HOURS: int = 8

#: Per-line SKU catalogue (rotated by shift) and nominal rate. Keyed by the packaging line
#: number; a line with no number (the unit-test fleet's "L1") falls back to its name length.
_SKU_CATALOGUE: tuple[tuple[str, str], ...] = (
    ("SKU-1001 Still water 0.5 L", "PET bottle"),
    ("SKU-1002 Sparkling water 1.0 L", "PET bottle"),
    ("SKU-2001 Orange juice 1.0 L", "carton"),
    ("SKU-2002 Apple juice 0.2 L", "carton"),
    ("SKU-3001 Cola 0.33 L", "aluminium can"),
    ("SKU-3002 Lemonade 0.33 L", "aluminium can"),
    ("SKU-4001 Sauce 0.5 kg", "glass jar"),
    ("SKU-4002 Honey 0.35 kg", "glass jar"),
)
_BASE_RATE_CASES_PER_H: int = 1800
_RATE_STEP_CASES_PER_H: int = 90

#: PLC tag -> (telemetry field, unit). One-to-one; the value is the field's value unchanged.
PLC_SENSOR_TAGS: tuple[tuple[str, str, str], ...] = (
    ("Speed_Act", "rpm", "rpm"),
    ("Load_Pct", "load_pct", "%"),
    ("Motor_Current", "current_a", "A"),
    ("Bearing_Temp", "bearing_temp_c", "degC"),
    ("Winding_Temp", "motor_temp_c", "degC"),
    ("Vib_RMS", "vibration_rms", "mm/s"),
    ("Vib_Kurtosis", "vibration_kurtosis", ""),
)
RUN_BIT_RPM: float = 100.0


def is_compressor_line(line: str) -> bool:
    """True when the line name denotes the Compressor Hall (contains "compressor")."""
    return "compressor" in line.lower()


def _line_number(line: str) -> int:
    """Trailing integer of a line name ("Packaging Line 3" -> 3, "L2" -> 2), else len(name)."""
    digits = ""
    for ch in reversed(line.strip()):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else len(line)


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def shift_of(plant_ts: datetime) -> tuple[str, datetime]:
    """(shift name, shift start) for a plant time: night 00-08, day 08-16, evening 16-24 UTC."""
    ts = _as_utc(plant_ts)
    for start_h, end_h, name in SHIFTS:
        if start_h <= ts.hour < end_h:
            start = ts.replace(hour=start_h, minute=0, second=0, microsecond=0)
            return name, start
    raise ValueError(f"hour {ts.hour} outside 0..23")  # pragma: no cover - unreachable


def simulated_mes_context(line: str, plant_ts: datetime) -> MesContext | None:
    """Simulated MES feed for one packaging line at `plant_ts` (plant time, not wall clock).

    Inputs: line name and an aware/naive (UTC) plant timestamp. Output: `MesContext` with the
    shift, a SKU + container rotated per shift from a fixed per-line catalogue, the quantity
    produced so far in the shift (`round(rate_per_h * hours into the shift)` with a per-line
    rate), uom "cases" and a `basis` spelling out exactly that arithmetic; None for a
    Compressor-Hall line (no SKU runs there). Deterministic: same inputs -> same output.
    """
    if is_compressor_line(line):
        return None
    ts = _as_utc(plant_ts)
    shift, shift_start = shift_of(ts)
    shift_index = int(ts.timestamp() // (SHIFT_HOURS * 3600))
    line_no = _line_number(line)
    sku, container = _SKU_CATALOGUE[(shift_index + line_no) % len(_SKU_CATALOGUE)]
    rate_per_h = _BASE_RATE_CASES_PER_H + _RATE_STEP_CASES_PER_H * (line_no % 5)
    hours_in = (ts - shift_start) / timedelta(hours=1)
    quantity = int(round(rate_per_h * hours_in))
    basis = (
        f"simulated MES feed: quantity_produced = round({rate_per_h} cases/h x {hours_in:.3f} h "
        f"into the {shift} shift (started {shift_start.isoformat()})) = {quantity}; "
        f"SKU = catalogue[(shift #{shift_index} + line {line_no}) mod {len(_SKU_CATALOGUE)}]. "
        "Context only: feeds no model, score or decision."
    )
    return MesContext(
        line=line,
        sku=sku,
        container=container,
        quantity_produced=quantity,
        uom="cases",
        shift=shift,
        plant_ts=ts,
        basis=basis,
    )


def plc_tags(
    latest: ObservedTelemetryRow | None,
    regime: str | None,
    baseline: dict[str, Any] | None,
) -> list[PlcTag]:
    """PLC tags of the X-Ray panel for one motor.

    Inputs: the latest observed sample (None -> every value None), the inferred regime id
    and the production model's baseline (for the simulated speed setpoint). Output: one
    `PlcTag` per tag. `Run_Bit` and the seven sensor tags carry their source telemetry field
    (`Run_Bit` is derived: rpm > 100). `Speed_SP` has no telemetry source: it is the inferred
    regime's centroid rpm from the production baseline (`_regime_centroids[f"{regime}_rpm"]`)
    and is flagged `simulated=True`; value None with basis "simulated: no production baseline"
    when there is no regime or no centroid.
    """
    values: dict[str, Any] = latest.model_dump() if latest is not None else {}
    rpm = values.get("rpm")
    tags: list[PlcTag] = [
        PlcTag(
            tag="Run_Bit",
            value=(float(rpm) > RUN_BIT_RPM) if rpm is not None else None,
            unit="",
            source="rpm",
            basis=f"rpm > {RUN_BIT_RPM:g}",
        )
    ]
    for tag, field, unit in PLC_SENSOR_TAGS:
        v = values.get(field)
        tags.append(
            PlcTag(
                tag=tag,
                value=float(v) if v is not None else None,
                unit=unit,
                source=field,
                basis=f"telemetry field {field}",
            )
        )
    centroids = (baseline or {}).get("_regime_centroids") or {}
    setpoint = centroids.get(f"{regime}_rpm") if regime else None
    if setpoint is not None:
        tags.append(
            PlcTag(
                tag="Speed_SP",
                value=float(setpoint),
                unit="rpm",
                source=None,
                simulated=True,
                basis=f"simulated: centroid rpm of the inferred operating state {regime} (production baseline)",
            )
        )
    else:
        tags.append(
            PlcTag(
                tag="Speed_SP",
                value=None,
                unit="rpm",
                source=None,
                simulated=True,
                basis="simulated: no production baseline",
            )
        )
    return tags


__all__ = [
    "PLC_SENSOR_TAGS",
    "SHIFTS",
    "is_compressor_line",
    "plc_tags",
    "shift_of",
    "simulated_mes_context",
]
