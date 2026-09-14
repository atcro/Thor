"""Synthetic fleet telemetry generator -- Thor's primary demo data.

Produces 30 days (default) of 10-minute sensor samples for 24 induction motors at a single
plant site. Every asset cycles through three operating regimes on a shift schedule:

* R1 idle / startup  (rpm ~ 600,  load ~ 10 %)
* R2 nominal         (rpm ~ 1480, load ~ 55 %)
* R3 high load       (rpm ~ 1500, load ~ 85 %)

Healthy sensor values depend on regime (vibration and temperatures rise with load), so a
load change alone must never look like a fault. Two assets receive an injected drive-end
bearing degradation that ends in failure at the end of the horizon:

* MTR-042 (the demo motor)  onset at 55 % of the horizon
* MTR-017 (subtler)         onset at 80 % of the horizon

Ground truth columns: `health` (1 -> 0) and `failure_within_h` (hours to failure, NaN when
no failure lies ahead). Models may use `failure_within_h` only to build labels, never as a
feature (docs/INTERFACES.md).

Everything is deterministic under `seed`. Runnable as:

    python -m data.simulator.generate [--days 30] [--seed 42] [--step-min 10] [--out DIR]

This module is shipped into the replay container without `apps/`, so the shared Pydantic
contracts are imported when available and mirrored locally otherwise.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:  # pragma: no cover - exercised in the api container / repo checkout
    from apps.api.schemas import Asset, TelemetryRow
except ModuleNotFoundError:  # pragma: no cover - replay container ships only data/ + streaming/

    class TelemetryRow(BaseModel):  # type: ignore[no-redef]
        """Local mirror of apps.api.schemas.TelemetryRow (same fields, same order)."""

        asset_id: str
        ts: datetime
        vibration_rms: float
        vibration_kurtosis: float
        vibration_crest: float
        bearing_temp_c: float
        motor_temp_c: float
        current_a: float
        rpm: float
        load_pct: float
        regime: str | None = None
        health: float | None = Field(default=None, ge=0.0, le=1.0)
        failure_within_h: float | None = None

    class Asset(BaseModel):  # type: ignore[no-redef]
        """Local mirror of apps.api.schemas.Asset."""

        asset_id: str
        name: str
        site: str
        line: str
        asset_type: str = "induction_motor"
        rated_kw: float = 75.0
        criticality: Literal["low", "medium", "high"] = "medium"


# --------------------------------------------------------------------------------------
# Fleet definition
# --------------------------------------------------------------------------------------

DEMO_ASSET = "MTR-042"

#: 24 asset ids. MTR-004 is replaced by MTR-042 so the demo asset id matches the architecture doc.
ASSET_IDS: list[str] = [f"MTR-{i:03d}" for i in range(1, 25)]
ASSET_IDS[3] = DEMO_ASSET

#: asset_id -> onset of bearing degradation as a fraction of the horizon.
FAULTY_ASSETS: dict[str, float] = {DEMO_ASSET: 0.55, "MTR-017": 0.80}

#: asset_id -> amplitude multiplier of the fault signature (MTR-017 is deliberately subtler).
FAULT_AMPLITUDE: dict[str, float] = {DEMO_ASSET: 1.0, "MTR-017": 0.6}

SENSOR_COLUMNS: tuple[str, ...] = (
    "vibration_rms",
    "vibration_kurtosis",
    "vibration_crest",
    "bearing_temp_c",
    "motor_temp_c",
    "current_a",
    "rpm",
    "load_pct",
)

TELEMETRY_COLUMNS: list[str] = [
    "asset_id",
    "ts",
    *SENSOR_COLUMNS,
    "regime",
    "health",
    "failure_within_h",
]

#: Deterministic start of the replay horizon (UTC).
DEFAULT_START = datetime(2025, 8, 1, 0, 0, tzinfo=UTC)

SITE = "Ludvika"
LINES: tuple[str, str, str] = ("Compressor Hall", "Pump House", "Conveyor A")

# (name, rated_kw, criticality) for each of the 8 motors per line, in ASSET_IDS order.
_CATALOGUE: dict[str, list[tuple[str, float, str]]] = {
    "Compressor Hall": [
        ("Main air compressor K-1 drive motor", 160.0, "high"),
        ("Instrument air compressor K-3 drive motor", 55.0, "medium"),
        ("Chiller CH-1 compressor motor", 110.0, "high"),
        ("Main air compressor K-2 drive motor", 160.0, "high"),  # MTR-042 (demo asset)
        ("Cooling tower fan CT-1 motor", 37.0, "medium"),
        ("Cooling tower fan CT-2 motor", 37.0, "medium"),
        ("Dryer regeneration blower B-1 motor", 22.0, "low"),
        ("Aftercooler circulation pump P-12 motor", 30.0, "low"),
    ],
    "Pump House": [
        ("Cooling water pump P-101 motor", 75.0, "high"),
        ("Cooling water pump P-102 motor", 75.0, "high"),
        ("Boiler feed pump P-201 motor", 110.0, "high"),
        ("Boiler feed pump P-202 motor", 110.0, "medium"),
        ("Fire water jockey pump P-301 motor", 15.0, "medium"),
        ("Process water pump P-110 motor", 55.0, "medium"),
        ("Condensate return pump P-120 motor", 30.0, "low"),
        ("Sump pump P-130 motor", 11.0, "low"),
    ],
    "Conveyor A": [
        ("Conveyor A1 head drive motor", 75.0, "medium"),  # MTR-017
        ("Conveyor A1 tail drive motor", 45.0, "low"),
        ("Conveyor A2 head drive motor", 75.0, "medium"),
        ("Conveyor A2 tail drive motor", 45.0, "low"),
        ("Transfer tower T-1 drive motor", 55.0, "medium"),
        ("Screen feeder F-1 drive motor", 30.0, "low"),
        ("Bucket elevator E-1 drive motor", 90.0, "high"),
        ("Stacker S-1 slew drive motor", 37.0, "medium"),
    ],
}


def build_assets() -> list[Asset]:
    """Return the 24-asset catalogue.

    Inputs: none. Output: list[Asset] in ASSET_IDS order (8 motors per line, 3 lines,
    site "Ludvika"). Names/kW/criticality look like a real plant register.
    """
    out: list[Asset] = []
    i = 0
    for line in LINES:
        for name, kw, crit in _CATALOGUE[line]:
            out.append(
                Asset(
                    asset_id=ASSET_IDS[i],
                    name=name,
                    site=SITE,
                    line=line,
                    asset_type="induction_motor",
                    rated_kw=kw,
                    criticality=crit,  # type: ignore[arg-type]
                )
            )
            i += 1
    return out


# --------------------------------------------------------------------------------------
# Regime schedule
# --------------------------------------------------------------------------------------

# Regime index 0/1/2 == R1/R2/R3
REGIME_LABELS = np.array(["R1", "R2", "R3"])
REGIME_RPM = np.array([600.0, 1480.0, 1500.0])
REGIME_LOAD = np.array([10.0, 55.0, 85.0])
REGIME_VIB = np.array([1.1, 2.1, 2.8])  # healthy vibration RMS (mm/s) per regime
REGIME_RPM_SD = np.array([15.0, 8.0, 8.0])
REGIME_LOAD_SD = np.array([2.5, 4.0, 4.0])

# P(R1, R2, R3) per shift kind: 0 = night (00-08), 1 = day (08-16), 2 = evening (16-24)
_SHIFT_REGIME_P = {
    0: np.array([0.45, 0.50, 0.05]),
    1: np.array([0.05, 0.35, 0.60]),
    2: np.array([0.10, 0.60, 0.30]),
}
_SHIFT_MIN = 8 * 60
_STARTUP_ROWS = 2  # first 20 min of every shift is startup (R1)


def _regime_schedule(n: int, step_min: int, phase_min: int, rng: np.random.Generator) -> np.ndarray:
    """Integer regime index (0/1/2) per row for one asset on a three-shift day.

    Inputs: n rows, sampling step in minutes, a per-asset phase offset in minutes (so the
    fleet is not synchronized), and a Generator. Output: int array of shape (n,).
    """
    minute = np.arange(n) * step_min + phase_min
    shift_no = minute // _SHIFT_MIN
    shift_kind = (minute // _SHIFT_MIN) % 3
    n_shifts = int(shift_no.max()) + 1
    per_shift = np.empty(n_shifts, dtype=int)
    for s in range(n_shifts):
        kind = int(s % 3)
        per_shift[s] = rng.choice(3, p=_SHIFT_REGIME_P[kind])
    regime = per_shift[shift_no]
    # Startup rows at the beginning of every shift.
    within = (minute % _SHIFT_MIN) // step_min
    regime = np.where(within < _STARTUP_ROWS, 0, regime)
    # Occasional short idle breaks (40 min) inside a working shift.
    rows_per_shift = _SHIFT_MIN // step_min
    for s in range(n_shifts):
        if per_shift[s] == 0 or rng.random() > 0.3:
            continue
        start = s * rows_per_shift + int(rng.integers(_STARTUP_ROWS + 3, rows_per_shift - 6))
        start -= phase_min // step_min
        lo, hi = max(start, 0), min(start + 4, n)
        if lo < hi:
            regime[lo:hi] = 0
    _ = shift_kind  # kept for readability of the schedule model
    return regime


def _ewm(x: np.ndarray, alpha: float, init: float | None = None) -> np.ndarray:
    """First-order lag y[t] = (1-alpha)*y[t-1] + alpha*x[t] (pandas ewm, adjust=False)."""
    if init is None:
        return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    y = pd.Series(np.concatenate([[init], x])).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return y[1:]


def _lag_alpha(step_min: int, tau_min: float) -> float:
    return float(1.0 - np.exp(-step_min / tau_min))


# --------------------------------------------------------------------------------------
# Per-asset sensor physics
# --------------------------------------------------------------------------------------


def _simulate_asset(
    asset_id: str,
    ts: pd.DatetimeIndex,
    step_min: int,
    rng: np.random.Generator,
    rated_kw: float,
) -> pd.DataFrame:
    """Simulate one asset's sensor history.

    Inputs: asset id, tz-aware timestamp index, step in minutes, seeded Generator, rated kW.
    Output: DataFrame with TELEMETRY_COLUMNS for that asset.
    """
    n = len(ts)
    phase_min = int(rng.integers(0, 180))
    regime = _regime_schedule(n, step_min, phase_min, rng)

    # Per-asset "personality" offsets so motors are distinguishable.
    vib_factor = float(rng.uniform(0.85, 1.2))
    rpm_offset = float(rng.uniform(-6, 6))
    load_offset = float(rng.uniform(-4, 4))
    temp_offset = float(rng.uniform(-3, 3))
    kurt_base = float(rng.uniform(2.85, 3.15))
    crest_base = float(rng.uniform(2.95, 3.35))

    # Slow in-regime wander (AR(1) via EWM of white noise).
    wander = _ewm(rng.normal(0, 1, n), alpha=0.08) * 3.0

    load = REGIME_LOAD[regime] + load_offset + rng.normal(0, REGIME_LOAD_SD[regime]) + wander
    load = np.clip(load, 0.0, 100.0)
    load_dev = load - REGIME_LOAD[regime]  # deviation from the regime's nominal load

    rpm = REGIME_RPM[regime] + rpm_offset + rng.normal(0, REGIME_RPM_SD[regime]) - 0.12 * load_dev
    rpm = np.clip(rpm, 0.0, None)

    # Rated current for a 400 V motor at ~0.88 pf and ~94 % efficiency.
    rated_a = rated_kw * 1000.0 / (np.sqrt(3) * 400.0 * 0.88 * 0.94)
    current = rated_a * (0.10 + 0.90 * load / 100.0) + rng.normal(0, 0.012 * rated_a, n)

    vib_wander = _ewm(rng.normal(0, 1, n), alpha=0.15) * 0.15
    vib = REGIME_VIB[regime] * vib_factor + 0.006 * load_dev + vib_wander + rng.normal(0, 0.07, n)
    kurt = kurt_base + rng.normal(0, 0.18, n) + np.where(regime == 0, 0.25, 0.0)
    crest = crest_base + 0.06 * regime + rng.normal(0, 0.14, n)

    hour = (ts.hour + ts.minute / 60.0).to_numpy()
    ambient = 21.0 + 3.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    # Healthy: motor housing ~ 60 C and drive-end bearing ~ 66 C at 85 % load, 21 C ambient.
    motor_target = ambient + 0.46 * load + temp_offset
    bearing_target = ambient + 0.50 * load + 4.0 + temp_offset

    # --- drive-end bearing degradation -------------------------------------------------
    health = np.ones(n)
    failure_within_h = np.full(n, np.nan)
    if asset_id in FAULTY_ASSETS:
        amp = FAULT_AMPLITUDE.get(asset_id, 1.0)
        onset = int(round(FAULTY_ASSETS[asset_id] * (n - 1)))
        idx = np.arange(n)
        progress = np.clip((idx - onset) / max(n - 1 - onset, 1), 0.0, 1.0)
        severity = progress**2.0  # smooth, monotone, accelerating
        active = idx >= onset
        vib = (
            vib + amp * severity * (5.5 + 0.4 * load / 55.0) + rng.normal(0, 1, n) * 0.35 * severity
        )
        kurt = kurt + amp * 4.5 * severity + np.abs(rng.normal(0, 1, n)) * 1.5 * severity
        crest = crest + amp * 2.2 * severity + rng.normal(0, 1, n) * 0.25 * severity
        bearing_target = bearing_target + amp * 30.0 * severity
        current = current + amp * 0.03 * rated_a * severity
        health = np.clip(1.0 - severity, 0.0, 1.0)
        hours_left = (n - 1 - idx) * step_min / 60.0
        failure_within_h = np.where(active, hours_left, np.nan)

    motor_temp = _ewm(motor_target, _lag_alpha(step_min, 35.0), init=float(motor_target[0]))
    bearing_temp = _ewm(bearing_target, _lag_alpha(step_min, 25.0), init=float(bearing_target[0]))
    motor_temp = motor_temp + rng.normal(0, 0.25, n)
    bearing_temp = bearing_temp + rng.normal(0, 0.3, n)

    return pd.DataFrame(
        {
            "asset_id": asset_id,
            "ts": ts,
            "vibration_rms": np.round(np.clip(vib, 0.05, None), 4),
            "vibration_kurtosis": np.round(np.clip(kurt, 1.5, None), 4),
            "vibration_crest": np.round(np.clip(crest, 1.4, None), 4),
            "bearing_temp_c": np.round(bearing_temp, 3),
            "motor_temp_c": np.round(motor_temp, 3),
            "current_a": np.round(np.clip(current, 0.0, None), 3),
            "rpm": np.round(rpm, 2),
            "load_pct": np.round(load, 3),
            "regime": REGIME_LABELS[regime],
            "health": np.round(health, 5),
            "failure_within_h": np.round(failure_within_h, 4),
        }
    )


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def generate_fleet(
    days: int = 30,
    seed: int = 42,
    step_min: int = 10,
    start: datetime = DEFAULT_START,
) -> tuple[pd.DataFrame, list[Asset]]:
    """Generate the whole fleet's telemetry.

    Inputs: horizon in days, RNG seed, sampling step in minutes, tz-aware UTC start.
    Output: (telemetry DataFrame in the Telemetry shape, sorted by (asset_id, ts);
    list[Asset]). `health`/`failure_within_h` are ground truth; healthy assets have
    health == 1.0 and failure_within_h == NaN. Deterministic under `seed`.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    n = int(days * 24 * 60 / step_min)
    ts = pd.date_range(start, periods=n, freq=f"{step_min}min", tz="UTC")
    assets = build_assets()
    kw = {a.asset_id: a.rated_kw for a in assets}
    frames: list[pd.DataFrame] = []
    for k, asset_id in enumerate(ASSET_IDS):
        rng = np.random.default_rng([seed, k])
        frames.append(_simulate_asset(asset_id, ts, step_min, rng, kw[asset_id]))
    df = pd.concat(frames, ignore_index=True)
    df = df[TELEMETRY_COLUMNS].sort_values(["asset_id", "ts"], kind="mergesort")
    return df.reset_index(drop=True), assets


def write_outputs(df: pd.DataFrame, assets: list[Asset], out_dir: Path) -> None:
    """Write telemetry.parquet, telemetry.csv and assets.json into `out_dir` (created)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "telemetry.parquet", index=False)
    df.to_csv(out_dir / "telemetry.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    (out_dir / "assets.json").write_text(
        json.dumps([a.model_dump() for a in assets], indent=2), encoding="utf-8"
    )


def df_to_rows(df: pd.DataFrame) -> list[TelemetryRow]:
    """Convert a Telemetry-shaped DataFrame to TelemetryRow models (NaN -> None)."""
    records = df.to_dict("records")
    out: list[TelemetryRow] = []
    for rec in records:
        for key in ("health", "failure_within_h", "regime"):
            v = rec.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[key] = None
        out.append(TelemetryRow(**rec))
    return out


def main() -> None:
    """CLI entrypoint: python -m data.simulator.generate [--days 30 --out data/simulator/out]."""
    parser = argparse.ArgumentParser(description="Generate Thor synthetic fleet telemetry")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-min", type=int, default=10)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "out", help="output directory"
    )
    args = parser.parse_args()
    df, assets = generate_fleet(days=args.days, seed=args.seed, step_min=args.step_min)
    write_outputs(df, assets, args.out)
    faulty = ", ".join(f"{a}@{int(f * 100)}%" for a, f in FAULTY_ASSETS.items())
    print(
        f"wrote {len(df)} rows x {len(df.columns)} cols for {df['asset_id'].nunique()} assets "
        f"({args.days} days @ {args.step_min} min) to {args.out}; faulty: {faulty}"
    )
    horizon_end = df["ts"].max()
    print(f"horizon: {df['ts'].min().isoformat()} -> {horizon_end.isoformat()}")


if __name__ == "__main__":
    main()
