"""Plant layout: which motor drives which machine on which line (asset metadata only).

The 24 motors of the synthetic fleet are grouped into five packaging lines of four motors
(infeed conveyor, filler, capper/labeler, case packer) plus a Compressor Hall of four
air-compressor drives. This module is the ONE place that defines the mapping and the
reasoning behind it. It changes names, lines and roles only: `rated_kw` and `criticality`
stay exactly what they were per asset id (the generator uses `rated_kw` for the current
signal), the telemetry generator's signals are untouched, and MTR-042 remains the live demo
fault (the fault schedule itself was re-timed on 2026-10-12 so that only MTR-042 degrades
during the replay; see `FAULTY_ASSETS`).

Why these motors are on these lines
-----------------------------------
The rule is written in code (`derive_groups`) and the committed `PLANT_LAYOUT` is tested to
equal what the rule derives from the default 30-day fleet (seed 42). Reproduce the figures
with ``python -m data.simulator.layout``.

1. Every motor gets a **baseline duty index** (`baseline_duty_index`): the equal-weight mean
   of the population z-scores of its mean load, number of starts, mean bearing temperature and
   mean vibration RMS, computed over its *baseline* rows (rows before the known fault onset for
   the motors in `FAULTY_ASSETS`, all rows otherwise), using observable sensors only.
2. **Lines 3-4** hold the seven motors with a drive-end bearing fault in their history
   (`FAULTY_ASSETS`: MTR-042 live demo, MTR-003/009/013/017/021/023 completed historical
   failures) plus the healthy motor with the highest baseline duty index.
3. **Lines 1-2** hold the eight healthy motors with the lowest baseline duty index.
4. Of the remaining eight healthy motors, the **Compressor Hall** takes the four with the
   highest rated power (as air-compressor drives would); **Line 5** takes the other four.
5. Inside a packaging line the roles follow rated power: filler (largest) > case packer >
   capper/labeler > infeed conveyor (smallest); the two lines of a pair are balanced by
   alternating the ranked motors. Ties on rated power keep asset-id order.

Every motor runs the same three-shift regime schedule, so mean load and the number of starts
are statistically flat across the fleet (load 48-57 %, 88-105 starts in 30 days). What
separates motors over the history the plant actually lived through is bearing temperature
and vibration, and that separation is driven by the degradation events. Measured over the
full history (`duty_profile`), per group of motors:

    group          bearing_temp mean / max   vibration_rms mean / max   load   starts
    Lines 1-2      50.6 C / 76.6 C           2.07 / 3.38 mm/s           52.0   95
    Lines 3-4      53.4 C / 104.6 C          2.67 / 10.07 mm/s          53.4   97
    Line 5         52.9 C / 76.1 C           2.30 / 3.60 mm/s           52.1   98
    Compressor     52.7 C / 75.7 C           2.21 / 3.60 mm/s           55.7   94

So "why do Lines 3-4 struggle while 1-2 run well?" has an answer that is true in the data:
their motors have the failure history, and over that history they ran hotter on average
(28 C hotter at the worst sample) and vibrated more (3.0x at the worst sample), at
essentially the same load and start count. Nothing is tuned to make that so; the line
assignment follows the measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

SITE = "Ludvika"

#: Packaging roles in machine order along a line (infeed first, case packer last).
PACKAGING_ROLES: tuple[str, str, str, str] = (
    "infeed conveyor",
    "filler",
    "capper/labeler",
    "case packer",
)
COMPRESSOR_ROLE = "air compressor"
COMPRESSOR_HALL = "Compressor Hall"
PACKAGING_LINES: tuple[str, ...] = tuple(f"Packaging Line {i}" for i in range(1, 6))
LINES: tuple[str, ...] = (*PACKAGING_LINES, COMPRESSOR_HALL)

#: Starts are counted as rpm crossings from below this value to at or above it.
START_RPM = 900.0


@dataclass(frozen=True)
class Slot:
    """One position in the plant: the line, the machine role and the motor's display name."""

    line: str
    role: str
    name: str


def _packaging(line_no: int, infeed: str, filler: str, capper: str, packer: str) -> dict[str, Slot]:
    line = f"Packaging Line {line_no}"
    return {
        infeed: Slot(line, "infeed conveyor", f"Line {line_no} infeed conveyor drive motor"),
        filler: Slot(line, "filler", f"Line {line_no} filler drive motor"),
        capper: Slot(line, "capper/labeler", f"Line {line_no} capper/labeler drive motor"),
        packer: Slot(line, "case packer", f"Line {line_no} case packer drive motor"),
    }


#: asset_id -> Slot. Derived by `derive_groups` from the default fleet (see the module
#: docstring) and committed here so the plant never depends on a generation at runtime;
#: `tests/test_simulator.py` asserts the two agree.
PLANT_LAYOUT: dict[str, Slot] = {
    # Lines 1-2: the eight healthy motors with the lowest baseline duty index.
    **_packaging(1, infeed="MTR-022", filler="MTR-001", capper="MTR-005", packer="MTR-019"),
    **_packaging(2, infeed="MTR-024", filler="MTR-011", capper="MTR-006", packer="MTR-010"),
    # Lines 3-4: the seven fault-history motors + MTR-018 (highest healthy duty index).
    **_packaging(3, infeed="MTR-013", filler="MTR-042", capper="MTR-021", packer="MTR-009"),
    **_packaging(4, infeed="MTR-018", filler="MTR-003", capper="MTR-017", packer="MTR-023"),
    # Line 5: the four lowest-rated of the remaining healthy motors.
    **_packaging(5, infeed="MTR-016", filler="MTR-015", capper="MTR-007", packer="MTR-008"),
    # Compressor Hall: the four highest-rated of the remaining healthy motors.
    "MTR-012": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-1 drive motor"),
    "MTR-002": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-2 drive motor"),
    "MTR-014": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-3 drive motor"),
    "MTR-020": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-4 drive motor"),
}


def line_members(line: str) -> list[str]:
    """Asset ids on `line` in machine order (packaging role order; compressors by K-number).

    Input: a line name from `LINES`. Output: list of asset ids (4 for every line).
    """
    members = [(aid, slot) for aid, slot in PLANT_LAYOUT.items() if slot.line == line]
    if line == COMPRESSOR_HALL:
        return [aid for aid, _ in sorted(members, key=lambda m: m[1].name)]
    order = {r: i for i, r in enumerate(PACKAGING_ROLES)}
    return [aid for aid, _ in sorted(members, key=lambda m: order[m[1].role])]


def _starts(rpm: pd.Series) -> int:
    return int(((rpm.shift(1) < START_RPM) & (rpm >= START_RPM)).sum())


def baseline_duty_index(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline duty of every motor from observable sensors (rule 1 of the module docstring).

    Input: telemetry in the Telemetry shape (asset_id, ts, rpm, load_pct, bearing_temp_c,
    vibration_rms; any other column is ignored). Output: DataFrame indexed by asset_id with
    `load_mean`, `starts`, `bearing_temp_mean`, `vibration_mean` over the motor's baseline
    rows (before the fault onset for `FAULTY_ASSETS`, all rows otherwise) and `duty_index` =
    the mean of the four population z-scores. Ground truth columns are never read; the onset
    comes from the simulator's own fault schedule.
    """
    from data.simulator.generate import FAULTY_ASSETS

    rows = []
    for aid, d in df.sort_values(["asset_id", "ts"]).groupby("asset_id"):
        if aid in FAULTY_ASSETS:
            onset = int(round(FAULTY_ASSETS[aid][0] * (len(d) - 1)))
            d = d.iloc[:onset]
        rows.append(
            {
                "asset_id": aid,
                "load_mean": float(d["load_pct"].mean()),
                "starts": _starts(d["rpm"]),
                "bearing_temp_mean": float(d["bearing_temp_c"].mean()),
                "vibration_mean": float(d["vibration_rms"].mean()),
            }
        )
    out = pd.DataFrame(rows).set_index("asset_id")
    cols = ["load_mean", "starts", "bearing_temp_mean", "vibration_mean"]
    z = (out[cols] - out[cols].mean()) / out[cols].std(ddof=0)
    out["duty_index"] = z.mean(axis=1)
    return out


def derive_groups(df: pd.DataFrame, rated_kw: dict[str, float]) -> dict[str, set[str]]:
    """Apply rules 2-4 of the module docstring to a fleet.

    Inputs: telemetry (Telemetry shape) and rated kW per asset id. Output: {"lines_3_4",
    "lines_1_2", "compressor_hall", "line_5"} -> set of asset ids. Deterministic.
    """
    from data.simulator.generate import FAULTY_ASSETS

    duty = baseline_duty_index(df)["duty_index"]
    healthy = duty.drop(index=[a for a in FAULTY_ASSETS if a in duty.index]).sort_values()
    hardest_healthy = str(healthy.index[-1])
    lines_3_4 = set(FAULTY_ASSETS) | {hardest_healthy}
    lines_1_2 = {str(a) for a in healthy.index[:8]}
    rest = [str(a) for a in healthy.index if a not in lines_1_2 and a != hardest_healthy]
    rest.sort(key=lambda a: (-rated_kw[a], a))
    return {
        "lines_3_4": lines_3_4,
        "lines_1_2": lines_1_2,
        "compressor_hall": set(rest[:4]),
        "line_5": set(rest[4:]),
    }


def duty_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-motor duty statistics over the FULL telemetry history (what the plant lived).

    Input: telemetry with asset_id, ts, rpm, load_pct, bearing_temp_c, vibration_rms.
    Output: DataFrame indexed by asset_id with `line`, `role`, `load_mean`, `starts`,
    `bearing_temp_mean`, `bearing_temp_max`, `vibration_mean`, `vibration_max`, sorted by
    line then role order. Only observable sensor columns are used; ground truth (`health`,
    `failure_within_h`) is never read.
    """
    rows = []
    for aid, d in df.sort_values(["asset_id", "ts"]).groupby("asset_id"):
        slot = PLANT_LAYOUT.get(aid)
        rows.append(
            {
                "asset_id": aid,
                "line": slot.line if slot else "",
                "role": slot.role if slot else "",
                "load_mean": float(d["load_pct"].mean()),
                "starts": _starts(d["rpm"]),
                "bearing_temp_mean": float(d["bearing_temp_c"].mean()),
                "bearing_temp_max": float(d["bearing_temp_c"].max()),
                "vibration_mean": float(d["vibration_rms"].mean()),
                "vibration_max": float(d["vibration_rms"].max()),
            }
        )
    out = pd.DataFrame(rows).set_index("asset_id")
    role_order = {r: i for i, r in enumerate((*PACKAGING_ROLES, COMPRESSOR_ROLE))}
    line_order = {ln: i for i, ln in enumerate(LINES)}
    out["_l"] = out["line"].map(line_order)
    out["_r"] = out["role"].map(role_order)
    return out.sort_values(["_l", "_r"]).drop(columns=["_l", "_r"])


def main() -> None:
    """Print the duty index, the derived groups and the full-history profile behind the layout:
    python -m data.simulator.layout [--days 30 --seed 42]."""
    parser = argparse.ArgumentParser(description="Thor plant layout duty profile")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from data.simulator.generate import generate_fleet

    df, assets = generate_fleet(days=args.days, seed=args.seed)
    pd.set_option("display.width", 200)
    print("Baseline duty index (rule 1):")
    print(baseline_duty_index(df).sort_values("duty_index", ascending=False).round(3).to_string())
    groups = derive_groups(df, {a.asset_id: a.rated_kw for a in assets})
    print("\nDerived groups (rules 2-4):")
    for name, ids in groups.items():
        committed = {a for a, s in PLANT_LAYOUT.items() if s.line in _GROUP_LINES[name]}
        print(f"  {name:<15} {sorted(ids)}  committed layout agrees: {ids == committed}")
    prof = duty_profile(df)
    print("\nFull-history profile per motor:")
    print(prof.round(2).to_string())
    print()
    for name, lines in _GROUP_LINES.items():
        g = prof[prof["line"].isin(lines)]
        print(
            f"{name:<15} bearing {g.bearing_temp_mean.mean():.1f}/{g.bearing_temp_max.mean():.1f} C  "
            f"vibration {g.vibration_mean.mean():.2f}/{g.vibration_max.mean():.2f} mm/s  "
            f"load {g.load_mean.mean():.1f} %  starts {g.starts.mean():.0f}"
        )


_GROUP_LINES: dict[str, tuple[str, ...]] = {
    "lines_1_2": PACKAGING_LINES[:2],
    "lines_3_4": PACKAGING_LINES[2:4],
    "line_5": (PACKAGING_LINES[4],),
    "compressor_hall": (COMPRESSOR_HALL,),
}


if __name__ == "__main__":
    main()
