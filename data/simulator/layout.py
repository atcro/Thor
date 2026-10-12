"""Plant layout: which motor drives which machine on which line (asset metadata only).

The 24 motors of the synthetic fleet are grouped into five packaging lines of four motors
(infeed conveyor, filler, capper/labeler, case packer) plus a Compressor Hall of four
air-compressor drives. This module is the ONE place that defines the mapping and the
reasoning behind it. It changes names, lines and roles only: `rated_kw` and `criticality`
stay exactly what they were per asset id (the generator uses `rated_kw` for the current
signal), the telemetry generator and `FAULTY_ASSETS` are untouched, and MTR-042 remains the
live demo fault.

Why these motors are on these lines
-----------------------------------
Motors were ranked by their *measured* duty in the generated history (30 days, seed 42;
reproduce with ``python -m data.simulator.layout``). Every motor runs the same three-shift
regime schedule, so mean load and the number of starts are statistically flat across the
fleet (load 47-59 %, 88-105 starts in 30 days). What does separate motors is bearing
temperature and vibration over the history the plant actually lived through, and that
separation is driven by the degradation events: the seven motors with a drive-end bearing
fault in their history (`FAULTY_ASSETS`) ran hotter and rougher than the rest.

Lines 3-4 therefore hold those seven motors plus MTR-018, the healthy motor with the
highest baseline duty (highest healthy mean load and vibration). Lines 1-2 hold the eight
healthy motors with the lowest duty. Line 5 and the Compressor Hall take the eight healthy
motors in between; the Compressor Hall gets the four with the highest rated power among
them, as air-compressor drives would. Within a packaging line the roles follow rated power:
filler (largest) > case packer > capper/labeler > infeed conveyor (smallest).

Measured over the full history (``duty_profile``), per group of motors:

    group          bearing_temp mean / max   vibration_rms mean / max   load   starts
    Lines 1-2      50.8 C / 72.9 C           2.07 / 3.13 mm/s           52.2   92
    Lines 3-4      53.4 C / 91.2 C           2.66 / 8.18 mm/s           53.4   97
    Line 5         52.6 C / 74.0 C           2.32 / 3.43 mm/s           52.7   100
    Compressor     52.6 C / 72.8 C           2.17 / 3.16 mm/s           54.6   98

So "why do Lines 3-4 struggle while 1-2 run well?" has an answer that is true in the data:
their motors have the failure history, and over that history they ran 2.5 C hotter on
average (18 C hotter at the worst sample) and vibrated 28 % more (2.6x at the worst sample),
at essentially the same load and start count. Nothing here is tuned to make that so; the
line assignment follows the measurement.
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


#: asset_id -> Slot. Rated power (kW) per motor is fixed by asset id in generate.py and
#: decides the role inside a line (filler largest ... infeed conveyor smallest).
PLANT_LAYOUT: dict[str, Slot] = {
    # Lines 1-2: the eight healthy motors with the lowest measured duty.
    **_packaging(1, infeed="MTR-007", filler="MTR-001", capper="MTR-022", packer="MTR-014"),
    **_packaging(2, infeed="MTR-024", filler="MTR-011", capper="MTR-005", packer="MTR-006"),
    # Lines 3-4: the seven motors with a bearing-fault history (MTR-042 live demo, MTR-021
    # fails live, MTR-017 subtle, MTR-003/009/013/023 completed historical failures) plus
    # MTR-018, the healthy motor with the highest baseline duty.
    **_packaging(3, infeed="MTR-013", filler="MTR-042", capper="MTR-021", packer="MTR-009"),
    **_packaging(4, infeed="MTR-018", filler="MTR-003", capper="MTR-017", packer="MTR-023"),
    # Line 5: four mid-duty healthy motors.
    **_packaging(5, infeed="MTR-016", filler="MTR-020", capper="MTR-008", packer="MTR-015"),
    # Compressor Hall: the four highest-rated motors among the remaining healthy ones.
    "MTR-012": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-1 drive motor"),
    "MTR-010": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-2 drive motor"),
    "MTR-019": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-3 drive motor"),
    "MTR-002": Slot(COMPRESSOR_HALL, COMPRESSOR_ROLE, "Air compressor K-4 drive motor"),
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


def duty_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-motor duty statistics over a telemetry DataFrame (the Telemetry shape).

    Input: telemetry with asset_id, ts, rpm, load_pct, bearing_temp_c, vibration_rms.
    Output: DataFrame indexed by asset_id with `line`, `role`, `load_mean`, `starts`
    (rpm crossings from below 900 to at or above 900), `bearing_temp_mean`,
    `bearing_temp_max`, `vibration_mean`, `vibration_max`, sorted by line then role order.
    Only observable sensor columns are used; ground truth (`health`, `failure_within_h`)
    is never read.
    """
    rows = []
    for aid, d in df.sort_values(["asset_id", "ts"]).groupby("asset_id"):
        rpm = d["rpm"]
        slot = PLANT_LAYOUT.get(aid)
        rows.append(
            {
                "asset_id": aid,
                "line": slot.line if slot else "",
                "role": slot.role if slot else "",
                "load_mean": float(d["load_pct"].mean()),
                "starts": int(((rpm.shift(1) < 900) & (rpm >= 900)).sum()),
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
    """Print the duty profile that justifies the layout: python -m data.simulator.layout."""
    parser = argparse.ArgumentParser(description="Thor plant layout duty profile")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from data.simulator.generate import generate_fleet

    df, _ = generate_fleet(days=args.days, seed=args.seed)
    prof = duty_profile(df)
    pd.set_option("display.width", 200)
    print(prof.round(2).to_string())
    groups = {
        "Lines 1-2": [a for a, s in PLANT_LAYOUT.items() if s.line in PACKAGING_LINES[:2]],
        "Lines 3-4": [a for a, s in PLANT_LAYOUT.items() if s.line in PACKAGING_LINES[2:4]],
        "Line 5": [a for a, s in PLANT_LAYOUT.items() if s.line == PACKAGING_LINES[4]],
        "Compressor": [a for a, s in PLANT_LAYOUT.items() if s.line == COMPRESSOR_HALL],
    }
    print()
    for name, ids in groups.items():
        g = prof.loc[ids]
        print(
            f"{name:<11} bearing {g.bearing_temp_mean.mean():.1f}/{g.bearing_temp_max.mean():.1f} C  "
            f"vibration {g.vibration_mean.mean():.2f}/{g.vibration_max.mean():.2f} mm/s  "
            f"load {g.load_mean.mean():.1f} %  starts {g.starts.mean():.0f}"
        )


if __name__ == "__main__":
    main()
