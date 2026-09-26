"""Carve the rotating-equipment unplanned-maintenance slice out of the raw FMUCD export.

Input:  data/external/fmucd.csv  (Facility Management Unified Classification Database, 1.4 GB,
        CC BY-NC, https://data.mendeley.com/datasets/cb8d2nsjss/1 -- not committed)
Output: data/field_history/fmucd_rotating_upm.csv (committed; a few MB)

Kept rows: PPM/UPM == UPM, component names a fan / pump / motor / compressor / chiller /
air handler / drive, and the free text names a symptom (bearing, vibration, noise, seized,
belt, overheat, motor ...). Exact (university, WOID) repeats and identical
(component, description) pairs are dropped. Names after "per"/"attn"/"contact", e-mail
addresses and phone numbers are scrubbed from the description.

Run: python data/field_history/build_fmucd_slice.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "fmucd.csv"
OUT = ROOT / "data" / "field_history" / "fmucd_rotating_upm.csv"

COMPONENT_RE = re.compile(r"fan|pump|motor|compressor|chiller|air handler|drive", re.I)
SYMPTOM_RE = re.compile(
    r"\b(?:bearing|vibrat\w*|noise|noisy|grind\w*|squeal\w*|belt|overheat\w*|seized|seize|"
    r"burnt|burned|motor|hums?|humming|smell|smoke|tripp?\w*|amps?|overcurrent|current)\b",
    re.I,
)
SCRUB = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), ""),
    (re.compile(r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), ""),
    (
        re.compile(
            r"\b(per|attn?|contact|call|ask for|see)\b\s*[:.-]?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
            re.I,
        ),
        "",
    ),
    (re.compile(r"\s{2,}"), " "),
]
COLS = [
    "UniversityID",
    "WOID",
    "ComponentDescription",
    "WODescription",
    "WOStartDate",
    "PPM/UPM",
    "LaborHours",
    "TotalCost",
]


def scrub(text: str) -> str:
    for pat, rep in SCRUB:
        text = pat.sub(rep, text)
    return text.strip(" -:;,")


def main() -> None:
    parts = []
    reader = pd.read_csv(SRC, usecols=COLS, chunksize=250_000, dtype=str, keep_default_na=False)
    for chunk in reader:
        m = (
            (chunk["PPM/UPM"] == "UPM")
            & chunk["ComponentDescription"].str.contains(COMPONENT_RE)
            & chunk["WODescription"].str.contains(SYMPTOM_RE)
        )
        parts.append(chunk.loc[m])
    df = pd.concat(parts, ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["UniversityID", "WOID"])
    df["description"] = df["WODescription"].map(scrub)
    df["component"] = df["ComponentDescription"].str.strip()
    df = df[df["description"].str.len() >= 12]
    df["_key"] = df["description"].str.lower()
    df = df.drop_duplicates(subset=["component", "_key"])
    out = pd.DataFrame(
        {
            "case_id": "fmucd-" + df["UniversityID"] + "-" + df["WOID"],
            "university": df["UniversityID"].astype(int),
            "component": df["component"],
            "description": df["description"],
            "start_date": df["WOStartDate"].str[:10],
            "labor_hours": pd.to_numeric(df["LaborHours"], errors="coerce"),
            "total_cost": pd.to_numeric(df["TotalCost"], errors="coerce"),
        }
    ).sort_values(["start_date", "case_id"], kind="stable")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, lineterminator="\n")
    mb = OUT.stat().st_size / 1e6
    print(f"matched {n_raw:,} rows -> {len(out):,} unique cases -> {OUT} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
