# data/field_history -- real unplanned work orders as Bolt precedent

`fmucd_rotating_upm.csv` is a committed slice (about 8,400 rows, 1.2 MB) of the
**Facility Management Unified Classification Database (FMUCD)**: 3.7 million maintenance work
orders from twelve North American universities, 2002-2021.

- Source: https://data.mendeley.com/datasets/cb8d2nsjss/1
- Paper: "Comprehensive maintenance dataset of building facilities for planned preventive and
  unplanned maintenance in North American universities", Data in Brief (2024),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11465139/
- License: **CC BY-NC 4.0** -- attribution required, non-commercial use only. This slice is
  redistributed under the same license; keep this notice with it.

## What is in the slice

Unplanned (UPM) work orders whose component is a fan, pump, motor, compressor, chiller, air
handler or drive **and** whose free text names a symptom (bearing, vibration, noise, seized,
belt, overheat, hums, tripping, overcurrent ...). Exact repeats are dropped; e-mail addresses,
phone numbers and "per / attn <Name>" clauses are scrubbed. Columns:

| column | meaning |
|---|---|
| `case_id` | `fmucd-<university>-<WOID>` |
| `university` | anonymous 1-12 id from the source |
| `component` | FMUCO component label (e.g. `Exhaust Fan`, `Circulation Pump, Hot Water`) |
| `description` | the technician / requester text, as written |
| `start_date` | YYYY-MM-DD |
| `labor_hours` | may be empty |
| `total_cost` | may be empty (missing for ~1/3 of rows; campus facilities, not industrial) |

## How Thor uses it

At API startup `apps/api/main.py` indexes the slice into a **second** Chroma collection,
`field_history`, next to the `manuals` collection. Bolt's `search_field_history` tool
retrieves the closest cases for a symptom and returns them with medians of labor hours and
cost computed in Python. It is precedent only:

- it is never merged into the manual index, so `search_manuals` and the Decision Contract
  explanation still cite plant manuals exclusively (CLAUDE.md section 7);
- Bolt presents cases as "field history: <component>, <year>" from a public campus dataset,
  never as guidance for this plant;
- the pipeline (profile, train, validate, explain, cost) does not read it at all.

## Rebuilding

Download the raw CSV to `data/external/fmucd.csv` (git-ignored, see `data/external/README.md`),
then:

    python data/field_history/build_fmucd_slice.py

The script is deterministic; rerunning it on the same source reproduces the committed file.
