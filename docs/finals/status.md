# Phase 0 — correctness pass: progress log

Session started 2026-10-11 (Claude Fable 5.1). Branch `finals`, baseline commit `1c4475f`.
One entry per milestone: time (UTC), phase/milestone, files, tests actually run with their
result, and the specific next step. If this session stops, resume from the last "next" line.

Fix order (from `docs/finals/prompt-0-correctness.md`): governance (1, 2) → ground truth (3) →
thresholds (6) → leakage (4) → edge parity (5) → analysis reuse (7) → demo prep → training
substance (model card, AutoML Studio view, AI4I rerun, Bolt eval case).

---

## 2026-10-11 — start: orientation and baseline

- **Phase:** 0 / orientation.
- **Done:** read CLAUDE.md §7 and §15, the brief, and every file named in the seven defects
  (`orchestrator.py`, `routes/approvals.py`, `routes/fleet.py`, `lifecycle.py`, `checks.py`,
  `automl.py`, `features.py`, `profiling.py`, `edge/*`, `replay.py`, `cost.py`, tests).
  All seven defects re-verified against the code at `1c4475f`.
- **Files changed:** `docs/finals/status.md` (this file).
- **Tests:** full `pytest` baseline running in the background (result recorded in the next entry).
- **Next:** fix 1+2 (governance): split maintenance approval from model promotion in
  `orchestrator._finalize`, open promotion requests only for passed validation, gate
  `lifecycle.request_promotion`/`promote` on `validation_passed`, fix `reject_promotion` call in
  `routes/approvals.py`; add tests; commit.
