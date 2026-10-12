# Thor metrics: the leak-free run on the re-timed data

Every number in the right-hand column comes from one real run recorded in
`docs/finals/prep-report.json` (training run `15e68a9abab84d63`, model `v82031`, 2026-10-12,
`scripts/demo_prep.py --n-trials 4` on the re-timed fault
schedule: six completed failures inside the seeded 80 %, MTR-042 healthy until after the seed
point). The middle column is the previous leak-free run on the old schedule (four completed
failures, 2026-10-11), kept so the slides can explain why the numbers moved; the left column is
the prototype. **Never carry the older numbers forward as current.**

| Metric | Prototype (before 2026-10-11) | Previous run (v57257, four failures) | Current run (v82031, six failures) |
|---|---|---|---|
| Champion | LightGBM | LightGBM | Random forest |
| Recall | 91.5 % | 86.5 % | **91.9 %** (test motors, P ≥ 0.5, 578 positive rows) |
| Precision | not reported honestly | 100 % | **96.2 %** (18,364 negative test rows) |
| F1 | – | 92.8 % | 94.0 % |
| AUROC | – | 0.933 | 1.000 |
| Warning lead time (median) | 43.7 h | 41.5 h (1 event) | **53.1 h** (2 completed failures among the test motors, p10 47.7 h / p90 58.5 h, alarm = 3 rows ≥ 0.3) |
| Industrial Model Score | 0.858 | 0.844 | **0.882** |
| Brier score (calibrated) | – | 0.0021 | 0.0039 on test; 0.00779 → 0.00547 on the calibration motors |
| Calibration | isotonic, same rows | Platt on log-odds, 6 shared motors | sigmoid on the probability, fitted on 3 calibration motors never used for training, selection or the final score |
| Validation data | last 25 % of ids | 12 / 6 / 6 (shared) / 6 | fixed seeded split: 12 train / 3 validation / 3 calibration / 6 test motors, four separate groups |
| Operating states and baselines | fitted on all 24 motors before the split | fitted on the 12 training motors only | same |
| Thresholds | 0.3 and 0.5 mixed | one meaning each | same: 0.5 classification, 0.3 early warning (3 sustained rows) |
| Edge vs centre | calibration dropped at the edge | max |Δp| 0.0 | max probability difference 1.6e-06 on 200 held-out vectors, feature difference 1.6e-15 on 18 raw windows |
| Training time | 60–100 s | 55–70 s (6 trials) | 171 s for 4 trials × 3 families natively (validation + parity ≈ 35 s of it) |
| Displayed probability | newest sample | median of the last 3 samples | same; status is the latched alarm (`apps/api/alarms.py`) |

## What changed and why

The fault schedule was re-timed on 2026-10-12 (team-approved) so that only MTR-042 degrades
during the replay: MTR-017 and MTR-021 now complete their failures inside the seeded history,
which gives the model six completed failures instead of four. With five or more,
`make_asset_split` forms four separate groups (validation and calibration no longer share
motors) and the test group holds two failure events (003, 013), so the lead time is a
median of two measurements rather than one. The champion is chosen on the validation motors
(002, 009, 014, failure 009), calibrated on the calibration
motors (008, 020, 023, failure 023) and reported on six test
motors (001, 003, 011, 013, 018, 042) the pipeline never touched. The numbers move between runs because
Optuna's search differs; the record is the committed report. Nothing was tuned for nicer
numbers: the first prep run on this schedule produced a LightGBM champion whose raw scores
reached −1.7e6 on pure leaves, which collapsed Platt scaling to a flat line (test recall 0.33)
and failed edge parity; the fix was to calibrate every family on the bounded probability
(`agents/validation/checks.py::_ProbabilityOnly`), not to pick a better run.

Caveats to say out loud: a synthetic fleet, 24 motors, six completed failures, two failure
events among the test motors, 10-minute aggregates (no spectra).

## Where the numbers come from

- Split, candidates, validation metrics, gate and parity: `docs/finals/prep-report.json`
  (`training.*`), written by `scripts/demo_prep.py` from `GET /pipeline/{run_id}`.
- Test metrics: `ValidationReport.test_metrics` in `agents/validation/checks.py::validate`
  (calibrated champion on `split.test` rows with a known label).
- Lead time: `agents/metrics.py::lead_times_h` (first of three consecutive rows with
  P ≥ 0.3 to the failure instant) — the same sustained rule the live alarm uses.
- Parity: `agents/mlops/lifecycle.py::check_edge_parity` (ONNX + sidecar through
  `edge.inference` vs the calibrated joblib).

## Candidates on the validation motors (what Optuna saw)

| Family | Validation IMS | Recall | Precision | Lead time | Latency (ms / 1000 rows) |
|---|---|---|---|---|---|
| Random forest (champion) | 0.933 | 1.000 | 0.707 | 72.2 h | 8.11 |
| XGBoost | 0.931 | 1.000 | 0.654 | 80.7 h | 3.75 |
| LightGBM | 0.929 | 1.000 | 0.628 | 237.0 h | 2.46 |

The champion is picked on these validation numbers; the headline numbers above are its
*test* numbers.

## AI4I 2020 benchmark

Rerun 2026-10-11 with the corrected methodology (6 trials, seed 0): champion lightgbm,
recall 0.784, precision 0.896, F1 0.836, AUROC 0.974,
Brier 0.0109 → 0.0095 on 2502 held-out rows (3.4 % positives).
Full table and caveats: `docs/finals/model-card.md` section 13 and `data/benchmark/README.md`.
