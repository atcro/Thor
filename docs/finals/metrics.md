# Thor metrics: prototype numbers vs the leak-free run

Every number in the right-hand column comes from one real run recorded in
`docs/finals/prep-report.json` (training run `a7d611188ec64383`, model `v57257`, 2026-10-11,
`scripts/demo_prep.py --n-trials 6`). The left-hand column is what the prototype reported before
the correctness pass. **Never carry the prototype numbers forward as current.**

| Metric | Prototype (before 2026-10-11) | Leak-free run (current) |
|---|---|---|
| Champion | LightGBM | LightGBM `v57257` |
| Recall | 91.5 % | **86.5 %** (test assets, P ≥ 0.5, 289 positive rows) |
| Precision | not reported honestly | **100 %** (no false alarm on 18,653 negative test rows) |
| F1 | – | 92.8 % |
| AUROC | – | 0.933 |
| Warning lead time (median) | 43.7 h | **41.5 h** (1 completed failure among test assets, alarm = 3 rows ≥ 0.3) |
| Industrial Model Score | 0.858 | **0.844** |
| Brier score (calibrated) | – | 0.0021 on test; 0.00095 → 0.00044 on the calibration assets |
| Calibration | isotonic, fitted on the same rows it was scored on | Platt (sigmoid), fitted on 6 calibration motors never used for training or the final score; isotonic only with ≥ 2 failure events there |
| Validation data | last 25 % of asset ids; same rows for selection, calibration and lead time | fixed seeded split: 12 train / 6 validation / 6 calibration / 6 test motors |
| Operating states and baselines | fitted on all 24 motors before the split | fitted on the 12 training motors only; others assigned |
| Thresholds | 0.3 and 0.5 mixed across stages | one meaning each: 0.5 classification, 0.3 early warning (3 sustained rows) |
| Edge vs centre | calibration dropped at the edge, baseline format mismatch | max probability difference 0.0 on 200 held-out vectors, feature difference 1.6e-15 on 18 raw windows |
| Training time (6 trials × 3 families) | 60–100 s | 55–70 s |
| Displayed probability | newest sample | median of the last 3 samples (the alarm's sustained-rows rule) |

## What changed and why

The prototype's numbers were optimistic because the data preparation leaked: operating states
and the healthy baselines were fitted on every motor before the split, the Optuna objective,
the calibration and the lead-time measurement all used the same held-out rows, and the search
ranked models on recall at 0.3 while the gate checked recall at 0.5. The leak-free run fixes the
asset-level split first (seed 0, recorded with the run), fits every statistic on the training
motors only, selects the champion on validation motors, calibrates on calibration motors and
reports the final metrics from six test motors the pipeline had never touched. Recall drops
from 91.5 % to 86.5 % and lead time from 43.7 h to 41.5 h; precision is 100 % and the Brier
score is 0.002 because the synthetic fault signature is strong. These are the numbers to quote
on stage, with their caveats: a synthetic fleet, 24 motors, only four completed failures, and a
single failure event among the test motors, so the lead time is one measurement, not a
distribution.

## Where the numbers come from

- Split, candidates, validation metrics, gate and parity: `docs/finals/prep-report.json`
  (`training.*`), written by `scripts/demo_prep.py` from `GET /pipeline/{run_id}`.
- Test metrics: `ValidationReport.test_metrics` in `agents/validation/checks.py::validate`
  (calibrated champion on `split.test` rows with a known label).
- Lead time: `agents/metrics.py::lead_times_h` (first of three consecutive rows with
  P ≥ 0.3 to the failure instant).
- Parity: `agents/mlops/lifecycle.py::check_edge_parity` (ONNX + sidecar through
  `edge.inference` vs the calibrated joblib).

## Candidates on the validation motors (what Optuna saw)

| Family | Validation IMS | Recall | Precision | Lead time | Latency (ms / 1000 rows) |
|---|---|---|---|---|---|
| LightGBM (champion) | 0.950 | 1.000 | 0.932 | 61.3 h | 0.98 |
| Random forest | 0.926 | 0.969 | 0.979 | 56.0 h | 2.08 |
| XGBoost | 0.916 | 1.000 | 0.929 | 51.7 h | 0.99 |

The champion is picked on these validation numbers; the headline numbers above are its
*test* numbers, which is why they are lower.

## AI4I 2020 benchmark

Rerun 2026-10-11 with the corrected methodology (6 trials, seed 0): champion lightgbm,
recall 0.784, precision 0.896, F1 0.836, AUROC 0.974,
Brier 0.0109 → 0.0095 on 2502 held-out rows (3.4 % positives).
Full table and caveats: `docs/finals/model-card.md` section 13 and `data/benchmark/README.md`.
