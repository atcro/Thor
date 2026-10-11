# AI4I 2020 credibility benchmark

Thor's primary demo runs on a synthetic regime-based fleet (`data/simulator/`). That is the
right choice for a reproducible "one motor, one fault, one decision" story, but it invites a
fair question: does the methodology only work because the data was written to make it work?

This directory answers that by running the **same model-selection and validation discipline**
on a real public dataset, the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(UCI, 10,000 rows, ~3.4% failures), and reporting honest held-out numbers.

**Honesty note (CLAUDE.md section 9).** This benchmark proves the methodology is not
cherry-picked to the synthetic story. It does *not* claim the synthetic demo's numbers
transfer to AI4I, and it is not the demo. Quote AI4I numbers as AI4I numbers.

## Run it

```bash
python -m data.benchmark.ai4i_benchmark            # downloads the CSV on first run
python -m data.benchmark.ai4i_benchmark --n-trials 20 --seed 1 --out data/benchmark/results.json
```

The CSV (`ai4i2020.csv`) and `results.json` are git-ignored; regenerate them locally.

## What is reused from the Thor pipeline, verbatim

| Thor component | Used here |
|---|---|
| `agents.data_agent.profiling.detect_regime` | KMeans regimes on (rpm, torque); torque stands in for `load_pct` |
| `agents.ml_architect.automl.industrial_model_score` | Champion ranking, `lead_time_h` pinned to 0 (see below) |
| Grouped holdout (`GroupShuffleSplit` on product buckets) | Whole groups move together; no row-level shuffle |
| `agents.validation.checks.detect_leakage` | Forbidden-name + high-correlation + group-overlap checks |
| `agents.validation.checks.calibrate_probabilities` | Isotonic / sigmoid calibration, Brier before vs after |

## What is NOT comparable to the synthetic demo

- AI4I is tabular with **no time axis**: the rolling-window feature pipeline, warning lead
  time, RUL interval and time-ordered backtest do not apply. The IMS therefore ranks on
  recall / precision / calibration / latency only; every family loses the same lead-time weight.
- AI4I has **no asset identity**: product ids are unique per row. Groups are the last two
  digits of the product number (100 buckets). That guarantees the grouped-split mechanics Thor
  insists on, not machine-level generalisation like the synthetic fleet's asset holdout.
- The five failure-mode flags (TWF/HDF/PWF/OSF/RNF) are post-hoc labels and are **never**
  features; `build_features` asserts this and `results.json` records it under `leakage_checks`.

## Results

### Results (rerun 2026-10-11, `python -m data.benchmark.ai4i_benchmark --n-trials 6 --seed 0`)

Grouped 75/25 split on product-id buckets (no row shuffle; 75 / 25 buckets),
the training buckets further split into fit / tune / calibration groups (4519 / 1490 / 1489 rows),
champion = highest Industrial Model Score on the tune group with the lead-time term pinned to 0
(no time axis), final metrics for the calibrated champion at the classification threshold 0.5
on the held-out buckets. Record: `data/benchmark/results.json` (git-ignored; regenerate with the
command).

| metric | value |
|---|---|
| champion family | lightgbm (isotonic calibration) |
| recall | 0.784 |
| precision | 0.896 |
| F1 | 0.836 |
| AUROC | 0.974 |
| Brier before / after calibration | 0.0109 / 0.0095 |
| positive rate (all / test) | 3.390% / 3.517% |
| n_train / n_test | 7498 / 2502 |

Per family on the tune group (what the search saw):

| family | IMS (no lead-time term) | recall | precision | Brier |
|---|---|---|---|---|
| random_forest | 0.624 | 0.852 | 0.657 | 0.0186 |
| xgboost | 0.646 | 0.852 | 0.730 | 0.0133 |
| lightgbm | 0.652 | 0.815 | 0.863 | 0.0096 |

Leakage checks: failure-mode flags in features = [],
group overlap = 0, `detect_leakage` passed = True.

What differs from the synthetic run and why the numbers are lower: AI4I has no time axis and
no asset identity, so there is no rolling-window feature pipeline, no regime baseline over a
motor's own history, no warning lead time and no asset-level holdout, and its 3.4 % failure
rate mixes five failure modes. Quote AI4I numbers as AI4I numbers: they show the selection and
validation discipline holds on real data, not that the synthetic numbers transfer.
