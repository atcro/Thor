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

Fill in from `results.json` after running (calibrated champion, threshold 0.5, grouped test set):

| metric | value |
|---|---|
| champion family | _fill in_ |
| recall | _fill in_ |
| precision | _fill in_ |
| F1 | _fill in_ |
| AUROC | _fill in_ |
| Brier before / after calibration | _fill in_ |
| n_train / n_test | _fill in_ |
