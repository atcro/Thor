# Model card: `thor-bearing-classifier` v57257

How the demo model was trained, so a judge can trace every number to the code and to one real
run. Source of the numbers: `docs/finals/prep-report.json` (run `a7d611188ec64383`,
2026-10-11, `scripts/demo_prep.py --n-trials 6`). Metrics side by side with the prototype's:
`docs/finals/metrics.md`.

## 1. Task

Binary classification: "will this motor fail within the next 48 h?" from its last 2 h of
sensor readings. P(failure) is calibrated. Two named thresholds, one meaning each
(`apps/api/schemas.py`): **0.5** = predicted failure (recall, precision, F1); **0.3** = early
warning, sustained over 3 consecutive 10-minute samples (warning lead time, and the maintenance
window must end before P exceeds it).

## 2. Data and labels

- Synthetic fleet (`data/simulator/generate.py`): 24 induction motors, 30 days, one sample
  every 10 minutes, eight sensors (vibration RMS / kurtosis / crest, bearing and motor
  temperature, current, rpm, load). Three operating regimes on a shift schedule.
- The control plane seeds the first 80 % of the timeline (82,944 rows); the replay streams
  the remaining 20 % live. The model is trained on the seeded 80 % only.
- **Labels come only from completed failures.** A motor whose countdown reaches the failure
  inside the data is a recorded event; its rows within 48 h of that instant are positive. Four
  motors qualify at 80 %: MTR-003, MTR-009, MTR-013, MTR-023. An in-progress degradation
  (MTR-042, MTR-017, MTR-021 at 80 %) is an unknown future: no label, never a positive.
- **Right-censoring:** the last 48 h of any motor with no failure ahead has an unknown label and
  is excluded from training and evaluation (6,912 of 82,680 feature rows). It is still scored.
- Simulator ground truth (`health`, `failure_within_h`) is used only here, for labels and
  offline evaluation. It never reaches MQTT, the fleet view, Bolt, Decision Contracts or the
  edge (`tests/test_ground_truth_isolation.py`).

## 3. Split (fixed before anything was fitted)

`agents/ml_architect/split.py::make_asset_split`, seed 0, whole motors only, no row shuffle.
The asset under analysis (MTR-042) is forced into the test group so its score is
out-of-sample. Completed failures are dealt train, test, tune in that order.

| Group | Motors | Completed failures |
|---|---|---|
| train (12) | MTR-002, 005, 006, 007, 010, 013, 015, 016, 017, 021, 023, 024 | MTR-013, MTR-023 |
| validation (6) | MTR-008, 009, 011, 014, 019, 020 | MTR-009 |
| calibration (6) | same six motors as validation | MTR-009 |
| test (6) | MTR-001, 003, 012, 018, 022, **042** | MTR-003 |

Validation and calibration share one group because four completed failures cannot give four
groups a positive each; the code splits them when there are at least five. This is recorded in
`AssetSplit.notes` with the run.

## 4. Preprocessing, fitted on the 12 training motors only

- **Regimes:** KMeans (k = 3) on standardised (rpm, load) fitted on training rows; every other
  row is assigned to its nearest centroid (`agents/data_agent/profiling.py::detect_regime`).
  Result: R1 idle/startup (600 rpm, 10 % load, 23 % of rows), R2 nominal (1480 rpm, 55 %,
  47 %), R3 high load (1500 rpm, 85 %, 30 %).
- **Healthy baseline:** per regime, mean and std of each sensor over the first 20 % of each
  training motor's history, plus the regime centroids and scales
  (`agents/ml_architect/features.py::baseline_for`). This baseline ships to the edge in the
  sidecar and is reused verbatim by "Run analysis" (nothing refitted).

## 5. Features (13, one window of 12 rows = 2 h)

| Feature | Meaning |
|---|---|
| `vib_rms_mean`, `vib_kurt_mean`, `vib_crest_mean` | window means of vibration RMS, kurtosis, crest factor |
| `vib_rms_slope` | OLS slope of vibration RMS over the window (per 10 min) |
| `bearing_temp_mean`, `bearing_temp_slope` | drive-end bearing temperature level and trend |
| `temp_delta` | bearing minus motor housing temperature |
| `current_mean`, `rpm_mean`, `load_mean` | duty of the window |
| `vib_rms_z`, `bearing_temp_z` | z-scores against the healthy baseline of the window's regime (per-row regime inferred from rpm and load, baselines averaged over the window) |
| `vib_rms_resid` | vibration RMS minus the regime's healthy mean |

The regime-normalised features are what keep a load change from looking like a fault. The edge
computes the same 13 numbers with numpy only (`edge/features.py`); parity is tested.

## 6. Search (Optuna, 6 trials per family, seed 0)

`agents/ml_architect/automl.py::launch_trial`: each family is fitted on the training motors and
scored on the validation motors; the objective is the Industrial Model Score (section 7).
Class imbalance: `class_weight="balanced_subsample"` (random forest) or
`scale_pos_weight = 64.5` (XGBoost, LightGBM).

| Family | Tuned | Best params (this run) | Validation IMS |
|---|---|---|---|
| LightGBM | n_estimators, max_depth, num_leaves, learning_rate, min_child_samples, subsample, colsample_bytree | 50 trees, depth 3, 8 leaves, lr 0.19, min_child_samples 40 | **0.950** |
| Random forest | n_estimators, max_depth, min_samples_leaf, max_features | 200 trees, depth 11, min_samples_leaf 20, sqrt features | 0.926 |
| XGBoost | n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree | 100 trees, depth 7, lr 0.069, min_child_weight 6 | 0.916 |

Every trial is logged to MLflow (experiment `thor-automl-MTR-042`) with the split's asset ids.

## 7. Champion selection: Industrial Model Score

`agents/metrics.py::industrial_model_score`, computed on the validation motors:

| Component | Weight | Definition |
|---|---|---|
| recall | 0.35 | recall at P ≥ 0.5 |
| lead_time_score | 0.25 | min(1, median warning lead time / 72 h), alarm at P ≥ 0.3 for 3 rows |
| precision | 0.15 | precision at P ≥ 0.5 |
| calibration_score | 0.15 | 1 − min(1, Brier / 0.25) |
| latency_score | 0.10 | 1 − min(1, ms per 1000 rows / 50) |

The champion (highest validation IMS) is LightGBM. Its reported IMS of **0.844** is recomputed
from the test motors, which is why it is lower than the 0.950 that selected it.

## 8. Calibration

`agents/validation/checks.py::calibrate_probabilities`, on the six calibration motors, on top
of the frozen champion. Method: isotonic regression when there are ≥ 200 rows **and** at least
two failing motors among them; otherwise Platt scaling (sigmoid). The demo fleet has one
failing motor in the calibration group, so the production model is **Platt-calibrated**
(a two-parameter sigmoid on the champion's log-odds; the fitted a and b travel in the edge
sidecar). Isotonic on a single failure
event degenerates into a near-step curve (probability jumps from 0 to 1 within one sample), which
made the 0.3 early-warning and 0.5 thresholds fire together; that is why the rule exists.
Brier on the calibration rows 0.00095 → 0.00044 (isotonic would have reached 0.00034 by
over-fitting that one event). scikit-learn fits the calibrator on LightGBM's log-odds; the
exported transform records that (`"input": "decision_function"`) and the edge recovers the
log-odds from the ONNX probability before applying the same sigmoid.

## 9. Validation gate and final test metrics (6 untouched test motors)

| Check | Threshold | Value | Result |
|---|---|---|---|
| leakage (asset overlap, forbidden or label-correlated features) | none | clean | pass |
| completed failures among test motors | ≥ 1 | 1 (MTR-003) | pass |
| recall at P ≥ 0.5 | ≥ 0.60 | **0.865** (289 positive rows) | pass |
| median warning lead time at P ≥ 0.3 | ≥ 12 h | **41.5 h** | pass |
| edge parity | max |Δp| ≤ 1e-3 | **0.0** (200 vectors), features 1.6e-15 (18 raw windows) | pass |

Also on test: precision 1.000, F1 0.928, AUROC 0.933, Brier 0.0021 over 18,942 rows.

## 10. Promotion and deployment

The train run ends with the champion registered as a *candidate* and a promotion request
pending. A named approver (in the demo prep: `demo-prep@thor`) approves it in ModelOps;
`lifecycle.promote()` re-checks validation and parity, moves the row to production and
`deploy_edge()` writes `thor-bearing-classifier_v57257.onnx` + `.json` (features, window,
baseline, calibration transform). A maintenance approval never promotes a model, and a
promotion never approves maintenance.

## 11. Inference path

The edge container subscribes to MQTT, keeps a 12-row buffer per motor, computes the 13
features with numpy, runs ONNX Runtime and applies the sidecar calibration: no Python ML
stack, no control plane, **no LLM anywhere on the telemetry path**. Every sample's probability
is stored. What is *displayed* (fleet, Asset 360, Bolt) is the median of the motor's last
three predictions, and "Run analysis" explains the window whose probability is the median of
the last three windows: the same three-sample rule that defines the early-warning alarm, so a
single noisy sample never drives a colour or a Decision Contract
(`DISPLAY_MEDIAN_ROWS` in `apps/api/schemas.py`). The control plane's "Run analysis" uses the
same production artifact and baseline. The LLM is called only to pick the next pipeline node
and to phrase the explanation of numbers already computed.

## 12. Limitations

- Synthetic data: the fault signature (vibration, kurtosis, crest, bearing temperature rising
  together) is cleaner than a real plant's, hence precision 1.0 and Brier 0.002.
- 24 motors, four completed failures: one failure event among the test motors, so the lead
  time is a single measurement (41.5 h), not a distribution; validation and calibration share
  motors.
- 10-minute aggregates only: no waveform or frequency-domain analysis, so bearing fault
  frequencies are not available; the model sees trends, not spectra.
- Thresholds (0.5, 0.3) and the plant-cost assumptions are configuration, not plant data.

## 13. AI4I 2020 benchmark (real public data)

Rerun with the corrected methodology (grouped split, calibration on a separate group,
classification threshold 0.5 by name, the shared Industrial Model Score). See
`data/benchmark/README.md` for what is and is not comparable.

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
