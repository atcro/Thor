"""AI4I 2020 credibility benchmark: Thor's model-selection + validation methodology on real data.

Why this exists (CLAUDE.md section 9)
--------------------------------------
The demo runs on a synthetic regime-based fleet so the "one motor, one fault, one decision"
story is reproducible. That alone would let a reviewer ask whether the methodology only works
because the data was written to make it work. This module runs the same *selection and
validation discipline* on the public AI4I 2020 Predictive Maintenance Dataset (UCI, 10,000
rows) and reports honest, held-out numbers. It proves the methodology is not cherry-picked to
the synthetic story. It does NOT claim the synthetic demo numbers transfer to AI4I.

What is and is not comparable
-----------------------------
AI4I is tabular, one row per product, with no time axis and no asset identity. Therefore:

* NOT reused: the rolling-window feature pipeline (``agents/ml_architect/features.py``),
  warning lead time, RUL intervals, and the time-ordered backtest. There is nothing to roll
  over and no failure instant to measure lead time against.
* Reused as-is:
  1. regime detection on operating state (``agents.data_agent.profiling.detect_regime``,
     fed rpm + torque with torque standing in for load_pct),
  2. the Industrial Model Score ranking (``agents.ml_architect.automl.industrial_model_score``)
     with ``lead_time_h`` fixed at 0 because the dataset has no time axis - i.e. the score
     ranks on recall / precision / calibration / latency only and every family loses the same
     0.25 weight, so the ranking is still a fair comparison between families,
  3. grouped (never random-row-shuffle) holdout via ``GroupShuffleSplit`` on product buckets,
  4. the leakage check (``agents.validation.checks.detect_leakage``) plus an explicit guard
     that the five failure-mode flags (TWF/HDF/PWF/OSF/RNF) are never features - they are
     post-hoc labels of *which* failure happened and would leak the target outright,
  5. probability calibration and Brier score (``agents.validation.checks.calibrate_probabilities``).

Grouping caveat
---------------
AI4I product ids are unique per row, so there is no true machine id to hold out. The split
groups rows by the last two digits of the product number (100 buckets). This guarantees the
*mechanics* Thor insists on - whole groups move together, no row-level shuffle - but it is not
evidence of machine-level generalisation the way the synthetic fleet's asset holdout is.

Nothing here calls an LLM or MLflow. ``results.json`` is the record.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from agents.data_agent.profiling import detect_regime as _thor_detect_regime
from agents.ml_architect.automl import industrial_model_score
from agents.validation.checks import calibrate_probabilities, detect_leakage
from apps.api.schemas import FeatureSpec

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

AI4I_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
DEFAULT_CSV = Path(__file__).resolve().parent / "ai4i2020.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "results.json"

# UCI header -> Thor snake_case. Matching is done on a normalised form of the header so minor
# punctuation differences between mirrors of the file still resolve.
COLUMN_MAP: dict[str, str] = {
    "udi": "udi",
    "product_id": "product_id",
    "type": "type",
    "air_temperature_k": "air_temp_k",
    "process_temperature_k": "process_temp_k",
    "rotational_speed_rpm": "rpm",
    "torque_nm": "torque_nm",
    "tool_wear_min": "tool_wear_min",
    "machine_failure": "machine_failure",
    "twf": "twf",
    "hdf": "hdf",
    "pwf": "pwf",
    "osf": "osf",
    "rnf": "rnf",
}
FAILURE_MODE_FLAGS: tuple[str, ...] = ("twf", "hdf", "pwf", "osf", "rnf")
LABEL = "machine_failure"
# Never features: the label, the five per-mode flags, identifiers.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset({LABEL, *FAILURE_MODE_FLAGS, "udi", "product_id"})
RAW_FEATURES: tuple[str, ...] = (
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
    "power_w",
    "temp_delta_k",
)
FAMILIES: tuple[str, ...] = ("random_forest", "xgboost", "lightgbm")
ALARM_THRESHOLD = 0.5
TEST_GROUP_FRACTION = 0.25
CAL_GROUP_FRACTION = 0.2
Z_EPS = 1e-9


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def download_ai4i(dest: Path) -> Path:
    """Fetch the AI4I 2020 CSV from UCI if ``dest`` does not already exist.

    Input: destination path. Output: the same path, now guaranteed to exist. Uses ``httpx``
    with a 60 s timeout and follows redirects. Never called by the test suite.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s -> %s", AI4I_URL, dest)
    resp = httpx.get(AI4I_URL, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _normalise_header(name: str) -> str:
    """'Air temperature [K]' -> 'air_temperature_k'; 'Product ID' -> 'product_id'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return s.strip("_")


def load_ai4i(path: Path) -> pd.DataFrame:
    """Read the raw UCI CSV and normalise it to Thor's snake_case benchmark frame.

    Input: path to ``ai4i2020.csv`` (original UCI headers). Output: DataFrame with columns
    ``udi, product_id, type, air_temp_k, process_temp_k, rpm, torque_nm, tool_wear_min,
    machine_failure, twf, hdf, pwf, osf, rnf`` plus two derived physical quantities:
    ``power_w = torque_nm * rpm * 2 * pi / 60`` (mechanical power) and
    ``temp_delta_k = process_temp_k - air_temp_k``. Raises ``ValueError`` if a required
    column is missing after normalisation.
    """
    raw = pd.read_csv(path)
    renamed = {c: COLUMN_MAP.get(_normalise_header(c), _normalise_header(c)) for c in raw.columns}
    df = raw.rename(columns=renamed).copy()
    required = ["product_id", "type", *RAW_FEATURES[:5], LABEL, *FAILURE_MODE_FLAGS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AI4I frame is missing columns after normalisation: {missing}")
    for c in [*RAW_FEATURES[:5], LABEL, *FAILURE_MODE_FLAGS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["product_id"] = df["product_id"].astype(str)
    df["type"] = df["type"].astype(str)
    df["power_w"] = df["torque_nm"] * df["rpm"] * (2.0 * np.pi / 60.0)
    df["temp_delta_k"] = df["process_temp_k"] - df["air_temp_k"]
    df[LABEL] = df[LABEL].fillna(0).astype(int)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Regimes + features
# --------------------------------------------------------------------------------------


def detect_regime(df: pd.DataFrame, k: int = 3, seed: int = 0) -> pd.Series:
    """Cluster operating state on (rpm, torque) and return a per-row regime id.

    Reuses ``agents.data_agent.profiling.detect_regime`` unchanged: its contract wants ``rpm``
    and ``load_pct``, and torque is the load proxy in AI4I, so ``torque_nm`` is passed under
    the ``load_pct`` name. Input: benchmark frame with ``rpm`` and ``torque_nm``; ``k``
    clusters; ``seed``. Output: ``pd.Series`` of ``"R1".."Rk"`` aligned to ``df.index``,
    labelled ascending by ``rpm_mean * torque_mean`` (R1 = lowest mechanical load).
    """
    view = df[["rpm", "torque_nm"]].rename(columns={"torque_nm": "load_pct"}).astype(float)
    report = _thor_detect_regime(view.reset_index(drop=True), k=k, seed=seed)
    return pd.Series(list(report.row_regime), index=df.index, name="regime", dtype="object")


def build_features(df: pd.DataFrame, regime: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Regime-normalised feature frame with the failure-mode flags guaranteed absent.

    Input: benchmark frame from ``load_ai4i`` and the per-row ``regime`` Series from
    ``detect_regime``. Output: ``(frame, feature_names)`` where ``frame`` has columns
    ``product_id, regime, <features...>, label`` and ``feature_names`` is the raw values of
    ``RAW_FEATURES`` plus one ``<name>_z`` per raw feature = z-score against that regime's own
    mean / std (so a high-torque regime is not mistaken for a fault). ``label`` =
    ``machine_failure``. Leakage guard: asserts none of ``twf, hdf, pwf, osf, rnf`` (nor the
    label or identifiers) is in the feature list.
    """
    out = pd.DataFrame(
        {"product_id": df["product_id"].astype(str).to_numpy(), "regime": regime.to_numpy()},
        index=df.index,
    )
    names: list[str] = []
    for col in RAW_FEATURES:
        x = df[col].astype(float)
        out[col] = x
        names.append(col)
        grp = x.groupby(regime)
        mu = grp.transform("mean")
        sd = grp.transform("std").fillna(0.0)
        out[f"{col}_z"] = ((x - mu) / (sd + Z_EPS)).astype(float)
        names.append(f"{col}_z")
    out["label"] = df[LABEL].astype(int)

    leaked = sorted(set(names) & FORBIDDEN_COLUMNS)
    assert not leaked, f"failure-mode flags / label leaked into features: {leaked}"
    for flag in FAILURE_MODE_FLAGS:
        assert flag not in names, f"failure-mode flag {flag} must never be a feature"
    return out, names


# --------------------------------------------------------------------------------------
# Grouped split
# --------------------------------------------------------------------------------------


def product_group(df: pd.DataFrame) -> pd.Series:
    """Group key for the grouped holdout: last two digits of the product number (100 buckets).

    Input: frame with ``product_id`` like ``"M14860"``. Output: int Series 0..99 aligned to
    ``df.index``. Rationale: ``type`` (L/M/H) gives only 3 groups, too coarse for a 25 %
    holdout; product ids are unique per row so they cannot be groups themselves. The two-digit
    bucket is a deterministic pseudo-asset key that keeps whole groups on one side of the split.
    """
    digits = df["product_id"].astype(str).str.extract(r"(\d+)")[0].fillna("0")
    return digits.str[-2:].astype(int).rename("group")


def grouped_split(df: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Grouped 75 / 25 train / test split with no product bucket straddling the boundary.

    Input: benchmark frame with ``product_id``; RNG seed. Output: ``(train_idx, test_idx)``
    positional index arrays into ``df``. Uses ``GroupShuffleSplit(test_size=0.25)`` on
    ``product_group`` - groups are shuffled, rows never are, so this honours CLAUDE.md
    section 7 (no random row shuffle on data with structure). The two index sets are
    disjoint and their group sets are disjoint.
    """
    groups = product_group(df).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRACTION, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(df)), groups=groups))
    return np.sort(train_idx), np.sort(test_idx)


def _grouped_subsplit(
    idx: np.ndarray, groups: np.ndarray, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a positional index array into (keep, held_out) by whole groups."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(splitter.split(np.zeros(len(idx)), groups=groups[idx]))
    return idx[np.sort(a)], idx[np.sort(b)]


# --------------------------------------------------------------------------------------
# Estimators (local: automl's search spaces are private and tied to the time-series holdout)
# --------------------------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    """Small Optuna search space per family; deliberately narrower than automl's."""
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if family == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        }
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def _make_estimator(family: str, params: dict[str, Any], scale_pos_weight: float, seed: int) -> Any:
    """Instantiate one family with class-imbalance handling and a fixed seed (mirrors automl)."""
    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            **params, class_weight="balanced_subsample", random_state=seed, n_jobs=1
        )
    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(X))
    if p.ndim == 2 and p.shape[1] >= 2:
        return np.clip(p[:, 1].astype(float), 0.0, 1.0)
    return np.clip(p.reshape(-1).astype(float), 0.0, 1.0)


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float = ALARM_THRESHOLD) -> dict[str, float]:
    """recall, precision, f1, auroc (0.5 if one class), brier at ``threshold``."""
    pred = (p >= threshold).astype(int)
    return {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else 0.5,
        "brier": float(brier_score_loss(y, p)),
    }


def _latency_ms_per_1000(model: Any, X: pd.DataFrame, repeats: int = 3) -> float:
    """Best-of-``repeats`` wall time of ``predict_proba``, scaled to ms per 1000 rows."""
    if len(X) == 0:
        return 0.0
    model.predict_proba(X.iloc[: min(len(X), 64)])
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(X)
        best = min(best, time.perf_counter() - t0)
    return float(best / len(X) * 1000.0 * 1000.0)


def _ims_no_time_axis(metrics: dict[str, float], latency_ms: float) -> dict[str, float]:
    """Industrial Model Score with lead_time_h pinned to 0 (AI4I has no time axis)."""
    ims = industrial_model_score({**metrics, "lead_time_h": 0.0}, latency_ms)
    return {
        "total": float(ims.total),
        "recall": float(ims.recall),
        "precision": float(ims.precision),
        "calibration_score": float(ims.calibration_score),
        "latency_score": float(ims.latency_score),
        "lead_time_score": float(ims.lead_time_score),
    }


# --------------------------------------------------------------------------------------
# run_benchmark
# --------------------------------------------------------------------------------------


def _search_family(
    family: str,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_tune: pd.DataFrame,
    y_tune: np.ndarray,
    n_trials: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Optuna search for one family; objective = IMS (lead time pinned at 0) on the tune slice.

    Returns ``(fitted best model on X_fit, summary)`` where summary carries the tune-slice
    metrics, latency, IMS breakdown and best params.
    """
    n_pos = int(y_fit.sum())
    spw = float(max(1.0, (len(y_fit) - n_pos) / max(1, n_pos)))

    def _fit(params: dict[str, Any]) -> tuple[Any, dict[str, float], float]:
        model = _make_estimator(family, params, spw, seed)
        model.fit(X_fit, y_fit)
        p = _positive_proba(model, X_tune)
        return model, _metrics(y_tune, p), _latency_ms_per_1000(model, X_tune)

    def objective(trial: optuna.Trial) -> float:
        _, m, lat = _fit(_suggest_params(trial, family))
        return _ims_no_time_axis(m, lat)["total"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(1, int(n_trials)), show_progress_bar=False)
    best = dict(study.best_trial.params)
    model, m, lat = _fit(best)
    summary = {
        "family": family,
        "params": best,
        "tune_metrics": m,
        "latency_ms_per_1000": lat,
        "ims": _ims_no_time_axis(m, lat),
        "n_trials": int(n_trials),
    }
    return model, summary


def run_benchmark(
    csv_path: Path,
    n_trials: int = 8,
    seed: int = 0,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run Thor's selection + validation discipline on AI4I and write honest test metrics.

    Input: path to the UCI CSV, Optuna trials per family, seed, optional output path
    (default ``data/benchmark/results.json``).

    Procedure:
      1. ``load_ai4i`` -> ``detect_regime`` -> ``build_features`` (flags never features).
      2. ``grouped_split``: 25 % of product buckets are the TEST set, untouched until step 6.
      3. The TRAIN buckets are split again by group into fit / tune / cal slices (60/20/20 of
         train). Optuna maximises the IMS (``lead_time_h`` pinned to 0 - no time axis, so the
         score is recall / precision / calibration / latency) on the tune slice for each of
         RF / XGBoost / LightGBM.
      4. Champion = highest IMS. It is refit on fit + tune and calibrated on the cal slice
         with ``agents.validation.checks.calibrate_probabilities`` (isotonic when >= 200 cal
         rows, else sigmoid).
      5. ``detect_leakage`` runs over the train / test bucket lists and feature names, plus an
         explicit check that no failure-mode flag is a feature and the group overlap is zero.
      6. Test metrics for the calibrated champion at threshold 0.5.

    Output: dict with ``recall, precision, f1, auroc, brier_before, brier_after,
    positive_rate, n_train, n_test, champion_family, calibration_method, families`` (per-family
    IMS table), ``test_metrics_uncalibrated``, ``leakage_checks`` and run metadata. The same
    dict is written as JSON to ``out_path``.
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path) if out_path is not None else DEFAULT_OUT

    df = load_ai4i(csv_path)
    regime = detect_regime(df, seed=seed)
    frame, feature_names = build_features(df, regime)
    groups = product_group(df).to_numpy()
    train_idx, test_idx = grouped_split(df, seed=seed)

    fit_tune_idx, cal_idx = _grouped_subsplit(train_idx, groups, CAL_GROUP_FRACTION, seed + 1)
    fit_idx, tune_idx = _grouped_subsplit(fit_tune_idx, groups, TEST_GROUP_FRACTION, seed + 2)

    X = frame[feature_names]
    y = frame["label"].to_numpy(dtype=int)
    if len(np.unique(y[fit_idx])) < 2:
        raise ValueError("fit slice contains a single class; cannot train")

    families: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for family in FAMILIES:
        try:
            model, summary = _search_family(
                family, X.iloc[fit_idx], y[fit_idx], X.iloc[tune_idx], y[tune_idx], n_trials, seed
            )
        except Exception as exc:  # noqa: BLE001 - one broken library must not sink the benchmark
            logger.warning("family %s failed: %s", family, exc)
            families.append({"family": family, "error": str(exc)})
            continue
        families.append(summary)
        models[family] = model
    if not models:
        raise RuntimeError("no family could be trained")
    ranked = sorted((f for f in families if "ims" in f), key=lambda f: -f["ims"]["total"])
    champion_family = ranked[0]["family"]
    champion = _make_estimator(
        champion_family,
        ranked[0]["params"],
        float(
            max(1.0, (len(fit_tune_idx) - y[fit_tune_idx].sum()) / max(1, y[fit_tune_idx].sum()))
        ),
        seed,
    )
    champion.fit(X.iloc[fit_tune_idx], y[fit_tune_idx])

    calibrated, cal_report = calibrate_probabilities(
        champion, X.iloc[cal_idx], pd.Series(y[cal_idx])
    )

    # Leakage: reuse the validation toolbox with product buckets in the asset_id slot.
    train_groups = sorted({str(g) for g in groups[train_idx]})
    test_groups = sorted({str(g) for g in groups[test_idx]})
    spec = FeatureSpec(
        features=feature_names,
        window_rows=1,
        regime_normalized=True,
        description="AI4I raw + per-regime z-scores (no rolling window: tabular data)",
    )
    leak_frame = frame.assign(asset_id=[str(g) for g in groups], ts=pd.Timestamp("2020-01-01"))
    leak = detect_leakage(leak_frame, spec, train_groups, test_groups)
    leakage_checks = {
        "forbidden_columns_excluded": sorted(FORBIDDEN_COLUMNS),
        "failure_mode_flags_in_features": [f for f in FAILURE_MODE_FLAGS if f in feature_names],
        "group_overlap_count": len(set(train_groups) & set(test_groups)),
        "row_overlap_count": int(len(np.intersect1d(train_idx, test_idx))),
        "high_corr_features": list(leak.feature_leakage),
        "detect_leakage_passed": bool(leak.passed),
        "notes": list(leak.notes),
    }

    X_test, y_test = X.iloc[test_idx], y[test_idx]
    p_raw = _positive_proba(champion, X_test)
    p_cal = _positive_proba(calibrated, X_test)
    test_cal = _metrics(y_test, p_cal)
    test_raw = _metrics(y_test, p_raw)

    result: dict[str, Any] = {
        "dataset": "AI4I 2020 Predictive Maintenance (UCI)",
        "source_csv": str(csv_path),
        "seed": int(seed),
        "n_trials_per_family": int(n_trials),
        "n_rows": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_fit": int(len(fit_idx)),
        "n_tune": int(len(tune_idx)),
        "n_cal": int(len(cal_idx)),
        "n_groups_train": len(train_groups),
        "n_groups_test": len(test_groups),
        "positive_rate": float(y.mean()),
        "positive_rate_test": float(y_test.mean()),
        "n_regimes": int(regime.nunique()),
        "regime_shares": {k: float(v) for k, v in regime.value_counts(normalize=True).items()},
        "features": feature_names,
        "champion_family": champion_family,
        "calibration_method": cal_report.method,
        "recall": test_cal["recall"],
        "precision": test_cal["precision"],
        "f1": test_cal["f1"],
        "auroc": test_cal["auroc"],
        "brier_before": test_raw["brier"],
        "brier_after": test_cal["brier"],
        "test_metrics_uncalibrated": test_raw,
        "families": families,
        "leakage_checks": leakage_checks,
        "notes": [
            "IMS computed with lead_time_h = 0: AI4I has no time axis, so the score ranks on "
            "recall / precision / calibration / latency only.",
            "Groups are product-number buckets (last two digits), not true machine ids.",
            "Test buckets were never used for tuning or calibration.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def format_markdown(result: dict[str, Any]) -> str:
    """Compact markdown summary of a ``run_benchmark`` result (test metrics + family IMS)."""
    lines = [
        f"AI4I 2020 benchmark - champion: {result['champion_family']} "
        f"(calibration: {result['calibration_method']}, seed {result['seed']})",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key in ("recall", "precision", "f1", "auroc", "brier_before", "brier_after"):
        lines.append(f"| {key} | {result[key]:.4f} |")
    lines.append(f"| positive_rate | {result['positive_rate']:.4f} |")
    lines.append(f"| n_train / n_test | {result['n_train']} / {result['n_test']} |")
    lines += [
        "",
        "| family | IMS | recall | precision | brier | latency ms/1k |",
        "|---|---|---|---|---|---|",
    ]
    for fam in result["families"]:
        if "ims" not in fam:
            lines.append(f"| {fam['family']} | failed: {fam.get('error', '?')} | | | | |")
            continue
        m = fam["tune_metrics"]
        lines.append(
            f"| {fam['family']} | {fam['ims']['total']:.4f} | {m['recall']:.3f} | "
            f"{m['precision']:.3f} | {m['brier']:.4f} | {fam['latency_ms_per_1000']:.2f} |"
        )
    lc = result["leakage_checks"]
    lines += [
        "",
        f"leakage: flags in features = {lc['failure_mode_flags_in_features']}, "
        f"group overlap = {lc['group_overlap_count']}, passed = {lc['detect_leakage_passed']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m data.benchmark.ai4i_benchmark [--csv PATH] [--n-trials N] [--seed S] [--out PATH]``.

    Downloads the CSV if ``--csv`` is missing, runs ``run_benchmark`` and prints a markdown table.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    csv_path = download_ai4i(args.csv)
    result = run_benchmark(csv_path, n_trials=args.n_trials, seed=args.seed, out_path=args.out)
    print(format_markdown(result))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
