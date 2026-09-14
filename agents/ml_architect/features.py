"""05 ML Architect: feature pipeline shared by training and the edge container.

Design
------
Every feature is a function of one asset's last ``window_rows`` raw sensor rows (a *window*)
plus a static per-regime *baseline*. The same numpy helper (``_features_from_windows``) is
used for the batch training pipeline (``build_feature_pipeline``, via sliding windows) and for
the single-window reference implementation (``compute_features_np``), so the two are identical
by construction. ``edge/features.py`` (Agent A) re-implements ``compute_features_np`` with numpy
only; the formulas are spelled out in that docstring.

Nothing here reads ``health``, ``failure_within_h``, regime strings or ``ts`` as a feature.
``failure_within_h`` is used only to build ``label`` / ``rul_h``; regime labels are used only
to build the baseline (normalisation), never as a model input.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from apps.api.schemas import SENSOR_COLUMNS, FeatureSpec, RegimeReport

FEATURE_NAMES: list[str] = [
    "vib_rms_mean",
    "vib_rms_slope",
    "vib_kurt_mean",
    "vib_crest_mean",
    "bearing_temp_mean",
    "bearing_temp_slope",
    "temp_delta",
    "current_mean",
    "rpm_mean",
    "load_mean",
    "vib_rms_z",
    "bearing_temp_z",
    "vib_rms_resid",
]

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "health",
    "failure_within_h",
    "regime",
    "ts",
    "asset_id",
    "label",
    "rul_h",
)

BASELINE_META_KEY = "_regime_centroids"
BASELINE_FRACTION = 0.2
STD_FLOOR = 1e-6
DEFAULT_REGIME = "ALL"

# Column indices in SENSOR_COLUMNS order.
_I = {name: i for i, name in enumerate(SENSOR_COLUMNS)}
_VIB, _KURT, _CREST = _I["vibration_rms"], _I["vibration_kurtosis"], _I["vibration_crest"]
_BT, _MT, _CUR, _RPM, _LOAD = (
    _I["bearing_temp_c"],
    _I["motor_temp_c"],
    _I["current_a"],
    _I["rpm"],
    _I["load_pct"],
)


# --------------------------------------------------------------------------------------
# Regime labels + baseline
# --------------------------------------------------------------------------------------


def _regime_labels(df: pd.DataFrame, regimes: RegimeReport | None) -> pd.Series:
    """Per-row regime id used for baseline construction (never a feature).

    Preference: ``regimes.row_regime`` (aligned to df order) -> df['regime'] column ->
    a single regime ``DEFAULT_REGIME``.
    """
    if regimes is not None and len(regimes.row_regime) == len(df):
        return pd.Series(list(regimes.row_regime), index=df.index, dtype=object)
    if "regime" in df.columns and df["regime"].notna().any():
        return df["regime"].astype(object).fillna(DEFAULT_REGIME)
    return pd.Series(DEFAULT_REGIME, index=df.index, dtype=object)


def _regime_ids(baseline: dict[str, dict[str, float]]) -> list[str]:
    meta = baseline.get(BASELINE_META_KEY, {})
    return sorted(k[: -len("_rpm")] for k in meta if k.endswith("_rpm"))


def baseline_for(
    df: pd.DataFrame,
    regimes: RegimeReport | None,
    baseline_fraction: float = BASELINE_FRACTION,
) -> dict[str, dict[str, float]]:
    """Per-regime healthy baseline used for the regime-normalised features.

    Input: telemetry DataFrame sorted by ``(asset_id, ts)``; ``regimes`` from
    ``detect_regime`` (or ``None`` to use df['regime'] / a single regime); the fraction of each
    asset's earliest rows treated as healthy (default 20 %).

    Output (JSON-serialisable, ships in the edge sidecar): keyed by raw sensor name, e.g.
    ``{"vibration_rms": {"mean": m, "std": s, "R1_mean": m1, "R1_std": s1, "R2_mean": ...},
    "bearing_temp_c": {...}, ...}`` for every column in ``SENSOR_COLUMNS``. ``mean``/``std``
    are pooled over all baseline rows (fallback when a regime has no baseline rows);
    ``<regime>_mean``/``<regime>_std`` are per regime. ``std`` is the population std (ddof=0)
    floored at ``STD_FLOOR``. The special key ``"_regime_centroids"`` holds
    ``{"<regime>_rpm", "<regime>_load_pct"}`` = mean rpm/load of every row of that regime in
    ``df`` and ``{"rpm_scale", "load_pct_scale"}`` = fleet-wide std of rpm/load (ddof=0, floored),
    which ``compute_features_np`` uses to assign a window to its regime.
    """
    if len(df) == 0:
        raise ValueError("baseline_for needs at least one row")
    labels = _regime_labels(df, regimes)
    sensors = list(SENSOR_COLUMNS)
    raw = df[sensors].astype(float)

    counts = df.groupby("asset_id", sort=False)["asset_id"].transform("size")
    pos = df.groupby("asset_id", sort=False).cumcount()
    n_base = np.maximum(1, np.ceil(baseline_fraction * counts.to_numpy())).astype(int)
    base_mask = pos.to_numpy() < n_base
    base = raw[base_mask]
    base_labels = labels[base_mask]

    def _stats(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        mean = frame.mean(skipna=True)
        std = frame.std(ddof=0, skipna=True)
        return mean.fillna(0.0), std.fillna(0.0).clip(lower=STD_FLOOR)

    pooled_mean, pooled_std = _stats(base)
    out: dict[str, dict[str, float]] = {
        s: {"mean": float(pooled_mean[s]), "std": float(pooled_std[s])} for s in sensors
    }
    regime_ids = sorted(str(r) for r in labels.unique())
    for rid in regime_ids:
        sub = base[(base_labels == rid).to_numpy()]
        if len(sub) >= 2:
            r_mean, r_std = _stats(sub)
        else:
            r_mean, r_std = pooled_mean, pooled_std
        for s in sensors:
            out[s][f"{rid}_mean"] = float(r_mean[s])
            out[s][f"{rid}_std"] = float(r_std[s])

    meta: dict[str, float] = {
        "rpm_scale": float(max(STD_FLOOR, raw["rpm"].std(ddof=0, skipna=True) or 0.0)),
        "load_pct_scale": float(max(STD_FLOOR, raw["load_pct"].std(ddof=0, skipna=True) or 0.0)),
    }
    for rid in regime_ids:
        m = (labels == rid).to_numpy()
        meta[f"{rid}_rpm"] = float(np.nan_to_num(raw.loc[m, "rpm"].mean(), nan=0.0))
        meta[f"{rid}_load_pct"] = float(np.nan_to_num(raw.loc[m, "load_pct"].mean(), nan=0.0))
    out[BASELINE_META_KEY] = meta
    return out


# --------------------------------------------------------------------------------------
# Window math (shared by batch pipeline and single-window reference)
# --------------------------------------------------------------------------------------


def _assign_regime(
    rpm: np.ndarray, load: np.ndarray, baseline: dict[str, dict[str, float]]
) -> np.ndarray:
    """Per-row nearest regime centroid in scaled (rpm, load) space.

    ``rpm``/``load`` may have any shape (e.g. ``(m, w)``); returns an int array of the same
    shape indexing the sorted regime ids. First minimum wins on ties.
    """
    ids = _regime_ids(baseline)
    if not ids:
        return np.zeros(np.shape(rpm), dtype=int)
    meta = baseline[BASELINE_META_KEY]
    rs, ls = meta["rpm_scale"], meta["load_pct_scale"]
    d = np.stack(
        [
            ((rpm - meta[f"{r}_rpm"]) / rs) ** 2 + ((load - meta[f"{r}_load_pct"]) / ls) ** 2
            for r in ids
        ],
        axis=-1,
    )
    return np.argmin(d, axis=-1)


def _regime_stat(
    baseline: dict[str, dict[str, float]], sensor: str, stat: str, regime_idx: np.ndarray
) -> np.ndarray:
    """Look up ``<regime>_<stat>`` for every entry of ``regime_idx`` (same shape out)."""
    ids = _regime_ids(baseline)
    entry = baseline[sensor]
    if not ids:
        return np.full(np.shape(regime_idx), float(entry[stat]))
    table = np.array([float(entry.get(f"{r}_{stat}", entry[stat])) for r in ids])
    return table[regime_idx]


def _features_from_windows(
    windows: np.ndarray, baseline: dict[str, dict[str, float]]
) -> np.ndarray:
    """windows: (m, n_sensors, w) -> (m, len(FEATURE_NAMES)). See compute_features_np."""
    if windows.ndim != 3 or windows.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f"expected windows of shape (m, {len(SENSOR_COLUMNS)}, w)")
    w = windows.shape[2]
    if w < 2:
        raise ValueError("window must contain at least 2 rows")
    i = np.arange(w, dtype=float)
    ic = i - i.mean()
    denom = float((ic**2).sum())

    means = windows.mean(axis=2)  # (m, n_sensors)
    slopes = (windows * ic).sum(axis=2) / denom  # (m, n_sensors), units per row

    # Regime per raw row (m, w); the window's expected value is the mean of the per-row
    # regime baselines, so a window straddling a load change is not mistaken for a fault.
    reg = _assign_regime(windows[:, _RPM, :], windows[:, _LOAD, :], baseline)
    vib_mu = _regime_stat(baseline, "vibration_rms", "mean", reg).mean(axis=1)
    vib_sd = np.maximum(_regime_stat(baseline, "vibration_rms", "std", reg).mean(axis=1), STD_FLOOR)
    bt_mu = _regime_stat(baseline, "bearing_temp_c", "mean", reg).mean(axis=1)
    bt_sd = np.maximum(_regime_stat(baseline, "bearing_temp_c", "std", reg).mean(axis=1), STD_FLOOR)

    out = np.column_stack(
        [
            means[:, _VIB],  # vib_rms_mean
            slopes[:, _VIB],  # vib_rms_slope
            means[:, _KURT],  # vib_kurt_mean
            means[:, _CREST],  # vib_crest_mean
            means[:, _BT],  # bearing_temp_mean
            slopes[:, _BT],  # bearing_temp_slope
            means[:, _BT] - means[:, _MT],  # temp_delta
            means[:, _CUR],  # current_mean
            means[:, _RPM],  # rpm_mean
            means[:, _LOAD],  # load_mean
            (means[:, _VIB] - vib_mu) / vib_sd,  # vib_rms_z
            (means[:, _BT] - bt_mu) / bt_sd,  # bearing_temp_z
            means[:, _VIB] - vib_mu,  # vib_rms_resid
        ]
    )
    return out.astype(float)


def compute_features_np(buffer: np.ndarray, baseline: dict[str, dict[str, float]]) -> np.ndarray:
    """Reference single-window feature vector (numpy only). Agent A copies this to the edge.

    Input: ``buffer`` of shape ``(window_rows, 8)`` holding one asset's most recent raw rows,
    oldest first, columns in ``SENSOR_COLUMNS`` order
    ``[vibration_rms, vibration_kurtosis, vibration_crest, bearing_temp_c, motor_temp_c,
    current_a, rpm, load_pct]``; ``baseline`` from ``baseline_for``.
    Output: 1-D float array aligned to ``FEATURE_NAMES``.

    Formulas (``x[c]`` = column c of the buffer, ``w`` = number of rows, ``i = 0..w-1``,
    ``ic = i - mean(i)``):
      mean(c)  = sum(x[c]) / w
      slope(c) = sum(ic * x[c]) / sum(ic**2)          (OLS slope, units per row = per 10 min)
      vib_rms_mean       = mean(vibration_rms)
      vib_rms_slope      = slope(vibration_rms)
      vib_kurt_mean      = mean(vibration_kurtosis)
      vib_crest_mean     = mean(vibration_crest)
      bearing_temp_mean  = mean(bearing_temp_c)
      bearing_temp_slope = slope(bearing_temp_c)
      temp_delta         = mean(bearing_temp_c) - mean(motor_temp_c)
      current_mean       = mean(current_a)
      rpm_mean           = mean(rpm)
      load_mean          = mean(load_pct)
      per-row regime R_i (for each row i of the buffer) = argmin over regime ids r
                 (sorted ascending, first minimum wins) of
                 ((rpm_i - C[r + "_rpm"]) / C["rpm_scale"])**2
                 + ((load_pct_i - C[r + "_load_pct"]) / C["load_pct_scale"])**2
                 where C = baseline["_regime_centroids"]
      mu_v = mean over i of baseline["vibration_rms"][R_i + "_mean"]
      sd_v = max(mean over i of baseline["vibration_rms"][R_i + "_std"], 1e-6)
      mu_t = mean over i of baseline["bearing_temp_c"][R_i + "_mean"]
      sd_t = max(mean over i of baseline["bearing_temp_c"][R_i + "_std"], 1e-6)
      vib_rms_z          = (vib_rms_mean - mu_v) / sd_v
      bearing_temp_z     = (bearing_temp_mean - mu_t) / sd_t
      vib_rms_resid      = vib_rms_mean - mu_v
    Averaging the per-row regime baselines means a window that straddles a load change gets a
    blended expected value, so the transition itself does not look like a fault.
    If the baseline has no regime centroids, the pooled ``mean``/``std`` are used instead.
    NaNs in the buffer propagate (the training pipeline forward/back-fills per asset first).
    """
    buf = np.asarray(buffer, dtype=float)
    if buf.ndim != 2 or buf.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f"buffer must be (window_rows, {len(SENSOR_COLUMNS)}), got {buf.shape}")
    windows = buf.T[np.newaxis, :, :]  # (1, n_sensors, w)
    return _features_from_windows(windows, baseline)[0]


# --------------------------------------------------------------------------------------
# build_feature_pipeline
# --------------------------------------------------------------------------------------


def build_feature_pipeline(
    df: pd.DataFrame,
    regimes: RegimeReport | None,
    horizon_h: float = 48.0,
    window_rows: int = 12,
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Turn raw telemetry into the model-ready feature frame.

    Input: telemetry DataFrame (Telemetry shape, sorted by ``(asset_id, ts)``; it is re-sorted
    defensively), ``regimes`` from ``detect_regime`` (may be ``None``), label horizon in hours
    and the rolling window length in rows (12 rows = 2 h at a 10-min step).
    Output: ``(features_df, spec)`` where ``features_df`` has columns
    ``asset_id, ts, *FEATURE_NAMES, label, rul_h``: one row per raw row after the first
    ``window_rows - 1`` rows of each asset (assets shorter than the window yield no rows);
    ``ts`` is the timestamp of the window's last row; ``label = 1`` iff
    ``failure_within_h <= horizon_h`` (NaN -> 0); ``rul_h = failure_within_h`` (NaN when no
    failure ahead). ``spec.features == FEATURE_NAMES``. The regime baseline used is attached
    as ``features_df.attrs["baseline"]`` (also recomputable with ``baseline_for(df, regimes)``).
    Raw sensor NaNs are forward- then back-filled per asset, remaining NaNs become 0.0.
    """
    if window_rows < 2:
        raise ValueError("window_rows must be >= 2")
    missing = [c for c in ("asset_id", "ts", *SENSOR_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"telemetry frame is missing columns: {missing}")

    order = np.lexsort((df["ts"].to_numpy(), df["asset_id"].to_numpy()))
    is_sorted = bool(np.all(order == np.arange(len(df))))
    if not is_sorted:
        df = df.iloc[order].reset_index(drop=True)
        if regimes is not None and len(regimes.row_regime) == len(df):
            regimes = regimes.model_copy(
                update={"row_regime": [regimes.row_regime[i] for i in order]}
            )
    baseline = baseline_for(df, regimes)

    fail = (
        df["failure_within_h"].astype(float)
        if "failure_within_h" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    label_all = (fail <= horizon_h).fillna(False).astype(int).to_numpy()
    rul_all = fail.to_numpy(dtype=float)

    parts: list[pd.DataFrame] = []
    for asset_id, g in df.groupby("asset_id", sort=True):
        if len(g) < window_rows:
            continue
        raw = g[list(SENSOR_COLUMNS)].astype(float).ffill().bfill().fillna(0.0).to_numpy()
        windows = np.lib.stride_tricks.sliding_window_view(raw, window_rows, axis=0)
        feats = _features_from_windows(windows, baseline)
        idx = g.index.to_numpy()[window_rows - 1 :]
        part = pd.DataFrame(feats, columns=FEATURE_NAMES)
        part.insert(0, "ts", g["ts"].to_numpy()[window_rows - 1 :])
        part.insert(0, "asset_id", asset_id)
        part["label"] = label_all[idx]
        part["rul_h"] = rul_all[idx]
        parts.append(part)

    columns = ["asset_id", "ts", *FEATURE_NAMES, "label", "rul_h"]
    if parts:
        features_df = pd.concat(parts, ignore_index=True)[columns]
    else:
        features_df = pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
        features_df["asset_id"] = features_df["asset_id"].astype(object)
    features_df["label"] = features_df["label"].astype(int)
    features_df.attrs["baseline"] = baseline
    features_df.attrs["horizon_h"] = float(horizon_h)

    spec = FeatureSpec(
        features=list(FEATURE_NAMES),
        window_rows=int(window_rows),
        regime_normalized=True,
        description=(
            f"Rolling {window_rows}-row window per asset (mean, OLS slope) over the raw sensors; "
            f"vib_rms_z / bearing_temp_z / vib_rms_resid are normalised against the healthy "
            f"baseline of the window's operating regime (first {int(BASELINE_FRACTION * 100)}% "
            f"of each asset's history, {len(_regime_ids(baseline))} regimes). "
            f"label = failure_within_h <= {horizon_h:g} h."
        ),
    )
    assert not any(c in FORBIDDEN_FEATURE_COLUMNS for c in spec.features)
    assert math.isfinite(horizon_h)
    return features_df, spec
