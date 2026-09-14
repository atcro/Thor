"""Pure-numpy feature computation for the edge container.

This is the dependency-free twin of `agents/ml_architect/features.py` (Agent B). Both must
produce identical numbers for the same raw window; `tests/test_edge.py` checks parity.

Raw row layout (`RAW_COLUMNS`, same order as `apps.api.schemas.SENSOR_COLUMNS`):

    0 vibration_rms, 1 vibration_kurtosis, 2 vibration_crest, 3 bearing_temp_c,
    4 motor_temp_c, 5 current_a, 6 rpm, 7 load_pct

Features over a window of `window_rows` consecutive rows (oldest first), for a row index
x = 0..w-1:

    vib_rms_mean        = mean(vibration_rms)
    vib_rms_slope       = least-squares slope of vibration_rms vs x  (units per row)
                          = sum((x - mean(x)) * y) / sum((x - mean(x))**2)
    vib_kurt_mean       = mean(vibration_kurtosis)
    vib_crest_mean      = mean(vibration_crest)
    bearing_temp_mean   = mean(bearing_temp_c)
    bearing_temp_slope  = least-squares slope of bearing_temp_c vs x  (deg C per row)
    temp_delta          = mean(bearing_temp_c) - mean(motor_temp_c)
    current_mean        = mean(current_a)
    rpm_mean            = mean(rpm)
    load_mean           = mean(load_pct)
    vib_rms_z           = (vib_rms_mean - B.vibration_rms.mean) / max(B.vibration_rms.std, EPS)
    bearing_temp_z      = (bearing_temp_mean - B.bearing_temp_c.mean) / max(B.bearing_temp_c.std, EPS)
    vib_rms_resid       = vib_rms_mean - B.vibration_rms.mean

where B is the healthy baseline for the current regime, supplied by the model sidecar JSON:

    "baseline": {"vibration_rms": {"mean": m, "std": s}, "bearing_temp_c": {...}}          (flat)
    "baseline": {"R1": {"vibration_rms": {...}, ...}, "R2": {...}, "R3": {...}}             (per regime)

When no baseline is available the fallback is mean=0, std=1, so the z-scores collapse to the
raw means and vib_rms_resid == vib_rms_mean.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EPS = 1e-9

RAW_COLUMNS: list[str] = [
    "vibration_rms",
    "vibration_kurtosis",
    "vibration_crest",
    "bearing_temp_c",
    "motor_temp_c",
    "current_a",
    "rpm",
    "load_pct",
]

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

#: Raw sensor each baseline-relative feature is normalized against.
BASELINE_RAW: dict[str, str] = {
    "vib_rms_z": "vibration_rms",
    "bearing_temp_z": "bearing_temp_c",
    "vib_rms_resid": "vibration_rms",
}

_COL = {name: i for i, name in enumerate(RAW_COLUMNS)}

Baseline = Mapping[str, Mapping[str, float]]


def slope(y: np.ndarray) -> float:
    """Least-squares slope of `y` against its row index 0..n-1. Returns 0.0 when n < 2."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    return float(xc @ y / (xc @ xc))


def select_baseline(baseline: Mapping[str, Any] | None, regime: str | None = None) -> Baseline:
    """Pick the baseline table for `regime` from a flat or per-regime sidecar baseline.

    Inputs: sidecar "baseline" (flat {raw: {mean, std}} or {regime: {raw: {mean, std}}}) or
    None; the current regime label. Output: flat {raw: {"mean", "std"}} (may be empty).
    Per-regime tables fall back to the flat table entries, then to any "default" key.
    """
    if not baseline:
        return {}
    per_regime = {
        k: v for k, v in baseline.items() if k not in RAW_COLUMNS and isinstance(v, Mapping)
    }
    flat = {k: v for k, v in baseline.items() if k in RAW_COLUMNS}
    if per_regime:
        table = per_regime.get(regime or "") or per_regime.get("default") or {}
        merged: dict[str, Mapping[str, float]] = dict(flat)
        merged.update({k: v for k, v in table.items() if k in RAW_COLUMNS})
        return merged
    return flat


def _baseline_stats(b: Baseline, raw: str) -> tuple[float, float]:
    entry = b.get(raw) if b else None
    if not entry:
        return 0.0, 1.0
    mean = float(entry.get("mean", 0.0))
    std = float(entry.get("std", 1.0))
    return mean, max(std, EPS)


def compute_features_np(
    buffer: np.ndarray,
    baseline: Mapping[str, Any] | None = None,
    regime: str | None = None,
) -> np.ndarray:
    """Compute the 13 FEATURE_NAMES from one raw window.

    Inputs: `buffer` of shape (window_rows, 8) in RAW_COLUMNS order, oldest row first; optional
    sidecar baseline; optional regime label used to select a per-regime baseline.
    Output: float64 array of shape (13,) in FEATURE_NAMES order.
    """
    buf = np.asarray(buffer, dtype=float)
    if buf.ndim != 2 or buf.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"buffer must be (window_rows, {len(RAW_COLUMNS)}), got {buf.shape}")
    if buf.shape[0] == 0:
        raise ValueError("buffer is empty")
    b = select_baseline(baseline, regime)
    means = buf.mean(axis=0)
    vib_mean = means[_COL["vibration_rms"]]
    bearing_mean = means[_COL["bearing_temp_c"]]
    vib_b_mean, vib_b_std = _baseline_stats(b, "vibration_rms")
    brg_b_mean, brg_b_std = _baseline_stats(b, "bearing_temp_c")
    return np.array(
        [
            vib_mean,
            slope(buf[:, _COL["vibration_rms"]]),
            means[_COL["vibration_kurtosis"]],
            means[_COL["vibration_crest"]],
            bearing_mean,
            slope(buf[:, _COL["bearing_temp_c"]]),
            bearing_mean - means[_COL["motor_temp_c"]],
            means[_COL["current_a"]],
            means[_COL["rpm"]],
            means[_COL["load_pct"]],
            (vib_mean - vib_b_mean) / vib_b_std,
            (bearing_mean - brg_b_mean) / brg_b_std,
            vib_mean - vib_b_mean,
        ],
        dtype=float,
    )


def rows_to_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """List of telemetry dicts (JSON rows) -> (n, 8) float matrix in RAW_COLUMNS order."""
    return np.array([[float(r[c]) for c in RAW_COLUMNS] for r in rows], dtype=float).reshape(
        -1, len(RAW_COLUMNS)
    )


def compute_features_dict(
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None = None,
    regime: str | None = None,
) -> dict[str, float]:
    """Convenience wrapper: telemetry dict rows (oldest first) -> {feature_name: value}.

    If `regime` is None the last row's "regime" key is used when present.
    """
    if regime is None and rows and rows[-1].get("regime"):
        regime = str(rows[-1]["regime"])
    vec = compute_features_np(rows_to_matrix(rows), baseline, regime)
    return dict(zip(FEATURE_NAMES, vec.tolist(), strict=True))


def rolling_features(
    matrix: np.ndarray,
    window_rows: int,
    baseline: Mapping[str, Any] | None = None,
    regimes: Sequence[str] | None = None,
) -> np.ndarray:
    """Vectorized rolling version for one asset's full history.

    Inputs: (n, 8) matrix in RAW_COLUMNS order (time-ordered), window size, optional baseline,
    optional per-row regime labels (len n) for per-regime baselines. Output: (n - window_rows + 1,
    13) array; row i corresponds to the window ending at raw row i + window_rows - 1 (same as
    pandas `rolling(window_rows)` after dropping the first window_rows - 1 rows).
    """
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    w = int(window_rows)
    if w < 1 or n < w:
        return np.empty((0, len(FEATURE_NAMES)))
    win = np.lib.stride_tricks.sliding_window_view(m, w, axis=0)  # (n-w+1, 8, w)
    means = win.mean(axis=2)  # (n-w+1, 8)
    x = np.arange(w, dtype=float)
    xc = x - x.mean()
    denom = float(xc @ xc) if w > 1 else 1.0
    vib_slope = (win[:, _COL["vibration_rms"], :] @ xc) / denom if w > 1 else np.zeros(n - w + 1)
    brg_slope = (win[:, _COL["bearing_temp_c"], :] @ xc) / denom if w > 1 else np.zeros(n - w + 1)
    vib_mean = means[:, _COL["vibration_rms"]]
    brg_mean = means[:, _COL["bearing_temp_c"]]

    k = n - w + 1
    vib_bm = np.zeros(k)
    vib_bs = np.ones(k)
    brg_bm = np.zeros(k)
    brg_bs = np.ones(k)
    if baseline:
        if regimes is None:
            b = select_baseline(baseline, None)
            vib_bm[:], vib_bs[:] = _baseline_stats(b, "vibration_rms")
            brg_bm[:], brg_bs[:] = _baseline_stats(b, "bearing_temp_c")
        else:
            reg = np.asarray(list(regimes), dtype=object)[w - 1 :]
            cache: dict[str, tuple[float, float, float, float]] = {}
            for i, r in enumerate(reg):
                key = str(r)
                if key not in cache:
                    b = select_baseline(baseline, key)
                    cache[key] = (
                        *_baseline_stats(b, "vibration_rms"),
                        *_baseline_stats(b, "bearing_temp_c"),
                    )
                vib_bm[i], vib_bs[i], brg_bm[i], brg_bs[i] = cache[key]

    out = np.column_stack(
        [
            vib_mean,
            vib_slope,
            means[:, _COL["vibration_kurtosis"]],
            means[:, _COL["vibration_crest"]],
            brg_mean,
            brg_slope,
            brg_mean - means[:, _COL["motor_temp_c"]],
            means[:, _COL["current_a"]],
            means[:, _COL["rpm"]],
            means[:, _COL["load_pct"]],
            (vib_mean - vib_bm) / vib_bs,
            (brg_mean - brg_bm) / brg_bs,
            vib_mean - vib_bm,
        ]
    )
    return out
