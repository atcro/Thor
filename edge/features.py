"""Pure-numpy feature computation for the edge container.

This is the dependency-free twin of `agents/ml_architect/features.py`. Both must produce
identical numbers for the same raw window and the same baseline; `tests/test_edge.py` checks
parity against the training implementation and `agents.mlops.lifecycle.check_edge_parity`
re-checks it on real held-out windows before a model can be promoted.

Raw row layout (`RAW_COLUMNS`, same order as `apps.api.schemas.SENSOR_COLUMNS`):

    0 vibration_rms, 1 vibration_kurtosis, 2 vibration_crest, 3 bearing_temp_c,
    4 motor_temp_c, 5 current_a, 6 rpm, 7 load_pct

Features over a window of `window_rows` consecutive rows (oldest first), for a row index
i = 0..w-1 and ic = i - mean(i):

    vib_rms_mean        = mean(vibration_rms)
    vib_rms_slope       = sum(ic * vibration_rms) / sum(ic**2)       (OLS slope per row)
    vib_kurt_mean       = mean(vibration_kurtosis)
    vib_crest_mean      = mean(vibration_crest)
    bearing_temp_mean   = mean(bearing_temp_c)
    bearing_temp_slope  = sum(ic * bearing_temp_c) / sum(ic**2)
    temp_delta          = mean(bearing_temp_c) - mean(motor_temp_c)
    current_mean        = mean(current_a)
    rpm_mean            = mean(rpm)
    load_mean           = mean(load_pct)
    vib_rms_z           = (vib_rms_mean - mu_v) / sd_v
    bearing_temp_z      = (bearing_temp_mean - mu_t) / sd_t
    vib_rms_resid       = vib_rms_mean - mu_v

where the baseline B is the sidecar's "baseline" in the TRAINING format produced by
`agents.ml_architect.features.baseline_for`:

    B[sensor] = {"mean": m, "std": s, "R1_mean": m1, "R1_std": s1, "R2_mean": ..., ...}
    B["_regime_centroids"] = {"R1_rpm", "R1_load_pct", ..., "rpm_scale", "load_pct_scale"}

Regime per raw row r of the window (NOT the simulator's label, which the edge never sees):

    R_r = argmin over regime ids (sorted, first minimum wins) of
          ((rpm_r - C[R + "_rpm"]) / C["rpm_scale"])**2
          + ((load_pct_r - C[R + "_load_pct"]) / C["load_pct_scale"])**2

    mu_v = mean over r of B["vibration_rms"][R_r + "_mean"]
    sd_v = max(mean over r of B["vibration_rms"][R_r + "_std"], STD_FLOOR)
    mu_t, sd_t likewise from B["bearing_temp_c"]

Averaging the per-row regime baselines means a window that straddles a load change gets a
blended expected value, so the transition itself does not look like a fault. A baseline with
no "_regime_centroids" uses the pooled "mean"/"std"; no baseline at all means mean 0 / std 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EPS = 1e-9
STD_FLOOR = 1e-6
CENTROIDS_KEY = "_regime_centroids"

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
_VIB, _KURT, _CREST = _COL["vibration_rms"], _COL["vibration_kurtosis"], _COL["vibration_crest"]
_BT, _MT, _CUR, _RPM, _LOAD = (
    _COL["bearing_temp_c"],
    _COL["motor_temp_c"],
    _COL["current_a"],
    _COL["rpm"],
    _COL["load_pct"],
)

Baseline = Mapping[str, Any]


def slope(y: np.ndarray) -> float:
    """Least-squares slope of `y` against its row index 0..n-1. Returns 0.0 when n < 2."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    return float(xc @ y / (xc @ xc))


def regime_ids(baseline: Baseline | None) -> list[str]:
    """Sorted regime ids present in the baseline's centroid table ([] when none)."""
    if not baseline:
        return []
    meta = baseline.get(CENTROIDS_KEY) or {}
    return sorted(k[: -len("_rpm")] for k in meta if str(k).endswith("_rpm"))


def assign_regime(rpm: np.ndarray, load: np.ndarray, baseline: Baseline) -> np.ndarray:
    """Index (into `regime_ids`) of the nearest regime centroid per element of rpm / load.

    Same formula as the training pipeline (`_assign_regime`); first minimum wins on ties.
    """
    ids = regime_ids(baseline)
    rpm = np.asarray(rpm, dtype=float)
    load = np.asarray(load, dtype=float)
    if not ids:
        return np.zeros(rpm.shape, dtype=int)
    meta = baseline[CENTROIDS_KEY]
    rs, ls = float(meta["rpm_scale"]), float(meta["load_pct_scale"])
    d = np.stack(
        [
            ((rpm - float(meta[f"{r}_rpm"])) / rs) ** 2
            + ((load - float(meta[f"{r}_load_pct"])) / ls) ** 2
            for r in ids
        ],
        axis=-1,
    )
    return np.argmin(d, axis=-1)


def _baseline_table(baseline: Baseline | None, sensor: str, stat: str) -> np.ndarray | None:
    """Per-regime `<regime>_<stat>` values for `sensor` (pooled value as fallback), or None
    when the baseline carries no entry for the sensor."""
    if not baseline:
        return None
    entry = baseline.get(sensor)
    if not entry:
        return None
    pooled = float(entry.get(stat, 0.0 if stat == "mean" else 1.0))
    ids = regime_ids(baseline)
    if not ids:
        return np.array([pooled])
    return np.array([float(entry.get(f"{r}_{stat}", pooled)) for r in ids])


def _window_baselines(
    rpm_rows: np.ndarray, load_rows: np.ndarray, baseline: Baseline | None
) -> tuple[float, float, float, float]:
    """(mu_v, sd_v, mu_t, sd_t) for one window: mean of the per-row regime baselines."""
    if not baseline:
        return 0.0, 1.0, 0.0, 1.0
    idx = assign_regime(rpm_rows, load_rows, baseline)
    out: list[float] = []
    for sensor, floor in (("vibration_rms", None), ("bearing_temp_c", None)):
        mean_t = _baseline_table(baseline, sensor, "mean")
        std_t = _baseline_table(baseline, sensor, "std")
        if mean_t is None or std_t is None:
            out += [0.0, 1.0]
            continue
        mu = float(mean_t[idx].mean()) if len(mean_t) > 1 else float(mean_t[0])
        sd = float(std_t[idx].mean()) if len(std_t) > 1 else float(std_t[0])
        out += [mu, max(sd, STD_FLOOR)]
        _ = floor
    return out[0], out[1], out[2], out[3]


def compute_features_np(
    buffer: np.ndarray,
    baseline: Baseline | None = None,
    regime: str | None = None,
) -> np.ndarray:
    """Compute the 13 FEATURE_NAMES from one raw window.

    Inputs: `buffer` of shape (window_rows, 8) in RAW_COLUMNS order, oldest row first; the
    sidecar baseline in the training format (see module docstring). `regime` is accepted for
    backward compatibility and IGNORED: the regime is inferred per row from rpm and load, as
    in training -- the edge never trusts an externally supplied regime label.
    Output: float64 array of shape (13,) in FEATURE_NAMES order.
    """
    buf = np.asarray(buffer, dtype=float)
    if buf.ndim != 2 or buf.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"buffer must be (window_rows, {len(RAW_COLUMNS)}), got {buf.shape}")
    if buf.shape[0] == 0:
        raise ValueError("buffer is empty")
    means = buf.mean(axis=0)
    vib_mean = means[_VIB]
    bearing_mean = means[_BT]
    mu_v, sd_v, mu_t, sd_t = _window_baselines(buf[:, _RPM], buf[:, _LOAD], baseline)
    return np.array(
        [
            vib_mean,
            slope(buf[:, _VIB]),
            means[_KURT],
            means[_CREST],
            bearing_mean,
            slope(buf[:, _BT]),
            bearing_mean - means[_MT],
            means[_CUR],
            means[_RPM],
            means[_LOAD],
            (vib_mean - mu_v) / sd_v,
            (bearing_mean - mu_t) / sd_t,
            vib_mean - mu_v,
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
    baseline: Baseline | None = None,
    regime: str | None = None,
) -> dict[str, float]:
    """Convenience wrapper: telemetry dict rows (oldest first) -> {feature_name: value}.

    Any "regime" key on the rows is ignored (see `compute_features_np`).
    """
    vec = compute_features_np(rows_to_matrix(rows), baseline, regime)
    return dict(zip(FEATURE_NAMES, vec.tolist(), strict=True))


def rolling_features(
    matrix: np.ndarray,
    window_rows: int,
    baseline: Baseline | None = None,
    regimes: Sequence[str] | None = None,
) -> np.ndarray:
    """Vectorized rolling version for one asset's full history.

    Inputs: (n, 8) matrix in RAW_COLUMNS order (time-ordered), window size, optional baseline
    (training format). `regimes` is accepted for backward compatibility and ignored: the
    regime of every raw row is inferred from its rpm / load. Output: (n - window_rows + 1, 13)
    array; row i corresponds to the window ending at raw row i + window_rows - 1 (same as
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
    vib_slope = (win[:, _VIB, :] @ xc) / denom if w > 1 else np.zeros(n - w + 1)
    brg_slope = (win[:, _BT, :] @ xc) / denom if w > 1 else np.zeros(n - w + 1)
    vib_mean = means[:, _VIB]
    brg_mean = means[:, _BT]

    k = n - w + 1
    vib_bm = np.zeros(k)
    vib_bs = np.ones(k)
    brg_bm = np.zeros(k)
    brg_bs = np.ones(k)
    if baseline:
        idx = assign_regime(win[:, _RPM, :], win[:, _LOAD, :], baseline)  # (k, w)
        for table_mean, table_std, bm, bs in (
            (
                _baseline_table(baseline, "vibration_rms", "mean"),
                _baseline_table(baseline, "vibration_rms", "std"),
                vib_bm,
                vib_bs,
            ),
            (
                _baseline_table(baseline, "bearing_temp_c", "mean"),
                _baseline_table(baseline, "bearing_temp_c", "std"),
                brg_bm,
                brg_bs,
            ),
        ):
            if table_mean is None or table_std is None:
                continue
            if len(table_mean) > 1:
                bm[:] = table_mean[idx].mean(axis=1)
                bs[:] = np.maximum(table_std[idx].mean(axis=1), STD_FLOOR)
            else:
                bm[:] = table_mean[0]
                bs[:] = max(float(table_std[0]), STD_FLOOR)

    return np.column_stack(
        [
            vib_mean,
            vib_slope,
            means[:, _KURT],
            means[:, _CREST],
            brg_mean,
            brg_slope,
            brg_mean - means[:, _MT],
            means[:, _CUR],
            means[:, _RPM],
            means[:, _LOAD],
            (vib_mean - vib_bm) / vib_bs,
            (brg_mean - brg_bm) / brg_bs,
            vib_mean - vib_bm,
        ]
    )
