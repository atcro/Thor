"""04 Data Reliability toolbox: schema validation, missingness, regime detection, profiling.

Every function here is deterministic local compute (no LLM, no database). The orchestrator
calls ``profile_dataset(df, asset_id)`` on the whole-fleet telemetry frame and hands the
resulting ``DataQualityContract`` to the ML Architect (05).

Telemetry frame shape: see docs/INTERFACES.md "Data shapes".
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from apps.api.schemas import (
    SENSOR_COLUMNS,
    DataQualityContract,
    MissingnessReport,
    Regime,
    RegimeReport,
    SchemaIssue,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("asset_id", "ts", *SENSOR_COLUMNS)
OPTIONAL_COLUMNS: tuple[str, ...] = ("regime", "health", "failure_within_h")

# Physically plausible ranges; values outside produce a *warning*, never an error.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "vibration_rms": (0.0, 100.0),
    "vibration_kurtosis": (0.0, 100.0),
    "vibration_crest": (0.0, 100.0),
    "bearing_temp_c": (-40.0, 250.0),
    "motor_temp_c": (-40.0, 250.0),
    "current_a": (0.0, 5000.0),
    "rpm": (0.0, 20000.0),
    "load_pct": (0.0, 150.0),
}

REGIME_LABELS_K3: tuple[str, ...] = ("idle/startup", "nominal", "high-load")
SILHOUETTE_SAMPLE_ROWS = 5000
FLATLINE_RUN_FRACTION = 0.5
MIN_TRAINABLE_SCORE = 60.0
MIN_TRAINABLE_ROWS = 500
MIN_TRAINABLE_ASSETS = 4


# --------------------------------------------------------------------------------------
# validate_schema
# --------------------------------------------------------------------------------------


def validate_schema(df: pd.DataFrame) -> list[SchemaIssue]:
    """Check a telemetry frame against the Telemetry contract.

    Input: telemetry DataFrame (any subset of the contract columns).
    Output: list of ``SchemaIssue`` with severity ``error`` (blocks training: missing required
    column, non-numeric sensor, unparseable ``ts``, empty frame), ``warning`` (naive ``ts``,
    implausible values, duplicate ``(asset_id, ts)``, ``health`` outside [0, 1], negative
    ``failure_within_h``) or ``info`` (missing optional column, frame not sorted).
    An empty list means the frame is fully conformant.
    """
    issues: list[SchemaIssue] = []
    if df is None or len(df) == 0:
        issues.append(SchemaIssue(column="__frame__", issue="empty frame", severity="error"))
        return issues

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(
                SchemaIssue(column=col, issue="required column missing", severity="error")
            )
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            issues.append(
                SchemaIssue(column=col, issue="optional column missing", severity="info")
            )

    if "ts" in df.columns:
        ts = df["ts"]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            try:
                ts = pd.to_datetime(ts, utc=True)
            except (ValueError, TypeError):
                issues.append(
                    SchemaIssue(column="ts", issue="not parseable as datetime", severity="error")
                )
                ts = None
        if ts is not None and getattr(ts.dt, "tz", None) is None:
            issues.append(
                SchemaIssue(column="ts", issue="timezone-naive timestamps", severity="warning")
            )

    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            issues.append(SchemaIssue(column=col, issue="non-numeric dtype", severity="error"))
            continue
        vals = s.to_numpy(dtype=float)
        n_inf = int(np.isinf(vals).sum())
        if n_inf:
            issues.append(
                SchemaIssue(column=col, issue=f"{n_inf} infinite values", severity="warning")
            )
        lo, hi = PLAUSIBLE_RANGES[col]
        finite = vals[np.isfinite(vals)]
        n_out = int(((finite < lo) | (finite > hi)).sum())
        if n_out:
            issues.append(
                SchemaIssue(
                    column=col,
                    issue=f"{n_out} values outside plausible range [{lo}, {hi}]",
                    severity="warning",
                )
            )

    if "health" in df.columns and pd.api.types.is_numeric_dtype(df["health"]):
        h = df["health"].dropna()
        n_bad = int(((h < 0.0) | (h > 1.0)).sum())
        if n_bad:
            issues.append(
                SchemaIssue(column="health", issue=f"{n_bad} values outside [0, 1]", severity="warning")
            )
    if "failure_within_h" in df.columns and pd.api.types.is_numeric_dtype(df["failure_within_h"]):
        n_neg = int((df["failure_within_h"].dropna() < 0.0).sum())
        if n_neg:
            issues.append(
                SchemaIssue(
                    column="failure_within_h", issue=f"{n_neg} negative values", severity="warning"
                )
            )

    if "asset_id" in df.columns and "ts" in df.columns:
        n_dup = int(df.duplicated(subset=["asset_id", "ts"]).sum())
        if n_dup:
            issues.append(
                SchemaIssue(
                    column="ts", issue=f"{n_dup} duplicate (asset_id, ts) rows", severity="warning"
                )
            )
        try:
            key = df[["asset_id", "ts"]]
            if not key.equals(key.sort_values(["asset_id", "ts"])):
                issues.append(
                    SchemaIssue(column="ts", issue="frame not sorted by (asset_id, ts)", severity="info")
                )
        except (TypeError, ValueError):
            pass
    return issues


# --------------------------------------------------------------------------------------
# detect_missingness
# --------------------------------------------------------------------------------------


def _gap_windows(df: pd.DataFrame, step_min: int) -> list[tuple[datetime, datetime]]:
    """Per-asset consecutive timestamp gaps longer than 1.5 x step_min."""
    gaps: list[tuple[datetime, datetime]] = []
    tol = pd.Timedelta(minutes=1.5 * step_min)
    for _, g in df[["asset_id", "ts"]].sort_values(["asset_id", "ts"]).groupby("asset_id", sort=False):
        ts = g["ts"]
        delta = ts.diff()
        for prev, cur in zip(ts.shift(1)[delta > tol], ts[delta > tol]):
            gaps.append((prev.to_pydatetime(), cur.to_pydatetime()))
    gaps.sort()
    return gaps


def detect_missingness(df: pd.DataFrame, step_min: int = 10) -> MissingnessReport:
    """Quantify missing data in a telemetry frame.

    Input: telemetry DataFrame; ``step_min`` is the nominal sampling step in minutes.
    Output: ``MissingnessReport`` with ``missing_fraction`` per sensor column (NaN share),
    ``gap_windows`` = per-asset ``(prev_ts, next_ts)`` pairs whose spacing exceeds 1.5 x step,
    and ``flatlined_sensors`` = sensors whose value repeats on more than 50 % of consecutive rows
    (median over assets) or that hold a single value fleet-wide.
    """
    if df is None or len(df) == 0:
        return MissingnessReport()
    sensors = [c for c in SENSOR_COLUMNS if c in df.columns]
    missing_fraction = {c: float(df[c].isna().mean()) for c in sensors}

    gaps: list[tuple[datetime, datetime]] = []
    if "ts" in df.columns and "asset_id" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        gaps = _gap_windows(df, step_min)

    flat: list[str] = []
    for c in sensors:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].dropna().nunique() <= 1:
            flat.append(c)
            continue
        if "asset_id" in df.columns:
            per_asset = df.groupby("asset_id", sort=False)[c].apply(
                lambda s: float((s.diff().dropna() == 0.0).mean()) if len(s) > 1 else 0.0
            )
            if len(per_asset) and float(per_asset.median()) > FLATLINE_RUN_FRACTION:
                flat.append(c)
    return MissingnessReport(
        missing_fraction=missing_fraction, gap_windows=gaps, flatlined_sensors=flat
    )


# --------------------------------------------------------------------------------------
# detect_regime
# --------------------------------------------------------------------------------------


def _regime_label(rank: int, k: int) -> str:
    if k == 3:
        return REGIME_LABELS_K3[rank]
    return f"regime-{rank + 1}"


def detect_regime(df: pd.DataFrame, k: int = 3, seed: int = 0) -> RegimeReport:
    """Cluster operating state from ``rpm`` and ``load_pct``.

    Input: telemetry DataFrame with numeric ``rpm`` and ``load_pct`` (NaNs are median-filled
    for clustering only); ``k`` clusters; ``seed`` for KMeans and the silhouette sample.
    Output: ``RegimeReport`` with ``k`` ``Regime`` entries named ``R1..Rk`` in ascending order of
    ``rpm_mean * load_mean`` (so R1 = idle/startup, R3 = high-load when k=3), ``row_regime``
    aligned to the input row order (``len == len(df)``), ``method="kmeans"`` and a silhouette
    score computed on a seeded sample of at most 5000 rows (``None`` when undefined).
    Degenerate inputs (fewer distinct points than ``k``) fall back to fewer clusters.
    """
    n = 0 if df is None else len(df)
    if n == 0:
        return RegimeReport(regimes=[], row_regime=[], method="kmeans", silhouette=None)
    for col in ("rpm", "load_pct"):
        if col not in df.columns:
            raise ValueError(f"detect_regime needs column '{col}'")

    X = df[["rpm", "load_pct"]].astype(float).copy()
    X = X.fillna(X.median()).fillna(0.0).to_numpy()
    n_unique = len(np.unique(X, axis=0))
    k_eff = max(1, min(k, n_unique))

    if k_eff == 1:
        labels = np.zeros(n, dtype=int)
    else:
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=seed)
        labels = km.fit_predict(Xs)

    # Order clusters by rpm*load ascending -> R1..Rk
    stats = []
    for c in range(k_eff):
        m = labels == c
        stats.append((float(X[m, 0].mean()) * float(X[m, 1].mean()), c))
    stats.sort()
    remap = {old: rank for rank, (_, old) in enumerate(stats)}
    ranks = np.array([remap[c] for c in labels])

    regimes: list[Regime] = []
    for rank in range(k_eff):
        m = ranks == rank
        regimes.append(
            Regime(
                regime_id=f"R{rank + 1}",
                label=_regime_label(rank, k_eff),
                n_rows=int(m.sum()),
                rpm_mean=float(X[m, 0].mean()),
                load_mean=float(X[m, 1].mean()),
                share=float(m.mean()),
            )
        )

    silhouette: float | None = None
    if k_eff >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(SILHOUETTE_SAMPLE_ROWS, n), replace=False)
        try:
            if len(np.unique(ranks[idx])) >= 2:
                silhouette = float(silhouette_score(Xs[idx], ranks[idx]))
        except ValueError:
            silhouette = None

    row_regime = [f"R{r + 1}" for r in ranks.tolist()]
    return RegimeReport(regimes=regimes, row_regime=row_regime, method="kmeans", silhouette=silhouette)


# --------------------------------------------------------------------------------------
# profile_dataset
# --------------------------------------------------------------------------------------


def profile_dataset(df: pd.DataFrame, asset_id: str, step_min: int = 10) -> DataQualityContract:
    """Produce the Data Quality Contract for a training run targeting ``asset_id``.

    Input: whole-fleet telemetry DataFrame (the model trains across assets), the target
    asset id, and the nominal sampling step in minutes.
    Output: ``DataQualityContract``. ``quality_score = 100 - penalties`` where penalties are:
    missing data (40 x mean sensor NaN fraction, max 40), timestamp gaps (2 per gap, max 10),
    flatlined sensors (10 each, max 30), schema issues (15 per error, 5 per warning, max 40),
    short window (25 if the frame spans < 24 h, 10 if < 72 h) and 20 if the target asset has
    no rows. ``trainable = score >= 60 and n_rows >= 500 and n_assets >= 4``.
    ``notes`` lists every penalty applied plus regime shares, for the AutoML Studio screen.
    """
    now = datetime.now(UTC)
    n_rows = 0 if df is None else int(len(df))
    if n_rows == 0:
        return DataQualityContract(
            asset_id=asset_id,
            window_start=now,
            window_end=now,
            n_rows=0,
            n_assets=0,
            sampling_rate_hz=1.0 / (step_min * 60.0),
            columns=[] if df is None else [str(c) for c in df.columns],
            quality_score=0.0,
            missingness=MissingnessReport(),
            regimes=RegimeReport(regimes=[], row_regime=[]),
            schema_issues=[SchemaIssue(column="__frame__", issue="empty frame", severity="error")],
            trainable=False,
            notes=["no telemetry rows; nothing to profile"],
        )

    issues = validate_schema(df)
    missing = detect_missingness(df, step_min=step_min)
    notes: list[str] = []
    penalties: dict[str, float] = {}

    can_cluster = "rpm" in df.columns and "load_pct" in df.columns
    if can_cluster:
        try:
            regimes = detect_regime(df)
        except (ValueError, TypeError) as exc:
            logger.warning("regime detection failed: %s", exc)
            regimes = RegimeReport(regimes=[], row_regime=[])
            notes.append(f"regime detection failed: {exc}")
    else:
        regimes = RegimeReport(regimes=[], row_regime=[])

    n_assets = int(df["asset_id"].nunique()) if "asset_id" in df.columns else 0
    if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        ts_min = df["ts"].min().to_pydatetime()
        ts_max = df["ts"].max().to_pydatetime()
    else:
        ts_min = ts_max = now
    span_h = (ts_max - ts_min).total_seconds() / 3600.0

    if missing.missing_fraction:
        mean_missing = float(np.mean(list(missing.missing_fraction.values())))
        if mean_missing > 0:
            penalties["missing"] = min(40.0, 40.0 * mean_missing)
    if missing.gap_windows:
        penalties["gaps"] = min(10.0, 2.0 * len(missing.gap_windows))
    if missing.flatlined_sensors:
        penalties["flatline"] = min(30.0, 10.0 * len(missing.flatlined_sensors))
    n_err = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warning")
    if n_err or n_warn:
        penalties["schema"] = min(40.0, 15.0 * n_err + 5.0 * n_warn)
    if span_h < 24.0:
        penalties["short_window"] = 25.0
    elif span_h < 72.0:
        penalties["short_window"] = 10.0
    target_rows = int((df["asset_id"] == asset_id).sum()) if "asset_id" in df.columns else 0
    if target_rows == 0:
        penalties["target_absent"] = 20.0
        notes.append(f"target asset {asset_id} has no rows in the window")
    else:
        notes.append(f"target asset {asset_id}: {target_rows} rows")

    score = float(np.clip(100.0 - sum(penalties.values()), 0.0, 100.0))
    for name, value in penalties.items():
        notes.append(f"penalty {name}: -{value:.1f}")
    notes.append(f"window span {span_h:.1f} h across {n_assets} assets, {n_rows} rows")
    for r in regimes.regimes:
        notes.append(
            f"{r.regime_id} {r.label}: share {r.share:.2f}, rpm {r.rpm_mean:.0f}, load {r.load_mean:.0f}%"
        )
    if regimes.silhouette is not None:
        notes.append(f"regime silhouette {regimes.silhouette:.2f}")

    trainable = (
        score >= MIN_TRAINABLE_SCORE
        and n_rows >= MIN_TRAINABLE_ROWS
        and n_assets >= MIN_TRAINABLE_ASSETS
    )
    if not trainable:
        notes.append(
            f"not trainable: need score >= {MIN_TRAINABLE_SCORE:.0f}, rows >= {MIN_TRAINABLE_ROWS}, "
            f"assets >= {MIN_TRAINABLE_ASSETS}"
        )

    return DataQualityContract(
        asset_id=asset_id,
        window_start=ts_min,
        window_end=ts_max,
        n_rows=n_rows,
        n_assets=n_assets,
        sampling_rate_hz=1.0 / (step_min * 60.0),
        columns=[str(c) for c in df.columns],
        quality_score=round(score, 2) if math.isfinite(score) else 0.0,
        missingness=missing,
        regimes=regimes,
        schema_issues=issues,
        trainable=bool(trainable),
        notes=notes,
    )
