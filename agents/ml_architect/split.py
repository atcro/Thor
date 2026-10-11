"""Deterministic, seeded asset-level split -- made BEFORE any preprocessing is fitted.

Why this exists (CLAUDE.md section 7, "never validate with a random shuffle split"): regimes,
healthy baselines and scales must be fitted on training motors only, the Optuna objective
must look at motors the model never trained on, calibration must use motors used for neither
training nor the final score, and the reported metrics must come from untouched test motors.
That is only possible if the groups are fixed first and recorded with the run.

Groups
------
* ``train``        -- the model is fitted on these motors only; regimes / baselines too.
* ``validation``   -- the Optuna objective (model + hyper-parameter selection).
* ``calibration``  -- isotonic / sigmoid calibration of the champion.
* ``test``         -- final metrics, never touched before.

Completed failures (the only motors that carry positive labels) are scarce: the demo fleet
has four. They are dealt to the groups in the fixed order train, test, tune, train, test,
tune, ... so training and the final test always get one first. ``tune`` is one group that
serves as BOTH validation and calibration unless there are at least
``MIN_POSITIVES_FOR_FOUR_GROUPS`` completed failures, in which case it is split in two. The
report says which arrangement was used and why.

The asset under analysis can be forced into ``test`` so its score is out-of-sample.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from apps.api.schemas import AssetSplit

GROUP_FRACTIONS: dict[str, float] = {"train": 0.5, "test": 0.25, "tune": 0.25}
#: Completed failures are dealt in this order (train and test get one before tune).
POSITIVE_ORDER: tuple[str, ...] = ("train", "test", "tune")
MIN_POSITIVES_FOR_FOUR_GROUPS = 5
#: The simulator counts down to ~0 h at the failure instant; a motor whose countdown never
#: reaches this value did not complete its failure inside the data (same rule as
#: ``build_feature_pipeline``), so it has no positive label.
COMPLETED_FAILURE_MAX_H = 1.0


def completed_failure_assets(df: pd.DataFrame) -> list[str]:
    """Assets with a COMPLETED failure in `df` (sorted ids).

    Input: a telemetry frame with ``failure_within_h`` or a feature frame with ``rul_h``.
    Output: ids whose countdown reaches <= ``COMPLETED_FAILURE_MAX_H``. An in-progress
    degradation (countdown still above that at the last row) is an unknown future, not a
    label, so it does not count.
    """
    col = "failure_within_h" if "failure_within_h" in df.columns else "rul_h"
    if col not in df.columns or len(df) == 0:
        return []
    vals = pd.to_numeric(df[col], errors="coerce")
    low = vals.groupby(df["asset_id"].astype(str)).min()
    return sorted(str(a) for a, v in low.items() if pd.notna(v) and v <= COMPLETED_FAILURE_MAX_H)


def make_asset_split(
    df: pd.DataFrame,
    seed: int = 0,
    forced_test: Iterable[str] = (),
    fractions: dict[str, float] | None = None,
) -> AssetSplit:
    """Build the seeded asset-level split for one run.

    Input: a telemetry frame (``asset_id`` + ``failure_within_h``) or feature frame
    (``asset_id`` + ``rul_h``), the RNG seed, asset ids forced into ``test`` (the asset under
    analysis) and optional group fractions (train / test / tune of the asset count).
    Output: ``AssetSplit`` with disjoint ``train`` / ``validation`` / ``calibration`` /
    ``test`` lists covering every asset, ``positives`` per group, and ``notes`` explaining the
    arrangement. Deterministic in (seed, asset ids, completed failures, forced_test).
    """
    frac = dict(GROUP_FRACTIONS if fractions is None else fractions)
    assets = sorted(str(a) for a in df["asset_id"].unique())
    n = len(assets)
    positives = completed_failure_assets(df)
    forced = [a for a in assets if a in set(forced_test)]
    rng = np.random.default_rng(seed)

    groups: dict[str, list[str]] = {"train": [], "test": list(forced), "tune": []}
    pos_free = [a for a in positives if a not in forced]
    pos_free = [pos_free[i] for i in rng.permutation(len(pos_free))]
    for i, a in enumerate(pos_free):
        groups[POSITIVE_ORDER[i % len(POSITIVE_ORDER)]].append(a)

    healthy = [a for a in assets if a not in set(positives) and a not in forced]
    healthy = [healthy[i] for i in rng.permutation(len(healthy))]
    targets = {g: int(round(frac[g] * n)) for g in ("train", "test", "tune")}
    targets["train"] = max(targets["train"], n - targets["test"] - targets["tune"])
    for g in ("train", "test", "tune"):
        while healthy and len(groups[g]) < targets[g]:
            groups[g].append(healthy.pop(0))
    groups["train"].extend(healthy)  # any remainder trains

    notes: list[str] = [
        f"{n} assets, {len(positives)} with a completed failure: {positives or '-'}",
        f"seed {seed}; completed failures dealt in order {', '.join(POSITIVE_ORDER)}",
    ]
    if forced:
        notes.append(f"forced into test (asset under analysis, never trained on): {forced}")

    tune = sorted(groups["tune"])
    if len(positives) >= MIN_POSITIVES_FOR_FOUR_GROUPS and len(tune) >= 2:
        tune_pos = [a for a in tune if a in set(positives)]
        tune_neg = [a for a in tune if a not in set(positives)]
        validation = sorted(tune_pos[0::2] + tune_neg[0::2])
        calibration = sorted(tune_pos[1::2] + tune_neg[1::2])
        notes.append(
            f"{len(positives)} completed failures: validation and calibration are separate groups"
        )
    else:
        validation = calibration = tune
        notes.append(
            "validation and calibration share the tune group: only "
            f"{len(positives)} completed failure(s), fewer than "
            f"{MIN_POSITIVES_FOR_FOUR_GROUPS} needed for four groups with a positive each"
        )
    out = AssetSplit(
        seed=int(seed),
        train=sorted(groups["train"]),
        validation=validation,
        calibration=calibration,
        test=sorted(groups["test"]),
        forced_test=sorted(forced),
        positives={
            "train": sorted(a for a in groups["train"] if a in set(positives)),
            "validation": sorted(a for a in validation if a in set(positives)),
            "calibration": sorted(a for a in calibration if a in set(positives)),
            "test": sorted(a for a in groups["test"] if a in set(positives)),
        },
        notes=notes,
    )
    if not out.positives["train"]:
        out.notes.append("WARNING: no completed failure among training assets; training will fail")
    if not out.positives["test"]:
        out.notes.append("WARNING: no completed failure among test assets; lead time cannot be measured")
    return out


def rows_in(df: pd.DataFrame, assets: Iterable[str], known_only: bool = True) -> pd.DataFrame:
    """Rows of `df` belonging to `assets`, optionally only those with a known label.

    Input: a feature frame (``asset_id``, optional ``label_known``), asset ids. Output: the
    filtered frame (copy). ``known_only`` drops right-censored rows -- the last horizon of a
    motor with no completed failure, whose label cannot be known yet.
    """
    mask = df["asset_id"].astype(str).isin(set(assets))
    if known_only and "label_known" in df.columns:
        mask &= df["label_known"].astype(bool)
    return df.loc[mask].copy()
