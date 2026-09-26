"""Shared pytest fixtures for Thor.

`synthetic_fleet` is a small, seeded, dependency-free stand-in for the real simulator
(`data/simulator/generate.py`). It produces the exact Telemetry DataFrame shape described in
docs/INTERFACES.md ("Data shapes") so every agent's tests can run before Agent A's generator
lands. Keep it fast (< 1 s) and deterministic; other test modules reuse it.

Environment: MLflow tracking, the models dir and the SQLite database are pointed at a fresh
temp directory (via ``setdefault`` so an explicit environment still wins) before
``apps.api.settings.get_settings`` is first called.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="thor-tests-"))
os.environ.setdefault("MLFLOW_TRACKING_URI", (_TMP_ROOT / "mlruns").resolve().as_uri())
os.environ.setdefault("MODELS_DIR", (_TMP_ROOT / "models").as_posix())
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP_ROOT / 'thor-test.db').as_posix()}")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# Never let a real provider key from .env reach a test: tests that exercise LLM paths set a
# fake key themselves and patch the client. (Forced, not setdefault, on purpose.)
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "auto"

try:  # settings is lru_cached; make sure the temp env above is what it sees
    from apps.api.settings import get_settings

    get_settings.cache_clear()
except Exception:  # pragma: no cover - settings module missing in a partial checkout
    pass

TELEMETRY_COLUMNS: list[str] = [
    "asset_id",
    "ts",
    "vibration_rms",
    "vibration_kurtosis",
    "vibration_crest",
    "bearing_temp_c",
    "motor_temp_c",
    "current_a",
    "rpm",
    "load_pct",
    "regime",
    "health",
    "failure_within_h",
]

# (rpm, load_pct) centres. R1 idle/startup, R2 nominal, R3 high-load.
REGIME_PARAMS: dict[str, tuple[float, float]] = {
    "R1": (600.0, 10.0),
    "R2": (1480.0, 55.0),
    "R3": (1500.0, 85.0),
}


def _shift_regime(hour_of_day: np.ndarray) -> np.ndarray:
    """Map hour-of-day (0..24) to a regime id on a fixed three-shift schedule."""
    out = np.full(hour_of_day.shape, "R1", dtype=object)
    out[(hour_of_day >= 6) & (hour_of_day < 14)] = "R2"
    out[(hour_of_day >= 14) & (hour_of_day < 22)] = "R3"
    return out


def make_synthetic_fleet(
    n_assets: int = 8,
    days: int = 6,
    step_min: int = 10,
    seed: int = 7,
    degrading: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a Telemetry-shaped frame for `n_assets` motors over `days` days.

    Inputs: fleet size, horizon in days, sampling step in minutes, RNG seed and an optional
    ``{asset_id: onset_fraction}`` map of degrading assets. By default two assets degrade:
    the third asset (onset at 55 % of the horizon) and the last asset (onset at 60 %), so one
    failing motor lands in the training assets and one in the last-25 %-by-id holdout.

    Output: DataFrame with columns ``TELEMETRY_COLUMNS`` sorted by ``(asset_id, ts)``. ``ts`` is
    tz-aware UTC at ``step_min`` spacing. Degrading assets show a smooth monotone rise in
    vibration_rms / vibration_kurtosis / vibration_crest / bearing_temp_c after onset,
    ``health`` decaying 1 -> 0 and ``failure_within_h`` counting down to 0 at the last row.
    Healthy assets have ``health = 1.0`` and ``failure_within_h = NaN``.
    """
    rng = np.random.default_rng(seed)
    asset_ids = [f"MTR-{i + 1:03d}" for i in range(n_assets)]
    if degrading is None:
        degrading = {asset_ids[min(2, n_assets - 1)]: 0.55, asset_ids[-1]: 0.60}

    n_rows = days * 24 * 60 // step_min
    ts = pd.date_range("2025-01-01", periods=n_rows, freq=f"{step_min}min", tz="UTC")
    t = np.arange(n_rows, dtype=float)
    hours = t * step_min / 60.0

    frames: list[pd.DataFrame] = []
    for idx, asset_id in enumerate(asset_ids):
        hod = (hours + 3.0 * idx) % 24.0
        regime = _shift_regime(hod)
        rpm_base = np.array([REGIME_PARAMS[r][0] for r in regime])
        load_base = np.array([REGIME_PARAMS[r][1] for r in regime])

        rpm = rpm_base + rng.normal(0.0, 12.0, n_rows)
        load = np.clip(load_base + rng.normal(0.0, 3.0, n_rows), 0.0, 100.0)
        vib_rms = 0.6 + 0.020 * load + rng.normal(0.0, 0.06, n_rows)
        vib_kurt = 2.8 + 0.006 * load + rng.normal(0.0, 0.12, n_rows)
        vib_crest = 3.2 + 0.010 * load + rng.normal(0.0, 0.10, n_rows)
        bearing_t = 34.0 + 0.30 * load + rng.normal(0.0, 0.7, n_rows)
        motor_t = 32.0 + 0.34 * load + rng.normal(0.0, 0.7, n_rows)
        current = 18.0 + 1.10 * load + rng.normal(0.0, 0.8, n_rows)

        health = np.ones(n_rows)
        fail_h = np.full(n_rows, np.nan)
        if asset_id in degrading:
            onset = int(degrading[asset_id] * n_rows)
            sev = np.clip((t - onset) / max(1.0, n_rows - 1 - onset), 0.0, 1.0) ** 1.5
            vib_rms = vib_rms + 2.5 * sev
            vib_kurt = vib_kurt + 4.0 * sev
            vib_crest = vib_crest + 2.0 * sev
            bearing_t = bearing_t + 22.0 * sev
            health = np.clip(1.0 - sev, 0.0, 1.0)
            fail_h = (n_rows - 1 - t) * step_min / 60.0

        frames.append(
            pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "ts": ts,
                    "vibration_rms": vib_rms,
                    "vibration_kurtosis": vib_kurt,
                    "vibration_crest": vib_crest,
                    "bearing_temp_c": bearing_t,
                    "motor_temp_c": motor_t,
                    "current_a": current,
                    "rpm": rpm,
                    "load_pct": load,
                    "regime": regime.astype(str),
                    "health": health,
                    "failure_within_h": fail_h,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return df[TELEMETRY_COLUMNS].sort_values(["asset_id", "ts"]).reset_index(drop=True)


_FLEET_CACHE: dict[str, pd.DataFrame] = {}


@pytest.fixture
def synthetic_fleet() -> pd.DataFrame:
    """Default fleet (8 assets, 6 days, 10-min step). Returns a fresh copy per test."""
    if "default" not in _FLEET_CACHE:
        _FLEET_CACHE["default"] = make_synthetic_fleet()
    return _FLEET_CACHE["default"].copy()


@pytest.fixture
def degrading_assets() -> list[str]:
    """Asset ids that degrade in the default `synthetic_fleet` (train one, holdout one)."""
    return ["MTR-003", "MTR-008"]


@pytest.fixture
def tmp_artifacts_dir(tmp_path: Path) -> Path:
    """Per-test directory for model artifacts."""
    d = tmp_path / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    return d
