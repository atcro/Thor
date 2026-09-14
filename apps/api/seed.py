"""Startup seeding: assets + bulk telemetry from `data/simulator/out/`.

`seed_if_needed()` is idempotent: it only inserts assets when the assets table is empty and
only bulk-loads telemetry when the telemetry table is empty. When the simulator output is
missing and `SEED_ON_START=1`, it generates the fleet via `data.simulator.generate`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from apps.api import db
from apps.api.schemas import Asset, TelemetryRow
from apps.api.settings import get_settings

log = logging.getLogger("thor.seed")

CHUNK = 5000


def _telemetry_count(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(db.telemetry)).scalar() or 0)


def _generate(out_dir: Path) -> bool:
    """Generate simulator output into `out_dir` (Agent A's module); False if unavailable."""
    try:
        from data.simulator.generate import generate_fleet, write_outputs
    except ImportError as e:
        log.warning("simulator module unavailable, cannot generate seed data: %s", e)
        return False
    try:
        df, assets = generate_fleet()
        out_dir.mkdir(parents=True, exist_ok=True)
        write_outputs(df, assets, out_dir)
        return True
    except Exception as e:
        log.warning("simulator generation failed: %s", e)
        return False


def _load_assets(path: Path) -> list[Asset]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("assets", [])
    return [Asset.model_validate(a) for a in raw]


def _row_from_record(rec: dict[str, Any]) -> TelemetryRow:
    clean: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, float) and pd.isna(v):
            clean[k] = None
        elif isinstance(v, pd.Timestamp):
            clean[k] = v.to_pydatetime()
        else:
            clean[k] = v
    return TelemetryRow.model_validate(clean)


def load_telemetry_parquet(path: Path, engine: Engine) -> int:
    """Bulk-insert a telemetry parquet file in chunks. Returns rows inserted."""
    df = pd.read_parquet(path)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    keep = [c.name for c in db.telemetry.columns if c.name in df.columns]
    df = df[keep].sort_values(["asset_id", "ts"]).reset_index(drop=True)
    # Load only the head of the timeline; the MQTT replay streams the rest live so the demo
    # motor's degradation is watched happening, not pre-loaded. SEED_FRACTION=1 loads all.
    fraction = float(os.environ.get("SEED_FRACTION", "0.80"))
    if 0.0 < fraction < 1.0 and "ts" in df.columns and len(df):
        t0, t1 = df["ts"].min(), df["ts"].max()
        cutoff = t0 + (t1 - t0) * fraction
        df = df.loc[df["ts"] <= cutoff].reset_index(drop=True)
    total = 0
    for start in range(0, len(df), CHUNK):
        chunk = df.iloc[start : start + CHUNK]
        rows = [_row_from_record(r) for r in chunk.to_dict("records")]
        total += db.insert_telemetry(rows, engine=engine)
    return total


def seed_if_needed(engine: Engine | None = None) -> dict[str, int]:
    """Seed assets and telemetry if the tables are empty. Returns counts inserted."""
    engine = engine or db.get_engine()
    settings = get_settings()
    out_dir = Path(settings.simulator_out)
    assets_path = out_dir / "assets.json"
    parquet_path = out_dir / "telemetry.parquet"
    result = {"assets": 0, "telemetry": 0}

    n_assets = len(db.list_assets(engine))
    n_rows = _telemetry_count(engine)
    if n_assets and n_rows:
        return result
    if not assets_path.exists() and os.environ.get("SEED_ON_START", "0") == "1":
        _generate(out_dir)
    if n_assets == 0 and assets_path.exists():
        try:
            result["assets"] = db.upsert_assets(_load_assets(assets_path), engine=engine)
            log.info("seeded %d assets from %s", result["assets"], assets_path)
        except Exception as e:
            log.warning("asset seeding failed: %s", e)
    if n_rows == 0 and parquet_path.exists():
        try:
            result["telemetry"] = load_telemetry_parquet(parquet_path, engine)
            log.info("bulk-loaded %d telemetry rows from %s", result["telemetry"], parquet_path)
        except Exception as e:
            log.warning("telemetry bulk load failed: %s", e)
    return result
