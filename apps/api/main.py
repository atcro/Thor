"""Thor control plane -- FastAPI app (02).

Lifespan: init the database, seed assets/telemetry, build the manual RAG index in the
background (best effort), start the MQTT subscriber (best effort). All routers mount at root so
paths match docs/INTERFACES.md. Set `THOR_SKIP_BACKGROUND=1` to skip RAG + MQTT (tests).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api import db, seed
from apps.api.routes import approvals, copilot, fleet, models, pipeline
from apps.api.settings import get_settings

log = logging.getLogger("thor.main")

_mqtt_started = False


def _build_rag_index() -> None:
    """Best-effort Chroma index build from data/manuals (P1; never blocks startup)."""
    try:
        from agents.reliability.rag import build_index

        settings = get_settings()
        n = build_index(Path(settings.manuals_dir), Path(settings.chroma_path))
        log.info("RAG index ready: %d chunks", n)
        from agents.reliability.rag import build_field_history_index

        csv_path = Path(settings.field_history_csv)
        if csv_path.exists():
            n_cases = build_field_history_index(csv_path, Path(settings.chroma_path))
            log.info("field-history index ready: %d cases", n_cases)
        else:
            log.info("no field-history CSV at %s; Bolt precedent search disabled", csv_path)
    except ImportError:
        log.info("agents.reliability.rag not available; skipping RAG index")
    except Exception as e:
        log.warning("RAG index build failed: %s", e)


def _start_mqtt(engine: Any) -> None:
    global _mqtt_started
    try:
        from apps.api.routes.telemetry import start_mqtt_subscriber

        start_mqtt_subscriber(engine)
        _mqtt_started = True
    except ImportError:
        log.info("apps.api.routes.telemetry not available; MQTT subscriber not started")
    except Exception as e:
        log.warning("MQTT subscriber failed to start: %s", e)


def _stop_mqtt() -> None:
    if not _mqtt_started:
        return
    try:
        from apps.api.routes.telemetry import stop_mqtt_subscriber

        stop_mqtt_subscriber()
    except Exception as e:
        log.warning("MQTT subscriber stop failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown: init_db -> seed -> RAG (thread) -> MQTT; stop MQTT on exit."""
    engine = db.init_db()
    try:
        from apps.api.orchestrator import llm_status

        status = llm_status()
        log.info(
            "LLM mode: %s (provider %s, model %s)",
            status["mode"],
            status["provider"],
            status["model"],
        )
        if status["warning"]:
            log.warning(status["warning"])
    except Exception as e:
        log.warning("LLM status check failed: %s", e)
    try:
        counts = seed.seed_if_needed(engine)
        log.info("seed: %s", counts)
    except Exception as e:
        log.warning("seeding failed: %s", e)
    if os.environ.get("THOR_SKIP_BACKGROUND", "0") != "1":
        threading.Thread(target=_build_rag_index, name="thor-rag-index", daemon=True).start()
        _start_mqtt(engine)
    yield
    _stop_mqtt()


app = FastAPI(
    title="Thor control plane",
    version="0.1.0",
    description="Agentic AutoML + MLOps copilot for industrial predictive maintenance.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "thor-api"}


try:
    from apps.api.routes import telemetry as _telemetry_routes

    app.include_router(_telemetry_routes.router)
except ImportError:
    log.info("telemetry routes not available yet")

app.include_router(fleet.router)
app.include_router(pipeline.router)
app.include_router(approvals.router)
app.include_router(models.router)
app.include_router(copilot.router)
