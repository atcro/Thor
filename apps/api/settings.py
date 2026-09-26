"""Process-wide configuration, read once from the environment (and .env if present).

Only apps/api/orchestrator.py may read anthropic_api_key / openai_api_key -- see CLAUDE.md
section 7.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "sqlite:///./thor.db"
    mlflow_tracking_uri: str = "./mlruns"
    mqtt_broker_url: str = "mqtt://localhost:1883"
    chroma_path: str = "./.chroma"
    manuals_dir: str = str(REPO_ROOT / "data" / "manuals")
    # Committed FMUCD slice indexed into a second Chroma collection for Bolt's field history.
    field_history_csv: str = str(REPO_ROOT / "data" / "field_history" / "fmucd_rotating_upm.csv")
    models_dir: str = "./models"
    replay_speed: float = 1800.0
    simulator_out: str = str(REPO_ROOT / "data" / "simulator" / "out")

    # LLM provider. "auto" picks openai when OPENAI_API_KEY is set, else anthropic when
    # ANTHROPIC_API_KEY is set, else template mode. Keys are read only in orchestrator.py.
    llm_provider: Literal["auto", "anthropic", "openai"] = "auto"
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_model: str = "claude-sonnet-5"
    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-5"

    # Plant-cost assumptions for the prescriptive layer. Configurable, never presented as fact.
    cost_planned_maintenance: float = 4200.0
    cost_unplanned_repair: float = 18500.0
    cost_downtime_per_hour: float = 3100.0
    downtime_planned_h: float = 3.0
    downtime_unplanned_h: float = 14.0

    @property
    def mqtt_host(self) -> str:
        return self.mqtt_broker_url.replace("mqtt://", "").split(":")[0]

    @property
    def mqtt_port(self) -> int:
        tail = self.mqtt_broker_url.replace("mqtt://", "").split(":")
        return int(tail[1]) if len(tail) > 1 else 1883


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
