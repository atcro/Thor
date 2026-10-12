"""API Key Up / API Key Down: the one switch for every LLM path in Thor.

Thin HTTP wrapper over `orchestrator.key_up` / `key_down` / `key_state`. Nothing here reads,
holds or returns a provider key (CLAUDE.md section 7): the state is `{"key": "up" | "down",
"provider": "anthropic" | "openai" | "none"}` and never anything else. A person flips it from
the header of every screen; the startup state is Down (`LLM_KEY_ON_START`).
"""

from __future__ import annotations

from fastapi import APIRouter

from apps.api import orchestrator
from apps.api.schemas import LlmKeyState

router = APIRouter(tags=["llm"])


@router.get("/llm/key", response_model=LlmKeyState)
def get_key_state() -> LlmKeyState:
    """Whether the provider key is held in memory (up) or dropped (down). Never the key."""
    return LlmKeyState(**orchestrator.key_state())


@router.post("/llm/key/up", response_model=LlmKeyState)
def key_up() -> LlmKeyState:
    """API Key Up: re-read the key from its source into memory; every LLM path turns on."""
    return LlmKeyState(**orchestrator.key_up())


@router.post("/llm/key/down", response_model=LlmKeyState)
def key_down() -> LlmKeyState:
    """API Key Down: drop the key from memory; every LLM path falls back to templates."""
    return LlmKeyState(**orchestrator.key_down())
