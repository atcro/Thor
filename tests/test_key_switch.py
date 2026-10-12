"""API Key Up / API Key Down: one switch for every LLM path (Phase A.3, CLAUDE.md section 7).

Down: no client can be built, `_llm_enabled()` is False on every path, the cached Settings
hold blank key fields and the process environment / .env are untouched. Up re-reads the key
from its source. The status endpoint never carries the key. Startup state is Down unless
`LLM_KEY_ON_START=up`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api import orchestrator
from apps.api.settings import get_settings
from tests.test_orchestrator import configure_env, install_fake_toolbox

FAKE_ANTHROPIC = "sk-ant-api03-not-a-real-key-0123456789"
FAKE_OPENAI = "sk-proj-not-a-real-key-0123456789"


def _env(monkeypatch: pytest.MonkeyPatch, anthropic: str = "", openai: str = "") -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic)
    monkeypatch.setenv("OPENAI_API_KEY", openai)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("LLM_KEY_ON_START", raising=False)
    get_settings.cache_clear()
    orchestrator._KEY_UP = None


def test_starts_down_and_every_path_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, anthropic=FAKE_ANTHROPIC)
    # first use initialises from LLM_KEY_ON_START (default down)
    assert orchestrator.key_state() == {"key": "down", "provider": "none"}
    assert orchestrator._llm_enabled() is False
    assert orchestrator._api_key() == "" and orchestrator._openai_key() == ""
    with pytest.raises(RuntimeError):
        orchestrator._llm_client()
    with pytest.raises(RuntimeError):
        orchestrator._openai_client()
    status = orchestrator.llm_status()
    assert status["mode"] == "template" and status["key"] == "down"
    assert status["key_configured"] is False
    # nothing in Thor's runtime holds the key: the cached Settings fields are blank ...
    s = get_settings()
    assert s.anthropic_api_key == "" and s.openai_api_key == ""
    # ... while the source (the process environment here) is untouched
    assert os.environ["ANTHROPIC_API_KEY"] == FAKE_ANTHROPIC
    assert orchestrator.llm_ping()["ok"] is False


def test_up_down_up_restores_from_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, anthropic=FAKE_ANTHROPIC)
    assert orchestrator.key_up() == {"key": "up", "provider": "anthropic"}
    assert orchestrator._llm_enabled() is True and orchestrator._api_key() == FAKE_ANTHROPIC
    assert orchestrator.llm_status()["mode"] == "llm"
    assert orchestrator.key_down() == {"key": "down", "provider": "none"}
    assert orchestrator._llm_enabled() is False and orchestrator._api_key() == ""
    assert get_settings().anthropic_api_key == ""
    assert orchestrator.key_up() == {"key": "up", "provider": "anthropic"}
    assert orchestrator._api_key() == FAKE_ANTHROPIC
    # the OpenAI path is governed by the same switch
    _env(monkeypatch, openai=FAKE_OPENAI)
    assert orchestrator.key_state()["provider"] == "none"
    assert orchestrator.key_up() == {"key": "up", "provider": "openai"}
    assert orchestrator._llm_enabled() is True
    assert orchestrator.key_down()["provider"] == "none"
    with pytest.raises(RuntimeError):
        orchestrator._openai_client()


def test_up_without_a_configured_key_stays_template(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    assert orchestrator.key_up() == {"key": "up", "provider": "none"}
    assert orchestrator._llm_enabled() is False
    assert orchestrator.llm_status()["mode"] == "template"


def test_llm_key_on_start_up(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, anthropic=FAKE_ANTHROPIC)
    monkeypatch.setenv("LLM_KEY_ON_START", "up")
    assert orchestrator.key_state() == {"key": "up", "provider": "anthropic"}
    monkeypatch.setenv("LLM_KEY_ON_START", "down")
    orchestrator._KEY_UP = None
    assert orchestrator.key_state()["key"] == "down"


def test_copilot_and_draft_fall_back_to_templates_while_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bolt and the contract explanation use the template while Down, even with a key at the
    source, and the LLM is never called."""
    from apps.api import db
    from apps.api.schemas import CopilotMessage, CopilotRequest
    from tests.test_orchestrator import make_bundle, seed_small_fleet

    configure_env(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    get_settings.cache_clear()
    orchestrator._KEY_UP = None
    engine = db.init_db()
    seed_small_fleet(engine)
    calls: list[str] = []

    def boom() -> object:
        calls.append("client")
        raise AssertionError("no client may be built while the key is down")

    monkeypatch.setattr(orchestrator, "_llm_client", boom)
    monkeypatch.setattr(orchestrator, "_openai_client", boom)
    monkeypatch.setattr(orchestrator, "tool_template_explanation", lambda b: "template text")
    assert orchestrator.draft_explanation(make_bundle()) == ("template text", "template")
    resp = orchestrator.copilot_reply(
        CopilotRequest(messages=[CopilotMessage(role="user", content="How is MTR-042 doing?")]),
        engine=engine,
    )
    assert resp.source == "template" and resp.reply.startswith("MTR-042")
    assert calls == []


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    configure_env(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    get_settings.cache_clear()
    orchestrator._KEY_UP = None
    install_fake_toolbox(monkeypatch, orchestrator)
    from apps.api.main import app

    with TestClient(app) as c:
        yield c


def test_key_routes_never_carry_the_key(client: TestClient) -> None:
    r = client.get("/llm/key")
    assert r.status_code == 200 and r.json() == {"key": "down", "provider": "none"}
    health = client.get("/system/health").json()
    assert health["llm"]["key"] == "down" and health["llm"]["mode"] == "template"
    r = client.post("/llm/key/up")
    assert r.status_code == 200 and r.json() == {"key": "up", "provider": "anthropic"}
    assert FAKE_ANTHROPIC not in r.text and "not-a-real" not in r.text
    health = client.get("/system/health").json()
    assert health["llm"]["key"] == "up" and health["llm"]["mode"] == "llm"
    assert FAKE_ANTHROPIC not in client.get("/system/health").text
    r = client.post("/llm/key/down")
    assert r.json() == {"key": "down", "provider": "none"}
    assert client.get("/llm/key").json()["key"] == "down"
    assert set(r.json()) == {"key", "provider"}


def test_only_orchestrator_touches_the_switch_state() -> None:
    """The routes and the UI go through key_up / key_down / key_state; no other module reads
    or writes the switch, the key fields or the provider env vars (CLAUDE.md section 7)."""
    import re

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for p in list((root / "apps").rglob("*.py")) + list((root / "agents").rglob("*.py")):
        if p.name in ("orchestrator.py", "settings.py"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"_KEY_UP|_blank_key_fields|_key_is_up", text):
            offenders.append(str(p))
    assert offenders == []
