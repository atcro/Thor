"""API-key placeholder hygiene: .env never reaches git or Docker, and the LLM mode is
observable without exposing the key (CLAUDE.md section 7)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from apps.api.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]


def _git_ignored(path: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True, check=False
    )
    return proc.returncode == 0


def test_env_files_are_gitignored() -> None:
    assert _git_ignored(".env")
    assert _git_ignored(".env.local")
    assert not _git_ignored(".env.example"), ".env.example must stay committed"


def test_env_files_are_dockerignored() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines and ".env.*" in lines


def test_env_example_has_blank_key() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=\n" in text, "the template must ship with a blank key"


def _set_key(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    get_settings.cache_clear()


def test_llm_status_template_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    _set_key(monkeypatch, "")
    status = orchestrator.llm_status()
    assert status["mode"] == "template"
    assert status["key_configured"] is False
    assert status["warning"] is None
    assert "sk-ant" not in str(status)


def test_llm_status_llm_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    _set_key(monkeypatch, "sk-ant-api03-not-a-real-key")
    status = orchestrator.llm_status()
    assert status["mode"] == "llm" and status["key_configured"] is True
    assert status["warning"] is None
    assert "not-a-real-key" not in str(status), "status must never echo the key"

    _set_key(monkeypatch, "pasted-wrong-thing")
    assert orchestrator.llm_status()["warning"] is not None


def test_llm_ping_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    _set_key(monkeypatch, "")
    out = orchestrator.llm_ping()
    assert out["ok"] is False and "blank" in out["error"]


def test_llm_ping_with_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    _set_key(monkeypatch, "sk-ant-api03-not-a-real-key")
    seen: dict[str, Any] = {}

    class FakeMessages:
        def create(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="ready")])
            msg._request_id = "req_test"
            return msg

    monkeypatch.setattr(
        orchestrator, "_llm_client", lambda: SimpleNamespace(messages=FakeMessages())
    )
    out = orchestrator.llm_ping()
    assert out == {
        "ok": True,
        "model": get_settings().anthropic_model,
        "reply": "ready",
        "request_id": "req_test",
    }
    assert seen["max_tokens"] <= 64, "the ping must stay tiny"


def test_llm_ping_reports_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    _set_key(monkeypatch, "sk-ant-api03-not-a-real-key")

    def boom() -> Any:
        raise RuntimeError("no network")

    monkeypatch.setattr(orchestrator, "_llm_client", boom)
    out = orchestrator.llm_ping()
    assert out["ok"] is False and "RuntimeError: no network" in out["error"]
