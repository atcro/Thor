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
    monkeypatch.setenv("OPENAI_API_KEY", "")  # a real key in .env must never reach a test
    monkeypatch.setenv("LLM_PROVIDER", "auto")
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


def _set_openai(monkeypatch: pytest.MonkeyPatch, key: str = "sk-proj-not-a-real-key") -> None:
    _set_key(monkeypatch, "")
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5")
    get_settings.cache_clear()


def test_llm_status_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    _set_openai(monkeypatch)
    status = orchestrator.llm_status()
    assert status["provider"] == "openai" and status["mode"] == "llm"
    assert status["model"] == "gpt-5" and status["key_configured"] is True
    assert status["warning"] is None
    assert "not-a-real-key" not in str(status)
    assert orchestrator._api_key() == "", "the Anthropic client must never see an OpenAI key"
    assert orchestrator._llm_enabled() is True

    _set_openai(monkeypatch, "pasted-wrong")
    assert orchestrator.llm_status()["warning"] is not None

    # explicit provider pin wins over auto-detection
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    get_settings.cache_clear()
    assert orchestrator.llm_status()["provider"] == "none"
    assert orchestrator._llm_enabled() is False
    monkeypatch.delenv("LLM_PROVIDER")
    get_settings.cache_clear()


class _FakeOpenAI:
    """Minimal stand-in for openai.OpenAI: scripted chat.completions.create responses."""

    def __init__(self, scripted: list[Any]) -> None:
        self.scripted = list(scripted)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        msg = self.scripted.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            id="chatcmpl_test",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


def _oa_msg(content: str | None = None, tool_calls: list[tuple[str, str, dict]] | None = None):
    return SimpleNamespace(
        content=content,
        refusal=None,
        tool_calls=[
            SimpleNamespace(
                id=cid, function=SimpleNamespace(name=name, arguments=__import__("json").dumps(a))
            )
            for cid, name, a in (tool_calls or [])
        ],
    )


def test_openai_tools_translation() -> None:
    from apps.api import orchestrator

    tools = orchestrator._openai_tools(orchestrator.COPILOT_TOOLS)
    assert {t["type"] for t in tools} == {"function"}
    names = [t["function"]["name"] for t in tools]
    assert "search_manuals" in names and "field_history_stats" in names
    assert tools[0]["function"]["parameters"] == orchestrator.COPILOT_TOOLS[0]["input_schema"]


def test_llm_ping_openai_with_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator

    _set_openai(monkeypatch)
    fake = _FakeOpenAI([_oa_msg("ready")])
    monkeypatch.setattr(orchestrator, "_openai_client", lambda: fake)
    out = orchestrator.llm_ping()
    assert out == {
        "ok": True,
        "provider": "openai",
        "model": "gpt-5",
        "reply": "ready",
        "request_id": "chatcmpl_test",
    }
    req = fake.requests[0]
    assert req["model"] == "gpt-5" and req["max_completion_tokens"] >= 4096
    assert req["reasoning_effort"] == "low", "gpt-5 family is a reasoning model"


def test_openai_kwargs_skip_reasoning_effort_for_non_reasoning_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.api import orchestrator

    _set_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    get_settings.cache_clear()
    kw = orchestrator._openai_kwargs(1024)
    assert kw == {"max_completion_tokens": 4096}
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")
    get_settings.cache_clear()
    assert orchestrator._openai_kwargs(2000) == {
        "max_completion_tokens": 8000,
        "reasoning_effort": "low",
    }


def test_route_openai_tool_choice_is_advisory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The planner's choice is recorded but the legal transition always wins."""
    from apps.api import orchestrator
    from apps.api.schemas import GraphState, PipelineStage

    _set_openai(monkeypatch)
    fake = _FakeOpenAI([_oa_msg(None, [("call_1", "choose_next_node", {"node": "finalize"})])])
    monkeypatch.setattr(orchestrator, "_openai_client", lambda: fake)
    gs = GraphState(run_id="r-oa", asset_id="MTR-042", stage=PipelineStage.profiling)
    nxt = orchestrator.route(gs)
    assert nxt == orchestrator.STAGE_TO_NEXT[PipelineStage.profiling]
    ev = [e for e in gs.events if e.tool == "route_llm"][0]
    assert ev.payload["llm_choice"] == "finalize" and ev.payload["accepted"] is False
    assert gs.llm_calls == 1
    req = fake.requests[0]
    assert req["tools"][0]["function"]["name"] == "choose_next_node"
    assert req["parallel_tool_calls"] is False and req["messages"][0]["role"] == "system"


def test_draft_explanation_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import orchestrator
    from tests.test_orchestrator import make_bundle

    _set_openai(monkeypatch)
    fake = _FakeOpenAI([_oa_msg("  The bearing is the driver.  ")])
    monkeypatch.setattr(orchestrator, "_openai_client", lambda: fake)
    text, source = orchestrator.draft_explanation(make_bundle())
    assert (text, source) == ("The bearing is the driver.", "llm")
    assert fake.requests[0]["messages"][0]["content"] == orchestrator.EXPLANATION_SYSTEM_PROMPT

    refused = _oa_msg(None)
    refused.refusal = "no"
    fake2 = _FakeOpenAI([refused])
    monkeypatch.setattr(orchestrator, "_openai_client", lambda: fake2)
    assert orchestrator.draft_explanation(make_bundle())[1] == "template"


def test_copilot_openai_tool_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool call -> tool result message -> final text, with the tool actually executed."""
    from apps.api import orchestrator
    from apps.api.schemas import CopilotMessage, CopilotRequest

    _set_openai(monkeypatch)
    fake = _FakeOpenAI(
        [
            _oa_msg(None, [("call_a", "search_manuals", {"query": "kurtosis", "k": 1})]),
            _oa_msg("Kurtosis rising means impacts. (vibration guide)"),
        ]
    )
    monkeypatch.setattr(orchestrator, "_openai_client", lambda: fake)
    monkeypatch.setattr(
        orchestrator,
        "copilot_search_manuals",
        lambda q, k=3, engine=None, chroma_path=None: {"query": q, "passages": []},
    )
    resp = orchestrator.copilot_reply(
        CopilotRequest(messages=[CopilotMessage(role="user", content="what does kurtosis mean?")])
    )
    assert resp.source == "llm" and resp.reply.startswith("Kurtosis rising")
    assert resp.tool_calls == [
        {
            "name": "search_manuals",
            "args": {"query": "kurtosis", "k": 1},
            "result_summary": "0 manual passages",
        }
    ]
    second = fake.requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_a"
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "call_a",
        "content": '{"query": "kurtosis", "passages": []}',
    }
    assert fake.requests[0]["tools"][0]["type"] == "function"
    assert resp.usage == {"input_tokens": 200, "output_tokens": 40, "llm_calls": 2}


def test_copilot_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Input cap, history cap, and tool-result clipping bound one Bolt turn."""
    from pydantic import ValidationError

    from apps.api import orchestrator
    from apps.api.schemas import COPILOT_MAX_MESSAGE_CHARS, CopilotMessage, CopilotRequest

    # 1. input cap: an over-long message fails validation (the route returns 422)
    with pytest.raises(ValidationError):
        CopilotMessage(role="user", content="x" * (COPILOT_MAX_MESSAGE_CHARS + 1))
    CopilotMessage(role="user", content="x" * COPILOT_MAX_MESSAGE_CHARS)

    # 2. tool-result clipping keeps the payload bounded and says so
    big = "y" * 20_000
    clipped = orchestrator._clip_tool_result(big)
    assert len(clipped) < 20_000 and clipped.startswith("y" * 100)
    assert clipped.endswith("[truncated: result clipped to 6000 characters]")
    assert orchestrator._clip_tool_result("short") == "short"

    # 3. history cap: only the last COPILOT_MAX_HISTORY messages reach the model
    _set_openai(monkeypatch)
    fake = _FakeOpenAI([_oa_msg("ok")])
    monkeypatch.setattr(orchestrator, "_openai_client", lambda: fake)
    history = [
        CopilotMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(25)
    ]
    history.append(CopilotMessage(role="user", content="latest question"))
    orchestrator.copilot_reply(CopilotRequest(messages=history))
    sent = fake.requests[0]["messages"]
    assert sent[0]["role"] == "system"
    assert len(sent) == 1 + orchestrator.COPILOT_MAX_HISTORY
    assert sent[-1]["content"] == "latest question" and sent[1]["content"] == "m16"

    # 4. a large tool result is clipped before it goes back to the model
    fake2 = _FakeOpenAI(
        [
            _oa_msg(None, [("call_b", "get_fleet", {})]),
            _oa_msg("done"),
        ]
    )
    monkeypatch.setattr(orchestrator, "_openai_client", lambda: fake2)
    monkeypatch.setattr(
        orchestrator, "copilot_get_fleet", lambda engine=None: {"blob": "z" * 50_000}
    )
    resp = orchestrator.copilot_reply(
        CopilotRequest(messages=[CopilotMessage(role="user", content="fleet?")])
    )
    tool_msg = fake2.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and len(tool_msg["content"]) < 6200
    assert tool_msg["content"].endswith("[truncated: result clipped to 6000 characters]")
    assert resp.usage["llm_calls"] == 2


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
        "provider": "anthropic",
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
