# What to do when the Anthropic API key arrives

Thor runs without a key today (template mode). This is the 10-minute runbook for switching
the LLM paths on and proving they work. Only `apps/api/orchestrator.py` reads the key; two
calls per pipeline run (route, then draft the explanation) plus the Bolt copilot tool loop.

> **Provider note (2026-09-25).** Thor accepts either provider. `LLM_PROVIDER=auto` picks OpenAI
> when `OPENAI_API_KEY` is set, else Anthropic when `ANTHROPIC_API_KEY` is set, else template
> mode; pin with `anthropic` or `openai`. All three LLM paths (route, draft, Bolt tool loop) and
> `llm_ping()` run on whichever provider is active. The steps below name the Anthropic
> variables; substitute `OPENAI_API_KEY` / `OPENAI_MODEL` (key prefix `sk-`, needs model
> inference permission -- a read-only key cannot generate) and read `provider` in the status
> output.

## 0. Rules

- Never commit the key. `.env` and `.env.*` are git- and docker-ignored; `.env.example` is the
  only env file that belongs in the repo, and its key line stays blank.
- Never paste the key into chat, an issue, a screenshot, or the compose file. It goes in `.env`
  and nowhere else.
- If it ever leaks, revoke it in the console and create a new one. Rotation is a one-line edit.

## 1. Put the key in place

```bash
cp .env.example .env            # skip if .env already exists
# edit .env:
#   ANTHROPIC_API_KEY=sk-ant-api03-...
#   ANTHROPIC_MODEL=claude-sonnet-5   # or claude-opus-5 for stronger prose
#   -- or, for OpenAI --
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=gpt-5
```

Do not `export` the key in a shell you share or record; the `.env` file is enough for both the
local API and Docker Compose (compose reads `.env` from the repo root).

## 2. Verify without touching the network

```bash
python -c "from apps.api.orchestrator import llm_status; print(llm_status())"
# {'mode': 'llm', 'model': 'claude-sonnet-5', 'key_configured': True, 'warning': None}
```

`mode: template` means the file was not found or the line is blank. A `warning` means the value
does not start with `sk-ant-` (usually a paste error).

## 3. One tiny live call

```bash
python -c "from apps.api.orchestrator import llm_ping; print(llm_ping())"
# {'ok': True, 'model': 'claude-sonnet-5', 'reply': 'ready', 'request_id': 'req_...'}
```

| `error` starts with | Meaning | Fix |
|---|---|---|
| `AuthenticationError` | key rejected | re-copy the key; check for trailing spaces |
| `NotFoundError` | model id unknown to this account | set `ANTHROPIC_MODEL` to a listed model |
| `PermissionDeniedError` | key lacks access | check the workspace the key was created in |
| `RateLimitError` | quota or spend limit hit | check console usage limits |
| `APIConnectionError` | no network / proxy | test connectivity from this machine |

Nothing else in Thor needs to be running for steps 2 and 3.

## 4. Restart the stack and confirm it picked the key up

```bash
docker compose up --build -d        # or: docker compose up -d api   (only the API reads it)
docker compose logs api | grep "LLM mode"      # LLM mode: llm (model claude-sonnet-5)
curl -s localhost:8000/system/health           # ... "llm": {"mode": "llm", "model": "..."} ...
```

On the Fleet screen the System panel shows an **LLM** row; the dot turns on and the model name
replaces "template mode".

Local without Docker: `uvicorn apps.api.main:app --reload` reads the same `.env`.

## 5. Run one analysis end to end in LLM mode

1. Open Asset 360 for MTR-042 and click **Run analysis** (about 90 s).
2. When the Decision Contract panel appears, the chip on the explanation should read
   **drafted by LLM** (field `explanation_source: "llm"`). If it reads **template**, the LLM
   call failed and fell back; `docker compose logs api | grep draft_explanation` shows why.
3. Read the paragraph against the numbers in the panel. Every probability, cost, hour, and
   manual citation in the text must already appear in the contract JSON. If the model invented
   one, that is a prompt bug in `EXPLANATION_SYSTEM_PROMPT`, not a data bug; report it.
4. Open Copilot and ask Bolt "why is MTR-042 at risk?" and "what is pending approval?". The
   response `source` should be `llm` and `tool_calls` should list the tools it used. Ask it to
   approve the contract; it must refuse and point you to the UI.

## 6. If anything fails, nothing breaks

Every LLM path falls back to the deterministic template on any exception and logs a warning.
The demo is identical in either mode except for the chip and the prose. To force template mode
again, blank the key line and restart the API.

## 7. Cost expectation

Two short calls per pipeline run and a handful per Bolt conversation, each a few thousand
tokens. A full day of demo rehearsals stays in the low single-digit dollars on either model.
Set a monthly spend limit on the key in the console anyway.
