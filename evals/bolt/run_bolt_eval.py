"""Run the Bolt copilot eval (evals/bolt/cases.jsonl) and grade it programmatically.

    python evals/bolt/run_bolt_eval.py                 # template mode, offline embedding
    python evals/bolt/run_bolt_eval.py --mode llm      # needs ANTHROPIC_API_KEY in .env
    python evals/bolt/run_bolt_eval.py --embedding default   # MiniLM instead of hashed

What it does: points Thor at a throwaway SQLite database and Chroma directory, seeds the small
test fleet (MTR-042 degrading) plus the example Decision Contract, indexes the real manuals
and the committed FMUCD slice, then sends every case through `orchestrator.copilot_reply` --
the same function the /copilot/chat route calls -- and grades each reply:

  route     the tool calls match `expect.tools` (exact / prefix / first_in)
  content   every `expect.must` regex matches the reply, no `expect.must_not` does
  guardrail only read-only tools were used, and the answer came from the requested mode
  pass      all of the above

Outputs (git-ignored): evals/bolt/out/<mode>/results.jsonl (one row per case: prompt_id,
prompt, tags, grade, explanation, latency_s, tool_calls, model, meta) and
evals/bolt/out/<mode>/traces/<id>_rep0.json (user turn, tool calls, tool result summaries,
assistant reply). Exit code 1 if the pass rate is below --min-pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

READ_ONLY_TOOLS = {
    "get_fleet",
    "get_asset",
    "get_contract",
    "get_run",
    "list_pending_approvals",
    "search_manuals",
    "search_field_history",
    "field_history_stats",
}


def configure(mode: str, embedding: str, workdir: Path) -> None:
    os.environ["THOR_RAG_EMBEDDING"] = embedding
    os.environ["DATABASE_URL"] = f"sqlite:///{(workdir / 'bolt_eval.db').as_posix()}"
    os.environ["CHROMA_PATH"] = str(workdir / "chroma")
    os.environ["MODELS_DIR"] = str(workdir / "models")
    os.environ["MLFLOW_TRACKING_URI"] = str(workdir / "mlruns")
    os.environ["THOR_SKIP_BACKGROUND"] = "1"
    os.environ["MANUALS_DIR"] = str(ROOT / "data" / "manuals")
    os.environ["FIELD_HISTORY_CSV"] = str(
        ROOT / "data" / "field_history" / "fmucd_rotating_upm.csv"
    )
    if mode == "template":
        os.environ["ANTHROPIC_API_KEY"] = ""
    from apps.api import db
    from apps.api.settings import get_settings

    get_settings.cache_clear()
    db.get_engine.cache_clear()
    if mode == "llm" and not get_settings().anthropic_api_key.strip():
        sys.exit("--mode llm needs ANTHROPIC_API_KEY (put it in .env); see docs/API_KEY.md")


def seed() -> str:
    """Seed the fleet + example contract and build both Chroma collections; return contract id."""
    from apps.api import db
    from apps.api.schemas import DecisionContract
    from tests.test_orchestrator import seed_small_fleet

    engine = db.init_db()
    seed_small_fleet(engine)
    example = json.loads(
        (ROOT / "docs" / "decision-contracts" / "example.json").read_text(encoding="utf-8")
    )
    example["asset_id"] = "MTR-042"
    dc = DecisionContract.model_validate(example)
    db.insert_decision_contract(dc, engine=engine)

    from agents.reliability import rag
    from apps.api.settings import get_settings

    s = get_settings()
    n_chunks = rag.build_index(Path(s.manuals_dir), Path(s.chroma_path))
    n_cases = rag.build_field_history_index(Path(s.field_history_csv), Path(s.chroma_path))
    print(f"seeded fleet + contract {dc.contract_id}; {n_chunks} manual chunks, {n_cases} cases")
    return dc.contract_id


def grade(
    case: dict[str, Any], reply: str, tool_names: list[str], source: str, mode: str
) -> tuple[dict[str, float], dict[str, str]]:
    exp = case["expect"]
    notes: dict[str, str] = {}
    want = exp["tools"]
    tm = exp.get("tools_mode", "exact")
    if tm == "exact":
        route = tool_names == want
    elif tm == "prefix":
        route = tool_names[: len(want)] == want
    else:  # first_in
        route = bool(tool_names) and tool_names[0] in want
    if not route:
        notes["route"] = f"tools {tool_names} != expected {want} ({tm})"
    flags = re.IGNORECASE | re.MULTILINE
    missing = [m for m in exp["must"] if not re.search(m, reply, flags)]
    present = [m for m in exp["must_not"] if re.search(m, reply, flags)]
    content = not missing and not present
    if missing:
        notes["content"] = "missing " + ", ".join(repr(m) for m in missing)
    if present:
        notes["content"] = notes.get("content", "") + " forbidden " + ", ".join(map(repr, present))
    guardrail = set(tool_names) <= READ_ONLY_TOOLS and source == mode
    if not guardrail:
        extra = set(tool_names) - READ_ONLY_TOOLS
        notes["guardrail"] = f"non-read-only tools {extra or '{}'}; source={source}"
    g = {
        "pass": float(route and content and guardrail),
        "route": float(route),
        "content": float(content),
        "guardrail": float(guardrail),
    }
    return g, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--embedding", choices=["hashed", "default"], default="hashed")
    ap.add_argument("--cases", default=str(Path(__file__).with_name("cases.jsonl")))
    ap.add_argument("--out", default=str(Path(__file__).with_name("out")))
    ap.add_argument("--min-pass", type=float, default=0.0)
    ap.add_argument("--only", default="", help="substring filter on case id")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="bolt-eval-"))
    configure(args.mode, args.embedding, workdir)
    contract_id = seed()

    from apps.api import db, orchestrator
    from apps.api.schemas import CopilotMessage, CopilotRequest

    engine = db.get_engine()
    lines = Path(args.cases).read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    out_dir = Path(args.out) / args.mode
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    model_name = orchestrator.get_settings().anthropic_model if args.mode == "llm" else "template"
    rows: list[dict[str, Any]] = []
    for c in cases:
        prompt = c["prompt"].replace("{CONTRACT_ID}", contract_id)
        req = CopilotRequest(
            messages=[CopilotMessage(role="user", content=prompt)], asset_id=c.get("asset_id")
        )
        t0 = time.perf_counter()
        try:
            resp = orchestrator.copilot_reply(req, engine=engine)
            reply, calls, source, status = resp.reply, resp.tool_calls, resp.source, "ok"
        except Exception as e:  # noqa: BLE001 - record harness failures, keep going
            reply, calls, source, status = f"ERROR {type(e).__name__}: {e}", [], "error", "error"
        latency = time.perf_counter() - t0
        names = [tc["name"] for tc in calls]
        g, notes = grade(c, reply, names, source, args.mode)
        rows.append(
            {
                "prompt_id": c["id"],
                "prompt": prompt,
                "tags": c["tags"],
                "status": status,
                "stop_reason": "end_turn",
                "grade": g,
                "explanation": notes,
                "latency_s": round(latency, 3),
                "tool_calls": len(calls),
                "model": model_name,
                "meta": {"source": source, "tools": names},
            }
        )
        trace: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for tc in calls:
            trace.append(
                {
                    "role": "tool_call",
                    "name": tc["name"],
                    "content": json.dumps(tc["args"], indent=2),
                }
            )
            trace.append({"role": "tool_result", "content": tc.get("result_summary", "")})
        trace.append({"role": "assistant", "content": reply})
        (out_dir / "traces" / f"{c['id']}_rep0.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    with results_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tag[row["tags"][0]].append(row)
    print(f"\nBolt eval -- mode={args.mode} embedding={args.embedding} cases={len(rows)}")
    header = (
        f"{'category':<10}{'n':>4}{'pass':>7}{'route':>7}{'content':>9}{'guard':>7}{'lat s':>7}"
    )
    print(header)
    for tag, rs in by_tag.items():
        n = len(rs)
        rates = {
            k: sum(r["grade"][k] for r in rs) / n for k in ("pass", "route", "content", "guardrail")
        }
        lat = sum(r["latency_s"] for r in rs) / n
        print(
            f"{tag:<10}{n:>4}{rates['pass']:>7.0%}{rates['route']:>7.0%}"
            f"{rates['content']:>9.0%}{rates['guardrail']:>7.0%}{lat:>7.2f}"
        )
    n_all = len(rows)
    overall = sum(r["grade"]["pass"] for r in rows) / n_all if n_all else 0.0
    print(f"{'ALL':<10}{n_all:>4}{overall:>7.0%}")
    fails = [r for r in rows if r["grade"]["pass"] < 1]
    if fails:
        print("\nfailures:")
        for r in fails:
            print(f"  {r['prompt_id']:<12} {r['explanation']}")
    print(f"\nresults: {results_path}")
    return 0 if overall >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
