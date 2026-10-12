"""Phase A.3 rehearsal driver: the whole finals story over HTTP, timed.

    python -m scripts.rehearse_a3 --api http://localhost:8000 --out rehearsal.json
    python -m scripts.rehearse_a3 --skip-retrain --skip-llm      # the short, offline version

Runs against a RESET stack (`scripts/demo_reset.ps1`: no model, replay held, key Down) and
does what the presenter does by hand: train, sign off, wait for MTR-042 to latch red, Run
analysis (key Down -> template), Bolt's three questions, approve, wait for the repair, record
the technician's finding, ask Bolt "Have we seen this before?", read the label ledger, retrain
once with the finding (its promotion is rejected so production stays as signed off), then
Scenario 2: key Up -> Bolt, key Down -> the same question, "Approve the repair" -> refusal.
It MUTATES the stack (a model, a contract, a work order, a finding): reset again before the
real demo. Writes a JSON timeline and prints every step with its measured seconds. Every
number comes from the API; nothing here computes one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from typing import Any

import httpx

T0 = time.time()
LOG: list[dict[str, Any]] = []


def mark(step: str, **kw: Any) -> None:
    row = {"t": round(time.time() - T0, 1), "step": step, **kw}
    LOG.append(row)
    print(
        f"[{row['t']:7.1f}s] {step}: "
        + json.dumps({k: v for k, v in kw.items() if k != "reply"}, default=str)[:400]
    )
    if "reply" in kw:
        print("    reply: " + str(kw["reply"]).replace("\n", "\n           ")[:1500])
    sys.stdout.flush()


def wait_for(fn, label: str, timeout: float, every: float = 1.0) -> Any:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = fn()
        if out:
            return out
        time.sleep(every)
    raise TimeoutError(f"timed out waiting for {label}")


def fleet_row(api: httpx.Client, asset_id: str) -> dict[str, Any]:
    return next(r for r in api.get("/fleet").json() if r["asset"]["asset_id"] == asset_id)


def poll_run(api: httpx.Client, run_id: str, stages: set[str], timeout: float) -> dict[str, Any]:
    def done():
        s = api.get(f"/pipeline/{run_id}").json()
        return s if s["stage"] in stages else None

    return wait_for(done, f"run {run_id} -> {stages}", timeout, every=2.0)


def bolt(
    api: httpx.Client, text: str, asset_id: str | None, history: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    msgs = [*(history or []), {"role": "user", "content": text}]
    t = time.time()
    r = api.post("/copilot/chat", json={"messages": msgs, "asset_id": asset_id}, timeout=180.0)
    r.raise_for_status()
    out = r.json()
    out["latency_s"] = round(time.time() - t, 1)
    return out


NUM_RE = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")


def numbers_of(reply: str) -> dict[str, Any]:
    """Probabilities as they appear (2-3 decimals, rounded to 2), percentages, status words."""
    probs = sorted({round(float(x), 2) for x in NUM_RE.findall(reply) if 0 <= float(x) <= 1})
    pcts = sorted({float(x) for x in PCT_RE.findall(reply)})
    status = [
        w
        for w in ("critical", "warning", "healthy", "offline", "no prediction")
        if w in reply.lower()
    ]
    return {"probabilities": probs, "percentages": pcts, "status_words": status}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default="rehearsal.json")
    ap.add_argument("--hold-red-s", type=float, default=30.0)
    ap.add_argument("--skip-retrain", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()
    api = httpx.Client(base_url=args.api, timeout=60.0)

    # 0. clean start
    h = api.get("/system/health").json()
    key = api.get("/llm/key").json()
    fleet = api.get("/fleet").json()
    unscored = sum(1 for r in fleet if r["health_score"] is None)
    mark(
        "start",
        health=h["llm"],
        key=key,
        replay=api.get("/replay").json()["released"],
        motors=len(fleet),
        unscored=unscored,
    )
    assert key["key"] == "down", "the switch must start Down"

    # 1. train
    t = time.time()
    run_id = api.post(
        "/pipeline/run", json={"asset_id": "MTR-042", "mode": "train", "n_trials": args.trials}
    ).json()["run_id"]
    s = poll_run(api, run_id, {"trained", "failed"}, 900)
    v = s.get("validation") or {}
    gate = v.get("gate") or []
    mark(
        "train",
        seconds=round(time.time() - t, 1),
        stage=s["stage"],
        error=s.get("error"),
        family=v.get("champion_family"),
        version=v.get("model_version"),
        gate_passed=all(g.get("passed") for g in gate) if gate else None,
        gate=[(g["name"], g["passed"]) for g in gate],
        test_metrics=v.get("test_metrics"),
        lead_time_h=(v.get("lead_time") or {}).get("median_h"),
        ims=(v.get("ims") or {}).get("total"),
        split_positives=(s.get("split") or {}).get("positives"),
        label_sources=[
            (x["asset_id"], x["source"], x["group"]) for x in s.get("label_sources") or []
        ],
        llm_calls=s.get("llm_calls"),
    )
    if s["stage"] != "trained":
        print("TRAINING FAILED; stopping")
        return 1

    # 2. sign off
    promo = next(p for p in api.get("/models/promotions").json() if p["status"] == "pending")
    t = time.time()
    d = api.post(
        f"/promotions/{promo['promotion_id']}/decision",
        json={"decision": "approved", "approver": "rehearsal@thor", "note": ""},
        timeout=120.0,
    ).json()
    mark(
        "sign_off",
        seconds=round(time.time() - t, 1),
        detail={k: v for k, v in d.get("detail", {}).items() if k != "edge"},
        replay=api.get("/replay").json()["released"],
    )
    t_sign = time.time()

    # 3. green, then MTR-042 red
    wait_for(
        lambda: all(r["health_score"] is not None for r in api.get("/fleet").json()),
        "24 scored",
        120,
    )
    mark(
        "fleet_green",
        seconds_after_sign_off=round(time.time() - t_sign, 1),
        statuses=sorted({r["status"] for r in api.get("/fleet").json()}),
    )
    wait_for(
        lambda: fleet_row(api, "MTR-042")["status"] == "critical",
        "MTR-042 critical",
        600,
        every=1.0,
    )
    row = fleet_row(api, "MTR-042")
    mark(
        "mtr042_red",
        seconds_after_sign_off=round(time.time() - t_sign, 1),
        p=row["failure_probability"],
        alarm_since=row["alarm_since"],
        others_alarmed=[
            r["asset"]["asset_id"]
            for r in api.get("/fleet").json()
            if r["status"] in ("critical", "warning") and r["asset"]["asset_id"] != "MTR-042"
        ],
    )
    time.sleep(args.hold_red_s)

    # 4. run analysis (key Down -> template)
    t = time.time()
    arun = api.post("/pipeline/run", json={"asset_id": "MTR-042", "mode": "analyze"}).json()[
        "run_id"
    ]
    s = poll_run(api, arun, {"awaiting_approval", "failed"}, 300)
    dc = s.get("contract") or {}
    drivers = [f["feature"] for f in dc.get("top_features", [])][:3]
    mark(
        "analysis",
        seconds=round(time.time() - t, 1),
        stage=s["stage"],
        contract=dc.get("contract_id"),
        recommendation=dc.get("recommendation"),
        p=dc.get("failure_probability"),
        source=dc.get("explanation_source"),
        drivers=drivers,
        llm_calls=s.get("llm_calls"),
    )
    assert dc.get("explanation_source") == "template"

    # 5. Bolt, key Down
    hist: list[dict[str, str]] = []
    for q in (
        "What's happening with MTR-042?",
        "Give me the details.",
        "What are the possible fixes?",
    ):
        r = bolt(api, q, "MTR-042", hist)
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": r["reply"]}]
        mark(
            "bolt_down",
            q=q,
            source=r["source"],
            latency_s=r["latency_s"],
            tools=[c["name"] for c in r["tool_calls"]],
            reply=r["reply"],
        )
        assert r["source"] == "template"

    # 5b. (extra data for the team) Up/Down comparison while the motor is red
    red_compare = None
    if not args.skip_llm:
        up = api.post("/llm/key/up").json()
        mark("key_up_while_red", state=up)
        if up["provider"] != "none":
            r_up = bolt(api, "What's happening with MTR-042?", "MTR-042")
            mark(
                "bolt_up_red",
                source=r_up["source"],
                latency_s=r_up["latency_s"],
                tools=[c["name"] for c in r_up["tool_calls"]],
                usage=r_up.get("usage"),
                reply=r_up["reply"],
            )
            down = api.post("/llm/key/down").json()
            r_dn = bolt(api, "What's happening with MTR-042?", "MTR-042")
            mark(
                "bolt_down_red",
                state=down,
                source=r_dn["source"],
                latency_s=r_dn["latency_s"],
                reply=r_dn["reply"],
            )
            red_compare = {
                "up": numbers_of(r_up["reply"]),
                "down": numbers_of(r_dn["reply"]),
                "up_source": r_up["source"],
                "down_source": r_dn["source"],
            }
            mark("compare_red", **red_compare)
        else:
            api.post("/llm/key/down")
    assert api.get("/llm/key").json()["key"] == "down"

    # 6. approve the contract; the timer
    crit = [
        a
        for a in api.get("/alarms", params={"asset_id": "MTR-042"}).json()
        if a["level"] == "critical"
    ]
    t = time.time()
    appr = api.post(
        f"/approvals/{dc['contract_id']}/decision",
        json={"decision": "approved", "approver": "engineer@plant", "note": "rehearsal"},
    ).json()
    t_appr = time.time()
    timer_s = None
    if crit:
        a = datetime.fromisoformat(crit[-1]["recorded_at"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(appr["decided_at"].replace("Z", "+00:00"))
        timer_s = round((b - a).total_seconds(), 1)
    mark(
        "approve",
        seconds=round(time.time() - t, 1),
        approval=appr["approval_id"],
        alarm_to_decision_s=timer_s,
        critical_rows=len(crit),
    )

    # 7. offline, then back green with the work order completed
    wait_for(lambda: fleet_row(api, "MTR-042")["status"] == "offline", "offline", 120)
    mark("offline", seconds_after_approval=round(time.time() - t_appr, 1))
    wait_for(
        lambda: (
            lambda r: (
                r["status"] == "healthy"
                and (r.get("work_order") or {}).get("status") == "completed"
            )
        )(fleet_row(api, "MTR-042")),
        "healthy + completed",
        300,
    )
    row = fleet_row(api, "MTR-042")
    mark(
        "back_green",
        seconds_after_approval=round(time.time() - t_appr, 1),
        p=row["failure_probability"],
        work_order=row["work_order"]["status"],
        finding=row["work_order"].get("finding"),
    )

    # 8. the technician's finding
    trust0 = api.get("/findings/trust").json()
    r = api.post(
        "/findings",
        json={
            "contract_id": dc["contract_id"],
            "finding": "confirmed_bearing_damage",
            "note": "drive-end bearing outer race spalled; bearing replaced",
            "recorded_by": "J. Ortega",
        },
    )
    finding = r.json()
    trust = api.get("/findings/trust").json()
    row = fleet_row(api, "MTR-042")
    mark(
        "finding",
        status=r.status_code,
        finding_id=finding.get("finding_id"),
        trust_before=trust0,
        trust_after=trust,
        fleet_card_finding=(row["work_order"] or {}).get("finding", {}).get("finding"),
    )
    assert r.status_code == 201 and trust["acted_on"] == 1 and trust["confirmed"] == 1

    # 9. Bolt recalls it (key Down)
    r = bolt(api, "Have we seen this before?", "MTR-042")
    mark(
        "bolt_recall",
        source=r["source"],
        tools=[(c["name"], c.get("args")) for c in r["tool_calls"]],
        reply=r["reply"],
    )
    ok_recall = (
        "This plant" in r["reply"] and "J. Ortega" in r["reply"] and dc["contract_id"] in r["reply"]
    )
    mark(
        "bolt_recall_check",
        cites_plant_finding=ok_recall,
        mentions_public="not from this plant" in r["reply"] or "public" in r["reply"],
    )

    # 10. ledger
    ledger = api.get("/findings/ledger").json()
    mark(
        "ledger",
        queued=[(q["asset_id"], q["contract_id"], q["recorded_by"]) for q in ledger["queued"]],
    )

    # 11. retrain with the finding
    if not args.skip_retrain:
        t = time.time()
        run2 = api.post(
            "/pipeline/run", json={"asset_id": "MTR-042", "mode": "train", "n_trials": args.trials}
        ).json()["run_id"]
        s2 = poll_run(api, run2, {"trained", "failed"}, 900)
        v2 = s2.get("validation") or {}
        gate2 = v2.get("gate") or []
        sp = s2.get("split") or {}
        mark(
            "retrain_with_finding",
            seconds=round(time.time() - t, 1),
            stage=s2["stage"],
            error=s2.get("error"),
            family=v2.get("champion_family"),
            version=v2.get("model_version"),
            gate_passed=all(g.get("passed") for g in gate2) if gate2 else None,
            gate=[(g["name"], g["passed"], g.get("detail")) for g in gate2],
            test_metrics=v2.get("test_metrics"),
            lead_time_h=(v2.get("lead_time") or {}).get("median_h"),
            ims=(v2.get("ims") or {}).get("total"),
            split={
                "train": sp.get("train"),
                "test": sp.get("test"),
                "positives": sp.get("positives"),
                "forced_test": sp.get("forced_test"),
            },
            notes=[n for n in sp.get("notes", []) if "work order" in n],
            label_sources=[
                (x["asset_id"], x["source"], x["group"], x.get("recorded_by"), x.get("n_rows"))
                for x in s2.get("label_sources") or []
            ],
            events=[
                e["message"] for e in s2.get("events", []) if e.get("tool") == "make_asset_split"
            ],
        )
        # keep the production model as it is: reject the second promotion request
        for p in api.get("/models/promotions").json():
            if p["status"] == "pending":
                api.post(
                    f"/promotions/{p['promotion_id']}/decision",
                    json={
                        "decision": "rejected",
                        "approver": "rehearsal@thor",
                        "note": "rehearsal: the retrained model is not promoted",
                    },
                )
                mark("retrain_promotion_rejected", promotion_id=p["promotion_id"])

    # 12. Scenario 2: Up then Down, same question, same numbers; refusal
    s2_compare = None
    if not args.skip_llm:
        up = api.post("/llm/key/up").json()
        mark("key_up", state=up, health=api.get("/system/health").json()["llm"])
        if up["provider"] != "none":
            r_up = bolt(api, "What's happening with MTR-042?", "MTR-042")
            mark(
                "bolt_up",
                source=r_up["source"],
                latency_s=r_up["latency_s"],
                tools=[c["name"] for c in r_up["tool_calls"]],
                usage=r_up.get("usage"),
                reply=r_up["reply"],
            )
        down = api.post("/llm/key/down").json()
        mark("key_down", state=down, health=api.get("/system/health").json()["llm"])
        r_dn = bolt(api, "What's happening with MTR-042?", "MTR-042")
        mark(
            "bolt_down_after",
            source=r_dn["source"],
            latency_s=r_dn["latency_s"],
            reply=r_dn["reply"],
        )
        if up["provider"] != "none":
            s2_compare = {
                "up": numbers_of(r_up["reply"]),
                "down": numbers_of(r_dn["reply"]),
                "up_source": r_up["source"],
                "down_source": r_dn["source"],
            }
            mark("compare_scenario2", **s2_compare)
        r_ref = bolt(api, "Approve the repair on MTR-042.", "MTR-042")
        mark(
            "bolt_refusal",
            source=r_ref["source"],
            tools=[c["name"] for c in r_ref["tool_calls"]],
            reply=r_ref["reply"],
        )
        key_text = json.dumps(api.get("/llm/key").json()) + json.dumps(
            api.get("/system/health").json()
        )
        mark("key_never_in_responses", contains_sk=("sk-" in key_text))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {"log": LOG, "red_compare": red_compare, "scenario2_compare": s2_compare},
            fh,
            indent=2,
            default=str,
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
