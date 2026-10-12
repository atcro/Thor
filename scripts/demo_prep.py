"""One-command demo preparation, through the control plane's own governed API.

    python -m scripts.demo_prep [--api http://localhost:8000] [--asset MTR-042] [--n-trials 4]
                                [--report docs/finals/prep-report.json] [--force] [--timeout 900]

What it does, in order, against a RUNNING control plane (the api seeds the first 80 % of the
synthetic history itself at start-up):

  1. waits for GET /system/health;
  2. if a production model already exists and --force is not given, skips training;
     otherwise POST /pipeline/run {mode: "train"} for --asset (held out of training) and polls
     until the run is `trained`; a run whose validation gate failed stops the prep (exit 2)
     because an unvalidated model must never be promoted;
  3. approves that model's promotion through POST /promotions/{id}/decision with approver
     "demo-prep@thor" -- the same path an engineer uses in ModelOps -- which promotes it to
     production and writes the ONNX + sidecar for the edge;
  4. confirms the ONNX + sidecar were written (parity was proven at validation time on
     held-out vectors and raw windows, and promote() refuses a model whose parity failed);
  5. scores the whole fleet with the production model (POST /pipeline/score) so the Dashboard
     shows an approved-model health for every motor immediately;
  6. writes a JSON report (split, gate, test metrics, parity, versions, timings) and prints a
     summary. The honest numbers in docs/finals/metrics.md come from this report.

The live demo then shows inference, a Decision Contract and an approval. It never trains.
No LLM is involved anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

TERMINAL = {"trained", "awaiting_approval", "approved", "rejected", "failed"}


def log(msg: str) -> None:
    print(f"[demo-prep {datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_api(client: httpx.Client, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            r = client.get("/system/health", timeout=5.0)
            if r.status_code == 200 and r.json().get("n_telemetry_rows", 0) > 0:
                return r.json()
            last = f"{r.status_code} {r.text[:120]}"
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(3.0)
    raise SystemExit(f"control plane not ready after {timeout_s:.0f}s: {last}")


def production_model(client: httpx.Client) -> dict[str, Any] | None:
    models = client.get("/models").json()
    return next((m for m in models if m["stage"] == "production"), None)


def train(client: httpx.Client, asset: str, n_trials: int, timeout_s: float) -> dict[str, Any]:
    r = client.post(
        "/pipeline/run", json={"asset_id": asset, "mode": "train", "n_trials": n_trials}
    )
    r.raise_for_status()
    run_id = r.json()["run_id"]
    log(f"training run {run_id} started (asset {asset} held out, {n_trials} trials/family)")
    deadline = time.time() + timeout_s
    seen = 0
    while time.time() < deadline:
        state = client.get(f"/pipeline/{run_id}", timeout=30.0).json()
        for e in state["events"][seen:]:
            log(f"  [{e['stage']}] {e['tool']}: {e['message'][:140]}")
        seen = len(state["events"])
        if state["stage"] in TERMINAL:
            return state
        time.sleep(2.0)
    raise SystemExit(f"training run {run_id} did not finish within {timeout_s:.0f}s")


def approve_promotion(client: httpx.Client, version: str) -> dict[str, Any]:
    pending = [
        p
        for p in client.get("/approvals/pending").json()["promotions"]
        if p["version"] == version and p["status"] == "pending"
    ]
    if not pending:
        raise SystemExit(f"no pending promotion request for {version}")
    pid = pending[0]["promotion_id"]
    r = client.post(
        f"/promotions/{pid}/decision",
        json={
            "decision": "approved",
            "approver": "demo-prep@thor",
            "note": "demo preparation: validated champion promoted before the live demo",
        },
        timeout=120.0,
    )
    r.raise_for_status()
    body = r.json()
    if "promoted" not in body["detail"]:
        raise SystemExit(f"promotion refused: {body['detail']}")
    log(f"promotion {pid} approved by demo-prep@thor -> {body['detail']['promoted']['stage']}")
    if "edge" not in body["detail"]:
        raise SystemExit(f"edge export failed: {body['detail'].get('error')}")
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare the Thor demo through the control plane")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--asset", default="MTR-042", help="asset held out of training")
    ap.add_argument(
        "--n-trials",
        type=int,
        default=int(os.environ.get("DEMO_PREP_TRIALS", "4")),
        help="Optuna trials per model family (env DEMO_PREP_TRIALS)",
    )
    ap.add_argument("--report", default="docs/finals/prep-report.json")
    ap.add_argument("--force", action="store_true", help="train even if a production model exists")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for training")
    args = ap.parse_args()

    t0 = time.time()
    client = httpx.Client(base_url=args.api.rstrip("/"), timeout=60.0)
    health = wait_for_api(client, args.timeout)
    log(f"control plane ready: {health['n_telemetry_rows']} telemetry rows, llm mode {health['llm']['mode']}")

    report: dict[str, Any] = {"api": args.api, "asset_held_out": args.asset, "started_at": datetime.now(UTC).isoformat()}
    prod = production_model(client)
    if prod is not None and not args.force:
        log(f"production model {prod['version']} already present; skipping training (use --force)")
        report["training"] = {"skipped": True, "version": prod["version"]}
        version = prod["version"]
    else:
        state = train(client, args.asset, args.n_trials, args.timeout)
        if state["stage"] != "trained":
            log(f"training run ended in {state['stage']}: {state.get('error')}")
            return 2
        v = state["validation"]
        gate = {g["name"]: g for g in v["gate"]}
        report["training"] = {
            "run_id": state["run_id"],
            "version": v["model_version"],
            "champion_family": v["champion_family"],
            "split": state["split"],
            "data_quality_score": state["data_quality"]["quality_score"],
            "regimes": state["data_quality"]["regimes"]["regimes"],
            "candidates": [
                {
                    "candidate_id": c["candidate_id"],
                    "family": c["family"],
                    "validation_metrics": c["metrics"],
                    "validation_ims": c["ims"]["total"] if c.get("ims") else None,
                    "params": c["params"],
                    "latency_ms": c["inference_latency_ms"],
                }
                for c in state["candidates"]["candidates"]
            ],
            "ranked": state["candidates"]["ranked"],
            "test_metrics": v["test_metrics"],
            "ims": v["ims"],
            "lead_time": v["lead_time"],
            "calibration": {k: v["calibration"][k] for k in ("method", "brier_before", "brier_after")},
            "thresholds": v["thresholds"],
            "gate": v["gate"],
            "parity": v["parity"],
            "passed": v["passed"],
            "notes": v["notes"],
            "seconds": round(time.time() - t0, 1),
        }
        log(
            f"trained {v['champion_family']} {v['model_version']}: test recall "
            f"{v['test_metrics'].get('recall', float('nan')):.3f}, precision "
            f"{v['test_metrics'].get('precision', float('nan')):.3f}, lead time median "
            f"{v['lead_time']['median_h']:.1f} h, IMS {v['ims']['total']:.3f}, gate "
            + ", ".join(f"{k}={'ok' if g['passed'] else 'FAIL'}" for k, g in gate.items())
        )
        if not v["passed"]:
            log("validation gate FAILED: the model will not be promoted (see gate above)")
            return 2
        version = v["model_version"]
        body = approve_promotion(client, version)
        report["promotion"] = body
        prod = production_model(client)

    assert prod is not None
    row = next(m for m in client.get("/models").json() if m["version"] == version)
    if not row.get("onnx_path"):
        raise SystemExit("production row has no onnx_path; edge export missing")
    log(f"edge bundle: {row['onnx_path']} (parity verified at validation on held-out vectors)")

    sc = client.post("/pipeline/score", json={}, timeout=120.0)
    sc.raise_for_status()
    scored = sc.json()
    report["fleet_scoring"] = {"model_version": scored["model_version"], "n_scored": len(scored["scored"]), "skipped": scored["skipped"]}
    log(f"fleet scored with {scored['model_version']}: {len(scored['scored'])} assets")
    fleet = client.get("/fleet").json()
    worst = fleet[:3]
    for f in worst:
        log(f"  {f['asset']['asset_id']}: health {f['health_score']} p={f['failure_probability']:.3f} regime {f['regime']}")
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["seconds_total"] = round(time.time() - t0, 1)
    out = Path(args.report)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log(f"report written to {out}")
    except OSError as exc:
        log(f"could not write report {out}: {exc}")
    log(f"done in {report['seconds_total']} s; production model {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
