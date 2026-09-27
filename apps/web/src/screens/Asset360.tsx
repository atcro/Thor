import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { decideContract, getAsset, getPredictions, getRun, getTelemetry, runPipeline, runWhatIf, runWsUrl } from "../api/client";
import type { Approval, Decision, GraphState, PipelineStage, RunSummary, WhatIfResult } from "../api/types";
import { TERMINAL_STAGES } from "../api/types";
import { CHART, TimeSeriesChart, type TimePoint } from "../components/charts";
import { DecisionContractPanel } from "../components/DecisionContractPanel";
import { EmptyState, ErrorState, HealthPill, Loaded, LoadingState, Panel, Pill, RegimeBadge, StageChip, Stat, WorkOrderPill } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { STAGE_LABEL, fmtNum, fmtPct, fmtRel, fmtShort, fmtTime, probTone } from "../lib/format";

const STEPS: { stage: PipelineStage; label: string }[] = [
  { stage: "profiling", label: "Profiling" },
  { stage: "training", label: "Training" },
  { stage: "validating", label: "Validating" },
  { stage: "explaining", label: "Explaining" },
  { stage: "awaiting_approval", label: "Awaiting approval" },
  { stage: "approved", label: "Decision" },
];

const STAGE_INDEX: Record<PipelineStage, number> = {
  queued: -1,
  profiling: 0,
  training: 1,
  validating: 2,
  explaining: 3,
  awaiting_approval: 4,
  approved: 5,
  rejected: 5,
  failed: 99,
};

function isTerminal(stage: PipelineStage | undefined): boolean {
  return stage !== undefined && TERMINAL_STAGES.includes(stage);
}

/**
 * Live GraphState for a run: WebSocket to /ws/runs/:id, falling back to polling
 * GET /pipeline/:id every 2s if the socket fails or closes before a terminal stage.
 */
function useRunState(runId: string | null) {
  const [state, setState] = useState<GraphState | null>(null);
  const [transport, setTransport] = useState<"ws" | "poll" | "idle">("idle");
  const [error, setError] = useState<string | null>(null);
  const stageRef = useRef<PipelineStage | undefined>(undefined);
  stageRef.current = state?.stage;

  useEffect(() => {
    setState(null);
    setError(null);
    setTransport("idle");
    if (!runId) return;

    let cancelled = false;
    let pollTimer: number | undefined;
    let ws: WebSocket | null = null;

    const apply = (s: GraphState) => {
      if (cancelled) return;
      setState(s);
      setError(null);
    };

    const startPolling = () => {
      if (cancelled || pollTimer !== undefined) return;
      setTransport("poll");
      const tick = async () => {
        try {
          const s = await getRun(runId);
          apply(s);
          if (isTerminal(s.stage) && pollTimer !== undefined) {
            window.clearInterval(pollTimer);
            pollTimer = undefined;
          }
        } catch (e) {
          if (!cancelled) setError(e instanceof Error ? e.message : String(e));
        }
      };
      void tick();
      pollTimer = window.setInterval(() => void tick(), 2000);
    };

    // initial snapshot regardless of transport
    getRun(runId).then(apply).catch(() => undefined);

    try {
      ws = new WebSocket(runWsUrl(runId));
      ws.onopen = () => {
        if (!cancelled) setTransport("ws");
      };
      ws.onmessage = (ev: MessageEvent<string>) => {
        try {
          apply(JSON.parse(ev.data) as GraphState);
        } catch {
          /* ignore malformed frame */
        }
      };
      ws.onerror = () => {
        startPolling();
      };
      ws.onclose = () => {
        if (!cancelled && !isTerminal(stageRef.current)) startPolling();
      };
    } catch {
      startPolling();
    }

    return () => {
      cancelled = true;
      if (pollTimer !== undefined) window.clearInterval(pollTimer);
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) ws.close();
    };
  }, [runId]);

  const patch = useCallback((fn: (s: GraphState) => GraphState) => setState((s) => (s ? fn(s) : s)), []);
  return { state, transport, error, patch };
}

function StageStepper({ state }: { state: GraphState | null }) {
  const stage = state?.stage;
  const failed = stage === "failed";
  const rejected = stage === "rejected";
  const done = stage === "approved" || rejected;
  // On failure, the step that was running is the last non-failed stage in the event log.
  const failedAt = failed
    ? [...(state?.events ?? [])].reverse().find((e) => e.stage !== "failed" && e.stage !== "queued")?.stage
    : undefined;
  const idx = failed ? (failedAt ? STAGE_INDEX[failedAt] : 0) : stage ? STAGE_INDEX[stage] : -1;
  return (
    <div className="stepper">
      {STEPS.map((s, i) => {
        let cls = "step";
        if (i < idx) cls += " done";
        else if (i === idx) cls += failed ? " failed" : done ? " done" : " active";
        const label = i === 5 && rejected ? "Rejected" : i === 5 && stage === "approved" ? "Approved" : s.label;
        const glyph = i < idx || (i === idx && done) ? "✓" : i === idx && failed ? "!" : String(i + 1);
        return (
          <div className={cls} key={s.stage}>
            <span className="node">{glyph}</span>
            <span className="txt">{label}</span>
            {i < STEPS.length - 1 && <span className="bar" />}
          </div>
        );
      })}
    </div>
  );
}

function EventLog({ state }: { state: GraphState | null }) {
  const events = state?.events ?? [];
  if (events.length === 0) return <EmptyState>No events yet.</EmptyState>;
  return (
    <ul className="events">
      {[...events].reverse().map((e, i) => (
        <li key={`${e.ts}-${i}`}>
          <span className="t">{fmtTime(e.ts)}</span>
          <span className="s">{STAGE_LABEL[e.stage] ?? e.stage}</span>
          <span>
            {e.tool && <span className="dim">{e.tool}() · </span>}
            {e.message}
          </span>
        </li>
      ))}
    </ul>
  );
}

function WhatIfForm({ runId, baseline }: { runId: string; baseline: number | null }) {
  const [load, setLoad] = useState<string>("");
  const [rpm, setRpm] = useState<string>("");
  const [result, setResult] = useState<WhatIfResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async () => {
    const scenario: Record<string, number> = {};
    if (load.trim() !== "" && Number.isFinite(Number(load))) scenario.load_mean = Number(load);
    if (rpm.trim() !== "" && Number.isFinite(Number(rpm))) scenario.rpm_mean = Number(rpm);
    if (Object.keys(scenario).length === 0) {
      setErr("Enter at least one override.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      setResult(await runWhatIf({ run_id: runId, scenario }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <div className="row wrap" style={{ alignItems: "flex-end", gap: 10 }}>
        <label className="field">
          load_mean (%)
          <input className="input mono" type="number" value={load} onChange={(e) => setLoad(e.target.value)} placeholder="e.g. 40" style={{ width: 110 }} />
        </label>
        <label className="field">
          rpm_mean
          <input className="input mono" type="number" value={rpm} onChange={(e) => setRpm(e.target.value)} placeholder="e.g. 1200" style={{ width: 110 }} />
        </label>
        <button className="btn steel" disabled={busy} onClick={() => void submit()}>
          {busy ? "Scoring…" : "Score scenario"}
        </button>
        {err && <span className="small" style={{ color: "var(--ember-2)" }}>{err}</span>}
      </div>
      {result ? (
        <div className="grid grid-3">
          <Stat label="baseline" value={fmtPct(result.baseline_probability)} />
          <Stat label="scenario" value={fmtPct(result.scenario_probability)} tone={result.delta < 0 ? "green" : "ember"} />
          <Stat label="delta" value={`${result.delta >= 0 ? "+" : ""}${fmtPct(result.delta)}`} tone={result.delta < 0 ? "green" : "ember"} />
        </div>
      ) : (
        <div className="small muted">
          Re-scores the champion with feature overrides — deterministic, no LLM.
          {baseline !== null && <> Baseline p(fail) {fmtPct(baseline)}.</>}
        </div>
      )}
    </div>
  );
}

function pickLatestRun(runs: RunSummary[]): RunSummary | null {
  if (!runs || runs.length === 0) return null;
  const awaiting = runs.find((r) => r.stage === "awaiting_approval");
  if (awaiting) return awaiting;
  return [...runs].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""))[0] ?? null;
}

export function Asset360() {
  const { id = "" } = useParams<{ id: string }>();
  const assetId = decodeURIComponent(id);

  const detail = usePolling(() => getAsset(assetId), 5000, assetId, assetId !== "");
  const telemetry = usePolling(() => getTelemetry(assetId, 24), 3000, assetId, assetId !== "");
  const predictions = usePolling(() => getPredictions(assetId, 500), 10000, assetId, assetId !== "");

  const [runId, setRunId] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchErr, setLaunchErr] = useState<string | null>(null);
  const run = useRunState(runId);

  // Adopt the latest known run for this asset if none has been started in this session.
  useEffect(() => {
    if (runId === null && detail.data?.runs?.length) {
      const latest = pickLatestRun(detail.data.runs);
      if (latest) setRunId(latest.run_id);
    }
  }, [detail.data, runId]);

  useEffect(() => {
    setRunId(null);
  }, [assetId]);

  const launch = async () => {
    setLaunching(true);
    setLaunchErr(null);
    try {
      const res = await runPipeline({ asset_id: assetId });
      setRunId(res.run_id);
    } catch (e) {
      setLaunchErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLaunching(false);
    }
  };

  const [deciding, setDeciding] = useState(false);
  const [decideErr, setDecideErr] = useState<string | null>(null);
  const [localApproval, setLocalApproval] = useState<Approval | null>(null);
  useEffect(() => setLocalApproval(null), [runId]);

  const decide = async (decision: Decision, approver: string, note: string) => {
    const contractId = run.state?.contract?.contract_id;
    if (!contractId) return;
    setDeciding(true);
    setDecideErr(null);
    try {
      const approval = await decideContract(contractId, { decision, approver, note });
      setLocalApproval(approval);
      run.patch((s) => ({ ...s, approval, stage: decision === "approved" ? "approved" : "rejected" }));
      void detail.refresh();
    } catch (e) {
      setDecideErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDeciding(false);
    }
  };

  const series = useMemo(() => {
    const rows = telemetry.data ?? [];
    const toPts = (key: "vibration_rms" | "bearing_temp_c"): TimePoint[] =>
      rows.map((r) => ({ t: new Date(r.ts).getTime(), v: Number.isFinite(r[key]) ? r[key] : null }));
    const preds: TimePoint[] = (predictions.data ?? []).map((p) => ({
      t: new Date(p.ts).getTime(),
      v: p.failure_probability,
    }));
    return { vib: toPts("vibration_rms"), temp: toPts("bearing_temp_c"), preds };
  }, [telemetry.data, predictions.data]);

  const latest = detail.data?.latest ?? null;
  const currentP = detail.data?.prediction ?? run.state?.contract?.failure_probability ?? null;
  const health = latest?.health !== null && latest?.health !== undefined ? latest.health * 100 : currentP !== null ? 100 * (1 - currentP) : null;
  const contract = run.state?.contract ?? null;
  const approval = run.state?.approval ?? localApproval;
  const runActive = run.state !== null && !isTerminal(run.state.stage);
  const showContract = contract !== null && (run.state?.stage === "awaiting_approval" || approval !== null);

  if (!assetId) return <ErrorState error="No asset id in route." />;

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <div className="row" style={{ gap: 10 }}>
            <h1 className="mono">{assetId}</h1>
            <RegimeBadge regime={latest?.regime ?? null} />
            <HealthPill score={health} />
            {currentP !== null && (
              <Pill tone={probTone(currentP)}>p(fail) {fmtPct(currentP)}</Pill>
            )}
          </div>
          <div className="sub">
            {detail.data ? (
              <>
                {detail.data.asset.name} · {detail.data.asset.site} / {detail.data.asset.line} · {detail.data.asset.rated_kw} kW ·{" "}
                {detail.data.asset.criticality} criticality · last sample {fmtShort(latest?.ts)}
              </>
            ) : detail.error ? (
              <span style={{ color: "var(--ember-2)" }}>{detail.error}</span>
            ) : (
              "Loading asset…"
            )}
          </div>
        </div>
        <div className="row">
          {run.state && <StageChip stage={run.state.stage} />}
          <button className="btn primary" onClick={() => void launch()} disabled={launching || runActive}>
            {launching ? "Starting…" : runActive ? "Analysis running" : "Run analysis"}
          </button>
        </div>
      </div>
      {launchErr && <ErrorState error={launchErr} />}

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 3fr) minmax(0, 2fr)", alignItems: "start" }}>
        <Panel
          title="Telemetry · last 24h"
          actions={<span className="small muted mono">poll 3s · {telemetry.data?.length ?? 0} rows</span>}
        >
          <Loaded data={telemetry.data} error={telemetry.error} loading={telemetry.loading} onRetry={telemetry.refresh} empty="No telemetry in the last 24h for this asset.">
            {() => (
              <div className="stack">
                <TimeSeriesChart title="vibration_rms" unit="mm/s" data={series.vib} color={CHART.ember} />
                <TimeSeriesChart title="bearing_temp_c" unit="°C" data={series.temp} color={CHART.steel} digits={1} />
                {series.preds.length > 0 ? (
                  <TimeSeriesChart title="failure_probability" unit="edge / control plane" data={series.preds} color={CHART.amber} yDomain={[0, 1]} threshold={0.5} />
                ) : (
                  <div className="small muted">No predictions logged yet — deploy a model to the edge or run an analysis.</div>
                )}
              </div>
            )}
          </Loaded>
        </Panel>

        <div className="stack">
          <Panel title="Latest sample">
            {latest ? (
              <div className="grid grid-4">
                <Stat label="vib rms" value={fmtNum(latest.vibration_rms, 2)} unit="mm/s" />
                <Stat label="kurtosis" value={fmtNum(latest.vibration_kurtosis, 2)} />
                <Stat label="crest" value={fmtNum(latest.vibration_crest, 2)} />
                <Stat label="bearing" value={fmtNum(latest.bearing_temp_c, 1)} unit="°C" />
                <Stat label="motor" value={fmtNum(latest.motor_temp_c, 1)} unit="°C" />
                <Stat label="current" value={fmtNum(latest.current_a, 1)} unit="A" />
                <Stat label="rpm" value={fmtNum(latest.rpm, 0)} />
                <Stat label="load" value={fmtNum(latest.load_pct, 0)} unit="%" />
              </div>
            ) : detail.loading ? (
              <LoadingState />
            ) : (
              <EmptyState>No samples yet.</EmptyState>
            )}
          </Panel>

          <Panel
            title="Analysis run"
            actions={
              run.state ? (
                <span className="small muted mono">
                  {run.state.run_id} · {run.transport === "ws" ? "live ws" : run.transport === "poll" ? "poll 2s" : "…"} · llm calls {run.state.llm_calls}
                </span>
              ) : undefined
            }
          >
            {run.state ? (
              <div className="stack">
                <StageStepper state={run.state} />
                {run.state.stage === "failed" && run.state.error && <ErrorState error={run.state.error} />}
                {run.error && <div className="small" style={{ color: "var(--ember-2)" }}>{run.error}</div>}
                <EventLog state={run.state} />
              </div>
            ) : runId ? (
              <LoadingState label="Connecting to run…" />
            ) : (
              <EmptyState>
                No analysis run for this asset yet. <b>Run analysis</b> profiles the fleet, trains three candidates, validates a champion and drafts a Decision Contract for approval.
              </EmptyState>
            )}
          </Panel>
        </div>
      </div>

      {showContract && contract && (
        <DecisionContractPanel contract={contract} approval={approval ?? null} onDecide={decide} busy={deciding} error={decideErr} />
      )}

      {run.state?.validation && runId && (
        <Panel title="What-if · reliability sandbox">
          <WhatIfForm runId={runId} baseline={currentP} />
        </Panel>
      )}

      {detail.data && detail.data.contracts.length > 0 && (
        <Panel title="Decision contract history" tight>
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Contract</th>
                  <th>Created</th>
                  <th>Model</th>
                  <th className="right">p(fail)</th>
                  <th>Recommendation</th>
                  <th>Work order</th>
                  <th className="right">Expected cost</th>
                  <th>Hash</th>
                </tr>
              </thead>
              <tbody>
                {detail.data.contracts.map((c) => (
                  <tr key={c.contract_id}>
                    <td className="mono">{c.contract_id}</td>
                    <td className="mono small">{fmtRel(c.created_at)}</td>
                    <td className="mono">{c.model_version}</td>
                    <td className="num">{fmtPct(c.failure_probability)}</td>
                    <td>{c.recommendation}</td>
                    <td>
                      {(() => {
                        const wo = detail.data?.work_orders?.find((w) => w.contract_id === c.contract_id);
                        return wo ? <WorkOrderPill wo={wo} /> : <span className="dim">—</span>;
                      })()}
                    </td>
                    <td className="num">{fmtNum(c.expected_cost, 0)}</td>
                    <td className="mono dim">{c.evidence_hash.slice(0, 12)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
