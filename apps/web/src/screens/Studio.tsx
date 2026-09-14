import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getRun, listRuns } from "../api/client";
import type { CandidateModel, GraphState, IndustrialModelScore, RunSummary } from "../api/types";
import { TERMINAL_STAGES } from "../api/types";
import { AXIS_TICK, CHART, ChartTip } from "../components/charts";
import { BarH, EmptyState, ErrorState, Kv, LoadingState, ModelStageChip, Panel, Pill, StageChip } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtHours, fmtInt, fmtNum, fmtPct, fmtRel, STAGE_LABEL } from "../lib/format";

const IMS_PARTS: { key: keyof Omit<IndustrialModelScore, "weights" | "total">; label: string }[] = [
  { key: "recall", label: "recall" },
  { key: "lead_time_score", label: "lead time" },
  { key: "precision", label: "precision" },
  { key: "calibration_score", label: "calibration" },
  { key: "latency_score", label: "latency" },
];

const FAMILY_LABEL: Record<string, string> = {
  random_forest: "Random forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
};

function StageCol({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <Panel
      className="stage-col"
      title={
        <h3>
          <span className="idx">{String(n).padStart(2, "0")}</span>
          {title}
        </h3>
      }
    >
      {children}
    </Panel>
  );
}

function Pending({ label }: { label: string }) {
  return <EmptyState>{label}</EmptyState>;
}

function ImsStack({ candidates, championId }: { candidates: CandidateModel[]; championId: string | null }) {
  const data = candidates.map((c) => {
    const ims = c.ims;
    const w = ims?.weights ?? {};
    const row: Record<string, number | string> = { name: FAMILY_LABEL[c.family] ?? c.family, total: ims?.total ?? 0 };
    for (const p of IMS_PARTS) row[p.label] = ims ? (ims[p.key] ?? 0) * (w[p.key] ?? 0) : 0;
    row.champion = c.candidate_id === championId ? 1 : 0;
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={Math.max(110, data.length * 30 + 40)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 12, bottom: 0, left: 0 }} barCategoryGap={6}>
        <XAxis type="number" domain={[0, 1]} tick={AXIS_TICK} axisLine={{ stroke: CHART.axis }} tickLine={false} tickFormatter={(v: number) => fmtNum(v, 1)} />
        <YAxis type="category" dataKey="name" width={86} tick={{ ...AXIS_TICK, fill: "#aab5c0" }} axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: "rgba(255,255,255,0.03)" }} content={<ChartTip valueFormatter={(v) => (typeof v === "number" ? fmtNum(v, 3) : "—")} />} />
        <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "#74818e" }} iconSize={8} />
        {IMS_PARTS.map((p, i) => (
          <Bar key={p.label} dataKey={p.label} stackId="ims" fill={CHART.steelRamp[i]} stroke="#161d25" strokeWidth={1} isAnimationActive={false} maxBarSize={16} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function CalibrationChart({ bins }: { bins: [number, number, number][] }) {
  const data = bins.map(([x, y, n]) => ({ x, y, n }));
  if (data.length === 0) return <div className="small muted">No reliability bins.</div>;
  return (
    <ResponsiveContainer width="100%" height={130}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -22 }}>
        <CartesianGrid stroke={CHART.grid} />
        <XAxis dataKey="x" type="number" domain={[0, 1]} tick={AXIS_TICK} axisLine={{ stroke: CHART.axis }} tickLine={false} tickFormatter={(v: number) => fmtNum(v, 1)} />
        <YAxis type="number" domain={[0, 1]} tick={AXIS_TICK} axisLine={false} tickLine={false} tickFormatter={(v: number) => fmtNum(v, 1)} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={CHART.axis} strokeDasharray="3 3" />
        <Tooltip content={<ChartTip labelFormatter={(l) => `predicted ${fmtNum(Number(l), 2)}`} valueFormatter={(v) => (typeof v === "number" ? fmtNum(v, 2) : "—")} />} />
        <Line type="monotone" dataKey="y" name="observed" stroke={CHART.steel} strokeWidth={2} dot={{ r: 3, fill: CHART.steel, strokeWidth: 0 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function Studio() {
  const [params, setParams] = useSearchParams();
  const runs = usePolling(() => listRuns(), 5000, "runs");
  const selected = params.get("run");

  // default to the most recent run once the list arrives
  useEffect(() => {
    if (!selected && runs.data && runs.data.length > 0) {
      const latest = [...runs.data].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""))[0];
      if (latest) setParams({ run: latest.run_id }, { replace: true });
    }
  }, [runs.data, selected, setParams]);

  const [state, setState] = useState<GraphState | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const terminal = state !== null && TERMINAL_STAGES.includes(state.stage);
  const poll = usePolling(() => getRun(selected ?? ""), terminal ? 0 : 3000, selected ?? "", selected !== null);
  useEffect(() => {
    if (poll.data) setState(poll.data);
    setErr(poll.error);
  }, [poll.data, poll.error]);
  useEffect(() => setState(null), [selected]);

  const dq = state?.data_quality ?? null;
  const cs = state?.candidates ?? null;
  const val = state?.validation ?? null;
  const contract = state?.contract ?? null;
  const approval = state?.approval ?? null;

  const runOptions = useMemo(() => {
    const list: RunSummary[] = runs.data ? [...runs.data] : [];
    return list.sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
  }, [runs.data]);

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>AutoML Studio</h1>
          <div className="sub">Every number here is computed by deterministic tool functions and logged to MLflow — the LLM only routes and explains.</div>
        </div>
        <div className="row">
          <label className="field">
            run
            <select className="select mono" value={selected ?? ""} onChange={(e) => setParams({ run: e.target.value })} style={{ minWidth: 300 }}>
              {runOptions.length === 0 && <option value="">— no runs yet —</option>}
              {runOptions.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.asset_id} · {r.run_id} · {STAGE_LABEL[r.stage] ?? r.stage} · {fmtRel(r.created_at)}
                </option>
              ))}
            </select>
          </label>
          {state && <StageChip stage={state.stage} />}
        </div>
      </div>

      {runs.error && !runs.data && <ErrorState error={runs.error} onRetry={runs.refresh} />}
      {!runs.error && runs.data && runs.data.length === 0 && (
        <EmptyState>No pipeline runs yet. Open an asset and press <b>Run analysis</b>.</EmptyState>
      )}
      {err && !state && <ErrorState error={err} />}
      {selected && !state && !err && <LoadingState label="Loading run…" />}

      {state && (
        <div className="studio-cols">
          <StageCol n={1} title="DATA">
            {dq ? (
              <div className="stack">
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="muted">quality score</span>
                  <span className="mono" style={{ fontSize: 22, color: dq.quality_score >= 60 ? "var(--green)" : "var(--ember-2)" }}>
                    {fmtNum(dq.quality_score, 0)}
                  </span>
                </div>
                <BarH value={dq.quality_score / 100} color={dq.quality_score >= 60 ? CHART.green : CHART.ember} />
                <Kv
                  rows={[
                    ["rows", fmtInt(dq.n_rows)],
                    ["assets", fmtInt(dq.n_assets)],
                    ["window", `${fmtDateTime(dq.window_start)} → ${fmtDateTime(dq.window_end)}`],
                    ["sampling", `${fmtNum(dq.sampling_rate_hz, 5)} Hz`],
                    ["trainable", dq.trainable ? "yes" : "no"],
                    ["flatlined", dq.missingness?.flatlined_sensors?.length ? dq.missingness.flatlined_sensors.join(", ") : "none"],
                  ]}
                />
                <div>
                  <h3 style={{ marginBottom: 4 }}>Regimes · {dq.regimes?.method ?? "—"}</h3>
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>id</th>
                        <th className="right">rpm</th>
                        <th className="right">load</th>
                        <th className="right">share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(dq.regimes?.regimes ?? []).map((r) => (
                        <tr key={r.regime_id}>
                          <td className="mono" title={r.label}>{r.regime_id}</td>
                          <td className="num">{fmtNum(r.rpm_mean, 0)}</td>
                          <td className="num">{fmtNum(r.load_mean, 0)}</td>
                          <td className="num">{fmtPct(r.share, 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {dq.regimes?.silhouette !== null && dq.regimes?.silhouette !== undefined && (
                    <div className="small muted" style={{ marginTop: 4 }}>silhouette {fmtNum(dq.regimes.silhouette, 2)}</div>
                  )}
                </div>
                <div>
                  <h3 style={{ marginBottom: 4 }}>Schema issues</h3>
                  {dq.schema_issues.length === 0 ? (
                    <div className="small muted">none</div>
                  ) : (
                    <div className="col" style={{ gap: 4 }}>
                      {dq.schema_issues.map((s, i) => (
                        <div key={i} className="row small">
                          <Pill tone={s.severity === "error" ? "danger" : s.severity === "warning" ? "warn" : "neutral"}>{s.severity}</Pill>
                          <span className="mono">{s.column}</span>
                          <span className="muted">{s.issue}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <Pending label={state.stage === "profiling" ? "Profiling fleet telemetry…" : "Not profiled yet."} />
            )}
          </StageCol>

          <StageCol n={2} title="TASK">
            {cs?.task ? (
              <div className="stack">
                <Pill tone="info">{cs.task.task}</Pill>
                <Kv rows={[["target", cs.task.target], ["horizon", fmtHours(cs.task.horizon_h)], ["experiment", cs.mlflow_experiment]]} />
                <div className="small muted">{cs.task.rationale}</div>
              </div>
            ) : (
              <Pending label={state.stage === "training" ? "Inferring task…" : "Task not inferred yet."} />
            )}
          </StageCol>

          <StageCol n={3} title="FEATURES">
            {cs?.features ? (
              <div className="stack">
                <Kv rows={[["count", String(cs.features.features.length)], ["window", `${cs.features.window_rows} rows`], ["regime-normalized", cs.features.regime_normalized ? "yes" : "no"]]} />
                <div className="feat-list">
                  {cs.features.features.map((f) => (
                    <span className="chip" key={f}>{f}</span>
                  ))}
                </div>
                <div className="small muted">{cs.features.description}</div>
              </div>
            ) : (
              <Pending label="Feature pipeline not built yet." />
            )}
          </StageCol>

          <StageCol n={4} title="MODELS">
            {cs && cs.candidates.length > 0 ? (
              <div className="stack">
                <div>
                  <h3 style={{ marginBottom: 2 }}>Industrial Model Score</h3>
                  <ImsStack candidates={cs.candidates} championId={val?.champion_id ?? cs.ranked[0] ?? null} />
                </div>
                {cs.candidates.map((c) => {
                  const champ = c.candidate_id === (val?.champion_id ?? cs.ranked[0]);
                  return (
                    <div className={`cand${champ ? " champ" : ""}`} key={c.candidate_id}>
                      <div className="row" style={{ justifyContent: "space-between" }}>
                        <b>{FAMILY_LABEL[c.family] ?? c.family}</b>
                        <span className="row">
                          {champ && <Pill tone="info">champion</Pill>}
                          <span className="mono">IMS {fmtNum(c.ims?.total ?? null, 3)}</span>
                        </span>
                      </div>
                      <div className="small muted mono" style={{ marginTop: 4 }}>
                        {["recall", "precision", "auroc", "brier", "f1"]
                          .filter((k) => c.metrics && k in c.metrics)
                          .map((k) => `${k} ${fmtNum(c.metrics[k], 3)}`)
                          .join(" · ") || "no metrics"}
                        {" · "}
                        {fmtNum(c.inference_latency_ms, 2)} ms
                      </div>
                      <div className="small dim mono" style={{ marginTop: 2 }}>
                        {c.candidate_id}
                        {c.mlflow_run_id ? ` · mlflow ${c.mlflow_run_id.slice(0, 8)}` : ""}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <Pending label={state.stage === "training" ? "Optuna trials running for 3 families…" : "No candidates yet."} />
            )}
          </StageCol>

          <StageCol n={5} title="VALIDATE">
            {val ? (
              <div className="stack">
                <div className="row wrap">
                  <Pill tone={val.passed ? "good" : "danger"}>{val.passed ? "passed" : "failed"}</Pill>
                  <span className="mono small">{val.champion_family} · {val.model_version}</span>
                </div>
                <div>
                  <h3 style={{ marginBottom: 4 }}>Leakage</h3>
                  <Kv
                    rows={[
                      ["temporal", val.leakage.temporal_leakage ? "LEAK" : "clean"],
                      ["asset overlap", val.leakage.asset_overlap ? "OVERLAP" : "none"],
                      ["feature", val.leakage.feature_leakage.length ? val.leakage.feature_leakage.join(", ") : "none"],
                    ]}
                  />
                </div>
                <div>
                  <h3 style={{ marginBottom: 4 }}>Backtest · asset-level folds</h3>
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>fold</th>
                        <th className="right">test</th>
                        <th className="right">recall</th>
                        <th className="right">prec</th>
                        <th className="right">auroc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {val.backtest.map((f) => (
                        <tr key={f.fold} title={`train end ${fmtDateTime(f.train_end)} · test ${f.test_assets.join(", ")}`}>
                          <td className="mono">{f.fold}</td>
                          <td className="num">{f.test_assets.length}</td>
                          <td className="num">{fmtNum(f.metrics.recall, 2)}</td>
                          <td className="num">{fmtNum(f.metrics.precision, 2)}</td>
                          <td className="num">{fmtNum(f.metrics.auroc, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h3 style={{ marginBottom: 4 }}>Calibration · {val.calibration.method}</h3>
                  <CalibrationChart bins={val.calibration.reliability_bins ?? []} />
                  <div className="small muted mono">brier {fmtNum(val.calibration.brier_before, 3)} → {fmtNum(val.calibration.brier_after, 3)}</div>
                </div>
                <div>
                  <h3 style={{ marginBottom: 4 }}>Lead time</h3>
                  <Kv
                    rows={[
                      ["median", fmtHours(val.lead_time.median_h)],
                      ["p90", fmtHours(val.lead_time.p90_h)],
                      ["p10", fmtHours(val.lead_time.p10_h)],
                      ["events", `${val.lead_time.n_events} @ ${fmtNum(val.lead_time.threshold, 2)}`],
                      ...(val.rul ? ([["RUL p10/50/90", `${fmtHours(val.rul.p10_h)} / ${fmtHours(val.rul.p50_h)} / ${fmtHours(val.rul.p90_h)}`]] as [string, string][]) : []),
                    ]}
                  />
                </div>
              </div>
            ) : (
              <Pending label={state.stage === "validating" ? "Running leakage checks, backtest, calibration…" : "Not validated yet."} />
            )}
          </StageCol>

          <StageCol n={6} title="DEPLOY">
            {contract || approval ? (
              <div className="stack">
                {contract && (
                  <div>
                    <h3 style={{ marginBottom: 4 }}>Decision contract</h3>
                    <Kv
                      rows={[
                        ["id", contract.contract_id],
                        ["p(fail)", fmtPct(contract.failure_probability)],
                        ["recommendation", contract.recommendation],
                        ["hash", contract.evidence_hash.slice(0, 16)],
                      ]}
                    />
                  </div>
                )}
                <div>
                  <h3 style={{ marginBottom: 4 }}>Approval</h3>
                  {approval ? (
                    <div className="stack" style={{ gap: 4 }}>
                      <Pill tone={approval.decision === "approved" ? "good" : "danger"}>{approval.decision}</Pill>
                      <span className="small mono muted">{approval.approver} · {fmtDateTime(approval.decided_at)}</span>
                      {approval.note && <span className="small">{approval.note}</span>}
                    </div>
                  ) : (
                    <Pill tone="bad">awaiting engineer</Pill>
                  )}
                </div>
                <div>
                  <h3 style={{ marginBottom: 4 }}>Registry stage</h3>
                  <div className="row">
                    <ModelStageChip stage={approval?.decision === "approved" ? "validated" : "candidate"} />
                    <span className="small muted">see ModelOps for promotion</span>
                  </div>
                </div>
              </div>
            ) : (
              <Pending label={state.stage === "failed" ? `Run failed: ${state.error ?? "unknown"}` : "No contract drafted yet."} />
            )}
          </StageCol>
        </div>
      )}
    </div>
  );
}
