import { useState } from "react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Approval, Decision, DecisionContract, MaintenanceOption } from "../api/types";
import { fmtDateTime, fmtHours, fmtMoney, fmtNum, fmtPct, humanise, OPTION_LABEL, shortHash } from "../lib/format";
import { AXIS_TICK, CHART, ChartTip } from "./charts";
import { Kv, Mono, Pill, SourceChip, Stat } from "./ui";

const OPTION_ORDER: MaintenanceOption[] = ["maintain_now", "maintain_later", "run_to_failure"];

function ShapBars({ contract }: { contract: DecisionContract }) {
  const feats = (contract.top_features ?? []).slice().sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
  if (feats.length === 0) return <div className="state">No SHAP attribution on this contract.</div>;
  const data = feats.map((f) => ({
    name: humanise(f.feature),
    shap: f.shap_value,
    feature_value: f.feature_value,
    direction: f.direction,
  }));
  const maxAbs = Math.max(...data.map((d) => Math.abs(d.shap)), 1e-6);
  return (
    <div>
      <ResponsiveContainer width="100%" height={Math.max(120, data.length * 26 + 24)}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 8 }} barCategoryGap={4}>
          <XAxis
            type="number"
            domain={[-maxAbs * 1.1, maxAbs * 1.1]}
            tick={AXIS_TICK}
            axisLine={{ stroke: CHART.axis }}
            tickLine={false}
            tickFormatter={(v: number) => fmtNum(v, 2)}
          />
          <YAxis type="category" dataKey="name" width={130} tick={{ ...AXIS_TICK, fill: "#aab5c0" }} axisLine={false} tickLine={false} />
          <ReferenceLine x={0} stroke={CHART.axis} />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.03)" }}
            content={
              <ChartTip
                valueFormatter={(v) => (typeof v === "number" ? `${v >= 0 ? "+" : ""}${fmtNum(v, 4)}` : "—")}
              />
            }
          />
          <Bar dataKey="shap" name="SHAP" isAnimationActive={false} radius={[0, 3, 3, 0]} maxBarSize={14}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.direction === "raises_risk" ? CHART.ember : CHART.steel} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="row wrap small muted" style={{ gap: 14, marginTop: 4 }}>
        <span className="row" style={{ gap: 5 }}>
          <span className="dot" style={{ background: CHART.ember }} /> raises risk
        </span>
        <span className="row" style={{ gap: 5 }}>
          <span className="dot" style={{ background: CHART.steel }} /> lowers risk
        </span>
        <span className="mono">
          {feats.map((f) => `${f.feature}=${fmtNum(f.feature_value, 2)}`).join(" · ")}
        </span>
      </div>
    </div>
  );
}

function CostTable({ contract }: { contract: DecisionContract }) {
  const cc = contract.cost_comparison;
  const options = cc?.options ?? [];
  const recommended = cc?.recommended ?? contract.recommendation;
  const ordered = OPTION_ORDER.map((o) => options.find((x) => x.option === o)).filter(
    (x): x is NonNullable<typeof x> => Boolean(x),
  );
  if (ordered.length === 0) return <div className="state">No cost comparison on this contract.</div>;
  return (
    <div className="table-wrap">
      <table className="tbl">
        <thead>
          <tr>
            <th>Option</th>
            <th>When</th>
            <th className="right">P(fail before)</th>
            <th className="right">Downtime</th>
            <th className="right">Expected cost</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((o) => (
            <tr key={o.option} className={o.option === recommended ? "hl" : ""}>
              <td>
                {OPTION_LABEL[o.option]}
                {o.option === recommended && (
                  <Pill tone="bad" title="Lowest expected cost">
                    recommended
                  </Pill>
                )}
              </td>
              <td className="mono small">{o.when ? fmtDateTime(o.when) : "—"}</td>
              <td className="num">{fmtPct(o.p_failure_before)}</td>
              <td className="num">{fmtHours(o.downtime_h)}</td>
              <td className="num">{fmtMoney(o.expected_cost)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="small muted" style={{ padding: "6px 10px" }}>
        Expected costs under configured plant-cost assumptions
        {cc?.assumptions && Object.keys(cc.assumptions).length > 0 && (
          <span className="mono dim">
            {" "}
            (
            {Object.entries(cc.assumptions)
              .map(([k, v]) => `${k}=${fmtNum(v, 0)}`)
              .join(", ")}
            )
          </span>
        )}
        . These are configurable inputs, not measured savings.
      </div>
    </div>
  );
}

export function DecisionContractPanel({
  contract,
  approval,
  onDecide,
  busy,
  error,
}: {
  contract: DecisionContract;
  approval: Approval | null;
  onDecide: (decision: Decision, approver: string, note: string) => Promise<void>;
  busy: boolean;
  error: string | null;
}) {
  const [approver, setApprover] = useState("engineer@plant");
  const [note, setNote] = useState("");
  const locked = approval !== null;
  const p = contract.failure_probability;

  return (
    <section className={`panel contract${locked ? " locked" : ""}`}>
      <div className="banner">
        {locked ? (
          <>
            <span className="dot on" />
            Contract is immutable; approval recorded as linked row
            <Mono>{approval.approval_id}</Mono>
          </>
        ) : (
          <>
            <span className="dot off pulse" />
            Decision Contract awaiting engineer approval — no action executes until recorded
          </>
        )}
      </div>
      <div className="panel-body stack">
        <div className="grid" style={{ gridTemplateColumns: "180px 1fr", alignItems: "start" }}>
          <Stat label="Failure probability" value={fmtPct(p, 1)} tone={p >= 0.4 ? "ember" : p >= 0.15 ? undefined : "green"} big />
          <Kv
            rows={[
              ["contract", contract.contract_id],
              ["evidence hash", <span title={contract.evidence_hash}>{shortHash(contract.evidence_hash, 20)}</span>],
              ["model version", contract.model_version],
              ["mlflow run", contract.champion_mlflow_run_id ?? "—"],
              ["created", fmtDateTime(contract.created_at)],
              ["lead time", fmtHours(contract.lead_time_h)],
              ["calibration brier", fmtNum(contract.calibration_brier, 3)],
              ["data quality", fmtNum(contract.data_quality_score, 0)],
            ]}
          />
        </div>

        <div className="grid grid-2">
          <div>
            <h3 style={{ marginBottom: 6 }}>Why — SHAP attribution</h3>
            <ShapBars contract={contract} />
          </div>
          <div>
            <h3 style={{ marginBottom: 6 }}>Cost comparison</h3>
            <CostTable contract={contract} />
            <div className="row" style={{ marginTop: 10, gap: 10 }}>
              <span className="muted small">Recommended window</span>
              <span className="mono">
                {contract.window_start ? fmtDateTime(contract.window_start) : "—"} → {contract.window_end ? fmtDateTime(contract.window_end) : "—"}
              </span>
              <Pill tone="bad">{OPTION_LABEL[contract.recommendation] ?? contract.recommendation}</Pill>
            </div>
          </div>
        </div>

        <div>
          <div className="row" style={{ marginBottom: 6 }}>
            <h3>Explanation</h3>
            <SourceChip source={contract.explanation_source} />
          </div>
          <p className="explain" style={{ margin: 0 }}>
            {contract.explanation_text || <span className="muted">No explanation text.</span>}
          </p>
        </div>

        <div>
          <h3 style={{ marginBottom: 6 }}>Cited manual passages</h3>
          {(contract.manual_context ?? []).length === 0 ? (
            <div className="muted small">No manual passages retrieved — the recommendation stands on SHAP + cost alone.</div>
          ) : (
            <div className="col">
              {contract.manual_context.map((m, i) => (
                <div className="passage" key={i}>
                  <div className="src">
                    {m.source} · {m.section}
                    {m.page !== null && m.page !== undefined ? ` · p.${m.page}` : ""}
                    <span className="dim"> · score {fmtNum(m.score, 2)}</span>
                  </div>
                  <div className="txt">{m.text.length > 420 ? `${m.text.slice(0, 420)}…` : m.text}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {locked ? (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Approval</th>
                  <th>Decision</th>
                  <th>Approver</th>
                  <th>Note</th>
                  <th>Decided at</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="mono">{approval.approval_id}</td>
                  <td>
                    <Pill tone={approval.decision === "approved" ? "good" : "danger"}>{approval.decision}</Pill>
                  </td>
                  <td className="mono">{approval.approver}</td>
                  <td>{approval.note || <span className="dim">—</span>}</td>
                  <td className="mono">{fmtDateTime(approval.decided_at)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          <div className="row wrap" style={{ gap: 10, alignItems: "flex-end" }}>
            <label className="field">
              approver
              <input className="input mono" value={approver} onChange={(e) => setApprover(e.target.value)} style={{ width: 180 }} />
            </label>
            <label className="field" style={{ flex: 1, minWidth: 200 }}>
              note (optional)
              <input className="input" value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. schedule with night-shift crew" />
            </label>
            <button className="btn primary" disabled={busy || !approver.trim()} onClick={() => void onDecide("approved", approver.trim(), note)}>
              Approve
            </button>
            <button className="btn" disabled={busy || !approver.trim()} onClick={() => void onDecide("rejected", approver.trim(), note)}>
              Reject
            </button>
            {error && <span className="small" style={{ color: "var(--ember-2)" }}>{error}</span>}
          </div>
        )}
      </div>
    </section>
  );
}
