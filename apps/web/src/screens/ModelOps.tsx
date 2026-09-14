import { useState } from "react";
import { checkDrift, decidePromotion, getFleet, listDrift, listModels, listPromotions } from "../api/client";
import type { Decision, DriftReport, PromotionRequest } from "../api/types";
import { Loaded, ModelStageChip, Panel, Pill } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtNum, fmtRel } from "../lib/format";

function RegistryPanel() {
  const models = usePolling(listModels, 5000, "models");
  return (
    <Panel title="Model registry" tight actions={<span className="small muted mono">{models.data?.length ?? 0} versions</span>}>
      <Loaded data={models.data} error={models.error} loading={models.loading} onRetry={models.refresh} empty="No registered models. A validated run registers its champion as a candidate.">
        {(list) => (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Version</th>
                  <th>Family</th>
                  <th>Stage</th>
                  <th className="right">IMS</th>
                  <th>Registered</th>
                  <th>MLflow run</th>
                  <th>ONNX</th>
                </tr>
              </thead>
              <tbody>
                {[...list]
                  .sort((a, b) => b.registered_at.localeCompare(a.registered_at))
                  .map((m) => (
                    <tr key={`${m.name}:${m.version}`}>
                      <td className="mono">{m.name}</td>
                      <td className="mono">{m.version}</td>
                      <td>{m.family}</td>
                      <td>
                        <ModelStageChip stage={m.stage} />
                      </td>
                      <td className="num">{fmtNum(m.ims_total, 3)}</td>
                      <td className="mono small" title={fmtDateTime(m.registered_at)}>{fmtRel(m.registered_at)}</td>
                      <td className="mono small dim">{m.mlflow_run_id ? m.mlflow_run_id.slice(0, 10) : "—"}</td>
                      <td className="mono small">{m.onnx_path ? <span title={m.onnx_path}>{m.onnx_path.split(/[\\/]/).pop()}</span> : <span className="dim">not exported</span>}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </Loaded>
    </Panel>
  );
}

function PromotionCard({ p, onDecided }: { p: PromotionRequest; onDecided: () => void }) {
  const [approver, setApprover] = useState("engineer@plant");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const cmp = p.comparison;
  const delta = cmp?.delta ?? {};
  const pending = p.status === "pending";

  const decide = async (decision: Decision) => {
    setBusy(true);
    setErr(null);
    try {
      await decidePromotion(p.promotion_id, { decision, approver: approver.trim(), note: "" });
      onDecided();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="cand stack" style={{ gap: 8 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="row">
          <span className="mono">{p.model_name}</span>
          <span className="mono">v{p.version}</span>
          <ModelStageChip stage={p.from_stage} />
          <span className="dim">→</span>
          <ModelStageChip stage={p.to_stage} />
        </span>
        <span className="row">
          <Pill tone={p.status === "approved" ? "good" : p.status === "rejected" ? "danger" : "bad"}>{p.status}</Pill>
          <span className="small muted mono">{fmtRel(p.created_at)}</span>
        </span>
      </div>
      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th></th>
              <th>Challenger</th>
              <th>Champion</th>
              <th className="right">Δ</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="muted">version</td>
              <td className="mono">{cmp?.challenger?.version ?? "—"}</td>
              <td className="mono">{cmp?.champion?.version ?? <span className="dim">none in production</span>}</td>
              <td className="num"></td>
            </tr>
            <tr>
              <td className="muted">family</td>
              <td>{cmp?.challenger?.family ?? "—"}</td>
              <td>{cmp?.champion?.family ?? "—"}</td>
              <td className="num"></td>
            </tr>
            <tr>
              <td className="muted">IMS</td>
              <td className="mono">{fmtNum(cmp?.challenger?.ims_total, 3)}</td>
              <td className="mono">{cmp?.champion ? fmtNum(cmp.champion.ims_total, 3) : "—"}</td>
              <td className="num" style={{ color: (delta.ims_total ?? 0) >= 0 ? "var(--green)" : "var(--ember-2)" }}>
                {delta.ims_total !== undefined ? `${delta.ims_total >= 0 ? "+" : ""}${fmtNum(delta.ims_total, 3)}` : "—"}
              </td>
            </tr>
            {Object.entries(delta)
              .filter(([k]) => k !== "ims_total")
              .map(([k, v]) => (
                <tr key={k}>
                  <td className="muted">{k}</td>
                  <td></td>
                  <td></td>
                  <td className="num" style={{ color: v >= 0 ? "var(--green)" : "var(--ember-2)" }}>
                    {v >= 0 ? "+" : ""}
                    {fmtNum(v, 3)}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <div className="small muted">
        {cmp?.recommend_promote ? <Pill tone="info">recommends promote</Pill> : <Pill tone="neutral">no recommendation</Pill>} {cmp?.rationale}
      </div>
      {pending && (
        <div className="row wrap" style={{ alignItems: "flex-end" }}>
          <label className="field">
            approver
            <input className="input mono" value={approver} onChange={(e) => setApprover(e.target.value)} style={{ width: 170 }} />
          </label>
          <button className="btn primary" disabled={busy || !approver.trim()} onClick={() => void decide("approved")}>
            Approve promotion
          </button>
          <button className="btn" disabled={busy || !approver.trim()} onClick={() => void decide("rejected")}>
            Reject
          </button>
          {err && <span className="small" style={{ color: "var(--ember-2)" }}>{err}</span>}
        </div>
      )}
    </div>
  );
}

function PromotionsPanel() {
  const promos = usePolling(listPromotions, 5000, "promotions");
  const [showAll, setShowAll] = useState(false);
  return (
    <Panel
      title="Promotion requests"
      actions={
        <label className="row small muted" style={{ gap: 4 }}>
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> show decided
        </label>
      }
    >
      <Loaded
        data={promos.data}
        error={promos.error}
        loading={promos.loading}
        onRetry={promos.refresh}
        isEmpty={(l) => l.filter((p) => showAll || p.status === "pending").length === 0}
        empty={showAll ? "No promotion requests." : "No pending promotions. Every promotion needs a recorded approval before promote() runs."}
      >
        {(list) => (
          <div className="stack">
            {[...list]
              .filter((p) => showAll || p.status === "pending")
              .sort((a, b) => b.created_at.localeCompare(a.created_at))
              .map((p) => (
                <PromotionCard key={p.promotion_id} p={p} onDecided={() => void promos.refresh()} />
              ))}
          </div>
        )}
      </Loaded>
    </Panel>
  );
}

function DriftPanel() {
  const drift = usePolling(() => listDrift(), 5000, "drift");
  const fleet = usePolling(getFleet, 30000, "fleet-for-drift");
  const [assetId, setAssetId] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [last, setLast] = useState<DriftReport | null>(null);

  const run = async () => {
    if (!assetId) return;
    setBusy(true);
    setErr(null);
    try {
      setLast(await checkDrift({ asset_id: assetId }));
      void drift.refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const assets = fleet.data ? [...fleet.data].sort((a, b) => a.asset.asset_id.localeCompare(b.asset.asset_id)) : [];

  return (
    <Panel
      title="Drift reports · PSI"
      actions={
        <span className="row">
          <select className="select mono" value={assetId} onChange={(e) => setAssetId(e.target.value)}>
            <option value="">— asset —</option>
            {assets.map((a) => (
              <option key={a.asset.asset_id} value={a.asset.asset_id}>
                {a.asset.asset_id}
              </option>
            ))}
          </select>
          <button className="btn steel sm" disabled={!assetId || busy} onClick={() => void run()}>
            {busy ? "Checking…" : "Check drift"}
          </button>
        </span>
      }
      tight
    >
      {err && <div className="small" style={{ color: "var(--ember-2)", padding: "8px 12px" }}>{err}</div>}
      {last && (
        <div className="small" style={{ padding: "8px 12px", borderBottom: "1px solid var(--line)" }}>
          Latest check on <span className="mono">{last.asset_id}</span>:{" "}
          <Pill tone={last.drift_detected ? "bad" : "good"}>{last.drift_detected ? "drift detected" : "stable"}</Pill>{" "}
          {last.drifted_features.length > 0 && <span className="mono muted">{last.drifted_features.join(", ")}</span>}
        </div>
      )}
      <Loaded data={drift.data} error={drift.error} loading={drift.loading} onRetry={drift.refresh} empty="No drift reports yet. Pick an asset and check drift.">
        {(list) => (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Model</th>
                  <th>Window</th>
                  <th>Status</th>
                  <th className="right">max PSI</th>
                  <th>Drifted features</th>
                </tr>
              </thead>
              <tbody>
                {[...list]
                  .sort((a, b) => b.window_end.localeCompare(a.window_end))
                  .map((d, i) => {
                    const vals = Object.values(d.psi ?? {});
                    const maxPsi = vals.length ? Math.max(...vals) : null;
                    return (
                      <tr key={`${d.asset_id}-${d.window_end}-${i}`}>
                        <td className="mono">{d.asset_id}</td>
                        <td className="mono">{d.model_version}</td>
                        <td className="mono small">{fmtDateTime(d.window_start)} → {fmtDateTime(d.window_end)}</td>
                        <td>
                          <Pill tone={d.drift_detected ? "bad" : "good"}>{d.drift_detected ? "drift" : "stable"}</Pill>
                        </td>
                        <td className="num" title={`threshold ${fmtNum(d.threshold, 2)}`}>{fmtNum(maxPsi, 3)}</td>
                        <td className="mono small">{d.drifted_features.length ? d.drifted_features.join(", ") : <span className="dim">—</span>}</td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        )}
      </Loaded>
    </Panel>
  );
}

export function ModelOps() {
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>ModelOps</h1>
          <div className="sub">Registry lifecycle candidate → validated → shadow → production. Promotions execute only after a recorded approval; drift re-triggers training.</div>
        </div>
      </div>
      <RegistryPanel />
      <div className="grid grid-2" style={{ alignItems: "start" }}>
        <PromotionsPanel />
        <DriftPanel />
      </div>
    </div>
  );
}
