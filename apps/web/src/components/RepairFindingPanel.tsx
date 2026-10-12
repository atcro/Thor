import { useEffect, useState } from "react";
import { listFindings, postFinding } from "../api/client";
import type { FindingKind, RepairFinding, WorkOrder } from "../api/types";
import { FINDING_LABEL } from "../api/types";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtRel } from "../lib/format";
import { Mono, Panel, Pill } from "./ui";

const KINDS: FindingKind[] = ["confirmed_bearing_damage", "no_fault_found", "other"];

function kindTone(k: FindingKind): "bad" | "good" | "neutral" {
  return k === "confirmed_bearing_damage" ? "bad" : k === "no_fault_found" ? "good" : "neutral";
}

/** One finding row as listed; superseded rows stay visible and say so. */
function FindingRow({ f, superseded }: { f: RepairFinding; superseded: boolean }) {
  return (
    <tr style={superseded ? { opacity: 0.55 } : undefined}>
      <td>
        <Pill tone={kindTone(f.finding)}>{FINDING_LABEL[f.finding]}</Pill>
        {superseded && <span className="dim small"> superseded</span>}
        {f.sandbox && <span className="dim small"> sandbox</span>}
      </td>
      <td>{f.note || <span className="dim">—</span>}</td>
      <td className="mono">{f.recorded_by}</td>
      <td className="mono small" title={fmtDateTime(f.recorded_at)}>{fmtRel(f.recorded_at)}</td>
      <td className="mono small dim">
        {f.finding_id}
        {f.supersedes && <span title={`corrects ${f.supersedes}`}> ↶</span>}
      </td>
    </tr>
  );
}

/**
 * "What did the technician find?" -- the form that closes the loop on a completed work
 * order (POST /findings, insert-only). One finding per work order; "Correct this finding"
 * adds a new row that supersedes the current one, and every row stays listed.
 *
 * Shown on Asset 360 for each completed work order of the asset. Nothing here is derived
 * from the model or the simulator: the person chooses the finding and signs it.
 */
export function RepairFindingPanel({ workOrders, onRecorded }: { workOrders: WorkOrder[]; onRecorded?: () => void }) {
  const completed = workOrders.filter((w) => w.status === "completed");
  const assetId = completed[0]?.asset_id ?? workOrders[0]?.asset_id ?? "";
  const findings = usePolling(() => listFindings(assetId), 5000, `findings-${assetId}`, assetId !== "");
  if (completed.length === 0) return null;
  const rows = findings.data ?? [];
  return (
    <Panel title="What did the technician find?" tight>
      <div className="panel-body stack">
        <div className="small muted">
          The plant reported the repair complete. Record what was actually found so the alarm can be trusted
          (Fleet, ModelOps) and so the next training run can learn from it. Entered by a person, stored insert-only;
          a correction adds a new row.
        </div>
        {completed.map((wo) => (
          <WorkOrderFinding
            key={wo.contract_id}
            wo={wo}
            rows={rows.filter((f) => f.contract_id === wo.contract_id)}
            onRecorded={() => {
              void findings.refresh();
              onRecorded?.();
            }}
          />
        ))}
      </div>
    </Panel>
  );
}

function WorkOrderFinding({ wo, rows, onRecorded }: { wo: WorkOrder; rows: RepairFinding[]; onRecorded: () => void }) {
  const current = rows.length > 0 ? rows[rows.length - 1] : null;
  const [open, setOpen] = useState(current === null);
  const [kind, setKind] = useState<FindingKind>("confirmed_bearing_damage");
  const [note, setNote] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (current !== null) setOpen(false);
  }, [current]);

  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      await postFinding({ contract_id: wo.contract_id, finding: kind, note: note.trim(), recorded_by: name.trim(), supersedes: current?.finding_id ?? null });
      setNote("");
      setOpen(false);
      onRecorded();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack" data-testid={`finding-${wo.contract_id}`} style={{ gap: 8 }}>
      <div className="row wrap" style={{ gap: 8, justifyContent: "space-between" }}>
        <span className="row wrap" style={{ gap: 8 }}>
          <span className="muted small">work order</span>
          <Mono>{wo.contract_id}</Mono>
          <span className="small muted">
            offline {fmtDateTime(wo.started_ts)} → back {fmtDateTime(wo.completed_ts)} (plant time) · approved by <span className="mono">{wo.approver}</span>
          </span>
        </span>
        {current && !open && (
          <button type="button" className="btn sm" onClick={() => setOpen(true)}>
            Correct this finding
          </button>
        )}
      </div>
      {rows.length > 0 && (
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th>Finding</th>
                <th>Note</th>
                <th>Recorded by</th>
                <th>When</th>
                <th>Row</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((f) => (
                <FindingRow key={f.finding_id} f={f} superseded={f.finding_id !== current?.finding_id} />
              ))}
            </tbody>
          </table>
        </div>
      )}
      {open && (
        <form
          className="stack"
          style={{ gap: 8 }}
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <div className="row wrap" role="radiogroup" aria-label="What did the technician find?" style={{ gap: 14 }}>
            {KINDS.map((k) => (
              <label key={k} className="row small" style={{ gap: 6, cursor: "pointer" }}>
                <input type="radio" name={`finding-${wo.contract_id}`} value={k} checked={kind === k} onChange={() => setKind(k)} />
                {k === "other" ? "Other (describe)" : FINDING_LABEL[k]}
              </label>
            ))}
          </div>
          <div className="row wrap" style={{ gap: 10, alignItems: "flex-end" }}>
            <label className="field" style={{ flex: 1, minWidth: 220 }}>
              note{kind === "other" ? " (required)" : " (optional)"}
              <input className="input" value={note} onChange={(e) => setNote(e.target.value)} placeholder={kind === "confirmed_bearing_damage" ? "e.g. drive-end bearing outer race spalled; replaced" : kind === "no_fault_found" ? "e.g. bearing within spec; sensor bracket loose" : "describe what was found"} />
            </label>
            <label className="field">
              your name (required)
              <input className="input mono" value={name} onChange={(e) => setName(e.target.value)} placeholder="technician@plant" style={{ width: 180 }} />
            </label>
            <button type="submit" className="btn primary" disabled={busy || !name.trim() || (kind === "other" && !note.trim())}>
              {current ? "Record correction" : "Record finding"}
            </button>
            {current && (
              <button type="button" className="btn" onClick={() => setOpen(false)} disabled={busy}>
                Cancel
              </button>
            )}
            {err && <span className="small" style={{ color: "var(--ember-2)" }}>{err}</span>}
          </div>
        </form>
      )}
    </div>
  );
}
