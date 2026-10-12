import { getLabelLedger } from "../api/client";
import type { AssetSplit, LabelSource } from "../api/types";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtRel } from "../lib/format";
import { Pill } from "./ui";

const SIM_SOURCE = "simulator history (stands in for imported CMMS records)";

/** Older runs recorded no label_sources: rebuild the simulator-history rows from the split. */
function sourcesFromSplit(split: AssetSplit | null | undefined): LabelSource[] {
  if (!split) return [];
  const out: LabelSource[] = [];
  for (const group of ["train", "validation", "calibration", "test"]) {
    for (const asset_id of split.positives?.[group] ?? []) out.push({ asset_id, source: "simulator_history", group });
  }
  return out;
}

/**
 * Label ledger (AutoML Studio): every completed failure the selected run trains on, with its
 * source, plus the confirmed findings queued for the NEXT training run (GET /findings/ledger).
 * Every row comes from the API; a finding's label is a person's statement, never a model's.
 */
export function LabelLedgerPanel({ sources, split, runId }: { sources: LabelSource[] | null | undefined; split: AssetSplit | null | undefined; runId: string | null }) {
  const ledger = usePolling(getLabelLedger, 5000, "label-ledger");
  const rows = sources && sources.length > 0 ? sources : sourcesFromSplit(split);
  const inRun = new Set(rows.filter((r) => r.source === "work_order_finding").map((r) => r.finding_id));
  const queued = ledger.data?.queued ?? [];
  return (
    <div data-testid="label-ledger">
      <h3 style={{ marginBottom: 4 }} title="Which completed failures this run learns from and where each label came from">
        Label ledger
      </h3>
      {rows.length === 0 ? (
        <div className="small muted">{runId ? "No completed failure in this run's split." : "Select a train run."}</div>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>motor</th>
              <th>group</th>
              <th>label source</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.asset_id}-${r.source}`}>
                <td className="mono">{r.asset_id}</td>
                <td className="mono small">{r.group}</td>
                <td className="small">
                  {r.source === "simulator_history" ? (
                    <span className="muted">{SIM_SOURCE}</span>
                  ) : (
                    <span>
                      <Pill tone="bad">confirmed by work order</Pill> <span className="mono">{r.contract_id}</span>, recorded by{" "}
                      <span className="mono">{r.recorded_by}</span>
                      {r.event_ts && <span className="muted"> · offline {fmtDateTime(r.event_ts)}</span>}
                      {r.n_rows !== null && r.n_rows !== undefined && <span className="muted"> · {r.n_rows} rows labelled</span>}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="small" style={{ marginTop: 6 }}>
        <span className="muted">Queued for the next training run: </span>
        {ledger.data === null ? (
          <span className="dim">…</span>
        ) : queued.length === 0 ? (
          <span className="dim">no confirmed finding yet</span>
        ) : (
          <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
            {queued.map((q) => (
              <li key={q.finding_id}>
                <span className="mono">{q.asset_id}</span> · confirmed by work order <span className="mono">{q.contract_id}</span>, recorded by{" "}
                <span className="mono">{q.recorded_by}</span> <span className="dim" title={fmtDateTime(q.recorded_at)}>{fmtRel(q.recorded_at)}</span>
                {inRun.has(q.finding_id) ? <Pill tone="good">in this run</Pill> : <Pill tone="info">queued</Pill>}
              </li>
            ))}
          </ul>
        )}
      </div>
      {ledger.data && queued.length > 0 && <div className="small dim" style={{ marginTop: 4 }}>{ledger.data.rule}</div>}
    </div>
  );
}
