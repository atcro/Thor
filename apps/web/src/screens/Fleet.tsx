import { useMemo } from "react";
import { Link } from "react-router-dom";
import { getFleet, getSystemHealth } from "../api/client";
import type { FleetAsset } from "../api/types";
import { Dot, HealthPill, Loaded, ModelStageChip, Panel, Pill, RegimeBadge } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtInt, fmtPct, fmtRel, probTone } from "../lib/format";

/** Sort by failure risk desc; assets without a prediction sort by health asc afterwards. */
function sortByRisk(list: FleetAsset[]): FleetAsset[] {
  return [...list].sort((a, b) => {
    const pa = a.failure_probability ?? -1;
    const pb = b.failure_probability ?? -1;
    if (pb !== pa) return pb - pa;
    return a.health_score - b.health_score;
  });
}

function AssetCard({ item }: { item: FleetAsset }) {
  const { asset } = item;
  const p = item.failure_probability;
  const atRisk = (p ?? 0) >= 0.4 || item.health_score < 60;
  return (
    <Link to={`/assets/${encodeURIComponent(asset.asset_id)}`} className={`asset-card${atRisk ? " risk" : ""}`}>
      {item.open_contract_id && <span className="flag" title={`Open decision contract ${item.open_contract_id}`} />}
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="id">{asset.asset_id}</span>
        <HealthPill score={item.health_score} />
      </div>
      <div className="name">{asset.name}</div>
      <div className="loc">
        {asset.site} · {asset.line}
        {asset.criticality === "high" && <span className="dim"> · critical</span>}
      </div>
      <div className="row wrap" style={{ gap: 6 }}>
        <RegimeBadge regime={item.regime} />
        {p !== null && p !== undefined && (
          <Pill tone={probTone(p)} title="Current failure probability">
            p(fail) {fmtPct(p)}
          </Pill>
        )}
        {item.stage && <ModelStageChip stage={item.stage} />}
      </div>
      <div className="foot">
        <span title={fmtDateTime(item.last_ts)}>seen {fmtRel(item.last_ts)}</span>
        {item.open_contract_id && <span style={{ color: "var(--ember)" }}>contract open</span>}
      </div>
    </Link>
  );
}

function SystemPanel() {
  const sys = usePolling(getSystemHealth, 5000, "system");
  return (
    <Panel title="System">
      <Loaded data={sys.data} error={sys.error} loading={sys.loading} onRetry={sys.refresh}>
        {(h) => {
          const dbOk = h.db === true || h.db === "ok";
          return (
            <div className="sys-list">
              {(
                [
                  ["TimescaleDB", dbOk],
                  ["MQTT broker", h.mqtt],
                  ["MLflow", h.mlflow],
                  ["Edge inference", h.edge],
                ] as [string, boolean][]
              ).map(([name, ok]) => (
                <div className="row" key={name}>
                  <span className="row">
                    <Dot on={ok} />
                    {name}
                  </span>
                  <span className={`mono small ${ok ? "" : "muted"}`}>{ok ? "up" : "down"}</span>
                </div>
              ))}
              <div className="row">
                <span className="row">
                  <Dot on={h.llm?.mode === "llm"} />
                  LLM
                </span>
                <span className={`mono small ${h.llm?.mode === "llm" ? "" : "muted"}`}>
                  {h.llm ? (h.llm.mode === "llm" ? h.llm.model : "template mode") : "unknown"}
                </span>
              </div>
              <div className="row" style={{ borderTop: "1px solid var(--line)", paddingTop: 8 }}>
                <span className="muted">telemetry rows</span>
                <span className="mono">{fmtInt(h.n_telemetry_rows)}</span>
              </div>
              <div className="row">
                <span className="muted">last ingest</span>
                <span className="mono" title={fmtDateTime(h.last_ingest_ts)}>
                  {fmtRel(h.last_ingest_ts)}
                </span>
              </div>
            </div>
          );
        }}
      </Loaded>
    </Panel>
  );
}

export function Fleet() {
  const fleet = usePolling(getFleet, 5000, "fleet");
  const sorted = useMemo(() => (fleet.data ? sortByRisk(fleet.data) : []), [fleet.data]);
  const nRisk = sorted.filter((a) => (a.failure_probability ?? 0) >= 0.4 || a.health_score < 60).length;
  const nOpen = sorted.filter((a) => a.open_contract_id).length;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Fleet</h1>
          <div className="sub">
            {fleet.data
              ? `${sorted.length} assets · ${nRisk} at risk · ${nOpen} open contract${nOpen === 1 ? "" : "s"} · sorted by failure risk`
              : "Loading fleet…"}
          </div>
        </div>
        <span className="small muted mono">poll 5s{fleet.lastUpdated ? ` · ${fmtRel(new Date(fleet.lastUpdated).toISOString())}` : ""}</span>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "1fr 240px", alignItems: "start" }}>
        <div>
          <Loaded
            data={fleet.data}
            error={fleet.error}
            loading={fleet.loading}
            onRetry={fleet.refresh}
            empty="No assets registered. Start the API with SEED_ON_START=1 or run the replay script."
          >
            {() => (
              <div className="fleet-grid">
                {sorted.map((item) => (
                  <AssetCard key={item.asset.asset_id} item={item} />
                ))}
              </div>
            )}
          </Loaded>
        </div>
        <SystemPanel />
      </div>
    </div>
  );
}
