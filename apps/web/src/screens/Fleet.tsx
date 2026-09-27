import { useMemo, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { getFleet, getSystemHealth } from "../api/client";
import type { FleetAsset } from "../api/types";
import { MotorSprite, spriteParams, TIER_WORD, type SpriteStyle } from "../components/MotorSprite";
import { Dot, HealthPill, Loaded, ModelStageChip, Panel, Pill, RegimeBadge, WorkOrderPill } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtInt, fmtPct, fmtRel, fmtShort, probTone } from "../lib/format";

/** Sort by failure risk desc; assets without a prediction sort by health asc afterwards. */
function sortByRisk(list: FleetAsset[]): FleetAsset[] {
  return [...list].sort((a, b) => {
    const pa = a.failure_probability ?? -1;
    const pb = b.failure_probability ?? -1;
    if (pb !== pa) return pb - pa;
    return a.health_score - b.health_score;
  });
}

/** Sprite style preference; a per-viewer convenience, so browser storage is fine here. */
const STYLE_KEY = "thor.fleet.spriteStyle";

function readStyle(): SpriteStyle {
  try {
    return localStorage.getItem(STYLE_KEY) === "pixel" ? "pixel" : "iso";
  } catch {
    return "iso";
  }
}

function saveStyle(style: SpriteStyle): void {
  try {
    localStorage.setItem(STYLE_KEY, style);
  } catch {
    /* private mode or blocked storage: the toggle still works for this page view */
  }
}

function StyleToggle({ value, onChange }: { value: SpriteStyle; onChange: (s: SpriteStyle) => void }) {
  return (
    <div className="seg" role="group" aria-label="Sprite style">
      {(
        [
          ["iso", "Isometric"],
          ["pixel", "Pixel"],
        ] as [SpriteStyle, string][]
      ).map(([key, label]) => (
        <button
          key={key}
          type="button"
          className={`seg-btn${value === key ? " on" : ""}`}
          aria-pressed={value === key}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function AssetCard({ item, style }: { item: FleetAsset; style: SpriteStyle }) {
  const { asset } = item;
  const p = item.failure_probability;
  const { atRisk, heat, tier } = spriteParams(item);
  return (
    <Link
      to={`/assets/${encodeURIComponent(asset.asset_id)}`}
      className={`asset-card tier-${tier}${atRisk ? " risk" : ""}`}
      style={{ "--heat": heat } as CSSProperties}
    >
      <div className="sprite">
        <MotorSprite item={item} style={style} />
        <span className={`tier-tag ${tier}`}>{TIER_WORD[tier]}</span>
      </div>
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
        {item.work_order && <WorkOrderPill wo={item.work_order} />}
      </div>
      <div className="foot">
        <span title={fmtDateTime(item.last_ts)}>last sample {fmtShort(item.last_ts)}</span>
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
                <span
                  className={`mono small ${h.llm?.mode === "llm" ? "" : "muted"}`}
                  title={h.llm?.warning ?? undefined}
                >
                  {h.llm
                    ? h.llm.mode === "llm"
                      ? `${h.llm.provider ?? ""} ${h.llm.model}`.trim()
                      : h.llm.provider && h.llm.provider !== "none"
                        ? `template mode (${h.llm.provider} key set, client pending)`
                        : "template mode"
                    : "unknown"}
                </span>
              </div>
              <div className="row" style={{ borderTop: "1px solid var(--line)", paddingTop: 8 }}>
                <span className="muted">telemetry rows</span>
                <span className="mono">{fmtInt(h.n_telemetry_rows)}</span>
              </div>
              <div className="row">
                <span className="muted">plant time</span>
                <span className="mono" title={fmtDateTime(h.last_ingest_ts)}>
                  {fmtShort(h.last_ingest_ts)}
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
  const [style, setStyle] = useState<SpriteStyle>(readStyle);
  const changeStyle = (s: SpriteStyle) => {
    setStyle(s);
    saveStyle(s);
  };
  const sorted = useMemo(() => (fleet.data ? sortByRisk(fleet.data) : []), [fleet.data]);
  const nRisk = sorted.filter((a) => (a.failure_probability ?? 0) >= 0.4 || a.health_score < 60).length;
  const nOpen = sorted.filter((a) => a.open_contract_id).length;
  const nWork = sorted.filter((a) => a.work_order && a.work_order.status !== "completed").length;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Fleet</h1>
          <div className="sub">
            {fleet.data
              ? `${sorted.length} assets · ${nRisk} at risk · ${nOpen} open contract${nOpen === 1 ? "" : "s"} · ${nWork} work order${nWork === 1 ? "" : "s"} · sorted by failure risk`
              : "Loading fleet…"}
          </div>
        </div>
        <div className="row" style={{ gap: 12 }}>
          <StyleToggle value={style} onChange={changeStyle} />
          <span className="small muted mono">poll 5s{fleet.lastUpdated ? ` · ${fmtRel(new Date(fleet.lastUpdated).toISOString())}` : ""}</span>
        </div>
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
                  <AssetCard key={item.asset.asset_id} item={item} style={style} />
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
