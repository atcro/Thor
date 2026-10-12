import { useMemo, useState, type CSSProperties, type KeyboardEvent } from "react";
import { Link } from "react-router-dom";
import { getFleet, getSystemHealth, getTwin } from "../api/client";
import type { FleetAsset, TwinAsset } from "../api/types";
import { FINDING_LABEL } from "../api/types";
import { AlarmTrustLine } from "../components/AlarmTrustLine";
import { MotorSprite, spriteParams, TIER_WORD, type SpriteStyle } from "../components/MotorSprite";
import { AlarmPills, Dot, HealthPill, Loaded, ModelStageChip, Panel, Pill, StateBadge, WorkOrderPill } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtInt, fmtPct, fmtRel, fmtShort, probTone } from "../lib/format";
import { TwinThemeRoot } from "../twin/TwinThemeRoot";
import { useXRay } from "../twin/useXRay";
import { XRayHud } from "../twin/XRayHud";
import { XRayToggle } from "../twin/XRayToggle";
import { XRayTooltip } from "../twin/XRayTooltip";

/** Sort by failure risk desc; assets without a prediction ("no prediction yet") come last. */
function sortByRisk(list: FleetAsset[]): FleetAsset[] {
  return [...list].sort((a, b) => {
    const pa = a.failure_probability ?? -1;
    const pb = b.failure_probability ?? -1;
    if (pb !== pa) return pb - pa;
    return a.asset.asset_id.localeCompare(b.asset.asset_id);
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

/**
 * X-Ray mode on the Fleet: the card becomes a hover host (tooltip from GET /twin above it,
 * keyboard-reachable via focus) and a click opens the HUD drawer instead of navigating.
 */
interface XRayCardProps {
  /** The asset's /twin entry (null while the first /twin poll is in flight). */
  twin: TwinAsset | null;
  onOpen: (assetId: string) => void;
}

function AssetCard({ item, style, xray }: { item: FleetAsset; style: SpriteStyle; xray?: XRayCardProps }) {
  const { asset } = item;
  const p = item.failure_probability;
  const { atRisk, heat, tier } = spriteParams(item);
  const [hot, setHot] = useState(false);
  const className = `asset-card tier-${tier}${atRisk ? " risk" : ""}`;
  const cardStyle = { "--heat": heat } as CSSProperties;
  const body = (
    <>
      <div className="sprite">
        <MotorSprite item={item} style={style} />
        <span className={`tier-tag ${tier}`}>{TIER_WORD[tier]}</span>
      </div>
      {item.open_contract_id && <span className="flag" title={`Open decision contract ${item.open_contract_id}`} />}
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="id">{asset.asset_id}</span>
        <span className="row wrap" style={{ gap: 4, justifyContent: "flex-end" }}>
          <HealthPill score={item.health_score} />
          <AlarmPills alarm={item.alarm} alarmSince={item.alarm_since} alarmReason={item.alarm_reason} />
        </span>
      </div>
      <div className="name">{asset.name}</div>
      <div className="loc">
        {asset.site} · {asset.line}
        {asset.criticality === "high" && <span className="dim" title="Asset criticality (plant metadata), not its status"> · high criticality</span>}
      </div>
      <div className="row wrap" style={{ gap: 6 }}>
        <StateBadge regime={item.regime} />
        {p !== null && p !== undefined && (
          <Pill tone={probTone(p)} title="Current failure probability">
            p(fail) {fmtPct(p)}
          </Pill>
        )}
        {item.stage && <ModelStageChip stage={item.stage} />}
        {item.work_order && <WorkOrderPill wo={item.work_order} />}
      </div>
      {item.work_order?.status === "completed" && (
        <div className="small" data-testid={`finding-line-${asset.asset_id}`}>
          {item.work_order.finding ? (
            <span title={`Recorded by ${item.work_order.finding.recorded_by}${item.work_order.finding.note ? `: ${item.work_order.finding.note}` : ""}`}>
              <span className="muted">technician: </span>
              {FINDING_LABEL[item.work_order.finding.finding]} <span className="dim mono">· {item.work_order.finding.recorded_by}</span>
            </span>
          ) : (
            <span style={{ color: "var(--steel)" }} title="The repair is complete and no finding is recorded yet. Open Asset 360 to record what the technician found.">
              What did the technician find? →
            </span>
          )}
        </div>
      )}
      <div className="foot">
        <span title={fmtDateTime(item.last_ts)}>last sample {fmtShort(item.last_ts)}</span>
        {item.open_contract_id && <span style={{ color: "var(--ember)" }}>contract open</span>}
      </div>
    </>
  );
  if (!xray) {
    return (
      <Link to={`/assets/${encodeURIComponent(asset.asset_id)}`} className={className} style={cardStyle}>
        {body}
      </Link>
    );
  }
  const open = () => xray.onOpen(asset.asset_id);
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      open();
    }
  };
  return (
    <div
      className="xray-host"
      onMouseEnter={() => setHot(true)}
      onMouseLeave={() => setHot(false)}
      onFocus={() => setHot(true)}
      onBlur={() => setHot(false)}
      data-testid={`xray-host-${asset.asset_id}`}
    >
      <div
        role="button"
        tabIndex={0}
        className={className}
        style={cardStyle}
        onClick={open}
        onKeyDown={onKey}
        aria-label={`${asset.asset_id}: open X-Ray HUD`}
        aria-haspopup="dialog"
      >
        {body}
      </div>
      {hot && (
        <TwinThemeRoot className="xray-tip-anchor" role="tooltip">
          {xray.twin ? <XRayTooltip asset={xray.twin} /> : <div className="xray-tip xray-dim">loading X-Ray…</div>}
        </TwinThemeRoot>
      )}
    </div>
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

/** The Fleet: the isometric/pixel fleet view (the landing view at `/`). */
export function Fleet() {
  // 2 s: the amber phase between the warning and the latched critical lasts a few seconds.
  const fleet = usePolling(getFleet, 2000, "fleet");
  const xray = useXRay();
  // X-Ray facts come from GET /twin (status, operating state, decision); polled only while X-Ray is on.
  const twin = usePolling(getTwin, 2000, "twin", xray.enabled);
  const twinById = useMemo(() => {
    const m = new Map<string, TwinAsset>();
    for (const a of twin.data?.assets ?? []) m.set(a.asset.asset_id, a);
    return m;
  }, [twin.data]);
  const [style, setStyle] = useState<SpriteStyle>(readStyle);
  const changeStyle = (s: SpriteStyle) => {
    setStyle(s);
    saveStyle(s);
  };
  const sorted = useMemo(() => (fleet.data ? sortByRisk(fleet.data) : []), [fleet.data]);
  // Counts come from the API's status (the latched alarm rule), never from a threshold here.
  const nCritical = sorted.filter((a) => a.status === "critical").length;
  const nWarning = sorted.filter((a) => a.status === "warning").length;
  const nUnscored = sorted.filter((a) => a.health_score === null || a.health_score === undefined).length;
  const nOpen = sorted.filter((a) => a.open_contract_id).length;
  const nWork = sorted.filter((a) => a.work_order && a.work_order.status !== "completed").length;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Fleet</h1>
          <div className="sub">
            {fleet.data
              ? `${sorted.length} assets · ${nCritical} critical · ${nWarning} warning · ${nUnscored} without a prediction yet · ${nOpen} open contract${nOpen === 1 ? "" : "s"} · ${nWork} work order${nWork === 1 ? "" : "s"} · sorted by failure risk`
              : "Loading fleet…"}
          </div>
          <AlarmTrustLine compact />
        </div>
        <div className="row" style={{ gap: 12 }}>
          <XRayToggle />
          <StyleToggle value={style} onChange={changeStyle} />
          <span className="small muted mono">poll 2s{fleet.lastUpdated ? ` · ${fmtRel(new Date(fleet.lastUpdated).toISOString())}` : ""}</span>
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
                  <AssetCard
                    key={item.asset.asset_id}
                    item={item}
                    style={style}
                    xray={xray.enabled ? { twin: twinById.get(item.asset.asset_id) ?? null, onOpen: xray.openAsset } : undefined}
                  />
                ))}
              </div>
            )}
          </Loaded>
        </div>
        <SystemPanel />
      </div>
      {xray.enabled && xray.openedAsset && (
        <TwinThemeRoot className="xray-drawer" data-testid="xray-drawer">
          <XRayHud assetId={xray.openedAsset} onClose={xray.close} />
        </TwinThemeRoot>
      )}
    </div>
  );
}
