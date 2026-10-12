/**
 * X-Ray HUD: the readout for one asset, from GET /twin/hud/:asset_id (polled every 2 s).
 *
 * Every number shown is a field of the HUD JSON that deterministic Python produced; the only
 * arithmetic here is layout (sparkline scaling, bar widths). The LLM has no part in it.
 * Kept deliberately short: probability, sensors, drivers, decision. Plant-integration context
 * (PLC tags, MES) sits in one collapsed section; model provenance is on the stage badge.
 */
import { useEffect, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { getTwinHud } from "../api/client";
import type { DataState, PlcTag, PredictionPoint, ShapFeature, Thresholds, TwinHud } from "../api/types";
import { Loaded, WORK_ORDER_WORD } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, fmtShort, humanise, OPTION_LABEL } from "../lib/format";
import { DECISION_WORD, HEALTHY_MEANING, STATUS_EXPLAIN, STATUS_GLYPH, stateWord } from "./status";
import { StatusGlyph } from "./XRayTooltip";

const DATA_STATE_WORD: Record<DataState, string> = { live: "live", stale: "stale", none: "no data" };

const SENSORS: [keyof NonNullable<TwinHud["asset"]["latest"]>, string, string, number][] = [
  ["vibration_rms", "vibration rms", "mm/s", 2],
  ["vibration_kurtosis", "kurtosis", "", 2],
  ["vibration_crest", "crest factor", "", 2],
  ["bearing_temp_c", "bearing temp", "°C", 1],
  ["motor_temp_c", "motor temp", "°C", 1],
  ["current_a", "current", "A", 1],
  ["rpm", "speed", "rpm", 0],
  ["load_pct", "load", "%", 0],
];

/** Drivers shown in the HUD; the full SHAP list is in Asset 360. */
const MAX_DRIVERS = 3;

function Module({ title, children, testId }: { title: string; children: ReactNode; testId?: string }) {
  return (
    <section className="xray-frame xray-mod" data-testid={testId}>
      <h4 className="xray-mod-t" style={{ margin: 0 }}>
        {title}
      </h4>
      {children}
    </section>
  );
}

/**
 * Tiny stored-prediction sparkline (raw per-sample rows, nothing hidden) with the two threshold
 * lines. Only scaling happens here.
 */
function Sparkline({ points, thresholds }: { points: PredictionPoint[]; thresholds: Thresholds }) {
  const W = 200;
  const H = 56;
  const pad = 3;
  const y = (p: number) => pad + (1 - Math.max(0, Math.min(1, p))) * (H - 2 * pad);
  const n = points.length;
  const x = (i: number) => (n <= 1 ? W / 2 : pad + (i / (n - 1)) * (W - 2 * pad));
  const path = points.map((pt, i) => `${x(i).toFixed(1)},${y(pt.failure_probability).toFixed(1)}`).join(" ");
  const last = n > 0 ? points[n - 1] : null;
  return (
    <svg className="xray-spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <line x1={0} x2={W} y1={y(thresholds.classification)} y2={y(thresholds.classification)} stroke="var(--tw-crit)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
      <line x1={0} x2={W} y1={y(thresholds.early_warning)} y2={y(thresholds.early_warning)} stroke="var(--tw-warn)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
      {n > 1 && <polyline points={path} fill="none" stroke="var(--tw-accent)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
      {last && <circle cx={x(n - 1)} cy={y(last.failure_probability)} r={2.5} fill="var(--tw-primary)" />}
    </svg>
  );
}

/** "Status: critical · latched since <ts> · <reason>" from the API's latched alarm fields. */
function StatusLine({ hud }: { hud: TwinHud }) {
  const a = hud.asset;
  if (a.alarm === "critical") {
    return (
      <>
        Status: <span className="xray-bad">critical</span> · latched since{" "}
        <span className="xray-num" title={fmtDateTime(a.alarm_since)}>{fmtShort(a.alarm_since)}</span>
        {a.alarm_reason && <> · {a.alarm_reason}</>}
      </>
    );
  }
  if (a.alarm === "warning") {
    return (
      <>
        Status: <span className="xray-warn">warning</span> · since{" "}
        <span className="xray-num" title={fmtDateTime(a.alarm_since)}>{fmtShort(a.alarm_since)}</span>
        {a.alarm_reason && <> · {a.alarm_reason}</>}
      </>
    );
  }
  if (a.status === "healthy") {
    return (
      <>
        Status: <span className="xray-ok">{HEALTHY_MEANING}</span>
      </>
    );
  }
  return <>Status: {STATUS_EXPLAIN[a.status]}</>;
}

function tagValue(t: PlcTag): string {
  if (t.value === null || t.value === undefined) return "—";
  if (typeof t.value === "boolean") return t.value ? "1 (on)" : "0 (off)";
  if (typeof t.value === "number") return Number.isInteger(t.value) ? fmtInt(t.value) : fmtNum(t.value, 2);
  return String(t.value);
}

function Drivers({ drivers }: { drivers: ShapFeature[] }) {
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap_value)), 1e-9);
  return (
    <div className="xray-bars">
      {drivers.map((d) => {
        const raises = d.direction === "raises_risk";
        return (
          <div className="xray-bar" key={d.feature} style={{ color: raises ? "var(--tw-crit)" : "var(--tw-accent)" }}>
            <div className="lbl">
              <span>{humanise(d.feature)}</span>
              <span className="dir">{Math.abs(d.shap_value) < 0.01 ? "little effect" : raises ? "raises risk" : "lowers risk"}</span>
            </div>
            <div className="val">{(d.shap_value >= 0 ? "+" : "") + fmtNum(d.shap_value, 3)}</div>
            <div className="track">
              <i style={{ width: `${(Math.abs(d.shap_value) / maxAbs) * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HudBody({ hud }: { hud: TwinHud }) {
  const a = hud.asset;
  const p = a.failure_probability;
  const wo = a.work_order;
  const c = hud.contract;

  return (
    <>
      <Module title="Failure probability" testId="hud-probability">
        <div className="xray-prob-row">
          {p === null || p === undefined ? (
            <div className="xray-big dim">no prediction yet</div>
          ) : (
            <div className="xray-big" data-testid="hud-pfail">
              {fmtPct(p)}
            </div>
          )}
          <div className="side">
            <span className="k">trend</span>
            <span className="xray-num">{hud.trend ?? "—"}</span>
            <span className="k">scored at</span>
            <span className="xray-num" title={fmtDateTime(a.prediction_ts)}>
              {fmtShort(a.prediction_ts)}
            </span>
          </div>
        </div>
        <div className="xray-note" data-testid="hud-status-line">
          <StatusLine hud={hud} />
        </div>
        <Sparkline points={hud.prediction_trend} thresholds={hud.thresholds} />
        <div className="xray-note">
          warning <span className="xray-warn xray-num">{fmtPct(hud.thresholds.early_warning, 0)}</span> · critical{" "}
          <span className="xray-bad xray-num">{fmtPct(hud.thresholds.classification, 0)}</span> · median of the last 3 samples
        </div>
      </Module>

      <Module title="Live sensors" testId="hud-sensors">
        <div className="xray-state" data-testid="hud-regime">
          State: <b>{stateWord(a.regime)}</b>
        </div>
        {a.latest ? (
          <div className="xray-sensors">
            {SENSORS.map(([key, label, unit, digits]) => (
              <div className="xray-sensor" key={key}>
                <span className="k">{label}</span>
                <span className="v">
                  {fmtNum(a.latest![key] as number, digits)}
                  {unit && <span className="u">{unit}</span>}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="xray-dim">no telemetry received yet</div>
        )}
        <div className="xray-note">
          last sample <span className="xray-num" title={fmtDateTime(a.last_ts)}>{fmtShort(a.last_ts)}</span> · Data:{" "}
          <span className={`xray-num ${a.data_state === "live" ? "xray-ok" : a.data_state === "stale" ? "xray-warn" : "xray-dim"}`}>
            {DATA_STATE_WORD[a.data_state]}
          </span>
        </div>
      </Module>

      <Module title="Top drivers" testId="hud-drivers">
        {hud.drivers.length > 0 ? (
          <>
            <Drivers drivers={hud.drivers.slice(0, MAX_DRIVERS)} />
            {hud.drivers_source && (
              <div className="xray-note">
                SHAP · contract <span className="xray-num">{hud.drivers_source.contract_id}</span>
              </div>
            )}
          </>
        ) : (
          <div className="xray-dim">no analysis yet — Run analysis in Asset 360</div>
        )}
      </Module>

      <Module title="Decision" testId="hud-decision">
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: c || wo ? 6 : 0 }}>Decision: {DECISION_WORD[a.decision_state]}</div>
        {(c || wo) && (
          <dl className="xray-kv">
            {c && (
              <>
                <dt>recommendation</dt>
                <dd>{OPTION_LABEL[c.recommendation]}</dd>
                <dt>planned window</dt>
                <dd>{c.window_start ? `${fmtShort(c.window_start)} → ${fmtShort(c.window_end)}` : "—"}</dd>
              </>
            )}
            {wo && (
              <>
                <dt>work order</dt>
                <dd>{WORK_ORDER_WORD[wo.status]}</dd>
                <dt>approver</dt>
                <dd>{wo.approver}</dd>
              </>
            )}
          </dl>
        )}
      </Module>

      <details className="xray-frame xray-mod xray-more" data-testid="hud-plant">
        <summary className="xray-mod-t">Plant integration · PLC tags{hud.mes ? " & MES" : ""}</summary>
        {hud.plc_tags.length > 0 ? (
          <table className="xray-table" data-testid="hud-plc">
            <tbody>
              {hud.plc_tags.map((t) => (
                <tr
                  key={t.tag}
                  className={t.simulated ? "simulated" : "mapped"}
                  title={t.simulated ? undefined : `${t.tag} ← ${t.source ?? "?"}`}
                  data-testid={`plc-${t.tag}`}
                >
                  <td className="tag">
                    {t.tag}
                    {t.simulated && (
                      <>
                        {" "}
                        <span className="xray-chip sim" title={t.basis}>
                          simulated
                        </span>
                      </>
                    )}
                  </td>
                  <td className="v">{tagValue(t)}</td>
                  <td className="u">{t.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="xray-dim">no tags (no telemetry yet)</div>
        )}
        {hud.mes && (
          <div data-testid="hud-mes">
            <dl className="xray-kv" style={{ marginTop: 8 }}>
              <dt>SKU</dt>
              <dd>{hud.mes.sku}</dd>
              <dt>container</dt>
              <dd>{hud.mes.container}</dd>
              <dt>quantity</dt>
              <dd>
                {fmtInt(hud.mes.quantity_produced)} {hud.mes.uom}
              </dd>
            </dl>
            <div className="xray-label-sim" title={hud.mes.basis}>
              <span className="xray-chip sim" aria-hidden="true">
                sim
              </span>
              simulated MES feed · context only, feeds no model or decision
            </div>
          </div>
        )}
      </details>
    </>
  );
}

export function XRayHud({ assetId, onClose }: { assetId: string; onClose: () => void }) {
  const hud = usePolling(() => getTwinHud(assetId), 2000, assetId);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const a = hud.data?.asset ?? null;
  const g = a ? STATUS_GLYPH[a.status] : null;

  return (
    <aside className="xray-hud" role="dialog" aria-label={`X-Ray HUD ${assetId}`} data-testid="xray-hud">
      <header className="xray-frame xray-hud-head">
        {a && g && <StatusGlyph status={a.status} size={22} lit />}
        <div className="grow">
          <div className="xray-id">{assetId}</div>
          {a && (
            <>
              <div className="name">{a.asset.name}</div>
              <div className="where">
                {a.asset.line}
                {a.asset.role ? ` · ${a.asset.role}` : ""}
              </div>
              {g && (
                <div className="status" style={{ color: g.color }} data-testid="hud-status" title={STATUS_EXPLAIN[a.status]}>
                  {g.word}
                  {a.alarm === "critical" && <span className="xray-dim" style={{ textTransform: "none", letterSpacing: 0 }}>· latched since {fmtShort(a.alarm_since)}</span>}
                </div>
              )}
            </>
          )}
        </div>
        <div className="xray-hud-actions">
          <button type="button" className="xray-hud-close" onClick={onClose} aria-label="Close HUD" title="Close (Esc)">
            ×
          </button>
          <Link to={`/assets/${encodeURIComponent(assetId)}`} className="xray-btn primary">
            Open in Asset 360
          </Link>
        </div>
      </header>
      <Loaded data={hud.data} error={hud.error} loading={hud.loading} onRetry={hud.refresh}>
        {(d) => <HudBody hud={d} />}
      </Loaded>
    </aside>
  );
}
