/**
 * X-Ray HUD: the sci-fi readout for one asset, from GET /twin/hud/:asset_id (polled every 2 s).
 *
 * Every number shown is a field of the HUD JSON that deterministic Python produced; the only
 * arithmetic here is layout (sparkline scaling, bar widths). The LLM has no part in it.
 */
import { useEffect, type CSSProperties, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { getTwinHud } from "../api/client";
import type { DataState, PlcTag, PredictionPoint, ShapFeature, Thresholds, TwinHud } from "../api/types";
import { Loaded, WORK_ORDER_WORD } from "../components/ui";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtInt, fmtNum, fmtPct, fmtShort, humanise, OPTION_LABEL } from "../lib/format";
import { DECISION_WORD, regimeWord, STATUS_GLYPH } from "./status";
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

/** Tiny stored-prediction sparkline with the two threshold lines. Only scaling happens here. */
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
              <span className="dir">
                {raises ? "raises risk" : "lowers risk"} · value {fmtNum(d.feature_value, 2)}
              </span>
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

function HudBody({ hud, assetId }: { hud: TwinHud; assetId: string }) {
  const a = hud.asset;
  const p = a.failure_probability;
  const wo = a.work_order;
  const c = hud.contract;
  const m = hud.model;
  const regimes = ["R1", "R2", "R3"] as const;
  const regimeVar: Record<(typeof regimes)[number], string> = { R1: "var(--tw-r1)", R2: "var(--tw-r2)", R3: "var(--tw-r3)" };

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
            <span className="k">health score</span>
            <span className="xray-num">{a.health_score === null ? "—" : fmtNum(a.health_score, 0)}</span>
            <span className="k">trend</span>
            <span className="xray-num">{hud.trend ?? "—"}</span>
            <span className="k">model</span>
            <span className="xray-num">{a.model_version ?? "—"}</span>
            <span className="k">scored at</span>
            <span className="xray-num" title={fmtDateTime(a.prediction_ts)}>
              {fmtShort(a.prediction_ts)}
            </span>
          </div>
        </div>
        <Sparkline points={hud.prediction_trend} thresholds={hud.thresholds} />
        <div className="xray-note">
          {hud.prediction_trend.length} stored sample{hud.prediction_trend.length === 1 ? "" : "s"} · lines: early warning{" "}
          <span className="xray-warn xray-num">{fmtPct(hud.thresholds.early_warning, 0)}</span> · classification{" "}
          <span className="xray-bad xray-num">{fmtPct(hud.thresholds.classification, 0)}</span> · shown p(fail) is the median of last 3 samples
        </div>
      </Module>

      <Module title="Regime" testId="hud-regime">
        <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
          {regimeWord(a.regime)}
          {a.regime && <span className="xray-dim xray-num" style={{ fontWeight: 400, marginLeft: 8 }}>{a.regime}</span>}
        </div>
        <div className="xray-chips">
          {regimes.map((r) => (
            <span
              key={r}
              className={`xray-chip${a.regime === r ? " lit" : ""}`}
              style={{ "--chip-color": regimeVar[r] } as CSSProperties}
              aria-current={a.regime === r ? "true" : undefined}
            >
              {r} · {regimeWord(r)}
            </span>
          ))}
        </div>
        <div className="xray-note">inferred from rpm / load against the production model's baseline</div>
      </Module>

      <Module title="Live sensors" testId="hud-sensors">
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
          last sample <span className="xray-num" title={fmtDateTime(a.last_ts)}>{fmtShort(a.last_ts)}</span> · data{" "}
          <span className={`xray-num ${a.data_state === "live" ? "xray-ok" : a.data_state === "stale" ? "xray-warn" : "xray-dim"}`}>{DATA_STATE_WORD[a.data_state]}</span> ·{" "}
          <span className="xray-num">{a.msgs_last_60s}</span> msgs / 60 s
        </div>
      </Module>

      <Module title="Top drivers" testId="hud-drivers">
        {hud.drivers.length > 0 ? (
          <>
            <Drivers drivers={hud.drivers} />
            {hud.drivers_source && (
              <div className="xray-note">
                SHAP from contract <span className="xray-num">{hud.drivers_source.contract_id}</span> · model{" "}
                <span className="xray-num">{hud.drivers_source.model_version}</span> ·{" "}
                <span className="xray-num" title={fmtDateTime(hud.drivers_source.created_at)}>
                  {fmtShort(hud.drivers_source.created_at)}
                </span>
              </div>
            )}
          </>
        ) : (
          <div className="xray-dim">no analysis yet — Run analysis in Asset 360</div>
        )}
      </Module>

      <Module title="Decision & work order" testId="hud-decision">
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{DECISION_WORD[a.decision_state]}</div>
        {c && (
          <dl className="xray-kv">
            <dt>contract</dt>
            <dd>{c.contract_id}</dd>
            <dt>recommendation</dt>
            <dd>{OPTION_LABEL[c.recommendation]}</dd>
            <dt>window</dt>
            <dd>
              {c.window_start ? `${fmtShort(c.window_start)} → ${fmtShort(c.window_end)}` : "—"}
            </dd>
            <dt>contract status</dt>
            <dd>{hud.contract_status ?? "—"}</dd>
            <dt>p(fail) at creation</dt>
            <dd>{fmtPct(c.failure_probability)}</dd>
          </dl>
        )}
        {wo && (
          <dl className="xray-kv" style={{ marginTop: c ? 8 : 0 }}>
            <dt>work order</dt>
            <dd>{WORK_ORDER_WORD[wo.status]}</dd>
            <dt>approver</dt>
            <dd>{wo.approver}</dd>
            <dt>approved</dt>
            <dd title={fmtDateTime(wo.approved_at)}>{fmtShort(wo.approved_at)}</dd>
            <dt>started</dt>
            <dd title={fmtDateTime(wo.started_ts)}>{fmtShort(wo.started_ts)}</dd>
            <dt>completed</dt>
            <dd title={fmtDateTime(wo.completed_ts)}>{fmtShort(wo.completed_ts)}</dd>
            <dt>planned downtime</dt>
            <dd>{fmtNum(wo.planned_downtime_h, 1)} h</dd>
          </dl>
        )}
        {!c && !wo && <div className="xray-dim">no decision contract for this asset</div>}
        <div className="xray-note">approvals are recorded by an engineer; Thor never executes a work order</div>
      </Module>

      <Module title="Model provenance" testId="hud-provenance">
        {m ? (
          <dl className="xray-kv">
            <dt>model</dt>
            <dd>
              {m.name} {m.version}
            </dd>
            <dt>family · stage</dt>
            <dd>
              {m.family} · {m.stage}
            </dd>
            <dt>approved by</dt>
            <dd>{m.approver ?? "—"}</dd>
            <dt>approved at</dt>
            <dd title={fmtDateTime(m.approved_at)}>{fmtShort(m.approved_at)}</dd>
            <dt>edge parity</dt>
            <dd className={m.parity_passed ? "xray-ok" : m.parity_passed === false ? "xray-bad" : "xray-dim"}>
              {m.parity_passed === null
                ? "not recorded"
                : `${m.parity_passed ? "✓ passed" : "✗ failed"} · max |Δp| ${m.parity_max_abs_diff === null ? "—" : fmtNum(m.parity_max_abs_diff, 4)}`}
            </dd>
            <dt>validation</dt>
            <dd className={m.validation_passed ? "xray-ok" : m.validation_passed === false ? "xray-bad" : "xray-dim"}>
              {m.validation_passed === null ? "not recorded" : m.validation_passed ? "✓ passed" : "✗ failed"}
            </dd>
            <dt>IMS</dt>
            <dd>{fmtNum(m.ims_total, 3)}</dd>
            <dt>validated lead time</dt>
            <dd>{m.lead_time_median_h === null ? "—" : `${fmtNum(m.lead_time_median_h, 1)} h`}</dd>
            <dt>calibration</dt>
            <dd>{m.calibration_method ?? "—"}</dd>
          </dl>
        ) : (
          <div className="xray-dim">no production model — predictions are off until a promotion is approved in ModelOps</div>
        )}
      </Module>

      <Module title="PLC tags · plant integration" testId="hud-plc">
        {hud.plc_tags.length > 0 ? (
          <table className="xray-table">
            <thead>
              <tr>
                <th>tag</th>
                <th style={{ textAlign: "right" }}>value</th>
                <th>unit</th>
              </tr>
            </thead>
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
        <div className="xray-note">mapped tags read one-to-one from this motor's telemetry (hover a row for its source signal)</div>
      </Module>

      {hud.mes && (
        <Module title={`MES context · ${hud.mes.line}`} testId="hud-mes">
          <dl className="xray-kv">
            <dt>SKU</dt>
            <dd>{hud.mes.sku}</dd>
            <dt>container</dt>
            <dd>{hud.mes.container}</dd>
            <dt>quantity produced</dt>
            <dd>
              {fmtInt(hud.mes.quantity_produced)} {hud.mes.uom}
            </dd>
            <dt>shift</dt>
            <dd>{hud.mes.shift}</dd>
            <dt>plant time</dt>
            <dd title={fmtDateTime(hud.mes.plant_ts)}>{fmtShort(hud.mes.plant_ts)}</dd>
          </dl>
          <div className="xray-label-sim" title={hud.mes.basis}>
            <span className="xray-chip sim" aria-hidden="true">
              sim
            </span>
            simulated MES feed · context only, feeds no model or decision
          </div>
        </Module>
      )}

      <div className="xray-note" style={{ textAlign: "center" }}>
        {assetId} · every value above is a field of GET /twin/hud · polled every 2 s
      </div>
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
                <div className="status" style={{ color: g.color }} data-testid="hud-status">
                  {g.word}
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
        {(d) => <HudBody hud={d} assetId={assetId} />}
      </Loaded>
    </aside>
  );
}
