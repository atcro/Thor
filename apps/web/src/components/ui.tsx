import type { ReactNode } from "react";
import type { AlarmLevel, ExplanationSource, ModelStage, PipelineStage, TwinStatus, WorkOrder, WorkOrderStatus } from "../api/types";
import { fmtDateTime, fmtNum, fmtShort, healthTone, STAGE_LABEL, stageTone, stateName, type Tone } from "../lib/format";

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

export function Panel({
  title,
  actions,
  children,
  className = "",
  tight = false,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  tight?: boolean;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title !== undefined || actions !== undefined) && (
        <header className="panel-head">
          {typeof title === "string" ? <h3>{title}</h3> : <div>{title}</div>}
          {actions && <div className="row">{actions}</div>}
        </header>
      )}
      <div className={`panel-body${tight ? " tight" : ""}`}>{children}</div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Pills / chips / dots
// ---------------------------------------------------------------------------

export function Pill({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`pill ${tone}`} title={title}>
      {children}
    </span>
  );
}

export function HealthPill({ score }: { score: number | null | undefined }) {
  const tone = healthTone(score);
  if (score === null || score === undefined) {
    return (
      <Pill tone="neutral" title="No approved-model prediction for this motor yet">
        no prediction yet
      </Pill>
    );
  }
  return (
    <Pill
      tone={tone}
      title="Health = 100 × (1 − calibrated failure probability); green = within healthy baseline (below the early-warning threshold)"
    >
      <span className="dot" style={{ background: "currentColor", width: 6, height: 6 }} />
      {fmtNum(score, 0)}
    </Pill>
  );
}

/** Operating state the API inferred from rpm / load (API field: `regime`), as a plain name. */
export function StateBadge({ regime }: { regime: string | null | undefined }) {
  const title = "Operating state inferred from rpm and load (called an operating regime in the literature)";
  if (!regime) {
    return (
      <Pill tone="neutral" title={title}>
        State: unknown
      </Pill>
    );
  }
  return (
    <Pill tone="info" title={title}>
      State: {stateName(regime)}
    </Pill>
  );
}

/** The API's status word for the Fleet card and Asset 360 (green = within healthy baseline). */
export const STATUS_WORD: Record<TwinStatus, string> = {
  healthy: "within healthy baseline",
  warning: "warning",
  critical: "critical",
  no_prediction: "no prediction",
  offline: "offline · work order",
  no_data: "no data",
};

/**
 * Alarm facts next to a HealthPill: "latched since <plant time>" while the alarm is critical
 * (critical stays until a completed work order clears it) and "sandbox scenario" when the
 * displayed prediction came from sandbox readings. Both are API fields; nothing is derived.
 */
export function AlarmPills({
  alarm,
  alarmSince,
  alarmReason,
  sandbox,
}: {
  alarm: AlarmLevel | null | undefined;
  alarmSince: string | null | undefined;
  alarmReason: string | null | undefined;
  sandbox: boolean | undefined;
}) {
  return (
    <>
      {alarm === "critical" && (
        <Pill tone="bad" title={alarmReason ?? "critical alarm latched until a completed work order clears it"}>
          latched since {fmtShort(alarmSince)}
        </Pill>
      )}
      {sandbox && <SandboxPill />}
    </>
  );
}

/** Marks a value that came from sandbox (simulated) readings. */
export function SandboxPill({ title = "Sandbox scenario: produced from simulated sandbox readings, not the recorded plant history" }: { title?: string }) {
  return (
    <Pill tone="warn" title={title}>
      sandbox scenario
    </Pill>
  );
}

export function StageChip({ stage }: { stage: PipelineStage | null | undefined }) {
  if (!stage) return <Pill tone="neutral">no run</Pill>;
  return <Pill tone={stageTone(stage)}>{STAGE_LABEL[stage] ?? stage}</Pill>;
}

const MODEL_STAGE_TONE: Record<ModelStage, Tone> = {
  candidate: "neutral",
  validated: "info",
  shadow: "warn",
  production: "good",
  archived: "neutral",
};

export function ModelStageChip({ stage }: { stage: ModelStage | null | undefined }) {
  if (!stage) return <Pill tone="neutral">—</Pill>;
  return <Pill tone={MODEL_STAGE_TONE[stage] ?? "neutral"}>{stage}</Pill>;
}

export function SourceChip({ source }: { source: ExplanationSource | null | undefined }) {
  if (!source) return null;
  return (
    <span className={`chip ${source === "llm" ? "steel" : ""}`} title="Who drafted this text">
      {source === "llm" ? "drafted by LLM" : "template"}
    </span>
  );
}

export function Dot({ on, pulse = false, warn = false }: { on: boolean | null | undefined; pulse?: boolean; warn?: boolean }) {
  const cls = on === null || on === undefined ? "" : warn ? "warn" : on ? "on" : "off";
  return <span className={`dot ${cls}${pulse ? " pulse" : ""}`} />;
}

export function Mono({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span className="mono" title={title}>
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Stat tile
// ---------------------------------------------------------------------------

export function Stat({
  label,
  value,
  unit,
  tone,
  big = false,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: "ember" | "steel" | "green";
  big?: boolean;
}) {
  return (
    <div className="stat">
      <span className="lbl">{label}</span>
      <span className={`val${big ? " big" : ""}${tone ? ` ${tone}` : ""}`}>
        {value}
        {unit && <span className="unit"> {unit}</span>}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty / loading / error states
// ---------------------------------------------------------------------------

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state" role="status">
      <span className="dot pulse" style={{ background: "var(--steel)", marginRight: 8 }} />
      {label}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="state">{children}</div>;
}

export function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="state error" role="alert">
      <div>Request failed</div>
      <code className="small">{error}</code>
      {onRetry && (
        <div style={{ marginTop: 8 }}>
          <button className="btn sm" onClick={onRetry}>
            Retry
          </button>
        </div>
      )}
    </div>
  );
}

/** Wraps the three common states around loaded data. */
export function Loaded<T>({
  data,
  error,
  loading,
  onRetry,
  empty,
  isEmpty,
  children,
}: {
  data: T | null;
  error: string | null;
  loading: boolean;
  onRetry?: () => void;
  empty?: ReactNode;
  isEmpty?: (d: T) => boolean;
  children: (d: T) => ReactNode;
}) {
  if (data === null) {
    if (error) return <ErrorState error={error} onRetry={onRetry} />;
    if (loading) return <LoadingState />;
    return <EmptyState>{empty ?? "Nothing here yet."}</EmptyState>;
  }
  if (isEmpty ? isEmpty(data) : Array.isArray(data) && data.length === 0) {
    return <EmptyState>{empty ?? "Nothing here yet."}</EmptyState>;
  }
  return <>{children(data)}</>;
}

// ---------------------------------------------------------------------------
// Key/value list
// ---------------------------------------------------------------------------

export function Kv({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v ?? "—"}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Small inline horizontal progress bar (0-1). */
export function BarH({ value, color }: { value: number; color?: string }) {
  const w = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0)) * 100;
  return (
    <div className="bar-h">
      <i style={{ width: `${w}%`, background: color }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Work order (plant-side execution of an approved Decision Contract)
// ---------------------------------------------------------------------------

export const WORK_ORDER_WORD: Record<WorkOrderStatus, string> = {
  scheduled: "work order scheduled",
  in_progress: "offline · in maintenance",
  completed: "back in service",
};

/** Plant-reported status of the work order behind an approved contract. Thor only displays it. */
export function WorkOrderPill({ wo }: { wo: WorkOrder }) {
  const tone: Tone = wo.status === "completed" ? "good" : wo.status === "in_progress" ? "warn" : "info";
  const base =
    wo.status === "scheduled"
      ? `Approved by ${wo.approver}. Plant window ${fmtDateTime(wo.window_start)} → ${fmtDateTime(wo.window_end)} (plant time). Executed by the plant CMMS, not by Thor.`
      : wo.status === "in_progress"
        ? `Taken offline by plant maintenance at ${fmtDateTime(wo.started_ts)} (plant time); ${wo.planned_downtime_h} h planned downtime.`
        : `Returned to service by plant maintenance at ${fmtDateTime(wo.completed_ts)} (plant time).`;
  const title = wo.sandbox_scenario ? `${base} Sandbox scenario: the contract was raised on simulated sandbox readings.` : base;
  return (
    <Pill tone={tone} title={title}>
      {WORK_ORDER_WORD[wo.status]}
    </Pill>
  );
}
