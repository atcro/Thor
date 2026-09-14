import type { ReactNode } from "react";
import type { ExplanationSource, ModelStage, PipelineStage } from "../api/types";
import { fmtNum, healthTone, STAGE_LABEL, stageTone, type Tone } from "../lib/format";

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
  return (
    <Pill tone={tone} title="Health score (0-100)">
      <span className="dot" style={{ background: "currentColor", width: 6, height: 6 }} />
      {score === null || score === undefined ? "—" : fmtNum(score, 0)}
    </Pill>
  );
}

const REGIME_LABEL: Record<string, string> = {
  R1: "R1 idle/startup",
  R2: "R2 nominal",
  R3: "R3 high-load",
};

export function RegimeBadge({ regime }: { regime: string | null | undefined }) {
  if (!regime) return <Pill tone="neutral">regime —</Pill>;
  return (
    <Pill tone="info" title="Operating regime (clustered on rpm/load)">
      {REGIME_LABEL[regime] ?? regime}
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
