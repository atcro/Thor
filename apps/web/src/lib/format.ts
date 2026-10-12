import type { MaintenanceOption, PipelineStage } from "../api/types";

export type Tone = "good" | "warn" | "bad" | "info" | "neutral" | "danger";

/** Health pill tone: >=85 green, 60-85 amber, <60 ember. */
export function healthTone(score: number | null | undefined): Tone {
  if (score === null || score === undefined || Number.isNaN(score)) return "neutral";
  if (score >= 85) return "good";
  if (score >= 60) return "warn";
  return "bad";
}

/**
 * Pill tone for a failure probability, at the API's own thresholds (Thresholds: early warning
 * 0.3, classification 0.5). Display only: the status word itself always comes from the API.
 */
export function probTone(p: number | null | undefined): Tone {
  if (p === null || p === undefined) return "neutral";
  if (p >= 0.5) return "bad";
  if (p >= 0.3) return "warn";
  return "good";
}

/**
 * Plain name of an operating state (called an "operating regime" in the literature; the API
 * field stays `regime`). The ONE place these names live: R1 startup / idle, R2 running,
 * R3 heavy load; any other id reads "state <id>".
 */
export function stateName(regimeId: string): string {
  switch (regimeId) {
    case "R1":
      return "startup / idle";
    case "R2":
      return "running";
    case "R3":
      return "heavy load";
    default:
      return `state ${regimeId}`;
  }
}

export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

export function fmtPct(p: number | null | undefined, digits = 1): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function fmtMoney(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

export function fmtHours(h: number | null | undefined): string {
  if (h === null || h === undefined || Number.isNaN(h)) return "—";
  if (h >= 48) return `${(h / 24).toFixed(1)} d`;
  return `${h.toFixed(1)} h`;
}

function parse(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

const pad = (n: number) => String(n).padStart(2, "0");

/** HH:MM (UTC) */
export function fmtTime(iso: string | null | undefined): string {
  const d = parse(iso);
  if (!d) return "—";
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

/** YYYY-MM-DD HH:MM UTC */
export function fmtDateTime(iso: string | null | undefined): string {
  const d = parse(iso);
  if (!d) return "—";
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(
    d.getUTCHours(),
  )}:${pad(d.getUTCMinutes())}Z`;
}

/** MM-DD HH:MM */
export function fmtShort(iso: string | null | undefined): string {
  const d = parse(iso);
  if (!d) return "—";
  return `${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(
    d.getUTCMinutes(),
  )}`;
}

/** Relative time, e.g. "3m ago". */
export function fmtRel(iso: string | null | undefined, now: number = Date.now()): string {
  const d = parse(iso);
  if (!d) return "never";
  const s = Math.max(0, Math.round((now - d.getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

export function shortHash(h: string | null | undefined, n = 12): string {
  if (!h) return "—";
  return h.length > n ? `${h.slice(0, n)}…` : h;
}

export const STAGE_LABEL: Record<PipelineStage, string> = {
  queued: "Queued",
  profiling: "Profiling",
  training: "Training",
  validating: "Validating",
  trained: "Model registered",
  explaining: "Explaining",
  awaiting_approval: "Awaiting approval",
  approved: "Approved",
  rejected: "Rejected",
  failed: "Failed",
};

export const OPTION_LABEL: Record<MaintenanceOption, string> = {
  maintain_now: "Maintain now",
  maintain_later: "Maintain at next window",
  run_to_failure: "Run to failure",
};

export function stageTone(stage: PipelineStage | null | undefined): Tone {
  switch (stage) {
    case "approved":
    case "trained":
      return "good";
    case "awaiting_approval":
      return "bad";
    case "rejected":
    case "failed":
      return "danger";
    case "queued":
      return "neutral";
    default:
      return "info";
  }
}

/** Humanise snake_case feature names for labels. */
export function humanise(s: string): string {
  return s.replace(/_/g, " ");
}
