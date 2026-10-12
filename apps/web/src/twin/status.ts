/**
 * Display rules shared by the 3-D scene, the X-Ray HUD and the Dashboard X-Ray.
 * Status itself comes from the API (`TwinAsset.status`); these helpers only map it to a
 * glyph shape, a colour token and a word, so colour is never the only cue.
 */
import type { DecisionState, Thresholds, TwinStatus } from "../api/types";
import { STATUS_COLOR } from "./palette";

export type GlyphShape = "circle" | "triangle" | "octagon" | "square" | "ring" | "dash";

export interface Glyph {
  shape: GlyphShape;
  color: string;
  word: string;
}

/** Healthy = green circle, warning = amber triangle, critical = red stop-octagon (§15). */
export const STATUS_GLYPH: Record<TwinStatus, Glyph> = {
  healthy: { shape: "circle", color: STATUS_COLOR.healthy, word: "healthy" },
  warning: { shape: "triangle", color: STATUS_COLOR.warning, word: "warning" },
  critical: { shape: "octagon", color: STATUS_COLOR.critical, word: "critical" },
  no_prediction: { shape: "ring", color: STATUS_COLOR.no_prediction, word: "no prediction yet" },
  offline: { shape: "square", color: STATUS_COLOR.offline, word: "offline · work order" },
  no_data: { shape: "dash", color: STATUS_COLOR.no_data, word: "no data" },
};

export const REGIME_WORD: Record<string, string> = {
  R1: "startup / idle",
  R2: "run",
  R3: "heavy load",
};

export function regimeWord(regime: string | null | undefined): string {
  if (!regime) return "regime unknown";
  return REGIME_WORD[regime] ?? regime;
}

export const DECISION_WORD: Record<DecisionState, string> = {
  none: "no decision",
  contract_pending: "contract pending",
  approved: "approved · work order scheduled",
  work_order_in_progress: "work order in progress",
  work_order_completed: "work order completed",
  rejected: "contract rejected",
};

/**
 * Status of a STORED prediction at a rewind time. Mirrors the API's own rule
 * (apps/api/routes/twin.py: critical >= classification, warning >= early_warning) with the
 * thresholds the API sent; used only by the rewind scrubber, where the API has already
 * returned the probability and the thresholds and the scene needs a glyph for each.
 */
export function statusForProbability(
  p: number | null | undefined,
  thresholds: Thresholds,
): TwinStatus {
  if (p === null || p === undefined || Number.isNaN(p)) return "no_prediction";
  if (p >= thresholds.classification) return "critical";
  if (p >= thresholds.early_warning) return "warning";
  return "healthy";
}
