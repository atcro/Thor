/**
 * Display rules shared by the 3-D scene, the X-Ray HUD and the Fleet X-Ray.
 * Status itself comes from the API (`TwinAsset.status` / `FleetAsset.status`, the latched
 * alarm rule in apps/api/alarms.py); these helpers only map it to a glyph shape, a colour
 * token and a word, so colour is never the only cue.
 */
import type { DecisionState, Thresholds, TwinStatus } from "../api/types";
import { stateName } from "../lib/format";
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

/** What green means, wherever a legend or tooltip explains it. */
export const HEALTHY_MEANING = "within healthy baseline";

/** Longer explanation of a status for legends and tooltips (the glyph word stays short). */
export const STATUS_EXPLAIN: Record<TwinStatus, string> = {
  healthy: `healthy · ${HEALTHY_MEANING}`,
  warning: "warning · sustained above the early-warning threshold",
  critical: "critical · latched until a completed work order clears it",
  no_prediction: "no prediction yet",
  offline: "offline · work order in progress",
  no_data: "no data · messages stopped",
};

/** Plain name of the operating state the API inferred from rpm / load (API field: `regime`). */
export function stateWord(regime: string | null | undefined): string {
  if (!regime) return "state unknown";
  return stateName(regime);
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
 * Status of a STORED prediction at a rewind time. Mirrors the API's thresholds (critical >=
 * classification, warning >= early_warning) for stored points only; used by the rewind
 * scrubber, where the API has already returned the probability and the thresholds and the
 * scene needs a glyph for each. Live status is never computed here: it comes from the API's
 * latched alarm (`TwinAsset.status`), which stays critical until a work order clears it.
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
