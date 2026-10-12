import type { AlarmEvent, Approval, DecisionContract } from "../api/types";

/**
 * "Alarm → signed decision": wall-clock milliseconds from the `recorded_at` of the asset's
 * critical alarm event that preceded the contract to the approval's `decided_at`.
 *
 * Pure: picks the LATEST critical alarm row for the contract's asset whose `recorded_at` is
 * at or before the contract's `created_at` (the alarm the engineer acted on); returns null when
 * there is no such alarm, the approval is not an approval, or the clocks disagree (negative).
 * Both ends are wall clock (`recorded_at`, `decided_at`), never plant time.
 */
export function alarmToDecisionMs(
  alarms: readonly AlarmEvent[],
  contract: Pick<DecisionContract, "asset_id" | "created_at">,
  approval: Pick<Approval, "decided_at" | "decision"> | null | undefined,
): number | null {
  if (!approval || approval.decision !== "approved") return null;
  const created = Date.parse(contract.created_at);
  const decided = Date.parse(approval.decided_at);
  if (!Number.isFinite(created) || !Number.isFinite(decided)) return null;
  let best: number | null = null;
  for (const ev of alarms) {
    if (ev.asset_id !== contract.asset_id || ev.level !== "critical") continue;
    const at = Date.parse(ev.recorded_at);
    if (!Number.isFinite(at) || at > created) continue;
    if (best === null || at > best) best = at;
  }
  if (best === null) return null;
  const ms = decided - best;
  return ms >= 0 ? ms : null;
}

/** 83_000 -> "1m 23s"; 5_000 -> "0m 5s"; 3_723_000 -> "62m 3s" (minutes never roll into hours). */
export function fmtMinSec(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(total / 60)}m ${total % 60}s`;
}
