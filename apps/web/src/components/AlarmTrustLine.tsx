import { getAlarmTrust } from "../api/client";
import type { AlarmTrust } from "../api/types";
import { usePolling } from "../hooks/usePolling";

/** The one line: counts only; a rate only once enough alarms were acted on (API rule). */
export function alarmTrustText(t: AlarmTrust): string {
  const base = `Alarms acted on: ${t.acted_on} · confirmed: ${t.confirmed} · no fault found: ${t.no_fault_found} · awaiting finding: ${t.awaiting_finding}`;
  const withFinding = t.confirmed + t.no_fault_found + t.other;
  if (t.acted_on >= t.min_for_rate && withFinding > 0) {
    return `${base} · confirmed ${Math.round((100 * t.confirmed) / withFinding)}% of ${withFinding} with a finding`;
  }
  return base;
}

/**
 * Alarm trust: a deterministic count over approved maintenance contracts and what the
 * technician found (GET /findings/trust). "Acted on" = an approved maintenance contract.
 * Sandbox-tagged findings are excluded by the API. Never a percentage under five acted-on
 * alarms.
 */
export function AlarmTrustLine({ compact = false }: { compact?: boolean }) {
  const trust = usePolling(getAlarmTrust, 5000, "alarm-trust");
  if (!trust.data) return null;
  const t = trust.data;
  return (
    <div
      className={`small ${compact ? "muted" : ""}`}
      data-testid="alarm-trust"
      title="Alarm trust: counts over approved maintenance contracts and the technician's findings on their completed work orders (GET /findings/trust). Counts only below five acted-on alarms."
    >
      <span className="muted">Alarm trust · </span>
      {alarmTrustText(t)}
      {t.other > 0 && <span className="dim"> · other: {t.other}</span>}
    </div>
  );
}
