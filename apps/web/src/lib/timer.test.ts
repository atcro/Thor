import { describe, expect, it } from "vitest";
import type { AlarmEvent } from "../api/types";
import { alarmToDecisionMs, fmtMinSec } from "./timer";

function ev(asset_id: string, level: AlarmEvent["level"], recorded_at: string): AlarmEvent {
  return { event_id: `${asset_id}-${recorded_at}`, asset_id, level, reason: "r", plant_ts: "2025-08-25T14:50:00Z", recorded_at, probability: 0.8, model_version: "v1" };
}

const contract = { asset_id: "MTR-042", created_at: "2026-10-12T10:05:00Z" };
const approval = { decision: "approved" as const, decided_at: "2026-10-12T10:06:23Z" };

describe("alarmToDecisionMs", () => {
  it("measures from the latest critical alarm before the contract to the approval, wall clock", () => {
    const alarms = [
      ev("MTR-042", "warning", "2026-10-12T10:00:00Z"),
      ev("MTR-042", "critical", "2026-10-12T10:01:00Z"),
      ev("MTR-042", "critical", "2026-10-12T10:03:00Z"), // the latest one before the contract
      ev("MTR-003", "critical", "2026-10-12T10:04:00Z"), // another asset
      ev("MTR-042", "critical", "2026-10-12T10:05:30Z"), // after the contract: ignored
    ];
    expect(alarmToDecisionMs(alarms, contract, approval)).toBe(3 * 60_000 + 23_000);
  });

  it("returns null without a preceding critical alarm, without an approval, or when rejected", () => {
    expect(alarmToDecisionMs([], contract, approval)).toBeNull();
    expect(alarmToDecisionMs([ev("MTR-042", "warning", "2026-10-12T10:00:00Z")], contract, approval)).toBeNull();
    expect(alarmToDecisionMs([ev("MTR-042", "critical", "2026-10-12T10:00:00Z")], contract, null)).toBeNull();
    expect(alarmToDecisionMs([ev("MTR-042", "critical", "2026-10-12T10:00:00Z")], contract, { decision: "rejected", decided_at: approval.decided_at })).toBeNull();
    // a clock that runs backwards is not a measurement
    expect(alarmToDecisionMs([ev("MTR-042", "critical", "2026-10-12T10:00:00Z")], contract, { decision: "approved", decided_at: "2026-10-12T09:00:00Z" })).toBeNull();
  });

  it("formats minutes and seconds", () => {
    expect(fmtMinSec(83_000)).toBe("1m 23s");
    expect(fmtMinSec(5_400)).toBe("0m 5s");
    expect(fmtMinSec(3_723_000)).toBe("62m 3s");
  });
});
