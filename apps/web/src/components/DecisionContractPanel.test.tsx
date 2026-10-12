import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AlarmEvent, Approval, DecisionContract } from "../api/types";
import { DecisionContractPanel } from "./DecisionContractPanel";

const contract: DecisionContract = {
  contract_id: "dc_MTR-042_20261012115138",
  created_at: "2026-10-12T10:05:00Z",
  asset_id: "MTR-042",
  run_id: "run1",
  model_version: "v5797",
  champion_mlflow_run_id: null,
  failure_probability: 0.78,
  recommendation: "maintain_now",
  window_start: "2025-08-25T23:40:00Z",
  window_end: "2025-08-26T02:40:00Z",
  expected_cost: 13500,
  cost_comparison: { asset_id: "MTR-042", recommended: "maintain_now", options: [], assumptions: {} },
  top_features: [],
  manual_context: [],
  lead_time_h: 53.1,
  calibration_brier: 0.005,
  data_quality_score: 100,
  explanation_text: "template text",
  explanation_source: "template",
  evidence_hash: "0".repeat(64),
};

const approval: Approval = {
  approval_id: "a1",
  contract_id: contract.contract_id,
  decision: "approved",
  approver: "engineer@plant",
  note: "",
  decided_at: "2026-10-12T10:06:23Z",
};

const alarms: AlarmEvent[] = [
  { event_id: "e1", asset_id: "MTR-042", level: "critical", reason: "3 consecutive predictions >= 0.5 (latched)", plant_ts: "2025-08-25T14:50:00Z", recorded_at: "2026-10-12T10:03:00Z", probability: 0.81, model_version: "v5797" },
];

const noop = async () => {};

describe("DecisionContractPanel", () => {
  it("shows the alarm → signed decision timer on an approved contract, wall clock, with a tooltip", () => {
    render(<DecisionContractPanel contract={contract} approval={approval} onDecide={noop} busy={false} error={null} alarms={alarms} />);
    const timer = screen.getByTestId("alarm-to-decision");
    expect(timer).toHaveTextContent("3m 23s");
    expect(timer.getAttribute("title")).toMatch(/wall clock/i);
    expect(screen.getByText("Alarm → signed decision")).toBeInTheDocument();
    expect(screen.getByText("template")).toBeInTheDocument();
  });

  it("shows nothing when there was no alarm, or while the contract is still pending", () => {
    const { rerender } = render(<DecisionContractPanel contract={contract} approval={approval} onDecide={noop} busy={false} error={null} alarms={[]} />);
    expect(screen.queryByTestId("alarm-to-decision")).toBeNull();
    rerender(<DecisionContractPanel contract={contract} approval={null} onDecide={noop} busy={false} error={null} alarms={alarms} />);
    expect(screen.queryByTestId("alarm-to-decision")).toBeNull();
  });
});
