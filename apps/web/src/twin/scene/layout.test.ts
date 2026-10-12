import { describe, expect, it } from "vitest";
import { SCENE_FIXTURE } from "./fixture";
import {
  assetPose,
  COMPRESSOR_GAP,
  computeLayout,
  introPose,
  linePose,
  machineKindFor,
  MOTOR_OFFSET_Z,
  overviewPose,
  PULSE_HEIGHT,
  ROW_SPACING,
  SLOT_SPACING,
} from "./layout";

const layout = computeLayout(SCENE_FIXTURE);

describe("computeLayout", () => {
  it("places every fixture asset exactly once", () => {
    expect(layout.assets.size).toBe(24);
    for (const a of SCENE_FIXTURE.assets) {
      expect(layout.assets.has(a.asset.asset_id)).toBe(true);
    }
  });

  it("stacks packaging lines as parallel rows in index order", () => {
    const rows = layout.lines.filter((l) => l.kind === "packaging");
    expect(rows.map((l) => l.line)).toEqual([
      "Packaging Line 1",
      "Packaging Line 2",
      "Packaging Line 3",
      "Packaging Line 4",
      "Packaging Line 5",
    ]);
    rows.forEach((l, i) => {
      expect(l.row).toBe(i);
      expect(l.head[2]).toBeCloseTo(i * ROW_SPACING);
      for (const id of l.assetIds) {
        expect(layout.assets.get(id)!.machine[2]).toBeCloseTo(i * ROW_SPACING);
      }
    });
  });

  it("orders machines along a line in the API's asset order", () => {
    const line3 = SCENE_FIXTURE.lines.find((l) => l.line === "Packaging Line 3")!;
    expect(line3.assets).toEqual(["MTR-013", "MTR-042", "MTR-021", "MTR-009"]);
    line3.assets.forEach((id, slot) => {
      const p = layout.assets.get(id)!;
      expect(p.slot).toBe(slot);
      expect(p.machine[0]).toBeCloseTo(slot * SLOT_SPACING);
      expect(p.row).toBe(2);
      expect(p.line).toBe("Packaging Line 3");
    });
    expect(layout.assets.get("MTR-042")!.machineKind).toBe("filler");
    expect(layout.assets.get("MTR-021")!.machineKind).toBe("capper");
    expect(layout.assets.get("MTR-013")!.machineKind).toBe("conveyor");
    expect(layout.assets.get("MTR-009")!.machineKind).toBe("packer");
  });

  it("puts the motor beside its machine and starts the pulse path at the motor", () => {
    const p = layout.assets.get("MTR-042")!;
    expect(p.motor[0]).toBeCloseTo(p.machine[0]);
    expect(p.motor[2]).toBeCloseTo(p.machine[2] + MOTOR_OFFSET_Z);
    expect(p.path[0]).toEqual([p.motor[0], PULSE_HEIGHT, p.motor[2]]);
    expect(p.path[1]).toEqual(layout.edge);
    expect(p.path[2]).toEqual(layout.controlPlane);
  });

  it("places the Compressor Hall apart, beyond the last packaging row", () => {
    const hall = layout.lines.find((l) => l.kind === "compressor")!;
    expect(hall.line).toBe("Compressor Hall");
    expect(hall.row).toBe(5);
    const lastRowZ = 4 * ROW_SPACING;
    for (const id of hall.assetIds) {
      const p = layout.assets.get(id)!;
      expect(p.machineKind).toBe("compressor");
      expect(p.machine[2]).toBeGreaterThanOrEqual(lastRowZ + COMPRESSOR_GAP + ROW_SPACING);
    }
    const xs = hall.assetIds.map((id) => layout.assets.get(id)!.machine[0]);
    expect([...xs].sort((a, b) => a - b)).toEqual(xs); // K-1..K-4 left to right
  });

  it("puts the edge node at the side of the lines and the control plane further out", () => {
    const { edge, controlPlane, extent } = layout;
    expect(edge[0]).toBeGreaterThan(extent.maxX);
    expect(controlPlane[0]).toBeGreaterThan(edge[0]);
    expect(edge[1]).toBeGreaterThan(0);
    expect(controlPlane[1]).toBeGreaterThan(edge[1]);
    // Every asset shares the same two downstream waypoints.
    for (const p of layout.assets.values()) {
      expect(p.path[1]).toEqual(edge);
      expect(p.path[2]).toEqual(controlPlane);
    }
  });

  it("keeps assets that no line references instead of dropping them", () => {
    const snap = {
      lines: SCENE_FIXTURE.lines.slice(0, 1),
      assets: SCENE_FIXTURE.assets,
    };
    const l = computeLayout(snap);
    expect(l.assets.size).toBe(24);
    expect(l.lines.at(-1)!.line).toBe("Unassigned");
    expect(l.lines.at(-1)!.assetIds).toHaveLength(20);
  });

  it("handles an empty snapshot", () => {
    const l = computeLayout({ lines: [], assets: [] });
    expect(l.assets.size).toBe(0);
    expect(l.lines).toEqual([]);
    expect(overviewPose(l).position.every(Number.isFinite)).toBe(true);
  });
});

describe("machineKindFor", () => {
  it("maps the API roles to machine families", () => {
    expect(machineKindFor("infeed conveyor", "packaging")).toBe("conveyor");
    expect(machineKindFor("filler", "packaging")).toBe("filler");
    expect(machineKindFor("capper/labeler", "packaging")).toBe("capper");
    expect(machineKindFor("case packer", "packaging")).toBe("packer");
    expect(machineKindFor("air compressor", "compressor")).toBe("compressor");
    expect(machineKindFor("mystery", "packaging")).toBe("generic");
    expect(machineKindFor(undefined, "compressor")).toBe("compressor");
  });
});

describe("camera poses", () => {
  it("overview looks at the plant centre from above and in front", () => {
    const o = overviewPose(layout);
    expect(o.position[1]).toBeGreaterThan(10);
    expect(o.position[2]).toBeGreaterThan(layout.extent.maxZ);
    expect(o.target[0]).toBeCloseTo(layout.center[0] + 4);
    const intro = introPose(layout);
    expect(intro.position[1]).toBeGreaterThan(o.position[1]);
  });

  it("line focus targets the row centre; asset focus targets the machine + motor", () => {
    const lp = linePose(layout, "Packaging Line 3")!;
    expect(lp.target[2]).toBeCloseTo(2 * ROW_SPACING + MOTOR_OFFSET_Z / 2);
    const ap = assetPose(layout, "MTR-042")!;
    const p = layout.assets.get("MTR-042")!;
    expect(ap.target[0]).toBeCloseTo(p.machine[0]);
    expect(ap.target[2]).toBeCloseTo(p.machine[2] + MOTOR_OFFSET_Z / 2);
    expect(linePose(layout, "nope")).toBeNull();
    expect(assetPose(layout, "nope")).toBeNull();
  });
});
