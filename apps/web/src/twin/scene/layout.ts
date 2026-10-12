/**
 * Pure world layout for the 3-D Digital Twin. No three.js, no React: a TwinSnapshot in,
 * world positions out, so it can be unit-tested without WebGL.
 *
 * World units are metres-ish. Packaging lines run along +X (machine order from the API),
 * rows are stacked along +Z in `line.index` order. The Compressor Hall sits beyond the last
 * row. The edge node is at the side of the lines; the control plane further away.
 */
import type { TwinAsset, TwinLine, TwinSnapshot } from "../../api/types";

export type Vec3 = [number, number, number];

export type MachineKind = "conveyor" | "filler" | "capper" | "packer" | "compressor" | "generic";

export interface AssetPlacement {
  assetId: string;
  line: string;
  lineKind: TwinLine["kind"];
  /** 0-based row ordinal (packaging rows first, then compressor rows). */
  row: number;
  /** 0-based position along the line, in the API's asset order. */
  slot: number;
  machineKind: MachineKind;
  /** Centre of the machine body (floor level). */
  machine: Vec3;
  /** Centre of the drive motor (floor level), beside the machine. */
  motor: Vec3;
  /** Waypoints of the message path: motor (raised) -> edge node -> control plane. */
  path: [Vec3, Vec3, Vec3];
}

export interface LinePlacement {
  line: string;
  kind: TwinLine["kind"];
  row: number;
  /** Where the line title sits (head of the row). */
  head: Vec3;
  /** Centre of the row (camera target for a "line" focus). */
  center: Vec3;
  assetIds: string[];
}

export interface PlantLayout {
  assets: Map<string, AssetPlacement>;
  lines: LinePlacement[];
  edge: Vec3;
  controlPlane: Vec3;
  /** Centre of everything (camera target for the overview). */
  center: Vec3;
  /** Bounding extent of the plant on the floor, for camera distance. */
  extent: { minX: number; maxX: number; minZ: number; maxZ: number };
}

export const SLOT_SPACING = 7.5;
export const ROW_SPACING = 7;
/** Motor sits in front of its machine (toward +Z, i.e. toward the default camera). */
export const MOTOR_OFFSET_Z = 2.6;
export const MOTOR_OFFSET_X = 0;
/** Height at which a pulse leaves the motor and the node heights it travels to. */
export const PULSE_HEIGHT = 1.1;
export const EDGE_HEIGHT = 1.4;
export const CP_HEIGHT = 2.2;
export const COMPRESSOR_GAP = 5;
export const COMPRESSOR_SPACING = 6.5;
export const TITLE_OFFSET_X = -7.5;

/** Machine family from the API's `asset.role` (and the line kind for the hall). */
export function machineKindFor(role: string | undefined, lineKind: TwinLine["kind"]): MachineKind {
  const r = (role ?? "").toLowerCase();
  if (lineKind === "compressor" || r.includes("compressor")) return "compressor";
  if (r.includes("conveyor") || r.includes("infeed")) return "conveyor";
  if (r.includes("filler")) return "filler";
  if (r.includes("capper") || r.includes("labeler") || r.includes("labeller")) return "capper";
  if (r.includes("packer") || r.includes("case")) return "packer";
  return "generic";
}

function sortedLines(lines: TwinLine[], kind: TwinLine["kind"]): TwinLine[] {
  return lines.filter((l) => l.kind === kind).sort((a, b) => a.index - b.index);
}

/**
 * Compute the world layout of a snapshot.
 * Input: the TwinSnapshot (only `lines` and `assets[].asset.role` are read).
 * Output: a PlantLayout with one AssetPlacement per asset id the lines reference.
 * Assets that appear in `assets[]` but in no line are appended to an extra row so nothing
 * the API sent is silently dropped.
 */
export function computeLayout(snapshot: Pick<TwinSnapshot, "lines" | "assets">): PlantLayout {
  const byId = new Map<string, TwinAsset>();
  for (const a of snapshot.assets) byId.set(a.asset.asset_id, a);

  const packaging = sortedLines(snapshot.lines, "packaging");
  const compressors = sortedLines(snapshot.lines, "compressor");
  const assets = new Map<string, AssetPlacement>();
  const lines: LinePlacement[] = [];

  // Layout bounds are decided first so the edge/control-plane can sit beside them.
  const nRows = packaging.length;
  const maxSlots = Math.max(1, ...packaging.map((l) => l.assets.length));
  const lineWidth = (maxSlots - 1) * SLOT_SPACING;
  const compZ = nRows > 0 ? (nRows - 1) * ROW_SPACING + COMPRESSOR_GAP + ROW_SPACING : 0;
  const nComp = compressors.reduce((n, l) => n + l.assets.length, 0);
  const compWidth = Math.max(0, nComp - 1) * COMPRESSOR_SPACING;
  const compX0 = lineWidth / 2 - compWidth / 2;

  const maxX = Math.max(lineWidth, compX0 + compWidth);
  const maxZ = compressors.length > 0 ? compZ : Math.max(0, (nRows - 1) * ROW_SPACING);
  const edge: Vec3 = [maxX + 9, EDGE_HEIGHT, maxZ / 2];
  const controlPlane: Vec3 = [maxX + 20, CP_HEIGHT, maxZ / 2 - 8];

  const place = (
    line: TwinLine,
    row: number,
    slot: number,
    assetId: string,
    machine: Vec3,
  ): AssetPlacement => {
    const twinAsset = byId.get(assetId);
    const kind = machineKindFor(twinAsset?.asset.role, line.kind);
    const motor: Vec3 = [machine[0] + MOTOR_OFFSET_X, 0, machine[2] + MOTOR_OFFSET_Z];
    return {
      assetId,
      line: line.line,
      lineKind: line.kind,
      row,
      slot,
      machineKind: kind,
      machine,
      motor,
      path: [[motor[0], PULSE_HEIGHT, motor[2]], edge, controlPlane],
    };
  };

  packaging.forEach((line, row) => {
    const z = row * ROW_SPACING;
    line.assets.forEach((assetId, slot) => {
      assets.set(assetId, place(line, row, slot, assetId, [slot * SLOT_SPACING, 0, z]));
    });
    lines.push({
      line: line.line,
      kind: line.kind,
      row,
      head: [TITLE_OFFSET_X, 0, z],
      center: [((line.assets.length - 1) * SLOT_SPACING) / 2, 0, z + MOTOR_OFFSET_Z / 2],
      assetIds: [...line.assets],
    });
  });

  let compSlot = 0;
  compressors.forEach((line, i) => {
    const row = nRows + i;
    const firstX = compX0 + compSlot * COMPRESSOR_SPACING;
    line.assets.forEach((assetId, slot) => {
      const x = compX0 + compSlot * COMPRESSOR_SPACING;
      assets.set(assetId, place(line, row, slot, assetId, [x, 0, compZ]));
      compSlot += 1;
    });
    const lastX = compX0 + Math.max(0, compSlot - 1) * COMPRESSOR_SPACING;
    lines.push({
      line: line.line,
      kind: line.kind,
      row,
      head: [firstX + TITLE_OFFSET_X, 0, compZ],
      center: [(firstX + lastX) / 2, 0, compZ + MOTOR_OFFSET_Z / 2],
      assetIds: [...line.assets],
    });
  });

  // Anything the lines did not reference: an extra row so it is still visible.
  const orphans = snapshot.assets.filter((a) => !assets.has(a.asset.asset_id));
  if (orphans.length > 0) {
    const z = maxZ + ROW_SPACING + COMPRESSOR_GAP;
    const row = lines.length;
    const pseudo: TwinLine = { line: "Unassigned", kind: "packaging", index: 999, assets: [], mes: null };
    orphans.forEach((a, slot) => {
      pseudo.assets.push(a.asset.asset_id);
      assets.set(
        a.asset.asset_id,
        place(pseudo, row, slot, a.asset.asset_id, [slot * SLOT_SPACING, 0, z]),
      );
    });
    lines.push({
      line: pseudo.line,
      kind: pseudo.kind,
      row,
      head: [TITLE_OFFSET_X, 0, z],
      center: [((orphans.length - 1) * SLOT_SPACING) / 2, 0, z],
      assetIds: [...pseudo.assets],
    });
  }

  let minX = Infinity;
  let maxXAll = -Infinity;
  let minZ = Infinity;
  let maxZAll = -Infinity;
  for (const p of assets.values()) {
    minX = Math.min(minX, p.machine[0]);
    maxXAll = Math.max(maxXAll, p.machine[0]);
    minZ = Math.min(minZ, p.machine[2]);
    maxZAll = Math.max(maxZAll, p.motor[2]);
  }
  if (assets.size === 0) {
    minX = 0;
    maxXAll = 0;
    minZ = 0;
    maxZAll = 0;
  }
  const extent = { minX, maxX: maxXAll, minZ, maxZ: maxZAll };
  const center: Vec3 = [(minX + maxXAll) / 2, 0, (minZ + maxZAll) / 2];
  return { assets, lines, edge, controlPlane, center, extent };
}

/** Camera position + target for a focus kind, derived from the layout alone. */
export interface CameraPose {
  position: Vec3;
  target: Vec3;
}

export function overviewPose(layout: PlantLayout): CameraPose {
  const { extent, center } = layout;
  const span = Math.max(extent.maxX - extent.minX, extent.maxZ - extent.minZ, 20);
  // Three-quarter view from the front-right, high enough to see every row and both nodes.
  const target: Vec3 = [center[0] + 4, 0.5, center[2]];
  return {
    position: [center[0] + span * 0.5, span * 0.6, extent.maxZ + span * 0.5],
    target,
  };
}

/** The far, high start of the opening move (the camera eases from here to the overview). */
export function introPose(layout: PlantLayout): CameraPose {
  const o = overviewPose(layout);
  return {
    position: [o.position[0] * 1.6, o.position[1] * 2.6, o.position[2] * 2.1],
    target: [o.target[0], o.target[1] + 2, o.target[2]],
  };
}

export function linePose(layout: PlantLayout, line: string): CameraPose | null {
  const l = layout.lines.find((x) => x.line === line);
  if (!l) return null;
  const spacing = l.kind === "compressor" ? COMPRESSOR_SPACING : SLOT_SPACING;
  const width = Math.max(1, l.assetIds.length - 1) * spacing;
  return {
    position: [l.center[0] + width * 0.15, 13 + width * 0.25, l.center[2] + 14 + width * 0.35],
    target: l.center,
  };
}

export function assetPose(layout: PlantLayout, assetId: string): CameraPose | null {
  const p = layout.assets.get(assetId);
  if (!p) return null;
  const target: Vec3 = [(p.machine[0] + p.motor[0]) / 2, 0.9, (p.machine[2] + p.motor[2]) / 2];
  return {
    position: [target[0] + 6.5, 5.5, target[2] + 8],
    target,
  };
}
