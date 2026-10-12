/**
 * Props contract between the Twin screen (apps/web/src/screens/Twin.tsx) and the 3-D scene
 * (apps/web/src/twin/scene/PlantScene.tsx). The scene renders what the API said; it never
 * computes a status, a probability or a rate.
 */
import type { MutableRefObject } from "react";
import type { TwinSnapshot, TwinStatus } from "../api/types";
import type { TwinPalette } from "./palette";

/** One live arrival at the control plane (from /ws/twin or /twin/events), as a pulse. */
export interface PulseEvent {
  seq: number;
  kind: "ingest" | "prediction";
  /** Predictions: where it was scored. Only edge predictions travel edge -> control plane. */
  source: "edge" | "control_plane" | null;
  asset_id: string;
  /** Aggregated arrivals this event stands for (>= 1). */
  count: number;
  /** performance.now() when the browser received it. */
  at: number;
}

export type CameraFocus =
  | { kind: "overview" }
  | { kind: "line"; line: string }
  | { kind: "asset"; assetId: string };

/** Rewind mode: the scene colours every motor from a STORED prediction at the scrub time. */
export interface RewindFrame {
  ts: string;
  /** asset_id -> stored failure probability at or before `ts` (null = none stored yet). */
  probabilities: Record<string, number | null>;
  /** asset_id -> status derived from that probability with the API's thresholds. */
  statuses: Record<string, TwinStatus>;
}

export interface SceneProps {
  snapshot: TwinSnapshot | null;
  /** When set, status/glow come from stored predictions and machine motion is frozen. */
  rewind: RewindFrame | null;
  /** The screen pushes events here; the scene drains it every frame and spawns pulses. */
  pulseQueue: MutableRefObject<PulseEvent[]>;
  selected: string | null;
  hovered: string | null;
  onSelect: (assetId: string | null) => void;
  onHover: (assetId: string | null) => void;
  focus: CameraFocus;
  /** Bump to re-trigger a fly-to on the same focus (e.g. the "overview" button). */
  focusNonce: number;
  xray: boolean;
  palette: TwinPalette;
  /** Measured frames per second, reported about once a second. */
  onFps?: (fps: number) => void;
}
