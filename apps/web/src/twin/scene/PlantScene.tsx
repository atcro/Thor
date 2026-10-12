/**
 * The 3-D Digital Twin scene (CLAUDE.md §15, Phase A). Mounted by src/screens/Twin.tsx
 * inside a full-height `.twin-theme` div. Renders ONLY what the API said:
 *
 *   machine + fan motion  <- asset.latest.rpm   (0 when offline / data_state != live / rewind)
 *   glow, halo, glyph     <- asset.status        (rewind: rewind.statuses[asset_id])
 *   status word + p       <- STATUS_GLYPH.word, asset.failure_probability (rewind.probabilities)
 *   regime ring           <- asset.regime
 *   decision badge        <- asset.decision_state (DECISION_WORD)
 *   work-order tag        <- asset.work_order.status
 *   pulses                <- props.pulseQueue (ingest / prediction events from the API)
 *   layout                <- snapshot.lines (order) + asset.role (machine family)
 *
 * No idle animation anywhere: with still data the scene is still.
 */
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { TwinAsset, TwinStatus } from "../../api/types";
import type { TwinPalette } from "../palette";
import type { SceneProps } from "../sceneTypes";
import { DECISION_WORD, STATUS_GLYPH } from "../status";
import { CameraRig } from "./CameraRig";
import { SceneLabel, LineTitle } from "./labels";
import { computeLayout, type AssetPlacement, type PlantLayout } from "./layout";
import { Machine, MACHINE_SIZE } from "./machines";
import { Motor } from "./Motor";
import { ControlPlane, EdgeNode, PathLines } from "./Nodes";
import { Pulses } from "./Pulses";
import { createResources, NO_RAYCAST, ResourcesContext, RPM_TO_RAD_PER_S, useResources } from "./resources";
import "./scene.css";

const EMPTY_SNAPSHOT = { lines: [], assets: [] };
const PULSE_CAP = 400;
const PULSE_CAP_PERF = 120;

/** rad/s of visual spin for one asset, or 0 when nothing should move. */
function omegaFor(a: TwinAsset, frozen: boolean): number {
  if (frozen) return 0;
  if (a.status === "offline" || a.data_state !== "live" || a.latest === null) return 0;
  const rpm = a.latest.rpm;
  if (!Number.isFinite(rpm) || rpm <= 0) return 0;
  return rpm * RPM_TO_RAD_PER_S;
}

interface AssetGroupProps {
  twin: TwinAsset;
  placement: AssetPlacement;
  status: TwinStatus;
  probability: number | null;
  regime: string | null;
  omega: number;
  selected: boolean;
  hovered: boolean;
  showWord: boolean;
  showBadge: boolean;
  palette: TwinPalette;
  onSelect: SceneProps["onSelect"];
  onHover: SceneProps["onHover"];
}

function AssetGroup(props: AssetGroupProps) {
  const r = useResources();
  const { twin, placement, status, probability, regime, omega, selected, hovered } = props;
  const id = twin.asset.asset_id;
  const [w, h, d] = MACHINE_SIZE[placement.machineKind];
  const dim = status === "no_data";
  const glyph = STATUS_GLYPH[status];
  const decision = twin.decision_state;
  const wo = twin.work_order;

  const hitCenter: [number, number, number] = [
    (placement.machine[0] + placement.motor[0]) / 2,
    h / 2,
    (placement.machine[2] + placement.motor[2]) / 2,
  ];
  const hitSize: [number, number, number] = [
    Math.max(w, 2.0),
    h,
    Math.abs(placement.motor[2] - placement.machine[2]) + Math.max(d, 1.4),
  ];

  return (
    <group>
      <group position={placement.machine}>
        <Machine kind={placement.machineKind} omega={omega} dim={dim} />
      </group>
      <group position={placement.motor}>
        <Motor status={status} regime={regime} omega={omega} selected={selected} palette={props.palette} />
      </group>
      {/* one invisible hit box per asset does the picking; visual meshes opt out */}
      <mesh
        position={hitCenter}
        scale={hitSize}
        geometry={r.geo.box}
        visible={false}
        onPointerOver={(e) => {
          e.stopPropagation();
          props.onHover(id);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          props.onHover(null);
        }}
        onClick={(e) => {
          e.stopPropagation();
          props.onSelect(id);
        }}
      />
      {props.showWord && (
        <SceneLabel
          position={[placement.motor[0], 2.75, placement.motor[2]]}
          className={`tw-word tw-word-${status}${hovered ? " is-hovered" : ""}${selected ? " is-selected" : ""}`}
        >
          <span className="tw-word-id">{id}</span>
          <span className="tw-word-text">{glyph.word}</span>
          {probability !== null && <span className="tw-word-p">p {probability.toFixed(2)}</span>}
        </SceneLabel>
      )}
      {props.showBadge && decision !== "none" && (
        <SceneLabel
          position={[placement.machine[0], h + 0.9, placement.machine[2]]}
          className={`tw-badge tw-badge-${decision}`}
        >
          {DECISION_WORD[decision]}
        </SceneLabel>
      )}
      {props.showBadge && wo?.status === "in_progress" && (
        <SceneLabel position={[placement.machine[0], h + 0.25, placement.machine[2]]} className="tw-wo">
          work order in progress
        </SceneLabel>
      )}
    </group>
  );
}

function Floor({ layout, palette, rewind, onSelect }: { layout: PlantLayout; palette: TwinPalette; rewind: boolean; onSelect: SceneProps["onSelect"] }) {
  const size = Math.max(160, (layout.extent.maxX - layout.extent.minX) * 4);
  const cx = (layout.extent.minX + Math.max(layout.extent.maxX, layout.controlPlane[0])) / 2;
  const cz = (layout.extent.minZ + layout.extent.maxZ) / 2;
  const floorMat = useMemo(() => new THREE.MeshLambertMaterial({ color: palette.floor }), [palette.floor]);
  useEffect(() => () => floorMat.dispose(), [floorMat]);
  // Rewind: tint the grid with the accent so the whole scene reads as "stored, not live".
  const gridColor = rewind ? palette.accent : palette.grid;
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(size, Math.round(size / 2), gridColor, gridColor);
    const m = g.material as THREE.LineBasicMaterial;
    m.transparent = true;
    m.opacity = rewind ? 0.5 : 0.35;
    m.depthWrite = false;
    return g;
  }, [size, gridColor, rewind]);
  useEffect(() => () => grid.dispose(), [grid]);
  return (
    <group position={[cx, 0, cz]}>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -0.01, 0]}
        material={floorMat}
        onClick={() => onSelect(null)}
      >
        <planeGeometry args={[size, size]} />
      </mesh>
      <primitive object={grid} position={[0, 0.005, 0]} raycast={NO_RAYCAST} />
      {rewind && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.0, 0]} raycast={NO_RAYCAST}>
          <planeGeometry args={[size, size]} />
          <meshBasicMaterial color={palette.accent} transparent opacity={0.06} depthWrite={false} toneMapped={false} />
        </mesh>
      )}
    </group>
  );
}

function FrameStats({ onFps }: { onFps?: (fps: number) => void }) {
  const frames = useRef(0);
  const since = useRef(0);
  useFrame(({ clock }) => {
    frames.current += 1;
    const now = clock.elapsedTime;
    if (since.current === 0) since.current = now;
    const elapsed = now - since.current;
    if (elapsed >= 1) {
      onFps?.(Math.round(frames.current / elapsed));
      frames.current = 0;
      since.current = now;
    }
  });
  return null;
}

function Resources({ palette, perf, children }: { palette: TwinPalette; perf: boolean; children: React.ReactNode }) {
  const res = useMemo(() => createResources(palette, perf), [palette, perf]);
  useEffect(() => () => res.dispose(), [res]);
  return <ResourcesContext.Provider value={res}>{children}</ResourcesContext.Provider>;
}

/** A stable key for the layout: positions depend only on line structure and roles. */
function layoutKey(s: SceneProps["snapshot"]): string {
  if (!s) return "";
  const lines = s.lines.map((l) => `${l.line}|${l.kind}|${l.index}|${l.assets.join(",")}`).join(";");
  const roles = s.assets.map((a) => `${a.asset.asset_id}=${a.asset.role ?? ""}`).join(",");
  return `${lines}#${roles}`;
}

export default function PlantScene(props: SceneProps) {
  const { snapshot, rewind, palette, performanceMode, xray, selected, hovered } = props;
  const key = layoutKey(snapshot);
  const layout = useMemo(
    () => computeLayout(snapshot ?? EMPTY_SNAPSHOT),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key],
  );

  useEffect(() => {
    document.body.style.cursor = hovered ? "pointer" : "";
    return () => {
      document.body.style.cursor = "";
    };
  }, [hovered]);

  const frozen = rewind !== null;
  const assets = snapshot?.assets ?? [];
  const showStructureLabels = true;

  return (
    <Canvas
      dpr={performanceMode ? 1 : [1, 1.5]}
      gl={{ antialias: true, powerPreference: "high-performance", alpha: false }}
      shadows={false}
      frameloop="always"
      camera={{ fov: 42, near: 0.5, far: 500, position: [60, 60, 120] }}
      onPointerMissed={() => props.onSelect(null)}
      className="tw-canvas"
    >
      <color attach="background" args={[palette.bg]} />
      <fog attach="fog" args={[palette.bg, 90, 260]} />
      <hemisphereLight args={["#c6d6ec", palette.floor, 1.7]} />
      <directionalLight position={[30, 50, 20]} intensity={2.2} color="#ffffff" />

      <Resources palette={palette} perf={performanceMode}>
        <Floor layout={layout} palette={palette} rewind={frozen} onSelect={props.onSelect} />

        {assets.map((twin) => {
          const id = twin.asset.asset_id;
          const placement = layout.assets.get(id);
          if (!placement) return null;
          const status: TwinStatus = rewind ? (rewind.statuses[id] ?? "no_prediction") : twin.status;
          const probability = rewind ? (rewind.probabilities[id] ?? null) : twin.failure_probability;
          const isSel = selected === id;
          const isHov = hovered === id;
          const focused = isSel || isHov;
          return (
            <AssetGroup
              key={id}
              twin={twin}
              placement={placement}
              status={status}
              probability={probability}
              regime={rewind ? null : twin.regime}
              omega={omegaFor(twin, frozen)}
              selected={isSel}
              hovered={isHov}
              showWord={focused || (xray && !performanceMode)}
              showBadge={!frozen && (focused || !performanceMode)}
              palette={palette}
              onSelect={props.onSelect}
              onHover={props.onHover}
            />
          );
        })}

        {showStructureLabels &&
          layout.lines.map((l) => <LineTitle key={l.line} position={l.head} title={l.line} kind={l.kind} />)}

        <EdgeNode position={layout.edge} label={showStructureLabels} />
        <ControlPlane position={layout.controlPlane} label={showStructureLabels} />
        <PathLines layout={layout} palette={palette} />
        <Pulses layout={layout} queue={props.pulseQueue} palette={palette} cap={performanceMode ? PULSE_CAP_PERF : PULSE_CAP} />
      </Resources>

      <CameraRig layout={layout} focus={props.focus} focusNonce={props.focusNonce} />
      <FrameStats onFps={props.onFps} />
    </Canvas>
  );
}
