/**
 * The two infrastructure nodes and the message path.
 *   Edge node: a small rack with an antenna mast (ONNX Runtime sidecar, subscribed to MQTT).
 *   Control plane: a taller server rack (FastAPI + TimescaleDB).
 *   Path: one drei Line (segments) from every motor to the edge node, and edge -> control plane.
 */
import { Line } from "@react-three/drei";
import { useMemo } from "react";
import type { TwinPalette } from "../palette";
import type { PlantLayout, Vec3 } from "./layout";
import { NodeLabel } from "./labels";
import { NO_RAYCAST, useResources } from "./resources";

export function EdgeNode({ position, label }: { position: Vec3; label: boolean }) {
  const r = useResources();
  const [x, , z] = position;
  return (
    <group position={[x, 0, z]}>
      <mesh geometry={r.geo.box} material={r.mat.machineEdge} position={[0, 0.05, 0]} scale={[1.6, 0.08, 1.4]} raycast={NO_RAYCAST} />
      <mesh geometry={r.geo.box} material={r.mat.node} position={[0, 0.7, 0]} scale={[1.0, 1.3, 0.8]} raycast={NO_RAYCAST} />
      {[0.35, 0.65, 0.95].map((y) => (
        <mesh key={y} geometry={r.geo.box} material={r.mat.nodeAccent} position={[0, y, 0.41]} scale={[0.7, 0.05, 0.02]} raycast={NO_RAYCAST} />
      ))}
      {/* antenna mast */}
      <mesh geometry={r.geo.cylinder} material={r.mat.steel} position={[0.35, 2.0, -0.2]} scale={[0.06, 1.4, 0.06]} raycast={NO_RAYCAST} />
      <mesh geometry={r.geo.sphere} material={r.mat.nodeAccent} position={[0.35, 2.75, -0.2]} scale={[0.22, 0.22, 0.22]} raycast={NO_RAYCAST} />
      {label && <NodeLabel position={[0, 3.4, 0]} title="Edge node" sub="ONNX Runtime · MQTT" />}
    </group>
  );
}

export function ControlPlane({ position, label }: { position: Vec3; label: boolean }) {
  const r = useResources();
  const [x, , z] = position;
  return (
    <group position={[x, 0, z]}>
      <mesh geometry={r.geo.box} material={r.mat.machineEdge} position={[0, 0.05, 0]} scale={[2.2, 0.08, 1.8]} raycast={NO_RAYCAST} />
      <mesh geometry={r.geo.box} material={r.mat.node} position={[0, 1.4, 0]} scale={[1.3, 2.8, 1.0]} raycast={NO_RAYCAST} />
      {[0.5, 0.9, 1.3, 1.7, 2.1, 2.5].map((y) => (
        <mesh key={y} geometry={r.geo.box} material={r.mat.nodeAccent} position={[0, y, 0.51]} scale={[0.9, 0.05, 0.02]} raycast={NO_RAYCAST} />
      ))}
      <mesh geometry={r.geo.box} material={r.mat.node} position={[1.3, 0.9, 0]} scale={[0.9, 1.8, 1.0]} raycast={NO_RAYCAST} />
      {label && (
        <NodeLabel position={[0, 3.6, 0]} title="Control plane" sub="FastAPI · TimescaleDB · MLflow" />
      )}
    </group>
  );
}

export function PathLines({ layout, palette }: { layout: PlantLayout; palette: TwinPalette }) {
  const points = useMemo(() => {
    const pts: Vec3[] = [];
    for (const p of layout.assets.values()) {
      pts.push(p.path[0], p.path[1]);
    }
    if (layout.assets.size > 0) pts.push(layout.edge, layout.controlPlane);
    return pts;
  }, [layout]);
  if (points.length < 2) return null;
  return (
    <Line
      points={points}
      segments
      color={palette.accent}
      lineWidth={1}
      transparent
      opacity={0.28}
      depthWrite={false}
      raycast={NO_RAYCAST}
    />
  );
}
