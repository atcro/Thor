/**
 * Html labels (drei `Html`) styled by scene.css with the `--tw-*` variables that the parent
 * `.twin-theme` wrapper defines. Kept few: line titles, the two nodes, decision badges and the
 * status word (xray / hovered / selected only).
 */
import { Html } from "@react-three/drei";
import type { ReactNode } from "react";
import type { Vec3 } from "./layout";

interface LabelProps {
  position: Vec3;
  className?: string;
  children: ReactNode;
}

export function SceneLabel({ position, className, children }: LabelProps) {
  return (
    <Html position={position} center zIndexRange={[20, 0]} style={{ pointerEvents: "none" }}>
      <div className={`tw-label ${className ?? ""}`}>{children}</div>
    </Html>
  );
}

export function LineTitle({ position, title, kind }: { position: Vec3; title: string; kind: string }) {
  return (
    <SceneLabel position={[position[0], 1.2, position[2]]} className="tw-title">
      <span className="tw-title-kind">{kind === "compressor" ? "HALL" : "LINE"}</span>
      {title}
    </SceneLabel>
  );
}

export function NodeLabel({ position, title, sub }: { position: Vec3; title: string; sub: string }) {
  return (
    <SceneLabel position={position} className="tw-node">
      <div className="tw-node-title">{title}</div>
      <div className="tw-node-sub">{sub}</div>
    </SceneLabel>
  );
}
