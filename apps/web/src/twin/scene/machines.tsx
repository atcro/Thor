/**
 * Low-poly procedural machines. Each takes `omega` (rad/s, already derived from the drive
 * motor's `latest.rpm` by AssetGroup; 0 = stands still) and `dim` (no data). Nothing in here
 * moves unless omega is non-zero, and omega only ever comes from telemetry.
 */
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { MachineKind } from "./layout";
import { NO_RAYCAST, useResources } from "./resources";

export interface MachineProps {
  kind: MachineKind;
  omega: number;
  dim: boolean;
}

const HALF_PI = Math.PI / 2;

/** Phase accumulator driven by omega; `apply` is called only while the machine moves. */
function useSpin(omega: number, apply: (phase: number) => void): void {
  const omegaRef = useRef(omega);
  const phase = useRef(0);
  const applyRef = useRef(apply);
  applyRef.current = apply;
  useEffect(() => {
    omegaRef.current = omega;
  }, [omega]);
  useFrame((_, dt) => {
    const w = omegaRef.current;
    if (w === 0) return;
    phase.current = (phase.current + w * Math.min(dt, 0.1)) % (Math.PI * 2 * 1000);
    applyRef.current(phase.current);
  });
}

function Base({ w, h, d, dim, y = 0 }: { w: number; h: number; d: number; dim: boolean; y?: number }) {
  const r = useResources();
  return (
    <>
      <mesh
        geometry={r.geo.box}
        material={dim ? r.mat.machineDim : r.mat.machine}
        position={[0, y + h / 2, 0]}
        scale={[w, h, d]}
        raycast={NO_RAYCAST}
      />
      {/* accent skirt: the technical edge line along the base */}
      <mesh
        geometry={r.geo.box}
        material={r.mat.machineEdge}
        position={[0, y + 0.06, 0]}
        scale={[w + 0.06, 0.08, d + 0.06]}
        raycast={NO_RAYCAST}
      />
    </>
  );
}

const ROLLERS = 9;

function Conveyor({ omega, dim }: MachineProps) {
  const r = useResources();
  const inst = useRef<THREE.InstancedMesh>(null);
  const tmp = useMemo(() => new THREE.Matrix4(), []);
  const base = useMemo(() => {
    const m = new THREE.Matrix4().makeRotationX(HALF_PI);
    m.scale(new THREE.Vector3(0.36, 1.5, 0.36));
    return m;
  }, []);
  const xs = useMemo(
    () => Array.from({ length: ROLLERS }, (_, i) => -2.6 + (i * 5.2) / (ROLLERS - 1)),
    [],
  );
  const write = (phase: number) => {
    const im = inst.current;
    if (!im) return;
    for (let i = 0; i < ROLLERS; i++) {
      tmp.makeRotationZ(phase + i * 0.4).multiply(base);
      tmp.setPosition(xs[i], 0.78, 0);
      im.setMatrixAt(i, tmp);
    }
    im.instanceMatrix.needsUpdate = true;
  };
  useEffect(() => write(0), []); // eslint-disable-line react-hooks/exhaustive-deps
  useSpin(omega, write);
  const mat = dim ? r.mat.machineDim : r.mat.machine;
  return (
    <group>
      {/* two rails + four legs */}
      <mesh geometry={r.geo.box} material={mat} position={[0, 0.62, 0.8]} scale={[6, 0.22, 0.14]} raycast={NO_RAYCAST} />
      <mesh geometry={r.geo.box} material={mat} position={[0, 0.62, -0.8]} scale={[6, 0.22, 0.14]} raycast={NO_RAYCAST} />
      {[-2.4, 2.4].map((x) =>
        [-0.8, 0.8].map((z) => (
          <mesh key={`${x}:${z}`} geometry={r.geo.box} material={mat} position={[x, 0.28, z]} scale={[0.14, 0.56, 0.14]} raycast={NO_RAYCAST} />
        )),
      )}
      <mesh geometry={r.geo.box} material={r.mat.machineEdge} position={[0, 0.06, 0]} scale={[6.1, 0.08, 1.9]} raycast={NO_RAYCAST} />
      <instancedMesh ref={inst} args={[r.geo.roller, r.mat.steel, ROLLERS]} raycast={NO_RAYCAST} />
    </group>
  );
}

const PINS = 12;

function Filler({ omega, dim }: MachineProps) {
  const r = useResources();
  const carousel = useRef<THREE.Group>(null);
  const pins = useRef<THREE.InstancedMesh>(null);
  useEffect(() => {
    const im = pins.current;
    if (!im) return;
    const m = new THREE.Matrix4();
    for (let i = 0; i < PINS; i++) {
      const a = (i / PINS) * Math.PI * 2;
      m.makeScale(0.16, 0.5, 0.16);
      m.setPosition(Math.cos(a) * 0.95, 0.2, Math.sin(a) * 0.95);
      im.setMatrixAt(i, m);
    }
    im.instanceMatrix.needsUpdate = true;
  }, []);
  useSpin(omega, (phase) => {
    if (carousel.current) carousel.current.rotation.y = phase;
  });
  return (
    <group>
      <Base w={2.2} h={1.0} d={2.2} dim={dim} />
      <mesh geometry={r.geo.cylinder} material={r.mat.steel} position={[0, 1.8, 0]} scale={[0.5, 1.6, 0.5]} raycast={NO_RAYCAST} />
      <group ref={carousel} position={[0, 1.45, 0]}>
        <mesh geometry={r.geo.torus} material={r.mat.machineEdge} rotation={[HALF_PI, 0, 0]} scale={[0.95, 0.95, 0.95]} raycast={NO_RAYCAST} />
        <instancedMesh ref={pins} args={[r.geo.pin, r.mat.steel, PINS]} raycast={NO_RAYCAST} />
      </group>
      <mesh geometry={r.geo.disc} material={dim ? r.mat.machineDim : r.mat.machine} position={[0, 2.62, 0]} scale={[2.0, 0.12, 2.0]} raycast={NO_RAYCAST} />
    </group>
  );
}

function Capper({ omega, dim }: MachineProps) {
  const r = useResources();
  const turret = useRef<THREE.Group>(null);
  useSpin(omega, (phase) => {
    if (turret.current) turret.current.rotation.y = phase;
  });
  const heads = useMemo(() => Array.from({ length: 6 }, (_, i) => (i / 6) * Math.PI * 2), []);
  return (
    <group>
      <Base w={1.7} h={1.3} d={1.7} dim={dim} />
      <group ref={turret} position={[0, 1.3, 0]}>
        <mesh geometry={r.geo.cylinder} material={r.mat.steel} position={[0, 0.5, 0]} scale={[0.8, 1.0, 0.8]} raycast={NO_RAYCAST} />
        {heads.map((a) => (
          <mesh
            key={a}
            geometry={r.geo.box}
            material={r.mat.machineEdge}
            position={[Math.cos(a) * 0.55, 0.95, Math.sin(a) * 0.55]}
            scale={[0.18, 0.3, 0.18]}
            raycast={NO_RAYCAST}
          />
        ))}
      </group>
    </group>
  );
}

function Packer({ omega, dim }: MachineProps) {
  const r = useResources();
  const pusher = useRef<THREE.Mesh>(null);
  useSpin(omega, (phase) => {
    // Reciprocating pusher: one stroke per 1.5 revolutions of the visual spin.
    if (pusher.current) pusher.current.position.x = 1.0 + 0.55 * Math.sin(phase / 1.5);
  });
  return (
    <group>
      <Base w={3.0} h={1.4} d={2.0} dim={dim} />
      <mesh ref={pusher} geometry={r.geo.box} material={r.mat.steel} position={[1.0, 1.65, 0]} scale={[0.5, 0.32, 1.2]} raycast={NO_RAYCAST} />
      <mesh geometry={r.geo.box} material={r.mat.steel} position={[-0.9, 1.75, 0]} scale={[1.0, 0.7, 1.4]} raycast={NO_RAYCAST} />
    </group>
  );
}

function Compressor({ omega, dim }: MachineProps) {
  const r = useResources();
  const wheel = useRef<THREE.Mesh>(null);
  useSpin(omega, (phase) => {
    if (wheel.current) wheel.current.rotation.x = phase;
  });
  const mat = dim ? r.mat.machineDim : r.mat.machine;
  return (
    <group>
      <mesh geometry={r.geo.box} material={r.mat.machineEdge} position={[0, 0.06, 0]} scale={[3.4, 0.08, 1.8]} raycast={NO_RAYCAST} />
      {[-0.9, 0.9].map((x) => (
        <mesh key={x} geometry={r.geo.box} material={mat} position={[x, 0.3, 0]} scale={[0.4, 0.6, 1.3]} raycast={NO_RAYCAST} />
      ))}
      {/* horizontal receiver tank */}
      <mesh geometry={r.geo.cylinder} material={mat} position={[0, 1.05, 0]} rotation={[0, 0, HALF_PI]} scale={[1.2, 2.8, 1.2]} raycast={NO_RAYCAST} />
      {/* flywheel on the drive end */}
      <mesh ref={wheel} geometry={r.geo.roller} material={r.mat.steel} position={[-1.65, 1.05, 0]} rotation={[0, 0, HALF_PI]} scale={[1.3, 0.14, 1.3]} raycast={NO_RAYCAST} />
      {/* outlet pipe */}
      <mesh geometry={r.geo.cylinder} material={r.mat.steel} position={[1.1, 1.95, 0]} scale={[0.16, 0.6, 0.16]} raycast={NO_RAYCAST} />
    </group>
  );
}

function Generic({ dim }: MachineProps) {
  const r = useResources();
  return (
    <group>
      <Base w={1.8} h={1.2} d={1.6} dim={dim} />
      <mesh geometry={r.geo.box} material={r.mat.steel} position={[0, 1.4, 0]} scale={[0.8, 0.4, 0.8]} raycast={NO_RAYCAST} />
    </group>
  );
}

export function Machine(props: MachineProps) {
  switch (props.kind) {
    case "conveyor":
      return <Conveyor {...props} />;
    case "filler":
      return <Filler {...props} />;
    case "capper":
      return <Capper {...props} />;
    case "packer":
      return <Packer {...props} />;
    case "compressor":
      return <Compressor {...props} />;
    default:
      return <Generic {...props} />;
  }
}

/** Footprint (w, h, d) of each machine kind, for the hit box and badge height. */
export const MACHINE_SIZE: Record<MachineKind, [number, number, number]> = {
  conveyor: [6.2, 1.2, 2.0],
  filler: [2.4, 2.8, 2.4],
  capper: [1.9, 2.4, 1.9],
  packer: [3.2, 2.2, 2.2],
  compressor: [3.6, 2.3, 2.0],
  generic: [2.0, 1.7, 1.8],
};
