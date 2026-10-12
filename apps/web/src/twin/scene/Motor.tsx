/**
 * One drive motor: body + terminal box + fan disc, an operating-state ring on the floor, a status
 * glyph above it, a halo and (when selected) a primary outline ring.
 *
 * Data -> pixel:
 *   emissive colour + halo opacity  <- `status` (STATUS_GLYPH / STATUS_COLOR, palette.glow)
 *   glyph shape                      <- `status` (STATUS_GLYPH.shape)
 *   fan spin                         <- `omega` (from latest.rpm; 0 when offline / no data / rewind)
 *   ring colour + dash pattern       <- `regime` (the operating state; palette.regime, resources.geo.regime)
 */
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { TwinStatus } from "../../api/types";
import { STATUS_GLYPH } from "../status";
import type { TwinPalette } from "../palette";
import { NO_RAYCAST, useResources } from "./resources";

export interface MotorProps {
  status: TwinStatus;
  regime: string | null;
  omega: number;
  selected: boolean;
  palette: TwinPalette;
}

const HALF_PI = Math.PI / 2;

/** Halo strength per status, scaled by palette.glow ("restrained"). */
export const HALO_FACTOR: Record<TwinStatus, number> = {
  critical: 1.0,
  warning: 0.7,
  healthy: 0.18,
  no_prediction: 0.12,
  offline: 0.4,
  no_data: 0,
};

const EMISSIVE_FACTOR: Record<TwinStatus, number> = {
  critical: 0.95,
  warning: 0.6,
  healthy: 0.14,
  no_prediction: 0.08,
  offline: 0.4,
  no_data: 0,
};

type RegimeKey = "R1" | "R2" | "R3" | "unknown";
function regimeKey(regime: string | null): RegimeKey {
  return regime === "R1" || regime === "R2" || regime === "R3" ? regime : "unknown";
}

export function Motor({ status, regime, omega, selected, palette }: MotorProps) {
  const r = useResources();
  const fan = useRef<THREE.Mesh>(null);
  const glyph = useRef<THREE.Mesh>(null);
  const omegaRef = useRef(omega);
  useEffect(() => {
    omegaRef.current = omega;
  }, [omega]);

  const body = useMemo(() => new THREE.MeshLambertMaterial({ color: palette.motor }), [palette.motor]);
  useEffect(() => () => body.dispose(), [body]);
  useEffect(() => {
    body.emissive.set(STATUS_GLYPH[status].color);
    body.emissiveIntensity = EMISSIVE_FACTOR[status] * (0.4 + 0.6 * palette.glow);
  }, [body, status, palette.glow]);

  const haloMat = useMemo(
    () =>
      new THREE.SpriteMaterial({
        map: r.haloTexture,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        toneMapped: false,
      }),
    [r.haloTexture],
  );
  useEffect(() => () => haloMat.dispose(), [haloMat]);
  const haloOpacity = HALO_FACTOR[status] * palette.glow;
  useEffect(() => {
    haloMat.color.set(STATUS_GLYPH[status].color);
    haloMat.opacity = haloOpacity;
  }, [haloMat, status, haloOpacity]);

  useFrame((state, dt) => {
    const w = omegaRef.current;
    if (w !== 0 && fan.current) fan.current.rotation.x += w * 2.5 * Math.min(dt, 0.1);
    if (glyph.current) glyph.current.quaternion.copy(state.camera.quaternion);
  });

  const rk = regimeKey(regime);
  const glyphShape = STATUS_GLYPH[status].shape;

  return (
    <group>
      {/* feet */}
      <mesh geometry={r.geo.box} material={r.mat.steel} position={[0, 0.1, 0]} scale={[1.2, 0.2, 0.9]} raycast={NO_RAYCAST} />
      {/* body: horizontal cylinder along X */}
      <mesh geometry={r.geo.cylinder} material={body} position={[0, 0.62, 0]} rotation={[0, 0, HALF_PI]} scale={[0.9, 1.4, 0.9]} raycast={NO_RAYCAST} />
      {/* terminal box */}
      <mesh geometry={r.geo.box} material={r.mat.steel} position={[0.1, 1.12, 0]} scale={[0.4, 0.24, 0.34]} raycast={NO_RAYCAST} />
      {/* shaft toward the machine */}
      <mesh geometry={r.geo.cylinder} material={r.mat.steel} position={[0, 0.62, -0.95]} rotation={[HALF_PI, 0, 0]} scale={[0.18, 0.5, 0.18]} raycast={NO_RAYCAST} />
      {/* fan disc on the non-drive end */}
      <mesh ref={fan} geometry={r.geo.roller} material={r.mat.machineEdge} position={[-0.8, 0.62, 0]} rotation={[0, 0, HALF_PI]} scale={[0.8, 0.1, 0.8]} raycast={NO_RAYCAST} />
      {/* state ring on the floor (operating state; API field `regime`) */}
      <mesh geometry={r.geo.regime[rk]} material={r.mat.regime[rk]} rotation={[-HALF_PI, 0, 0]} position={[0, 0.02, 0]} raycast={NO_RAYCAST} />
      {selected && (
        <mesh geometry={r.geo.selectRing} material={r.mat.primary} rotation={[-HALF_PI, 0, 0]} position={[0, 0.03, 0]} raycast={NO_RAYCAST} />
      )}
      {/* status glyph, billboarded */}
      <mesh ref={glyph} geometry={r.geo.glyph[glyphShape]} material={r.mat.glyph[status]} position={[0, 2.0, 0]} raycast={NO_RAYCAST} />
      {haloOpacity > 0 && (
        <sprite material={haloMat} position={[0, 0.7, 0]} scale={[2.6, 2.6, 1]} raycast={NO_RAYCAST} />
      )}
    </group>
  );
}
