/**
 * Perspective camera + OrbitControls + a smooth fly-to.
 *   overview  2.5 s ease-out from a high, far position to a three-quarter view of all lines
 *             (runs on mount: the opening move).
 *   line      frame that row.
 *   asset     close three-quarter view of that machine + motor.
 * The move lerps camera position and the controls' target each frame; any user input on the
 * controls cancels an in-flight move, and once it is done the controls are left alone.
 */
import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { CameraFocus } from "../sceneTypes";
import { assetPose, introPose, linePose, overviewPose, type CameraPose, type PlantLayout } from "./layout";

export interface CameraRigProps {
  layout: PlantLayout;
  focus: CameraFocus;
  focusNonce: number;
}

interface Flight {
  active: boolean;
  t0: number;
  dur: number;
  from: THREE.Vector3;
  to: THREE.Vector3;
  targetFrom: THREE.Vector3;
  targetTo: THREE.Vector3;
}

const OVERVIEW_S = 2.5;
const FOCUS_S = 1.4;

function poseFor(layout: PlantLayout, focus: CameraFocus): CameraPose | null {
  switch (focus.kind) {
    case "overview":
      return overviewPose(layout);
    case "line":
      return linePose(layout, focus.line);
    case "asset":
      return assetPose(layout, focus.assetId);
  }
}

export function CameraRig({ layout, focus, focusNonce }: CameraRigProps) {
  const controls = useRef<OrbitControlsImpl>(null);
  const camera = useThree((s) => s.camera);
  const flight = useMemo<Flight>(
    () => ({
      active: false,
      t0: 0,
      dur: 1,
      from: new THREE.Vector3(),
      to: new THREE.Vector3(),
      targetFrom: new THREE.Vector3(),
      targetTo: new THREE.Vector3(),
    }),
    [],
  );
  const clockRef = useRef(0);
  const mounted = useRef(false);

  // Opening move: park the camera high and far, then ease down to the overview.
  useEffect(() => {
    if (mounted.current) return;
    mounted.current = true;
    const intro = introPose(layout);
    camera.position.set(...intro.position);
    const c = controls.current;
    if (c) {
      c.target.set(...intro.target);
      c.update();
    }
  }, [camera, layout]);

  useEffect(() => {
    const pose = poseFor(layout, focus);
    if (!pose) return;
    const c = controls.current;
    flight.from.copy(camera.position);
    flight.targetFrom.copy(c ? c.target : new THREE.Vector3());
    flight.to.set(...pose.position);
    flight.targetTo.set(...pose.target);
    flight.dur = focus.kind === "overview" ? OVERVIEW_S : FOCUS_S;
    flight.t0 = clockRef.current;
    flight.active = true;
  }, [layout, focus, focusNonce, flight, camera]);

  // User input cancels the move so the rig never fights the engineer.
  useEffect(() => {
    const c = controls.current;
    if (!c) return;
    const cancel = () => {
      flight.active = false;
    };
    c.addEventListener("start", cancel);
    return () => c.removeEventListener("start", cancel);
  }, [flight]);

  useFrame(({ clock }) => {
    clockRef.current = clock.elapsedTime;
    const c = controls.current;
    if (!c) return;
    if (flight.active) {
      const t = Math.min(1, (clock.elapsedTime - flight.t0) / flight.dur);
      const e = 1 - Math.pow(1 - t, 3);
      camera.position.lerpVectors(flight.from, flight.to, e);
      c.target.lerpVectors(flight.targetFrom, flight.targetTo, e);
      if (t >= 1) flight.active = false;
    }
    c.update();
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={4}
      maxDistance={180}
      maxPolarAngle={Math.PI / 2 - 0.05}
      enablePan
      screenSpacePanning={false}
    />
  );
}
