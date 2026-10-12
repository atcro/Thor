/**
 * Message-path pulses. Every frame the queue the screen filled is drained and each event
 * becomes one particle travelling the stored path:
 *   ingest      motor -> edge node -> control plane   (~1.2 s, palette.pulseIngest)
 *   prediction  edge node -> control plane            (~0.6 s, palette.pulsePrediction)
 * One InstancedMesh holds every live particle (ring buffer; oldest dropped over the cap).
 * `count` > 1 on an event makes the particle bigger instead of spawning more. With an empty
 * queue nothing travels: there is no idle animation. The per-frame path allocates nothing.
 */
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import type { MutableRefObject } from "react";
import * as THREE from "three";
import type { PulseEvent } from "../sceneTypes";
import type { TwinPalette } from "../palette";
import type { PlantLayout } from "./layout";
import { NO_RAYCAST, useResources } from "./resources";

export interface PulsesProps {
  layout: PlantLayout;
  queue: MutableRefObject<PulseEvent[]>;
  palette: TwinPalette;
  cap: number;
}

const INGEST_S = 1.2;
const PREDICTION_S = 0.6;
const KIND_INGEST = 0;
const KIND_PREDICTION = 1;

interface Pool {
  cap: number;
  head: number;
  n: number;
  t0: Float32Array;
  dur: Float32Array;
  scale: Float32Array;
  asset: Int32Array;
  kind: Uint8Array;
}

export function Pulses({ layout, queue, palette, cap }: PulsesProps) {
  const r = useResources();
  const mesh = useRef<THREE.InstancedMesh>(null);
  const pool = useMemo<Pool>(
    () => ({
      cap,
      head: 0,
      n: 0,
      t0: new Float32Array(cap),
      dur: new Float32Array(cap),
      scale: new Float32Array(cap),
      asset: new Int32Array(cap),
      kind: new Uint8Array(cap),
    }),
    [cap],
  );

  // Paths as flat arrays: per asset index, 3 waypoints (9 floats) + the two segment lengths.
  const paths = useMemo(() => {
    const ids = [...layout.assets.keys()];
    const index = new Map<string, number>();
    const pts = new Float32Array(ids.length * 9);
    const seg = new Float32Array(ids.length * 2);
    ids.forEach((id, i) => {
      index.set(id, i);
      const p = layout.assets.get(id)!.path;
      for (let k = 0; k < 3; k++) {
        pts[i * 9 + k * 3] = p[k][0];
        pts[i * 9 + k * 3 + 1] = p[k][1];
        pts[i * 9 + k * 3 + 2] = p[k][2];
      }
      seg[i * 2] = Math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]);
      seg[i * 2 + 1] = Math.hypot(p[2][0] - p[1][0], p[2][1] - p[1][1], p[2][2] - p[1][2]);
    });
    return { index, pts, seg };
  }, [layout]);

  const colors = useMemo(
    () => [new THREE.Color(palette.pulseIngest), new THREE.Color(palette.pulsePrediction)],
    [palette.pulseIngest, palette.pulsePrediction],
  );
  const material = useMemo(() => new THREE.MeshBasicMaterial({ toneMapped: false }), []);
  useEffect(() => () => material.dispose(), [material]);

  const tmpM = useMemo(() => new THREE.Matrix4(), []);
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpS = useMemo(() => new THREE.Vector3(), []);
  const q = useMemo(() => new THREE.Quaternion(), []);

  useFrame(({ clock }) => {
    const im = mesh.current;
    if (!im) return;
    const now = clock.elapsedTime;

    // 1. Drain the queue (splice everything the screen pushed since the last frame).
    const events = queue.current;
    if (events.length > 0) {
      const batch = events.splice(0, events.length);
      for (const ev of batch) {
        const ai = paths.index.get(ev.asset_id);
        if (ai === undefined) continue;
        // A control-plane score (Run analysis, /pipeline/score) never travelled the edge
        // path, so it draws no edge -> control plane pulse. Only edge predictions do.
        if (ev.kind === "prediction" && ev.source === "control_plane") continue;
        const slot = pool.head;
        pool.head = (pool.head + 1) % pool.cap;
        pool.n = Math.min(pool.cap, pool.n + 1);
        pool.t0[slot] = now;
        pool.kind[slot] = ev.kind === "prediction" ? KIND_PREDICTION : KIND_INGEST;
        pool.dur[slot] = ev.kind === "prediction" ? PREDICTION_S : INGEST_S;
        pool.scale[slot] = 0.22 * Math.min(2.2, 1 + 0.25 * Math.log2(Math.max(1, ev.count)));
        pool.asset[slot] = ai;
        im.setColorAt(slot, colors[pool.kind[slot]]);
      }
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
    }

    // 2. Write a matrix for every live particle, compacted into instances 0..j-1.
    let j = 0;
    for (let s = 0; s < pool.n; s++) {
      const age = now - pool.t0[s];
      if (age < 0 || age >= pool.dur[s]) continue;
      const u = age / pool.dur[s];
      const ai = pool.asset[s];
      const o = ai * 9;
      if (pool.kind[s] === KIND_INGEST) {
        const l0 = paths.seg[ai * 2];
        const l1 = paths.seg[ai * 2 + 1];
        const split = l0 / (l0 + l1);
        if (u < split) {
          const t = u / split;
          tmpP.set(
            paths.pts[o] + (paths.pts[o + 3] - paths.pts[o]) * t,
            paths.pts[o + 1] + (paths.pts[o + 4] - paths.pts[o + 1]) * t,
            paths.pts[o + 2] + (paths.pts[o + 5] - paths.pts[o + 2]) * t,
          );
        } else {
          const t = (u - split) / (1 - split);
          tmpP.set(
            paths.pts[o + 3] + (paths.pts[o + 6] - paths.pts[o + 3]) * t,
            paths.pts[o + 4] + (paths.pts[o + 7] - paths.pts[o + 4]) * t,
            paths.pts[o + 5] + (paths.pts[o + 8] - paths.pts[o + 5]) * t,
          );
        }
      } else {
        tmpP.set(
          paths.pts[o + 3] + (paths.pts[o + 6] - paths.pts[o + 3]) * u,
          paths.pts[o + 4] + (paths.pts[o + 7] - paths.pts[o + 4]) * u,
          paths.pts[o + 5] + (paths.pts[o + 8] - paths.pts[o + 5]) * u,
        );
      }
      const sc = pool.scale[s];
      tmpS.set(sc, sc, sc);
      tmpM.compose(tmpP, q, tmpS);
      im.setMatrixAt(j, tmpM);
      if (j !== s) im.setColorAt(j, colors[pool.kind[s]]);
      j++;
    }
    im.count = j;
    if (j > 0) {
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
    }
  });

  // Reset the pool when the layout or cap changes (paths no longer match).
  useEffect(() => {
    pool.n = 0;
    pool.head = 0;
    if (mesh.current) mesh.current.count = 0;
  }, [pool, paths]);

  return (
    <instancedMesh
      key={cap}
      ref={mesh}
      args={[r.geo.pulse, material, cap]}
      frustumCulled={false}
      raycast={NO_RAYCAST}
    />
  );
}
