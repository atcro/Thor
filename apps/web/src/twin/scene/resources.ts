/**
 * Geometries, materials and textures shared by every machine and motor in the scene.
 * Created once per palette and disposed when it changes, so the
 * per-frame path allocates nothing. Everything here is procedural: no fonts, HDRIs or models.
 */
import { createContext, useContext } from "react";
import * as THREE from "three";
import type { TwinStatus } from "../../api/types";
import type { GlyphShape } from "../status";
import type { TwinPalette } from "../palette";
import { STATUS_COLOR } from "../palette";

export interface SceneResources {
  segments: number;
  geo: {
    box: THREE.BoxGeometry;
    cylinder: THREE.CylinderGeometry;
    /** Six-sided roller so its rotation is visible. */
    roller: THREE.CylinderGeometry;
    disc: THREE.CylinderGeometry;
    pin: THREE.CylinderGeometry;
    torus: THREE.TorusGeometry;
    sphere: THREE.SphereGeometry;
    pulse: THREE.SphereGeometry;
    selectRing: THREE.RingGeometry;
    glyph: Record<GlyphShape, THREE.ShapeGeometry>;
    /** State rings: dash pattern per operating state so a load change is visible without colour. */
    regime: Record<"R1" | "R2" | "R3" | "unknown", THREE.BufferGeometry>;
  };
  mat: {
    machine: THREE.MeshLambertMaterial;
    machineDim: THREE.MeshLambertMaterial;
    machineEdge: THREE.MeshLambertMaterial;
    node: THREE.MeshLambertMaterial;
    nodeAccent: THREE.MeshBasicMaterial;
    primary: THREE.MeshBasicMaterial;
    steel: THREE.MeshLambertMaterial;
    glyph: Record<TwinStatus, THREE.MeshBasicMaterial>;
    regime: Record<"R1" | "R2" | "R3" | "unknown", THREE.MeshBasicMaterial>;
  };
  haloTexture: THREE.Texture;
  dispose: () => void;
}

/** Flat ring made of `dashes` arc segments (dashes = 1 -> solid ring). Lies in the XY plane. */
export function dashedRingGeometry(
  inner: number,
  outer: number,
  dashes: number,
  gapFraction: number,
  segmentsPerDash: number,
): THREE.BufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];
  const dashArc = (Math.PI * 2) / dashes;
  const fill = dashes === 1 ? 1 : 1 - gapFraction;
  for (let d = 0; d < dashes; d++) {
    const start = d * dashArc;
    const base = positions.length / 3;
    for (let s = 0; s <= segmentsPerDash; s++) {
      const a = start + (s / segmentsPerDash) * dashArc * fill;
      const c = Math.cos(a);
      const n = Math.sin(a);
      positions.push(c * inner, n * inner, 0, c * outer, n * outer, 0);
      if (s < segmentsPerDash) {
        const i = base + s * 2;
        indices.push(i, i + 1, i + 2, i + 1, i + 3, i + 2);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  g.setIndex(indices);
  g.computeVertexNormals();
  return g;
}

function glyphShape(shape: GlyphShape, r: number): THREE.Shape {
  const s = new THREE.Shape();
  switch (shape) {
    case "circle":
      s.absarc(0, 0, r, 0, Math.PI * 2, false);
      break;
    case "triangle":
      s.moveTo(0, r * 1.15);
      s.lineTo(-r, -r * 0.85);
      s.lineTo(r, -r * 0.85);
      s.closePath();
      break;
    case "octagon": {
      for (let i = 0; i < 8; i++) {
        const a = Math.PI / 8 + (i * Math.PI) / 4;
        const x = Math.cos(a) * r * 1.08;
        const y = Math.sin(a) * r * 1.08;
        if (i === 0) s.moveTo(x, y);
        else s.lineTo(x, y);
      }
      s.closePath();
      break;
    }
    case "square":
      s.moveTo(-r * 0.9, -r * 0.9);
      s.lineTo(r * 0.9, -r * 0.9);
      s.lineTo(r * 0.9, r * 0.9);
      s.lineTo(-r * 0.9, r * 0.9);
      s.closePath();
      break;
    case "ring": {
      s.absarc(0, 0, r, 0, Math.PI * 2, false);
      const hole = new THREE.Path();
      hole.absarc(0, 0, r * 0.58, 0, Math.PI * 2, true);
      s.holes.push(hole);
      break;
    }
    case "dash":
      s.moveTo(-r, -r * 0.22);
      s.lineTo(r, -r * 0.22);
      s.lineTo(r, r * 0.22);
      s.lineTo(-r, r * 0.22);
      s.closePath();
      break;
  }
  return s;
}

/** Radial-gradient sprite texture for the motor halo (procedural, 64 px). */
function haloTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, "rgba(255,255,255,0.9)");
    g.addColorStop(0.35, "rgba(255,255,255,0.35)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

export function createResources(palette: TwinPalette): SceneResources {
  const segments = 16;
  const lambert = (color: string, extra: Partial<THREE.MeshLambertMaterialParameters> = {}) =>
    new THREE.MeshLambertMaterial({ color, ...extra });
  const basic = (color: string, extra: Partial<THREE.MeshBasicMaterialParameters> = {}) =>
    new THREE.MeshBasicMaterial({ color, toneMapped: false, ...extra });

  const glyphGeo = {} as Record<GlyphShape, THREE.ShapeGeometry>;
  for (const shape of ["circle", "triangle", "octagon", "square", "ring", "dash"] as const) {
    glyphGeo[shape] = new THREE.ShapeGeometry(glyphShape(shape, 0.34), shape === "circle" || shape === "ring" ? segments : 1);
  }
  const glyphMat = {} as Record<TwinStatus, THREE.MeshBasicMaterial>;
  for (const st of Object.keys(STATUS_COLOR) as TwinStatus[]) {
    glyphMat[st] = basic(STATUS_COLOR[st], { side: THREE.DoubleSide });
  }

  const ringR = 1.15;
  const regimeGeo = {
    R1: dashedRingGeometry(ringR, ringR + 0.14, 4, 0.35, 14),
    R2: dashedRingGeometry(ringR, ringR + 0.14, 1, 0, 56),
    R3: dashedRingGeometry(ringR, ringR + 0.14, 12, 0.4, 5),
    unknown: dashedRingGeometry(ringR, ringR + 0.06, 2, 0.5, 24),
  };
  const regimeMat = {
    R1: basic(palette.regime.R1, { side: THREE.DoubleSide }),
    R2: basic(palette.regime.R2, { side: THREE.DoubleSide }),
    R3: basic(palette.regime.R3, { side: THREE.DoubleSide }),
    unknown: basic(palette.regime.unknown, { side: THREE.DoubleSide, transparent: true, opacity: 0.5 }),
  };

  const geo = {
    box: new THREE.BoxGeometry(1, 1, 1),
    cylinder: new THREE.CylinderGeometry(0.5, 0.5, 1, segments),
    roller: new THREE.CylinderGeometry(0.5, 0.5, 1, 6),
    disc: new THREE.CylinderGeometry(0.5, 0.5, 1, segments),
    pin: new THREE.CylinderGeometry(0.5, 0.5, 1, 6),
    torus: new THREE.TorusGeometry(1, 0.06, 6, 40),
    sphere: new THREE.SphereGeometry(0.5, segments, Math.max(4, segments / 2)),
    pulse: new THREE.SphereGeometry(0.5, 8, 6),
    selectRing: new THREE.RingGeometry(1.42, 1.52, 48),
    glyph: glyphGeo,
    regime: regimeGeo,
  };
  const mat = {
    machine: lambert(palette.machine),
    machineDim: lambert(palette.machine, { transparent: true, opacity: 0.45 }),
    machineEdge: lambert(palette.machineEdge, { emissive: palette.machineEdge, emissiveIntensity: 0.25 }),
    node: lambert(palette.node),
    nodeAccent: basic(palette.accent),
    primary: basic(palette.primary, { side: THREE.DoubleSide }),
    steel: lambert("#6b7a8c"),
    glyph: glyphMat,
    regime: regimeMat,
  };
  const halo = haloTexture();

  const dispose = () => {
    for (const g of Object.values(geo)) {
      if (g instanceof THREE.BufferGeometry) g.dispose();
      else for (const gg of Object.values(g)) gg.dispose();
    }
    for (const m of Object.values(mat)) {
      if (m instanceof THREE.Material) m.dispose();
      else for (const mm of Object.values(m)) mm.dispose();
    }
    halo.dispose();
  };

  return { segments, geo, mat, haloTexture: halo, dispose };
}

export const ResourcesContext = createContext<SceneResources | null>(null);

export function useResources(): SceneResources {
  const r = useContext(ResourcesContext);
  if (!r) throw new Error("useResources outside PlantScene");
  return r;
}

/** Visual meshes opt out of raycasting; one invisible hit box per asset does the picking. */
export const NO_RAYCAST = (): null => null;

/** Visual spin: 1500 rpm -> 0.8 rev/s so a roller's facets are readable, still ∝ rpm. */
export const RPM_TO_RAD_PER_S = (0.8 * Math.PI * 2) / 1500;
