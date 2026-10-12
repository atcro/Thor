/**
 * The 3-D Digital Twin's colour scheme, in ONE place.
 *
 * The rest of the app keeps its control-room theme (src/index.css). Only the twin scene, its
 * labels and the X-Ray HUD use this palette. Switch `TWIN_THEME` to "app" to render the twin
 * in the app's existing colours (a one-line change).
 *
 * Status colours are NOT part of the theme: healthy = green circle, warning = amber triangle,
 * critical = red stop-octagon (CLAUDE.md §15). The brand/neon yellow is never used for a status.
 */

export type TwinTheme = "neon" | "app";

/** One-line switch for the team's evaluation. */
export const TWIN_THEME: TwinTheme = "neon";

export interface TwinPalette {
  /** Scene background and fog. */
  bg: string;
  /** Floor plane and grid lines. */
  floor: string;
  grid: string;
  /** Primary chrome: outlines, selected highlights, HUD frame. */
  primary: string;
  /** Accent: data path, pulses, links, secondary highlights. */
  accent: string;
  /** Machine bodies (low-poly, matte). */
  machine: string;
  machineEdge: string;
  /** Motor bodies and the edge / control-plane nodes. */
  motor: string;
  node: string;
  /** Text on the HUD and labels. */
  ink: string;
  ink2: string;
  /** HUD panel surface and border. */
  panel: string;
  panelLine: string;
  /** Pulse colours by event kind. */
  pulseIngest: string;
  pulsePrediction: string;
  /** Regime ring colours: R1 idle/startup, R2 nominal, R3 high load, unknown. */
  regime: { R1: string; R2: string; R3: string; unknown: string };
  /** Glow intensity 0..1 (the "restrained" part). */
  glow: number;
}

const NEON: TwinPalette = {
  bg: "#070b12",
  floor: "#0c1420",
  grid: "#16304a",
  primary: "#e8ff47",
  accent: "#3d9bff",
  machine: "#1d2a3a",
  machineEdge: "#3d9bff",
  motor: "#2b3b50",
  node: "#13203a",
  ink: "#eef5ff",
  ink2: "#9fb3cc",
  panel: "rgba(7, 12, 22, 0.86)",
  panelLine: "rgba(61, 155, 255, 0.45)",
  pulseIngest: "#3d9bff",
  pulsePrediction: "#e8ff47",
  regime: { R1: "#7fd7ff", R2: "#3d9bff", R3: "#b48cff", unknown: "#4e5a66" },
  glow: 0.6,
};

/** The app's own tokens (src/index.css) so the twin can be switched back in one line. */
const APP: TwinPalette = {
  bg: "#0f141a",
  floor: "#161d25",
  grid: "#26313d",
  primary: "#e3793b",
  accent: "#84acc7",
  machine: "#1c2530",
  machineEdge: "#34424f",
  motor: "#222d3a",
  node: "#1c2530",
  ink: "#e6ebf0",
  ink2: "#aab5c0",
  panel: "rgba(22, 29, 37, 0.92)",
  panelLine: "#34424f",
  pulseIngest: "#84acc7",
  pulsePrediction: "#e3793b",
  regime: { R1: "#a9c7db", R2: "#84acc7", R3: "#f09a63", unknown: "#4e5a66" },
  glow: 0.35,
};

export const THEMES: Record<TwinTheme, TwinPalette> = { neon: NEON, app: APP };

export function twinPalette(theme: TwinTheme = TWIN_THEME): TwinPalette {
  return THEMES[theme];
}

/**
 * Status colours (shared by the scene glyphs, the HUD and the Dashboard X-Ray). Keyed by the
 * API's TwinStatus. Deliberately outside the theme: they mean the same thing everywhere.
 */
export const STATUS_COLOR = {
  healthy: "#2ee07a",
  warning: "#ff9f1c",
  critical: "#ff4d4d",
  no_prediction: "#8a97a8",
  offline: "#5fb8ff",
  no_data: "#55606d",
} as const;

/** CSS custom properties for the `.twin-theme` root (HUD, labels, toolbar). */
export function paletteCssVars(p: TwinPalette): Record<string, string> {
  return {
    "--tw-bg": p.bg,
    "--tw-primary": p.primary,
    "--tw-accent": p.accent,
    "--tw-ink": p.ink,
    "--tw-ink-2": p.ink2,
    "--tw-panel": p.panel,
    "--tw-line": p.panelLine,
    "--tw-ok": STATUS_COLOR.healthy,
    "--tw-warn": STATUS_COLOR.warning,
    "--tw-crit": STATUS_COLOR.critical,
    "--tw-none": STATUS_COLOR.no_prediction,
    "--tw-offline": STATUS_COLOR.offline,
    "--tw-nodata": STATUS_COLOR.no_data,
    "--tw-r1": p.regime.R1,
    "--tw-r2": p.regime.R2,
    "--tw-r3": p.regime.R3,
    "--tw-glow": String(p.glow),
  };
}
