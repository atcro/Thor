import type { CSSProperties } from "react";
import type { FleetAsset } from "../api/types";
import { fmtPct } from "../lib/format";

/** Two drawing styles the user can switch between; both use the same data mapping. */
export type SpriteStyle = "iso" | "pixel";

/**
 * Risk tier from the current failure probability. This is the only thing that
 * picks a colour, and every tier is also encoded by motion and by a text tag so
 * the screen works without colour (deuteranopia / protanopia safe by default).
 *
 *   low    p < 0.15   green outline, gears turning normally
 *   medium p < 0.40   amber outline, pulsates
 *   high   p >= 0.40  vermillion outline, bearing glows and smokes
 *   none   no prediction yet: neutral outline, gears turn if a regime is known
 */
export type RiskTier = "low" | "medium" | "high" | "none";

export function riskTier(p: number | null | undefined): RiskTier {
  if (p === null || p === undefined || Number.isNaN(p)) return "none";
  if (p >= 0.4) return "high";
  if (p >= 0.15) return "medium";
  return "low";
}

export const TIER_WORD: Record<RiskTier, string> = {
  low: "low risk",
  medium: "medium risk",
  high: "high risk",
  none: "no prediction",
};

/**
 * Data-driven visual parameters for one motor sprite.
 * Every value here is derived from numbers the API already computed — the
 * sprite only renders them, it never decides anything.
 */
export interface SpriteParams {
  tier: RiskTier;
  /** Seconds per fan revolution; null when the regime is unknown (motor drawn stopped). */
  spinSec: number | null;
  /** 0..1 heat at the drive-end bearing, from failure probability (high tier only). */
  heat: number;
  /** Vibration amplitude in px; 0 unless the tier is high. */
  shake: number;
  /** An open Decision Contract shows a beacon over the drive end. */
  beacon: boolean;
  /** At-risk flag, same rule the card border used before tiers existed. */
  atRisk: boolean;
  label: string;
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

/** Fan period per regime (R1 idle ~600 rpm, R2 nominal ~1480, R3 high load ~1500). */
const SPIN_SEC: Record<string, number> = { R1: 1.5, R2: 0.6, R3: 0.5 };

const REGIME_WORD: Record<string, string> = {
  R1: "idle/startup",
  R2: "nominal",
  R3: "high load",
};

export function spriteParams(item: FleetAsset): SpriteParams {
  const p = item.failure_probability;
  const tier = riskTier(p);
  const atRisk = (p ?? 0) >= 0.4 || item.health_score < 60;
  const heat = tier === "high" ? clamp01(0.5 + ((p as number) - 0.4) / 1.2) : 0;
  // A motor the plant has taken offline for an approved work order stands still: no spin, no
  // shake, whatever the (stale) last prediction says.
  const offline = item.work_order?.status === "in_progress";
  const shake = tier === "high" && !offline ? 0.5 + heat * 1.3 : 0;
  const spinSec = offline ? null : item.regime ? (SPIN_SEC[item.regime] ?? 0.6) : null;
  const regimeWord = offline
    ? "offline for maintenance"
    : item.regime
      ? (REGIME_WORD[item.regime] ?? item.regime)
      : "regime unknown";
  const pWord = p === null || p === undefined ? "no prediction" : `p(fail) ${fmtPct(p)}`;
  return {
    tier,
    spinSec,
    heat,
    shake,
    beacon: Boolean(item.open_contract_id),
    atRisk,
    label: `${item.asset.asset_id}: ${TIER_WORD[tier]}, ${regimeWord}, ${pWord}${item.open_contract_id ? ", contract open" : ""}`,
  };
}

// ---------------------------------------------------------------------------
// Isometric style: flat-shaded vector, crisp edges
// ---------------------------------------------------------------------------

function IsoMotor({ s }: { s: SpriteParams }) {
  return (
    <>
      <polygon className="floor" points="30,58 150,58 138,70 18,70" />
      <polygon className="floor-edge" points="18,70 138,70 138,72 18,72" />

      <g className="mech">
        <rect className="dark" x="52" y="52" width="12" height="6" />
        <rect className="dark" x="90" y="52" width="12" height="6" />

        <rect className="body" x="46" y="22" width="62" height="30" />
        <rect className="top" x="46" y="22" width="62" height="6" />
        <rect className="dark" x="46" y="46" width="62" height="6" />
        {Array.from({ length: 9 }, (_, i) => (
          <rect key={i} className="fin" x={52 + i * 6} y="28" width="2" height="18" />
        ))}

        <rect className="top" x="66" y="12" width="22" height="10" />
        <rect className="dark" x="66" y="20" width="22" height="2" />
        <rect className="lamp" x="70" y="15" width="4" height="4" />

        <ellipse className="dark" cx="46" cy="37" rx="8" ry="16" />
        <g transform="translate(46 37) scale(0.5 1) translate(-46 -37)">
          <g className="fan">
            <rect className="blade" x="45" y="24" width="2" height="26" />
            <rect className="blade" x="33" y="36" width="26" height="2" />
            <rect className="blade" x="45" y="24" width="2" height="26" transform="rotate(45 46 37)" />
            <rect className="blade" x="45" y="24" width="2" height="26" transform="rotate(-45 46 37)" />
          </g>
          <circle className="hub" cx="46" cy="37" r="3" />
        </g>

        <rect className="top" x="108" y="30" width="8" height="14" />
        <rect className="shaft" x="116" y="35" width="18" height="4" />
        <line className="shaft-dash" x1="116" y1="37" x2="134" y2="37" />
        <rect className="dark" x="134" y="31" width="10" height="12" />
        <rect className="top" x="134" y="31" width="10" height="2" />

        <circle className="halo" cx="112" cy="37" r="11" />
        <circle className="hot" cx="112" cy="37" r="5" />
      </g>

      {s.tier === "high" && (
        <g className="smoke">
          <circle className="puff" cx="112" cy="28" r="3" />
          <circle className="puff" cx="115" cy="28" r="4" />
          <circle className="puff" cx="109" cy="28" r="3.5" />
        </g>
      )}

      {s.beacon && (
        <g className="beacon">
          <rect x="106" y="8" width="12" height="7" />
          <rect x="111" y="15" width="2" height="9" />
        </g>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Pixel style: 40x14 bitmap, one <rect> per lit cell
// ---------------------------------------------------------------------------

/**
 * Legend: t top highlight · b body · f fin · d dark · k terminal box · L lamp ·
 * o bearing housing · s shaft · c coupling · g floor · G floor edge · . empty.
 * The fan interior (rows 5-9, cols 5-9) and shaft stripes are drawn by the
 * two animation frames below, not here.
 */
const PIX_BASE = [
  "........................................",
  "..............ttttt.....................",
  "..............kkkkk.....................",
  "..............kLkkk.....................",
  ".....ddddd.tttttttttttttttttttoo........",
  "....d.....dbfbfbfbfbfbfbfbfbfboo........",
  "....d.....dbfbfbfbfbfbfbfbfbfboosssssscc",
  "....d.....dbfbfbfbfbfbfbfbfbfboosssssscc",
  "....d.....dbfbfbfbfbfbfbfbfbfboo......cc",
  "....d.....dbfbfbfbfbfbfbfbfbfboo........",
  ".....ddddd.dddddddddddddddddddd.........",
  "...........ddd..............ddd.........",
  "..gggggggggggggggggggggggggggggggggggg..",
  ".GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG.",
];

/** Frame A: plus-shaped blades, shaft stripes on even cells. */
const PIX_FRAME_A = [
  [7, 5], [7, 6], [7, 8], [7, 9], [5, 7], [6, 7], [8, 7], [9, 7],
  [33, 6], [35, 6], [37, 6], [34, 7], [36, 7],
];
/** Frame B: cross-shaped blades, shaft stripes shifted one cell. */
const PIX_FRAME_B = [
  [5, 5], [6, 6], [8, 8], [9, 9], [9, 5], [8, 6], [6, 8], [5, 9],
  [34, 6], [36, 6], [33, 7], [35, 7], [37, 7],
];

const PIX_CLASS: Record<string, string> = {
  t: "top",
  b: "body",
  f: "fin",
  d: "dark",
  k: "top",
  L: "lamp",
  o: "top",
  s: "shaft",
  c: "dark",
  g: "floor",
  G: "floor-edge",
};

function PixelMotor({ s }: { s: SpriteParams }) {
  const cells: JSX.Element[] = [];
  PIX_BASE.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const cls = PIX_CLASS[row[x]];
      if (cls) cells.push(<rect key={`${x}-${y}`} className={cls} x={x} y={y} width="1" height="1" />);
    }
  });
  return (
    <>
      <g className="mech">
        {cells}
        <g className="frame-a">
          {PIX_FRAME_A.map(([x, y]) => (
            <rect key={`a${x}-${y}`} className={x < 20 ? "blade" : "shaft-stripe"} x={x} y={y} width="1" height="1" />
          ))}
        </g>
        <g className="frame-b">
          {PIX_FRAME_B.map(([x, y]) => (
            <rect key={`b${x}-${y}`} className={x < 20 ? "blade" : "shaft-stripe"} x={x} y={y} width="1" height="1" />
          ))}
        </g>
        <rect className="hub" x="7" y="7" width="1" height="1" />
        {/* drive-end bearing heat */}
        <rect className="halo" x="29" y="5" width="4" height="5" />
        <rect className="hot" x="30" y="6" width="2" height="3" />
      </g>

      {s.tier === "high" && (
        <g className="smoke">
          <rect className="puff" x="30" y="3" width="1" height="1" />
          <rect className="puff" x="31" y="3" width="2" height="1" />
          <rect className="puff" x="29" y="3" width="1" height="1" />
        </g>
      )}

      {s.beacon && (
        <g className="beacon">
          <rect x="29" y="0" width="4" height="2" />
          <rect x="30" y="2" width="1" height="2" />
        </g>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------

/**
 * Motor sprite for one fleet asset. Animation is CSS-only and keyed off custom
 * properties set from `spriteParams`; the tier class drives colour, and the
 * same tier is exposed as text in the label so colour is never the only cue.
 */
export function MotorSprite({ item, style = "iso" }: { item: FleetAsset; style?: SpriteStyle }) {
  const s = spriteParams(item);
  const cssVars = {
    "--spin": s.spinSec === null ? "0s" : `${s.spinSec}s`,
    "--heat": s.heat,
    "--shake": `${s.shake}px`,
  } as CSSProperties;
  const cls = [
    "motor",
    style,
    `tier-${s.tier}`,
    s.spinSec === null ? "stopped" : "",
    s.shake > 0 ? "vibrating" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <svg
      className={cls}
      style={cssVars}
      viewBox={style === "pixel" ? "0 0 40 14" : "18 4 132 68"}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={s.label}
      data-risk={s.atRisk ? "1" : "0"}
      data-tier={s.tier}
      shapeRendering="crispEdges"
    >
      {style === "pixel" ? <PixelMotor s={s} /> : <IsoMotor s={s} />}
    </svg>
  );
}
