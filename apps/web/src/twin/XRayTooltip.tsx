/**
 * X-Ray hover content: the status glyph and exactly three key facts, all from a TwinAsset
 * the API returned. Also exports StatusGlyph (an inline SVG shape per status) for the scene
 * labels and the Dashboard.
 */
import type { TwinAsset, TwinStatus } from "../api/types";
import { fmtPct } from "../lib/format";
import { DECISION_WORD, regimeWord, STATUS_GLYPH, type GlyphShape } from "./status";

function shapePath(shape: GlyphShape): JSX.Element {
  // All shapes are drawn in a 0..20 box and filled with the status colour.
  switch (shape) {
    case "circle":
      return <circle cx="10" cy="10" r="7.5" />;
    case "triangle":
      return <polygon points="10,2.5 18,17 2,17" />;
    case "octagon":
      return <polygon points="6.5,2 13.5,2 18,6.5 18,13.5 13.5,18 6.5,18 2,13.5 2,6.5" />;
    case "square":
      return <rect x="3" y="3" width="14" height="14" rx="1.5" />;
    case "ring":
      return <circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" strokeWidth="2.5" />;
    case "dash":
      return <rect x="3" y="8.5" width="14" height="3" rx="1" />;
  }
}

/** Status glyph: shape + colour from STATUS_GLYPH. Decorative (the status word is always text beside it). */
export function StatusGlyph({ status, size = 14, lit = false, className = "" }: { status: TwinStatus; size?: number; lit?: boolean; className?: string }) {
  const g = STATUS_GLYPH[status];
  return (
    <svg
      className={`xray-glyph${lit ? " lit" : ""}${className ? ` ${className}` : ""}`}
      width={size}
      height={size}
      viewBox="0 0 20 20"
      aria-hidden="true"
      focusable="false"
      style={{ color: g.color, fill: g.color }}
      data-status={status}
    >
      {shapePath(g.shape)}
    </svg>
  );
}

export function XRayTooltip({ asset }: { asset: TwinAsset }) {
  const g = STATUS_GLYPH[asset.status];
  const p = asset.failure_probability;
  return (
    <div className="xray-tip" data-testid={`xray-tip-${asset.asset.asset_id}`}>
      <div className="xray-tip-head">
        <StatusGlyph status={asset.status} size={16} lit />
        <span className="xray-id">{asset.asset.asset_id}</span>
        <span className="xray-status-word" style={{ color: g.color }}>
          {g.word}
        </span>
      </div>
      <div className="xray-tip-name">{asset.asset.name}</div>
      <dl className="xray-facts">
        <dt>p(fail)</dt>
        <dd className="xray-num">{p === null || p === undefined ? "no prediction yet" : fmtPct(p)}</dd>
        <dt>regime</dt>
        <dd>{regimeWord(asset.regime)}</dd>
        <dt>decision</dt>
        <dd>{DECISION_WORD[asset.decision_state]}</dd>
      </dl>
    </div>
  );
}
