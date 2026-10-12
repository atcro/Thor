/**
 * Rewind scrubber: replays STORED predictions (GET /twin/rewind) over plant time so the lead
 * time between MTR-042's early warning and its failure point is visible. Nothing here is a
 * live prediction and nothing is computed from telemetry: for every scrub position the bar
 * looks up, per asset, the last stored probability at or before that time (a step function)
 * and maps it to a status with the thresholds the API sent (statusForProbability).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getTwinRewind } from "../api/client";
import type { RewindMarker, RewindMarkerKind, RewindSeries, TwinAsset, TwinRewind, TwinStatus } from "../api/types";
import { fmtDateTime, fmtNum, fmtShort } from "../lib/format";
import type { RewindFrame } from "./sceneTypes";
import { statusForProbability } from "./status";

const DEFAULT_FOCUS = "MTR-042";
/** Play advances the scrub position by one plant-hour every 100 ms. */
const PLAY_STEP_MS = 3_600_000;
const PLAY_TICK_MS = 100;
/** Slider resolution: one plant-minute. */
const SLIDER_STEP_MS = 60_000;

interface SeriesMs {
  asset_id: string;
  ts: number[];
  p: number[];
}

function toMs(iso: string): number {
  return new Date(iso).getTime();
}

/** Index of the last point with ts <= t (binary search), or -1 when none is stored yet. */
function lastAtOrBefore(ts: number[], t: number): number {
  let lo = 0;
  let hi = ts.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

/** The frame the scene renders at scrub time `t` (ms): stored probability + status per asset. */
export function frameAt(t: number, series: SeriesMs[], assetIds: string[], rewind: TwinRewind): RewindFrame {
  const probabilities: Record<string, number | null> = {};
  const statuses: Record<string, TwinStatus> = {};
  for (const id of assetIds) {
    probabilities[id] = null;
    statuses[id] = "no_prediction";
  }
  for (const s of series) {
    const i = lastAtOrBefore(s.ts, t);
    const p = i >= 0 ? s.p[i] : null;
    probabilities[s.asset_id] = p;
    statuses[s.asset_id] = statusForProbability(p, rewind.thresholds);
  }
  return { ts: new Date(t).toISOString(), probabilities, statuses };
}

function pickFocus(rewind: TwinRewind, focusAssetId: string | null): RewindSeries | null {
  const want = focusAssetId ?? DEFAULT_FOCUS;
  return (
    rewind.assets.find((s) => s.asset_id === want) ??
    rewind.assets.find((s) => s.asset_id === DEFAULT_FOCUS) ??
    rewind.assets[0] ??
    null
  );
}

function markerOf(series: RewindSeries | null, kind: RewindMarkerKind): RewindMarker | null {
  return series?.markers.find((m) => m.kind === kind) ?? null;
}

export function RewindBar({
  snapshotAssets,
  focusAssetId,
  onFrame,
}: {
  snapshotAssets: TwinAsset[];
  focusAssetId: string | null;
  onFrame: (frame: RewindFrame | null) => void;
}) {
  const [rewind, setRewind] = useState<TwinRewind | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [t, setT] = useState<number | null>(null);
  const [live, setLive] = useState(true);
  const [playing, setPlaying] = useState(false);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const series = useMemo<SeriesMs[]>(
    () =>
      (rewind?.assets ?? []).map((s) => ({
        asset_id: s.asset_id,
        ts: s.points.map(([ts]) => toMs(ts)),
        p: s.points.map(([, p]) => p),
      })),
    [rewind],
  );
  const assetIds = useMemo(() => snapshotAssets.map((a) => a.asset.asset_id), [snapshotAssets]);
  const startMs = rewind?.start ? toMs(rewind.start) : null;
  const endMs = rewind?.end ? toMs(rewind.end) : null;
  const span = startMs !== null && endMs !== null && endMs > startMs ? endMs - startMs : null;

  const emit = useCallback(
    (ms: number, r: TwinRewind) => {
      setT(ms);
      setLive(false);
      onFrameRef.current(frameAt(ms, series, assetIds, r));
    },
    [series, assetIds],
  );

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getTwinRewind(400);
      setRewind(r);
      if (r.end) {
        // Start at the end of the stored range so the scene immediately shows "REWIND".
        const ms = toMs(r.end);
        setT(ms);
        setLive(false);
        onFrameRef.current(
          frameAt(
            ms,
            r.assets.map((s) => ({ asset_id: s.asset_id, ts: s.points.map(([ts]) => toMs(ts)), p: s.points.map(([, p]) => p) })),
            assetIds,
            r,
          ),
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const goLive = useCallback(() => {
    setPlaying(false);
    setLive(true);
    onFrameRef.current(null);
  }, []);

  // Play/pause: this is a scrub of STORED data at a fixed 1 plant-hour per 100 ms. It is not
  // decorative motion: every frame comes from the stored prediction rows at that plant time.
  useEffect(() => {
    if (!playing || !rewind || t === null || endMs === null) return;
    const id = window.setInterval(() => {
      setT((cur) => {
        if (cur === null) return cur;
        const next = Math.min(endMs, cur + PLAY_STEP_MS);
        onFrameRef.current(frameAt(next, series, assetIds, rewind));
        if (next >= endMs) setPlaying(false);
        return next;
      });
    }, PLAY_TICK_MS);
    return () => window.clearInterval(id);
    // `t` is intentionally not a dependency: the interval reads the current value via setT.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, rewind, endMs, series, assetIds]);

  // Leaving the bar returns the scene to live data.
  useEffect(() => () => onFrameRef.current(null), []);

  const focus = rewind ? pickFocus(rewind, focusAssetId) : null;
  const early = markerOf(focus, "early_warning");
  const failure = markerOf(focus, "predicted_failure");
  const workOrder = markerOf(focus, "work_order_started");

  if (!rewind) {
    return (
      <div className="xray-frame rewind-bar" data-testid="rewind-bar">
        <div className="rewind-row">
          <button type="button" className="xray-btn primary" onClick={() => void load()} disabled={loading}>
            {loading ? "Loading stored predictions…" : "Load stored predictions"}
          </button>
          <span className="xray-dim">Rewind replays stored predictions only; no telemetry is re-scored.</span>
          {error && (
            <span className="xray-bad" role="alert">
              {error}
            </span>
          )}
        </div>
      </div>
    );
  }

  if (startMs === null || endMs === null || span === null || t === null) {
    return (
      <div className="xray-frame rewind-bar" data-testid="rewind-bar">
        <div className="rewind-row xray-dim">no stored predictions yet — approve a promotion in ModelOps and let the replay run</div>
      </div>
    );
  }

  const pct = (ms: number) => `${(((ms - startMs) / span) * 100).toFixed(2)}%`;
  const jump = (m: RewindMarker | null) => {
    if (!m) return;
    setPlaying(false);
    emit(Math.max(startMs, Math.min(endMs, toMs(m.ts))), rewind);
  };

  return (
    <div className="xray-frame rewind-bar" data-testid="rewind-bar">
      <div className="rewind-row">
        <span className="xray-mod-t" style={{ margin: 0, flex: "none" }}>
          Rewind
        </span>
        <button type="button" className={`xray-btn${live ? " on" : ""}`} onClick={goLive} aria-pressed={live}>
          Live
        </button>
        <button
          type="button"
          className={`xray-btn${playing ? " on" : ""}`}
          aria-pressed={playing}
          onClick={() => {
            if (playing) {
              setPlaying(false);
              return;
            }
            // Replaying from the end restarts at the beginning of the stored range.
            const from = t >= endMs ? startMs : t;
            emit(from, rewind);
            setPlaying(true);
          }}
          title="Scrub through the stored predictions at 1 plant-hour per 100 ms"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <span className="rewind-time" title={fmtDateTime(new Date(t).toISOString())} data-testid="rewind-time">
          {live ? "live" : fmtShort(new Date(t).toISOString())}
        </span>
        <span className="spacer" />
        <span className="xray-dim">{focus ? `markers: ${focus.asset_id}` : "no marker series"}</span>
        <button type="button" className="xray-btn" disabled={!early} onClick={() => jump(early)} title={early?.detail}>
          early warning
        </button>
        <button type="button" className="xray-btn" disabled={!failure} onClick={() => jump(failure)} title={failure?.detail}>
          first ≥ {fmtNum(rewind.thresholds.classification, 1)}
        </button>
        <button type="button" className="xray-btn" disabled={!workOrder} onClick={() => jump(workOrder)} title={workOrder?.detail}>
          work order
        </button>
      </div>
      <div className="rewind-track">
        {focus?.markers.map((m) => {
          const ms = toMs(m.ts);
          if (ms < startMs || ms > endMs) return null;
          const frac = (ms - startMs) / span;
          return (
            <span
              key={`${m.kind}-${m.ts}`}
              className={`rewind-tick ${m.kind}${frac > 0.8 ? " flip" : ""}`}
              style={{ left: pct(ms) }}
              title={`${m.label} · ${fmtDateTime(m.ts)} · ${m.detail}`}
            >
              <span>{m.label}</span>
            </span>
          );
        })}
        <input
          type="range"
          min={startMs}
          max={endMs}
          step={SLIDER_STEP_MS}
          value={t}
          aria-label="Plant time"
          aria-valuetext={fmtDateTime(new Date(t).toISOString())}
          onChange={(e) => {
            setPlaying(false);
            emit(Number(e.target.value), rewind);
          }}
        />
      </div>
      <div className="rewind-legend">
        <span>stored predictions only</span>
        <span>
          {fmtShort(rewind.start)} → {fmtShort(rewind.end)} plant time
        </span>
        {focus && (
          <span title={focus.lead_time_basis}>
            {focus.asset_id} lead time <b>{focus.lead_time_h === null ? "—" : `${fmtNum(focus.lead_time_h, 1)} h`}</b> · {focus.lead_time_basis}
          </span>
        )}
        <span title={rewind.validated_lead_time_basis}>
          validated lead time <b>{rewind.validated_lead_time_h === null ? "—" : `${fmtNum(rewind.validated_lead_time_h, 1)} h`}</b> ·{" "}
          {rewind.validated_lead_time_basis}
        </span>
      </div>
    </div>
  );
}
