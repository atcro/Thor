/**
 * 3-D Digital Twin (Phase A): the plant's architecture running live.
 *
 * Every value on this screen comes from the control plane: GET /twin (status, operating state,
 * sensors, decisions, model provenance), the /ws/twin event stream (pulses and measured
 * counters), GET /twin/rewind (stored predictions) and GET /sandbox (overrides). The scene
 * renders; it decides nothing.
 */
import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from "react";
import { Navigate } from "react-router-dom";
import type { CameraFocus, RewindFrame } from "../twin/sceneTypes";
import { twinPalette } from "../twin/palette";
import { DECISION_WORD, STATUS_EXPLAIN, STATUS_GLYPH, stateWord } from "../twin/status";
import { SANDBOX_DEFAULT_MOTOR, SandboxPanel, useSandboxState } from "../twin/SandboxPanel";
import { SiteTabs } from "../twin/SiteTabs";
import { TwinThemeRoot } from "../twin/TwinThemeRoot";
import { useTwinLive } from "../twin/useTwinLive";
import { useXRay } from "../twin/useXRay";
import { XRayToggle } from "../twin/XRayToggle";
import { XRayHud } from "../twin/XRayHud";
import { StatusGlyph, XRayTooltip } from "../twin/XRayTooltip";
import { RewindBar } from "../twin/RewindBar";
import { supportsWebGL } from "../twin/webgl";
import { fmtNum, fmtPct, fmtShort } from "../lib/format";
import "../twin/twin.css";

// three.js is only loaded for this screen.
const PlantScene = lazy(() => import("../twin/scene/PlantScene"));

const PERF_KEY = "thor.twin.perf";

function readPerf(): boolean {
  try {
    return localStorage.getItem(PERF_KEY) === "1";
  } catch {
    return false;
  }
}

function savePerf(on: boolean): void {
  try {
    localStorage.setItem(PERF_KEY, on ? "1" : "0");
  } catch {
    /* private mode: the toggle still works for this page view */
  }
}

function ProvenanceBadge({ twin }: { twin: ReturnType<typeof useTwinLive>["snapshot"] }) {
  const m = twin?.model ?? null;
  return (
    <div className="twin-badge tr" data-testid="twin-provenance">
      <div className="t">Production model · registry</div>
      {m ? (
        <div className="kv">
          <span className="k">model</span>
          <span className="v">{m.name} {m.version}</span>
          <span className="k">family</span>
          <span className="v">{m.family}</span>
          <span className="k">approved by</span>
          <span className="v">{m.approver ?? "—"}</span>
          <span className="k">approved at</span>
          <span className="v">{m.approved_at ? fmtShort(m.approved_at) : "—"}</span>
          <span className="k">edge parity</span>
          <span className={`v ${m.parity_passed ? "twin-ok" : m.parity_passed === false ? "twin-bad" : "twin-dim"}`}>
            {m.parity_passed === null
              ? "not recorded"
              : m.parity_passed
                ? `passed · max |Δp| ${m.parity_max_abs_diff === null ? "—" : fmtNum(m.parity_max_abs_diff, 4)}`
                : "FAILED"}
          </span>
          <span className="k">IMS</span>
          <span className="v">{fmtNum(m.ims_total, 3)}</span>
          <span className="k">validated lead time</span>
          <span className="v">{m.lead_time_median_h === null ? "—" : `${fmtNum(m.lead_time_median_h, 1)} h`}</span>
        </div>
      ) : (
        <div className="twin-dim">no production model — predictions are off until a promotion is approved in ModelOps</div>
      )}
    </div>
  );
}

function CountersBadge({
  live,
  transport,
  eventsSeen,
  fps,
  plantTime,
}: {
  live: ReturnType<typeof useTwinLive>["live"];
  transport: string;
  eventsSeen: number;
  fps: number | null;
  plantTime: string | null | undefined;
}) {
  return (
    <div className="twin-badge bl" data-testid="twin-counters">
      <div className="t">Live data path · measured at the control plane</div>
      <div className="kv">
        <span className="k">messages</span>
        <span className="v">{live ? `${fmtNum(live.msgs_per_s, 1)} /s` : "—"}</span>
        <span className="k">predictions (edge + control plane)</span>
        <span className="v">{live ? `${fmtNum(live.preds_per_s, 1)} /s` : "—"}</span>
        <span className="k">edge inference p50 / p95</span>
        <span className="v">
          {live && live.edge_inference_ms_p50 !== null
            ? `${fmtNum(live.edge_inference_ms_p50, 1)} / ${fmtNum(live.edge_inference_ms_p95, 1)} ms`
            : "no samples yet"}
        </span>
        <span className="k">motor → edge → control plane (edge-scored)</span>
        <span className="v">
          {live && live.path_latency_ms_p50 !== null
            ? `${fmtNum(live.path_latency_ms_p50, 0)} / ${fmtNum(live.path_latency_ms_p95, 0)} ms`
            : "no samples yet"}
        </span>
        <span className="k">latency samples</span>
        <span className="v">{live ? live.n_latency_samples : "—"}</span>
        <span className="k">events this session</span>
        <span className="v">{eventsSeen}</span>
        <span className="k">stream</span>
        <span className="v">{transport === "ws" ? "websocket" : transport === "poll" ? "poll 500 ms" : "connecting"}</span>
        <span className="k">mqtt / edge</span>
        <span className="v">
          {live ? (
            <>
              <span className={live.mqtt_connected ? "twin-ok" : "twin-bad"}>{live.mqtt_connected ? "up" : "down"}</span>
              {" / "}
              <span className={live.edge_up ? "twin-ok" : "twin-bad"}>{live.edge_up ? "up" : "down"}</span>
            </>
          ) : (
            "—"
          )}
        </span>
        <span className="k">plant time</span>
        <span className="v">{fmtShort(plantTime)}</span>
        <span className="k">render</span>
        <span className="v">{fps === null ? "—" : `${Math.round(fps)} fps`}</span>
      </div>
    </div>
  );
}

function Legend({ palette }: { palette: ReturnType<typeof twinPalette> }) {
  return (
    <div className="twin-badge br" data-testid="twin-legend">
      <div className="t">Encoding</div>
      <div className="twin-legend">
        {(["healthy", "warning", "critical", "no_prediction", "offline", "no_data"] as const).map((s) => (
          <span className="row" key={s} title={STATUS_EXPLAIN[s]}>
            <StatusGlyph status={s} size={12} />
            {s === "healthy" ? STATUS_EXPLAIN.healthy : STATUS_GLYPH[s].word}
          </span>
        ))}
        <span className="row" style={{ color: palette.regime.R1 }}>
          <span className="twin-ring" /> <span style={{ color: "var(--tw-ink)" }}>ring: state {stateWord("R1")}</span>
        </span>
        <span className="row" style={{ color: palette.regime.R2 }}>
          <span className="twin-ring" /> <span style={{ color: "var(--tw-ink)" }}>ring: state {stateWord("R2")}</span>
        </span>
        <span className="row" style={{ color: palette.regime.R3 }}>
          <span className="twin-ring" /> <span style={{ color: "var(--tw-ink)" }}>ring: state {stateWord("R3")}</span>
        </span>
        <span className="row" style={{ color: "var(--tw-warn)" }}>
          ■ <span style={{ color: "var(--tw-ink)" }}>tag: sandbox scenario (simulated readings)</span>
        </span>
        <span className="row" style={{ color: palette.pulseIngest }}>
          ● <span style={{ color: "var(--tw-ink)" }}>pulse: telemetry message</span>
        </span>
        <span className="row" style={{ color: palette.pulsePrediction }}>
          ● <span style={{ color: "var(--tw-ink)" }}>pulse: edge prediction</span>
        </span>
      </div>
    </div>
  );
}

export function Twin() {
  const webgl = useMemo(() => supportsWebGL(), []);
  if (!webgl) {
    return <Navigate to="/" replace state={{ twinFallback: "WebGL is not available on this device" }} />;
  }
  return <TwinView />;
}

function TwinView() {
  const palette = useMemo(() => twinPalette(), []);
  const twin = useTwinLive(true);
  const xray = useXRay();
  const [perf, setPerf] = useState<boolean>(readPerf);
  const [fps, setFps] = useState<number | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [focus, setFocus] = useState<CameraFocus>({ kind: "overview" });
  const [focusNonce, setFocusNonce] = useState(0);
  const [rewind, setRewind] = useState<RewindFrame | null>(null);
  const [showRewind, setShowRewind] = useState(false);
  const [showSandbox, setShowSandbox] = useState(false);
  const sandbox = useSandboxState(showSandbox);
  const sandboxOn = showSandbox || Boolean(sandbox.state?.active);

  const assetsById = useMemo(() => {
    const m = new Map<string, NonNullable<typeof twin.snapshot>["assets"][number]>();
    for (const a of twin.snapshot?.assets ?? []) m.set(a.asset.asset_id, a);
    return m;
  }, [twin.snapshot]);

  const goTo = useCallback((f: CameraFocus) => {
    setFocus(f);
    setFocusNonce((n) => n + 1);
  }, []);

  const onSelect = useCallback(
    (id: string | null) => {
      setSelected(id);
      if (id) {
        goTo({ kind: "asset", assetId: id });
        if (xray.enabled) xray.openAsset(id);
      }
    },
    [goTo, xray],
  );

  // Keyboard: Esc clears the selection; O returns to the overview.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
      if ((e.key === "o" || e.key === "O") && !(e.target instanceof HTMLInputElement)) goTo({ kind: "overview" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goTo]);

  const hoveredAsset = hovered ? assetsById.get(hovered) ?? null : null;
  const selectedAsset = selected ? assetsById.get(selected) ?? null : null;
  const lines = twin.snapshot?.lines ?? [];

  return (
    <TwinThemeRoot className="twin-page">
      <div className="twin-toolbar">
        <h1>3-D Digital Twin</h1>
        <SiteTabs tabs={twin.snapshot?.sites} />
        <div className="row" style={{ gap: 4 }} role="group" aria-label="Fly to">
          <button type="button" className="twin-btn" onClick={() => goTo({ kind: "overview" })} title="Overview camera move (O)">
            Overview
          </button>
          {lines.map((l) => (
            <button
              key={l.line}
              type="button"
              className={`twin-btn${focus.kind === "line" && focus.line === l.line ? " on" : ""}`}
              onClick={() => goTo({ kind: "line", line: l.line })}
            >
              {l.kind === "compressor" ? "Compressors" : l.line.replace("Packaging ", "")}
            </button>
          ))}
        </div>
        <div className="spacer" />
        <XRayToggle />
        <button
          type="button"
          className={`twin-btn${showSandbox ? " on" : ""}`}
          aria-pressed={showSandbox}
          title="Sandbox: override one motor's recorded readings, inject a fault, shake up operations or reset (simulated readings, tagged all the way through)"
          onClick={() => setShowSandbox((v) => !v)}
        >
          Sandbox
        </button>
        <button
          type="button"
          className={`twin-btn${showRewind ? " on" : ""}`}
          aria-pressed={showRewind}
          onClick={() => {
            setShowRewind((v) => !v);
            if (showRewind) setRewind(null);
          }}
        >
          Rewind
        </button>
        <button
          type="button"
          className={`twin-btn${perf ? " on" : ""}`}
          aria-pressed={perf}
          title="Performance mode: no halos, fewer labels, device pixel ratio 1"
          onClick={() => {
            setPerf((v) => {
              savePerf(!v);
              return !v;
            });
          }}
        >
          Performance mode
        </button>
      </div>

      <div className={`twin-stage${sandboxOn ? " sandbox" : ""}`}>
        {sandboxOn && (
          <div className="twin-sandbox-banner" role="status" data-testid="sandbox-banner">
            SANDBOX — simulated readings
          </div>
        )}
        <div className="twin-canvas">
          <Suspense fallback={<div className="twin-fallback twin-dim">Loading the scene…</div>}>
            <PlantScene
              snapshot={twin.snapshot}
              rewind={rewind}
              pulseQueue={twin.pulseQueue}
              selected={selected}
              hovered={hovered}
              onSelect={onSelect}
              onHover={setHovered}
              focus={focus}
              focusNonce={focusNonce}
              xray={xray.enabled}
              performanceMode={perf}
              palette={palette}
              onFps={setFps}
            />
          </Suspense>
        </div>

        {twin.snapshotError && !twin.snapshot && (
          <div className="twin-badge tl twin-bad" role="alert">
            Control plane unreachable: {twin.snapshotError}
          </div>
        )}

        {showSandbox && (
          <SandboxPanel
            assetId={selected ?? SANDBOX_DEFAULT_MOTOR}
            asset={assetsById.get(selected ?? SANDBOX_DEFAULT_MOTOR) ?? null}
            state={sandbox.state}
            onState={sandbox.setState}
          />
        )}

        {xray.enabled && hoveredAsset && (
          <div className="twin-hover" data-testid="twin-hover">
            <XRayTooltip asset={hoveredAsset} />
          </div>
        )}
        {!xray.enabled && selectedAsset && (
          <div className="twin-hover" data-testid="twin-selected">
            <div className="row" style={{ gap: 8 }}>
              <StatusGlyph status={selectedAsset.status} size={14} />
              <b>{selectedAsset.asset.asset_id}</b> · {selectedAsset.asset.name}
            </div>
            <div className="twin-dim" style={{ fontSize: 12 }}>
              {STATUS_GLYPH[selectedAsset.status].word}
              {selectedAsset.alarm === "critical" ? ` · latched since ${fmtShort(selectedAsset.alarm_since)}` : ""} · p(fail){" "}
              {selectedAsset.failure_probability === null ? "—" : fmtPct(selectedAsset.failure_probability)} · state:{" "}
              {stateWord(selectedAsset.regime)} · Decision: {DECISION_WORD[selectedAsset.decision_state]}
              {selectedAsset.sandbox || selectedAsset.alarm_sandbox ? " · sandbox scenario" : ""} · turn on X-Ray for the HUD
            </div>
          </div>
        )}

        <ProvenanceBadge twin={twin.snapshot} />
        <CountersBadge
          live={twin.live}
          transport={twin.transport}
          eventsSeen={twin.eventsSeen}
          fps={fps}
          plantTime={rewind ? rewind.ts : twin.snapshot?.plant_time}
        />
        <Legend palette={palette} />

        {rewind && (
          <div className="twin-status-line" data-testid="twin-rewind-state">
            REWIND · stored predictions at {fmtShort(rewind.ts)} · machines frozen
          </div>
        )}

        {showRewind && (
          <div className="twin-rewind with-badges">
            <RewindBar
              snapshotAssets={twin.snapshot?.assets ?? []}
              focusAssetId={selected ?? "MTR-042"}
              onFrame={setRewind}
            />
          </div>
        )}

        {xray.openedAsset && (
          <div className="twin-drawer" data-testid="twin-drawer">
            <XRayHud assetId={xray.openedAsset} onClose={xray.close} />
          </div>
        )}
      </div>
    </TwinThemeRoot>
  );
}
