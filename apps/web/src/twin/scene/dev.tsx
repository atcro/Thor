/**
 * DEV HARNESS — not part of the app. Mounts PlantScene full-screen with SCENE_FIXTURE and a
 * synthetic pulse feeder (~30 events/s) so the scene can be looked at without the API:
 *   npx vite --port 5174   ->   http://localhost:5174/src/twin/scene/dev.html
 * The real app never imports this file; the fixture events here are synthetic by design.
 */
import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { paletteCssVars, twinPalette } from "../palette";
import type { CameraFocus, PulseEvent, RewindFrame } from "../sceneTypes";
import { statusForProbability } from "../status";
import { SCENE_FIXTURE } from "./fixture";
import PlantScene from "./PlantScene";

const palette = twinPalette();

function Harness() {
  const queue = useRef<PulseEvent[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [focus, setFocus] = useState<CameraFocus>({ kind: "overview" });
  const [nonce, setNonce] = useState(0);
  const [xray, setXray] = useState(true);
  const [feed, setFeed] = useState(true);
  const [rewind, setRewind] = useState<RewindFrame | null>(null);
  const [fps, setFps] = useState(0);

  // Synthetic feeder: ~30 events/s spread over the fixture's live motors.
  useEffect(() => {
    if (!feed) return;
    const live = SCENE_FIXTURE.assets.filter((a) => a.data_state === "live").map((a) => a.asset.asset_id);
    let seq = 0;
    const id = window.setInterval(() => {
      const now = performance.now();
      const asset = live[Math.floor(Math.random() * live.length)];
      queue.current.push({ seq: seq++, kind: "ingest", source: null, asset_id: asset, count: 1, at: now });
      if (Math.random() < 0.9) {
        queue.current.push({ seq: seq++, kind: "prediction", source: "edge", asset_id: asset, count: 1, at: now });
      }
    }, 1000 / 15); // 15 ticks/s × ~1.9 events = ~28 events/s
    return () => window.clearInterval(id);
  }, [feed]);

  const toggleRewind = () => {
    if (rewind) return setRewind(null);
    const probabilities: Record<string, number | null> = {};
    const statuses: RewindFrame["statuses"] = {};
    for (const a of SCENE_FIXTURE.assets) {
      const p = a.asset.asset_id === "MTR-042" ? 0.36 : a.failure_probability === null ? null : 0.05;
      probabilities[a.asset.asset_id] = p;
      statuses[a.asset.asset_id] = statusForProbability(p, SCENE_FIXTURE.thresholds);
    }
    setRewind({ ts: "2025-09-21T02:00:00Z", probabilities, statuses });
  };

  const fly = (f: CameraFocus) => {
    setFocus(f);
    setNonce((n) => n + 1);
  };

  return (
    <div className="twin-theme" style={{ ...paletteCssVars(palette), position: "fixed", inset: 0, background: palette.bg }}>
      <PlantScene
        snapshot={SCENE_FIXTURE}
        rewind={rewind}
        pulseQueue={queue}
        selected={selected}
        hovered={hovered}
        onSelect={(id) => {
          setSelected(id);
          if (id) fly({ kind: "asset", assetId: id });
        }}
        onHover={setHovered}
        focus={focus}
        focusNonce={nonce}
        xray={xray}
        palette={palette}
        onFps={setFps}
      />
      <div
        id="dev-bar"
        style={{
          position: "absolute",
          top: 8,
          left: 8,
          display: "flex",
          gap: 6,
          alignItems: "center",
          fontFamily: "ui-monospace, Consolas, monospace",
          fontSize: 12,
          color: palette.ink,
          background: palette.panel,
          border: `1px solid ${palette.panelLine}`,
          padding: "6px 10px",
          borderRadius: 4,
        }}
      >
        <strong style={{ color: palette.primary }}>DEV HARNESS</strong>
        <span>fixture + synthetic pulses</span>
        <span id="fps" data-fps={fps}>fps {fps}</span>
        <button onClick={() => fly({ kind: "overview" })}>overview</button>
        <button onClick={() => fly({ kind: "line", line: "Packaging Line 3" })}>line 3</button>
        <button onClick={() => fly({ kind: "asset", assetId: "MTR-042" })}>MTR-042</button>
        <button onClick={() => setXray((v) => !v)}>xray {xray ? "on" : "off"}</button>
        <button onClick={() => setFeed((v) => !v)}>feed {feed ? "on" : "off"}</button>
        <button onClick={toggleRewind}>rewind {rewind ? "on" : "off"}</button>
        <span>sel {selected ?? "-"} hov {hovered ?? "-"}</span>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Harness />
  </StrictMode>,
);
