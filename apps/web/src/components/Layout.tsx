import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { API_BASE_URL, getReplay, getSystemHealth, pauseReplay, resumeReplay } from "../api/client";
import type { ReplayState } from "../api/types";
import { usePolling } from "../hooks/usePolling";
import { fmtDateTime, fmtShort } from "../lib/format";
import { KeySwitch } from "./KeySwitch";
import { Dot } from "./ui";

/** Side menu, in order. The 3-D Digital Twin sits directly under Bolt. */
export const NAV: { to: string; label: string; key: string; end?: boolean }[] = [
  { to: "/", label: "Fleet", key: "1", end: true },
  { to: "/studio", label: "AutoML Studio", key: "2" },
  { to: "/modelops", label: "ModelOps", key: "3" },
  { to: "/copilot", label: "Copilot · Bolt", key: "4" },
  { to: "/twin", label: "3-D Digital Twin", key: "5" },
];

function crumbFor(pathname: string): { section: string; detail?: string } {
  if (pathname.startsWith("/assets/")) {
    return { section: "Asset 360", detail: decodeURIComponent(pathname.split("/")[2] ?? "") };
  }
  if (pathname.startsWith("/twin")) return { section: "3-D Digital Twin" };
  if (pathname.startsWith("/studio")) return { section: "AutoML Studio" };
  if (pathname.startsWith("/modelops")) return { section: "ModelOps" };
  if (pathname.startsWith("/copilot")) return { section: "Copilot · Bolt" };
  return { section: "Fleet" };
}

/** One word for the replay's position: held until a production model exists, else streaming / paused. */
function replayPhase(r: ReplayState): string {
  if (!r.released) return "held: no production model yet";
  return r.paused ? "paused" : "streaming";
}

/**
 * Topbar replay indicator: GET /replay every 2 s (the label, plant time and phase are API
 * fields), with the presenter's Pause / Resume recovery switch (POST /replay/pause|resume).
 */
function ReplayIndicator() {
  const replay = usePolling(getReplay, 2000, "replay");
  const [busy, setBusy] = useState(false);
  const [override, setOverride] = useState<{ state: ReplayState; at: number } | null>(null);
  // A POST response is newer than the last poll: show it until a later poll lands.
  const r = override && (replay.lastUpdated ?? 0) < override.at ? override.state : (replay.data ?? override?.state ?? null);
  if (!r) return null;
  const act = async (fn: () => Promise<ReplayState>) => {
    setBusy(true);
    try {
      setOverride({ state: await fn(), at: Date.now() });
      void replay.refresh();
    } catch {
      /* the next poll shows the real state */
    } finally {
      setBusy(false);
    }
  };
  const held = !r.released;
  return (
    <span className="replay" data-testid="replay-indicator" title={r.seeded_until ? `Seeded history until ${fmtDateTime(r.seeded_until)}; the replay streams the rest.` : undefined}>
      <span>
        {r.label} · plant time {fmtShort(r.plant_time)} · {replayPhase(r)}
      </span>
      <button type="button" className="btn sm" disabled={busy || held || r.paused} onClick={() => void act(pauseReplay)} title="Hold the replay where it is (POST /replay/pause)">
        Pause
      </button>
      <button type="button" className="btn sm" disabled={busy || !r.paused} onClick={() => void act(resumeReplay)} title="Let the replay continue (POST /replay/resume)">
        Resume
      </button>
    </span>
  );
}

export function Layout() {
  const { pathname, state } = useLocation();
  const fallbackNote = (state as { twinFallback?: string } | null)?.twinFallback;
  const health = usePolling(getSystemHealth, 5000, "system");
  const apiUp = health.data !== null && !health.error;
  const crumb = crumbFor(pathname);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="wordmark">
          <span className="glyph">T</span>
          <div>
            <div className="name">THOR</div>
            <div className="sub">predictive maintenance</div>
          </div>
        </div>
        <nav aria-label="Main">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `nav-link${isActive || (n.to === "/" && pathname.startsWith("/assets/")) ? " active" : ""}`
              }
            >
              <span className="lbl">
                {n.to === "/copilot" && (
                  <span className="bolt-avatar" aria-hidden="true" style={{ marginRight: 6 }}>⚡</span>
                )}
                {n.label}
              </span>
              <span className="k">{n.key}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          LLM plans · code computes · engineer decides
        </div>
      </aside>

      <header className="topbar">
        <div className="crumb">
          <b>{crumb.section}</b>
          {crumb.detail && (
            <>
              {" / "}
              <span className="mono">{crumb.detail}</span>
            </>
          )}
        </div>
        <div className="conn">
          <KeySwitch />
          <ReplayIndicator />
          {health.data && (
            <span title="Telemetry rows in the control-plane store">
              {health.data.n_telemetry_rows.toLocaleString("en-US")} rows
            </span>
          )}
          <span className="api" title={API_BASE_URL}>
            <Dot on={health.loading ? null : apiUp} pulse={health.loading} />
            {health.loading ? "connecting" : apiUp ? "API connected" : "API unreachable"}
          </span>
        </div>
      </header>

      <main className="main">
        {fallbackNote && pathname === "/" && (
          <div className="pill warn" role="status" style={{ marginBottom: 12 }}>
            3-D Digital Twin unavailable: {fallbackNote}. Showing the Fleet instead.
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}
