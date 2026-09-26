import { NavLink, Outlet, useLocation } from "react-router-dom";
import { API_BASE_URL, getSystemHealth } from "../api/client";
import { usePolling } from "../hooks/usePolling";
import { fmtRel } from "../lib/format";
import { Dot } from "./ui";

const NAV: { to: string; label: string; key: string; end?: boolean }[] = [
  { to: "/", label: "Fleet", key: "1", end: true },
  { to: "/studio", label: "AutoML Studio", key: "2" },
  { to: "/modelops", label: "ModelOps", key: "3" },
  { to: "/copilot", label: "Copilot · Bolt", key: "4" },
];

function crumbFor(pathname: string): { section: string; detail?: string } {
  if (pathname.startsWith("/assets/")) {
    return { section: "Asset 360", detail: decodeURIComponent(pathname.split("/")[2] ?? "") };
  }
  if (pathname.startsWith("/studio")) return { section: "AutoML Studio" };
  if (pathname.startsWith("/modelops")) return { section: "ModelOps" };
  if (pathname.startsWith("/copilot")) return { section: "Copilot · Bolt" };
  return { section: "Fleet" };
}

export function Layout() {
  const { pathname } = useLocation();
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
                <img
                  src="/bolt-logo.png"
                  alt=""
                  className="bolt-avatar"
                  width={16}
                  height={16}
                  style={{ marginRight: 6 }}
                />
              )}
              {n.label}
            </span>
            <span className="k">{n.key}</span>
          </NavLink>
        ))}
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
          {health.data && (
            <span title="Telemetry rows in the control-plane store">
              {health.data.n_telemetry_rows.toLocaleString("en-US")} rows · ingest{" "}
              {fmtRel(health.data.last_ingest_ts)}
            </span>
          )}
          <span className="api" title={API_BASE_URL}>
            <Dot on={health.loading ? null : apiUp} pulse={health.loading} />
            {health.loading ? "connecting" : apiUp ? "API connected" : "API unreachable"}
          </span>
        </div>
      </header>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
