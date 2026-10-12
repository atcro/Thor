import { render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReplayState, SystemHealth } from "../api/types";
import { Layout } from "./Layout";

const health: SystemHealth = {
  db: "ok",
  mqtt: true,
  mlflow: true,
  edge: true,
  n_telemetry_rows: 103680,
  last_ingest_ts: "2026-09-13T10:00:00Z",
};

const replay: ReplayState = {
  label: "Replay (recorded plant history)",
  released: true,
  paused: false,
  production_model_version: "v64634",
  plant_time: "2025-08-27T02:00:00Z",
  seeded_until: "2025-08-20T00:00:00Z",
  paused_at: null,
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("Layout", () => {
  beforeEach(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/system/health")) return jsonResponse(health);
      if (url.endsWith("/replay")) return jsonResponse(replay);
      return new Response("not found", { status: 404 });
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("lists the side menu in order with the twin directly under Bolt, and shows the replay indicator", async () => {
    render(
      <MemoryRouter initialEntries={["/assets/MTR-042"]}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/assets/:id" element={<div>asset page</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    const nav = screen.getByRole("navigation", { name: "Main" });
    const links = within(nav).getAllByRole("link");
    // the label span (the key hint and Bolt's avatar glyph are decoration)
    expect(links.map((l) => l.querySelector(".lbl")?.textContent?.replace("⚡", "").trim())).toEqual([
      "Fleet",
      "AutoML Studio",
      "ModelOps",
      "Copilot · Bolt",
      "3-D Digital Twin",
    ]);
    expect(links.map((l) => l.getAttribute("href"))).toEqual(["/", "/studio", "/modelops", "/copilot", "/twin"]);
    // Asset 360 keeps Fleet active
    expect(links[0]).toHaveClass("active");
    expect(screen.getByText("Asset 360")).toBeInTheDocument();
    // replay indicator from GET /replay: label, plant time and phase are API fields
    const replayEl = await screen.findByTestId("replay-indicator");
    expect(replayEl).toHaveTextContent("Replay (recorded plant history) · plant time 08-27 02:00 · streaming");
    expect(within(replayEl).getByRole("button", { name: "Pause" })).toBeEnabled();
    expect(within(replayEl).getByRole("button", { name: "Resume" })).toBeDisabled();
  });
});
