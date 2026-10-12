/**
 * Wording guard: the person-visible word is "state", never "regime" (the API field keeps its
 * name). Renders the Fleet, the Twin (scene mocked) and the HUD from fixtures and fails if the
 * rendered text contains "regime".
 */
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FleetAsset, SandboxState, SystemHealth, TwinEventBatch, TwinSnapshot } from "./api/types";
import { Fleet } from "./screens/Fleet";
import { SCENE_FIXTURE } from "./twin/scene/fixture";
import { hudFixture } from "./twin/testFixtures";
import { XRayProvider } from "./twin/useXRay";
import { XRayHud } from "./twin/XRayHud";

vi.mock("./twin/scene/PlantScene", () => ({
  default: ({ snapshot }: { snapshot: TwinSnapshot | null }) => (
    <div data-testid="plant-scene">{snapshot?.lines.map((l) => l.line).join(" | ") ?? "no snapshot"}</div>
  ),
}));
vi.mock("./twin/webgl", () => ({ supportsWebGL: () => true }));

import { Twin } from "./screens/Twin";

/** GET /fleet derived from the scene fixture (same assets, FleetAsset shape). */
const fleet: FleetAsset[] = SCENE_FIXTURE.assets.map((a) => ({
  asset: a.asset,
  health_score: a.health_score,
  failure_probability: a.failure_probability,
  prediction_ts: a.prediction_ts,
  model_version: a.model_version,
  regime: a.regime,
  last_ts: a.last_ts,
  open_contract_id: a.open_contract_id,
  stage: "production",
  work_order: a.work_order,
  status: a.status,
  alarm: a.alarm,
  alarm_since: a.alarm_since,
  alarm_reason: a.alarm_reason,
  sandbox: a.sandbox,
}));

const health: SystemHealth = { db: "ok", mqtt: true, mlflow: true, edge: true, n_telemetry_rows: 1000, last_ingest_ts: SCENE_FIXTURE.plant_time };
const batch: TwinEventBatch = { since: 0, next: 0, events: [], dropped: 0, live: SCENE_FIXTURE.live };
const sandbox: SandboxState = { active: false, overrides: [], retired_prediction_max_id: 0, last_reset_at: null, events: [] };

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("visible wording", () => {
  beforeEach(() => {
    vi.stubGlobal("WebSocket", undefined);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/system/health")) return jsonResponse(health);
      if (url.includes("/fleet")) return jsonResponse(fleet);
      if (url.includes("/twin/hud/")) return jsonResponse(hudFixture);
      if (url.includes("/twin/events")) return jsonResponse(batch);
      if (url.endsWith("/twin")) return jsonResponse(SCENE_FIXTURE);
      if (url.endsWith("/sandbox")) return jsonResponse(sandbox);
      return new Response("not found", { status: 404 });
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('never shows the word "regime" on the Fleet, the Twin or the HUD', async () => {
    const { container, unmount } = render(
      <MemoryRouter>
        <XRayProvider>
          <Fleet />
        </XRayProvider>
      </MemoryRouter>,
    );
    await screen.findByText("MTR-042");
    expect(container.textContent).toMatch(/State: running/);
    expect(container.textContent).not.toMatch(/regime/i);
    unmount();

    const twin = render(
      <MemoryRouter initialEntries={["/twin"]}>
        <XRayProvider>
          <Routes>
            <Route path="/twin" element={<Twin />} />
          </Routes>
        </XRayProvider>
      </MemoryRouter>,
    );
    await screen.findByTestId("plant-scene");
    await screen.findByTestId("twin-legend");
    expect(twin.container.textContent).toMatch(/ring: state heavy load/);
    expect(twin.container.textContent).not.toMatch(/regime/i);
    twin.unmount();

    const hud = render(
      <MemoryRouter>
        <XRayHud assetId="MTR-042" onClose={() => undefined} />
      </MemoryRouter>,
    );
    await screen.findByTestId("hud-pfail");
    expect(hud.container.textContent).toMatch(/State/);
    expect(hud.container.textContent).not.toMatch(/regime/i);
  });
});
