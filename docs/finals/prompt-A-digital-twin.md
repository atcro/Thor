# Prompt A — 3-D Digital Twin, X-Ray, Dashboard rename (send Sunday after Prompt 0, Claude Fable 5.1, effort xhigh)

Send this after Prompt 0 (correctness pass) has finished and its work is committed. Paste
everything below the line into a fresh Claude Code session started in `C:\Thor-main`.

---

We're a top-4 finalist in the ABB hackathon (Theme 1, Agentic Predictive Maintenance). On Tuesday we
present Thor in person to a judging panel: about 4 minutes of slides and 6 minutes of live demo, then
Q&A. The prototype works end to end today. What it lacks is the moment that makes a judge lean
forward. Other finalists also build with Claude, so what wins is polish and a coherent story, not
feature count. Read `CLAUDE.md` first, especially §7 (non-negotiable rules) and §15 (finals plan),
then `docs/finals/session-notes.md` from this morning's correctness pass. They are binding.

The correctness pass made a few things true that you must keep true:
- Simulator ground truth (`health`, `failure_within_h`) never reaches the UI.
- Motor health comes only from approved-model predictions, and a motor without one shows
  "no prediction yet".
- Maintenance approval and model promotion are separate decisions.
- "Run analysis" reuses the production model instead of retraining.

Don't regress any of it.

**Your goal for this session is Phase A from §15:** a 3-D Digital Twin view of the plant, an X-Ray
mode that turns any asset into a futuristic HUD readout, and a new color scheme tried inside the
3-D view only. When you finish, the demo must be more impressive than it is now and must still
work. Code freeze is Monday 5 pm, and a second session tomorrow builds Bolt Designer on top of your
work, so leave the codebase clean and documented.

## What "done" looks like

- **Plant layout.** The 24 existing motors are regrouped into Packaging Lines 1–5 (4 motors each)
  plus a Compressor Hall (4 motors). Packaging roles are infeed conveyor, filler, capper/labeler and
  case packer drives; compressor-hall roles are air-compressor drives. Change only asset metadata
  (name, line, role) in the simulator and seed. Telemetry signals and `FAULTY_ASSETS` timing stay
  exactly as they are, and MTR-042's live degradation remains the hero story.
  - Assign motors to lines from their actual telemetry character. Motors with heavier duty
    (load, starts, temperature) and the historical failures go together on Lines 3–4.
  - Tomorrow's Bolt Designer will investigate "why do Lines 3–4 struggle while 1–2 run well?" The
    answer must already be true in the data. Don't fake it.
  - Record the mapping and the reasoning in one place in code.
- **View names.** The plant has two views of the same live data, switchable from the main
  navigation, with exactly these names in the UI:
  - **Dashboard:** the existing isometric/pixel fleet view, renamed from "Fleet". Update routes,
    labels and tests accordingly.
  - **3-D Digital Twin:** the new view, and the default landing view.
- **3-D Digital Twin view: data-first, no decoration.** The judges are ABB engineers. The test is
  "can we prove every pixel?", not "does it look like a factory?". The twin should read as the
  system's architecture running live, with real wow but no oversell.
  - **What it shows:** each instrumented motor and the machine it drives, laid out as Packaging
    Lines 1–5 (infeed conveyor, filler, capper/labeler, case packer) plus the Compressor Hall. Add
    an edge node and the control plane as visible elements. Clean, low-poly, technical.
  - **What it leaves out:** buildings, vehicles, people, and any motion that doesn't come from data.
  - **Motion and status from real data:**
    - machine motion follows that motor's live rpm and stops when it's offline for a work order;
    - glow and status glyph come from the approved model's calibrated prediction, with the
      probability on hover;
    - a regime ring shows the inferred operating regime (startup, run, heavy load), so viewers can
      watch a load change shift the regime without raising an alarm. That's our regime-awareness
      differentiator, made visible;
    - a motor whose messages stop goes grey: "no data".
  - **The live data path:**
    - each time a telemetry message actually arrives, a pulse travels motor → edge node → control
      plane;
    - live counters show message rate and edge inference latency, measured, not assumed;
    - pulses are driven by real ingest and prediction events (stream them from the API, for
      example over a websocket or a short poll), never by an animation loop.
  - **Governance in place:** a badge on each machine shows its decision state (contract pending,
    approved, work order in progress, and later "upgraded spec installed"). A corner badge shows
    the production model's version, approver and parity status, from the registry.
  - **Rewind:** a time scrubber that replays stored predictions, so the team can show where
    MTR-042's early warning first fired, the motor's failure point, and the lead time between
    them. This makes our validation metric visible. It uses stored predictions only.
  - **Camera:** orbit, zoom, and a smooth fly-to when a line or asset is selected. Add a short
    "overview" camera move to open the demo on.
  - Molding and Batching appear as greyed-out site tabs, not built.
- **X-Ray toggle**, available in both the 3-D Digital Twin and the Dashboard.
  - Hover shows the asset's status glyph and three key facts.
  - Click opens a HUD panel styled like a sci-fi heads-up display. It shows:
    - calibrated failure probability and its trend;
    - the operating regime;
    - live sensor values;
    - top drivers if an analysis exists;
    - decision and work-order state;
    - model provenance;
    - an "Open in Asset 360" action.
  - **Plant data integration panels.** These show the kind of data a plant already captures, to
    demonstrate where Thor plugs in. They must be honest about what they are:
    - **PLC tags:** run bit, speed actual, motor current and similar, mapped one-to-one from the
      motor's real telemetry. Each tag shows its source signal on hover (for example
      "Speed_Act ← rpm"). Where a value has no telemetry source (for example a speed setpoint),
      derive it deterministically, mark it "simulated", and say from what.
    - **MES context (line-level HUD):** SKU, container type, quantity produced, unit of measure.
      Generated deterministically on the API side and labelled "simulated MES feed". It is
      context only: it feeds no model, score, recommendation or decision. Add a test asserting
      that.
  - Nothing is invented in the browser. All values come from the API.
- **Color scheme.**
  - The rest of the app keeps its current color scheme. Don't restyle it.
  - The 3-D Digital Twin scene and its HUD try a new scheme so the team can judge it: neon-yellow
    primary, electric-blue accents and restrained glows.
  - Keep the scene's palette in one place, so switching it to the app's existing colors is a
    one-line change if the team prefers that.
  - Status follows §15 everywhere: green circle, amber triangle, red stop-octagon. Never use the
    brand yellow to mean warning.
  - Keep text readable on a projector.

## Constraints

- **3D stack:** use three.js through React Three Fiber, plus helpers as needed. Pick versions
  compatible with the current React 18 setup, and don't upgrade React.
- **Performance:** it must hold at least 45 fps on Intel Iris Xe integrated graphics at 1080p.
  - Use instancing and low-poly procedural geometry, cap the device pixel ratio, and avoid
    expensive real-time shadows.
  - Any bloom or glow has a "performance mode" fallback.
  - If WebGL is unavailable, fall back to the Dashboard view.
- **Offline:** it must run fully offline. No CDN assets and no runtime downloads; everything is
  procedural or bundled.
- **Dependencies:** they must install through the normal `docker compose up --build`.
- **Data and LLM rules:** no number in the UI comes from the LLM or from frontend arithmetic posing
  as a measurement. No decorative motion: everything that moves is driven by data. The demo must
  run fully in LLM template mode.
- **Message rate:** at replay speed the message rate can be tens per second. Pulses may be
  visually aggregated to stay smooth, but the counters always show the true measured rate.
- **Must keep passing:** `docker compose build`, `pytest`, `ruff check .`, and
  `npm --prefix apps/web run build && npm --prefix apps/web run test`. Update tests the layout
  change breaks, and add a smoke test for the 3-D Digital Twin route.
- **Scope:** don't build Streamline swim lanes, Molding or Batching, Plant Personas, or Bolt
  Designer. Keep within the task; skip unrelated refactors.

## How to work

- **Commit after each working milestone,** in imperative mood. Any milestone can then be rolled
  back without losing the rest.
- **Testing budget.** Test where it protects the demo, not everywhere.
  - After each change, run only the tests for the code you touched. Run the full `pytest`,
    `ruff` and web suite at milestones, not after every edit.
  - Run `docker compose build` once, at the end.
  - Write tests only for what this brief asks for: the rename, the 3-D Digital Twin route smoke
    test, and the MES feed feeding no decision. No tests for 3D visuals or styling.
  - Measure frame rate once in a real browser at the end rather than assuming it.
- **Check claims, not just code.** Every progress or status claim must point to a tool result
  from this session. At the end, one fresh-context sub-agent checks the "done" list against the
  code and the actual test output, especially that every value shown in the 3-D view traces to
  an API field, and flags anything claimed but not shown. Fix what it flags before the final
  report.
- **Delegate independent work** (API endpoints for simulated tags, the Dashboard rename, the 3D
  scene) to sub-agents and keep working while they run.
- **Keep notes for tomorrow.** Update `CLAUDE.md` where your work changes its facts (screens,
  layout, commands). Keep a short `docs/finals/session-notes.md` of lessons and gotchas for
  tomorrow's session.
- **Progress log.** Keep `docs/finals/status.md` current; the team watches it to track progress
  against their usage limits. After every milestone, append an entry with the time, the phase and
  milestone, the files changed, the tests run with their actual results, and what's next. If
  this session stops early (usage limit or model switch), a new session must be able to resume
  from that file alone, so keep the "next" line specific.
- **Work autonomously.** The team is not watching in real time. For reversible actions within this
  brief, proceed without asking.
- **Docker:** you may run `docker compose build` to prove the images still build. Leave starting,
  stopping and resetting the stack to the team: give them the commands.
- **End-to-end checks:** when you need a running system, run it natively from the repo's `.venv`
  (`uvicorn apps.api.main:app` on SQLite, plus the replay with `python -m streaming.replay.replay`)
  rather than starting the Docker stack. Stop those processes when you're done.
- **Final report.** Audit every claim against a tool result from this session. Lead with what works,
  then what doesn't, with test output. Then give the exact commands the team runs to rebuild and
  view it.
