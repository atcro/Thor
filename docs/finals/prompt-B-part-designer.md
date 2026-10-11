# Prompt B — Bolt Designer + demo hardening (send Monday morning, Claude Fable 5.1, effort xhigh)

Before sending, make sure Prompt 0 and Prompt A are built, tested and committed. Read
`docs/finals/session-notes.md` and adjust anything below that those sessions changed. Then paste
everything below the line into a fresh Claude Code session started in `C:\Thor-main`.

---

We're a top-4 finalist in the ABB hackathon, presenting Thor in person on Tuesday: about 4 minutes
of slides and 6 minutes of live demo, then judge Q&A. Code freeze is **today at 5 pm**. After that,
the team only rehearses. Read `CLAUDE.md` (especially §7 and §15) and `docs/finals/session-notes.md`
from yesterday's sessions first. They are binding.

**Why this feature exists.** After an earlier round, a judge asked: "If this can monitor a motor,
could it be used to create a motor?" A second recurring question is whether Thor's AI ever makes
decisions. **Bolt Designer** answers both, without breaking our governance model:

- Deterministic code finds the signal in fleet spec and performance data and proposes an improved
  spec.
- Bolt explains the case in conversation.
- The engineer approves a Design Change Request.

The name is deliberate: Bolt is our copilot (a lightning bolt), and a bolt is also a part you design.

The line we want to be able to say on stage: *"Bolt explains your fleet. Bolt Designer shows you
how to build a better motor. You sign."*

**Your goal: Phase B from §15**, then make the whole demo rehearsal-ready by 5 pm.

## What "done" looks like

- **Nameplate specs.** Each motor carries a realistic nameplate record: rated kW, frame,
  poles/speed, insulation class, service factor, efficiency class, enclosure, and drive-end and
  non-drive-end bearing designations. Same-role motors across Lines 1–5 share identical nameplates,
  so the paperwork is the same while measured performance differs. That's the point of the story.
- **A deterministic design toolbox,** a new `agents/` module with docstrings stating inputs and
  outputs, per §11:
  - fetch specs on file;
  - compare lines and roles on measured operating stress and outcomes (load, starts, temperature
    rise, vibration, failures, time between failures). Use observable telemetry and recorded
    failures only, never simulator ground-truth fields;
  - derive a recommended spec envelope from the best performers and the stresses the weak lines
    actually see.
  - Each recommendation is tied to explicit engineering rules and thresholds kept in code. For
    example, a thermal margin argues for insulation class or service factor; starts and vibration
    argue for bearing selection; a load profile argues for the kW rating.
  - When the evidence says the cause is operational rather than the spec, it says so honestly.
    That's a credible answer, not a failure.
  - Support the packaging-line motors and the compressor-hall drive motors.
- **Design Change Request.**
  - It's created by the deterministic toolbox when the engineer clicks **Draft Design Change
    Request**, and stored insert-only.
  - It moves to approved only through `record_decision()` with a named approver. That decision is
    separate from maintenance approvals and model promotions.
  - Show it with the same evidence-linked rigor as a Decision Contract.
- **Bolt Designer screen,** its own item in the left navigation, with Bolt's look (same chat panel
  and persona; icon mixing a lightning bolt and a hex bolt) and the app's existing color scheme.
  - **Left: the Bolt chat,** guiding an engineering investigation. It must support, in template
    mode and with an LLM:
    1. "Pull up the specs for the motors on the pack lines."
    2. "Lines 3 and 4 keep having motor trouble while 1 and 2 run fine. Why?"
    3. "Propose a spec for the next motor purchase."
    4. "Draft the change request."
    - For step 4, Bolt points the engineer to the Draft button. It never creates the request
      itself.
  - **Right: a spec workbench** showing current vs recommended specs, each change with its
    evidence, and a chart of the line-to-line performance gap.
  - **Bolt stays strictly read-only:**
    - it reaches data only through read-only tools and quotes only numbers the tools returned;
    - it may suggest a draft, but drafting is the engineer's click and approval is the engineer's
      decision;
    - it asks a clarifying question when it would help, the way a senior reliability engineer would.
- **Fault hypothesis.** A small deterministic, rule-based layer answers "what's probably wrong?" for
  an analysed motor.
  - It produces one hypothesis: possible bearing degradation, possible thermal overload,
    vibration anomaly (unclassified), or insufficient evidence.
  - Each comes with a confidence level, supporting evidence, counter-evidence and limitations.
  - It uses only the aggregate features we actually have, with thresholds as named constants and
    unit tests.
  - Call it a hypothesis, never a confirmed root cause. Don't invent frequency-domain analysis:
    we have 10-minute aggregates, not waveforms.
  - Show it in the Decision Contract view and the HUD. It feeds Bolt Designer, so a bearing
    hypothesis points the redesign at the bearing and thermal specs.
- **Repair outcome.** When the simulated work order completes, record an insert-only outcome linked
  to the contract: action taken, parts replaced, the spec installed, and the confirmed failure mode
  the plant reports. Show it next to the original hypothesis ("predicted: possible bearing
  degradation, found: drive-end bearing damage"). This is how Thor would measure whether its calls
  were right. Don't add automatic retraining from outcomes.
- **A closed-loop hero story: detect → diagnose → redesign → install → plan.** This is the
  centerpiece of the demo:
  1. MTR-042 degrades live, and Run analysis drafts a Decision Contract recommending replacement.
  2. From that contract, **"Design the replacement in Bolt Designer"** opens Bolt Designer focused
     on MTR-042's role. The engineer approves a Design Change Request for an upgraded spec.
  3. Back on the contract, the engineer **approves maintenance with the approved upgraded spec as
     the replacement part.**
     - The maintenance approval record references the Design Change Request.
     - The Decision Contract itself stays immutable.
     - Like-for-like replacement remains the default when no Design Change Request is attached.
     - Design before approval, because at replay speed 1800 the work order can execute within
       seconds of approval.
  4. The simulated CMMS executes the work order. Its events record which spec was installed, and
     afterwards MTR-042's nameplate (in the API, the HUD and the 3-D Digital Twin) shows the
     upgraded spec, with an "Upgraded spec installed" marker.
  5. The HUD shows a **next planned inspection date** for the new motor (in plant time).
     - It comes from a deterministic bearing rated-life (L10) calculation under MTR-042's
       measured load and speed, comparing old and new bearing specs.
     - Use typical published catalog load ratings, stated as such in code.
     - Label it "rated-life estimate under measured duty". It is never a model prediction, because
       the model has no data on the new spec.
     - Live monitoring continues on the new motor as normal.
- **Don't fake the payoff.** After the repair, the replay streams the motor's normal healthy
  baseline exactly as before. Don't alter telemetry to show the upgraded motor outperforming. The
  claim is a longer rated life plus continued monitoring, and that's true.
- **Other entry points.** Bolt Designer is also reachable from a line's HUD in the 3-D Digital Twin
  and from its nav item.
- **Demo hardening.**
  - Write `docs/finals/demo-script.md`: the 6-minute click path with timings, what to say at each
    beat, a pre-demo checklist (prep and reset commands, browser zoom, performance mode,
    template-mode fallback if Wi-Fi drops), and recovery moves if a step stalls.
  - Write `docs/finals/judge-qa.md`: crisp answers to the likely questions:
    - AI decisions and governance;
    - the leakage we found and fixed, using `docs/finals/metrics.md`;
    - training-serving parity;
    - real data and integration with an MES/CMMS (never call Thor an MES);
    - scalability from asset to site to enterprise;
    - the AI4I benchmark;
    - cost assumptions (never claim savings as fact).
  - Run the full hero flow end to end yourself after a demo prep and reset, through the API, and fix
    anything that stalls or looks wrong.

## Constraints

- **§7 holds absolutely:** no LLM-computed numbers, keys read only in `apps/api/orchestrator.py`,
  insert-only records, approval only via `record_decision()`.
- **Keep Prompt 0's fixes:** ground-truth isolation, separate approvals, the validation gate,
  parity and analysis reuse. Don't regress them.
- **Data:** don't alter telemetry signals, `FAULTY_ASSETS` timing, or MTR-042's story.
- **Must keep passing:** `pytest`, `ruff check .`, the web build and tests, and
  `docker compose build`. Add tests for the toolbox rules, for the Design Change Request being
  insert-only and approval-gated, and for Bolt having no write path.
- **Time-box:** priority order is (1) the closed loop for MTR-042 with the fault hypothesis,
  (2) the repair outcome, (3) the general fleet-wide workbench. If Bolt Designer isn't solid by 2 pm, cut scope (fewer spec attributes, packaging motors
  only, MTR-042's role only) rather than ship something flaky. A working smaller feature beats a broken
  bigger one on stage. Stop building at 4:30 pm and use the rest for verification and the demo
  docs.
- **Scope:** stay within the task. No unrelated refactors, and nothing from §15's cut list.

## How to work

- **Commit after each working milestone** so any step can be rolled back.
- **Git:** commit only on the `finals` branch. Never push, never touch `main`, and never rewrite
  history. The team pushes themselves.
- **Testing budget.** Test where it protects the demo or a claim we make on stage, not everywhere.
  - After each change, run only the tests for the code you touched. Run the full `pytest`,
    `ruff` and web suite at milestones, not after every edit.
  - Run `docker compose build` once, at the end.
  - Write tests only for the invariants listed in this brief. No exhaustive edge cases, no
    snapshot tests, and no tests for styling.
- **Check claims, not just code.** Every progress or status claim must point to a tool result
  from this session. Near the end, one fresh-context sub-agent does two jobs in a single pass:
  - audit the "done" list against the code and the actual test output, flagging anything claimed
    but not shown;
  - play a skeptical ABB judge walking through the demo script, flagging any on-screen claim the
    data can't back.
  Fix what it flags before the final report.
- **Delegate independent work** to sub-agents and keep working while they run.
- **Keep the notes current.** Update `CLAUDE.md` (§5 screens, §6 agent table with the new toolbox,
  §15) and `docs/finals/session-notes.md` as facts change.
- **Progress log.** Keep `docs/finals/status.md` current; the team watches it to track progress
  against their usage limits. After every milestone, append an entry with the time, the phase and
  milestone, the files changed, the tests run with their actual results, and what's next. If
  this session stops early (usage limit or model switch), a new session must be able to resume
  from that file alone, so keep the "next" line specific.
- **Work autonomously.** The team is not watching in real time. For reversible actions within this
  brief, proceed without asking.
- **Docker:** you may run `docker compose build`. Leave starting, stopping and resetting the stack to
  the team: give them the commands.
- **End-to-end checks:** when you need a running system, run it natively from the repo's `.venv`
  (`uvicorn apps.api.main:app` on SQLite, plus the replay with `python -m streaming.replay.replay`)
  rather than starting the Docker stack. Stop those processes when you're done.
- **Final report.** Audit every claim against a tool result from this session. Lead with whether
  the hero flow works end to end, then what was cut and why, then the exact commands to prepare,
  reset and rehearse.
