# Live demo walkthrough (6 minutes, shown by hand)

Rehearsed on 2026-10-12 on the re-timed data (native stack at 900×, then the Docker stack after
`.\scripts\demo_reset.ps1`). Every timing below is measured, not estimated; the ones that depend
on the laptop's load (training) are given as the range seen. Only the plant data is simulated:
training, sign-off, scoring, analysis, Bolt's answers, approval and the repair really run.

The replay streams the recorded history at 900× (`REPLAY_SPEED`; one plant hour = 4 s). One pass
of the story is about ten minutes of wall clock, after which the recorded month replays with the
timeline shifted (the historical failures then happen "live" again; see "Later passes").

## Before the judges arrive (pre-demo checklist, 5 minutes)

1. **Reset:** `.\scripts\demo_reset.ps1` from `C:\Thor-main`. It wipes the data, restarts the
   stack, seeds the first 80 % of the history and stops: no model, every motor grey, the replay
   held. It ends by printing the replay state (`released=False`) and `24 motors, 24 without a
   prediction yet`. About 2 minutes.
2. **Health check:** the script prints `/system/health`; `db ok`, `mqtt true`, `edge true`. In the
   browser the top bar shows `API connected` and `Replay (recorded plant history) · held: no
   production model yet`. The Fleet page's System panel shows LLM `template mode` or the provider.
3. **Browser:** Chrome, http://localhost:5173, zoom 110 % on the 1080p projector (the cards and
   the HUD stay readable; the twin keeps its fps). Close other tabs.
4. **Twin performance mode:** open the 3-D Digital Twin once, press **Performance mode** if the
   `render` counter in the bottom-left badge shows under 45 fps on the venue's display, press
   **Overview**, then return to Fleet. The setting persists.
5. **Template mode if Wi-Fi drops:** nothing to do; if the provider is unreachable the API falls
   back to the deterministic template for the explanation and for Bolt (the chip says
   "template"). To force it on purpose, start the stack with the key blank in `.env`.
6. Do **not** train before the judges arrive: the model and the sign-off are the story.

## The 10 steps (click path, what to say, measured timings)

Timings are from the moment of the click. `t` is demo time.

### 1. Open on Fleet (t = 0:00)

Screen: **Fleet** (the landing view). 24 motors, every card grey with "no prediction yet", the
top bar says `held: no production model yet`.

Say: "Twenty-four motors on five packaging lines and a compressor hall. No model exists yet, so
Thor says so: grey, 'no prediction yet'. Nothing is scored by a rule of thumb."

### 2. Train (t = 0:10 → ~2:30)

Screen: **AutoML Studio** → **Train new model** (asset held out `MTR-042`, trials per family as
set in the form; the default is the measured one below).

Measured: `Train new model` → `Registered` in **90–150 s** natively while other work was running
(2 trials ≈ 90 s, 4 trials ≈ 150 s; the Docker rehearsal number is in the status log). The run
page updates every 3 s: the asset split appears within 2 s, the operating states within 9 s,
the three candidates at the end of the search, then the gate checklist and parity.

Talk track (honest): "The model learns from the fleet's own history: months of healthy running
in each operating state — startup, running, heavy load — plus the six motors that failed before.
It learned what normal looks like and what 'about to fail' looks like. We set the baseline now,
for when, not if, something goes wrong on the pack floor." Point at: the split (12 train / 3
validation / 3 calibration / 6 test motors; MTR-042 held out, never trained on), the three
families ranked by the Industrial Model Score, the gate (leakage, test failure events, recall,
lead time, edge parity), the final metrics from six untouched test motors.

### 3. Sign off (t ≈ 2:40)

Screen: **ModelOps** → the pending promotion request → type your name → **Approve promotion**.

Measured: the request returns in **5–11 s** (promotion, ONNX export, edge reload, fleet scored);
the top bar flips to `streaming` and Fleet turns green **within 6–12 s of the click** (24 green,
0 without a prediction). The replay starts by itself: `GET /replay` says `released` the moment
a production model exists.

Say: "A named person signs the model off. That signature is what releases the plant history:
the edge loads the model, every motor is scored, and green means within its healthy baseline —
a calibrated probability below the early-warning threshold."

### 4. Something goes wrong (t ≈ 3:40, no click)

Screen: **Fleet**. Watch MTR-042 (Line 3 filler).

Measured (three runs): green for **35–72 s** after the sign-off, then **red (critical, latched)**.
The calibrated probability steps from below 0.01 to 0.81 within three samples, so the amber
phase between the warning and the critical alarm lasts a few seconds at most and is sometimes
skipped (both alarms fire on the same sample). The card then reads `critical · latched since
<plant time>` and stays red: the status is latched until a completed work order clears it.

Say: "Here it is. Three consecutive samples over the line raised the alarm, and it stays red
until the plant acts — a motor does not talk its way out of an alarm."

### 5. Run analysis (t ≈ 4:00)

Screen: click the MTR-042 card → **Asset 360** → **Run analysis**.

Measured: **4–5 s** to `Awaiting approval` (model loaded, SHAP drivers, manual passages, cost
comparison, Decision Contract). The stepper shows `Load production model → Explaining →
Awaiting approval`; the probability on the contract is the card's.

Say: "This uses the model you just signed off — no retraining. SHAP names the drivers, the
manuals are searched for those symptoms, deterministic code compares maintain now, the next
low-load window and run to failure under the plant's own cost assumptions."

### 6. Bolt (t ≈ 4:10, while the analysis runs or right after)

Screen: **Copilot · Bolt**, asset context `MTR-042`. Ask, in this order:
- "What's happening with MTR-042?"
- "Give me the details."
- "What are the possible fixes?"

Measured: template mode answers in **< 1 s** each; in LLM mode a few seconds. Before the contract
exists the answers draw on the live readings, the latched status and the trend; after it, on the
contract's drivers, costs, manual passages and field-history precedent. The last answer points to
the approval on Asset 360; Bolt never approves.

Say: "Bolt reads the same evidence, adds precedent from real work orders, and has no write
tools."

### 7. Approve (t ≈ 5:00)

Screen: back to **Asset 360** (MTR-042), scroll to the Decision Contract, type a name, **Approve**.

Say: "In a real emergency the motor comes out within a few clicks; often you have to pivot.
The contract lays out the solution after the engineer has asked Bolt and read the evidence.
Thor stops here; it cannot schedule, dispatch or promote a model from this screen."

### 8. Back to green (t ≈ 5:05, no click)

Screen: **Fleet**.

Measured: `offline · in maintenance` **within 2–3 s** of the approval; back **green with the work
order completed 13–16 s** after the approval. The latched alarm clears on the completion event;
the rewind records the lead time of this run.

Say: "The plant took it offline for the planned three hours and brought it back on its healthy
baseline; the simulated CMMS executes at the approval so you can see it. The model registry did
not change."

### 9. 3-D Digital Twin (t ≈ 5:20)

Screen: **3-D Digital Twin** (last item in the side menu). Press **Overview**, turn on **X-Ray**,
hover MTR-042 (glyph + three facts), click it (HUD: status line, raw probability sparkline, state,
live sensors, drivers from the contract, decision and work order, provenance, PLC tags, the
simulated MES feed). Press **Rewind** → **Load stored predictions** → **early warning** /
**critical alarm** / **work order**: the bar shows **this run: first sustained warning → motor
taken offline** (24 h in the rehearsal) next to **validated on held-out motors** (53 h).

Say: "More than a 3-D fleet: each line in physical space, the live data path with measured
rates, and a rewind over stored predictions that makes the lead time visible — this run's next
to the validated figure."

### 10. Sandbox (t ≈ 5:45)

Screen: twin → **Sandbox**. The banner `SANDBOX — simulated readings` appears. Move the rpm or
temperature slider on the selected motor (the state ring follows; no alarm unless the model scores
it as risky), press **Troubleshoot <motor>** (red within about 15 s), **Random fault**, **Shake up
operations** (nothing breaks), **Reset sandbox**. Any sandbox motor that went red can be analysed
and approved like MTR-042 and comes back green; the contract and the work order are labelled
"sandbox scenario".

Say: "Sandbox readings take the real path — replay, MQTT, edge, control plane — and are tagged
all the way through; they never enter training."

## Recovery moves

- **Training fails or takes too long:** `.\scripts\demo_reset.ps1 -Prep` (or `docker compose run
  --rm prep` on the running stack) trains, validates, promotes as `demo-prep@thor` and scores the
  fleet in about a minute; continue at step 4.
- **WebGL fails:** the twin route falls back to Fleet with a notice; stay on Fleet and show the
  X-Ray HUD there (hover a card, click for the HUD).
- **A step stalls:** **Pause** in the top bar holds the replay where it is; **Resume** continues.
  While paused the twin shows motors as "no data" after 30 s — say so.
- **Nothing red after 2 minutes:** check the top bar says `streaming`; if it says `held`, the
  promotion was not approved (ModelOps); if `paused`, press Resume.
- **"Run analysis" returns 409:** no production model — approve the promotion in ModelOps.
- **Wi-Fi drops:** everything works in template mode; say so, it demonstrates the LLM is not
  load-bearing.

## Later passes

After about ten minutes the recorded month replays with the timeline shifted. The historical
failures (MTR-003/009/013/017/021/023) then degrade "live" and latch red in turn, and MTR-042's
degradation returns at the same point of the next pass; a repaired motor stays on its healthy
baseline. If the Q&A runs long, say "the replay loops the recorded month, so the history's
failures come around again" or press Pause.

## Numbers to quote (and their caveats)

From `docs/finals/metrics.md` (the 2026-10-12 run on the re-timed data; `docs/finals/prep-report.json`):
see that file for recall, precision, lead time and IMS. Caveats: synthetic fleet, six completed
failures, two failure events among the test motors, 10-minute aggregates (no spectra). Never state
a dollar saving as fact; the costs are configurable assumptions. Never quote the pre-2026-10-12
numbers (0.865 / 41.5 h / 0.844).
