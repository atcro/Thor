# Live demo walkthrough (finals: two scenarios, shown by hand)

Phase A.3 (2026-10-12) adds the beats after step 8 (the finding, Bolt's recall, the label
ledger) and Scenario 2 (API Key Up / Down). The header of every screen shows the switch; it
starts **API Key Down** and stays Down for the whole of Scenario 1. See "Phase A.3 beats" below
for the click path and the measured timings.

Rehearsed on 2026-10-12 on the re-timed data (native stack at 900×, then the Docker stack after
`.\scripts\demo_reset.ps1`). Every timing below is measured, not estimated; the ones that depend
on the laptop's load (training) are given as the range seen. Only the plant data is simulated:
training, sign-off, scoring, analysis, Bolt's answers, approval and the repair really run.

The replay streams the recorded history at 900× (`REPLAY_SPEED`; one plant hour = 4 s). One pass
of the story (the six days after the seed point) is about ten minutes of wall clock, after which
the same six days play again with plant time continuing (see "Later passes").

## Dry run, as built (the hero path in 11 clicks)

Run from `C:\Thor-main` in PowerShell. Docker Desktop must be running.

```
.\scripts\demo_reset.ps1        # 1-2 min: wipes data, restarts the stack, seeds, holds the replay
```

It ends with `production model: none`, `replay: released=False`, `fleet: 24 motors, 24 without a
prediction yet`. Open http://localhost:5173 (Chrome). Then, in this order:

| # | Click | What you should see | Say |
|---|---|---|---|
| 1 | Nothing; the app opens on **Fleet** | 24 grey cards, "no prediction yet"; top bar `held: no production model yet` | "No model exists yet, so Thor says so." |
| 2 | **AutoML Studio** → **Train new model** (asset `MTR-042`, trials `4`) | A run appears; split within 2 s, states within 9 s, candidates, gate, parity; stage `Registered` in **60–75 s** | "Learns from the fleet's own history: months of healthy running in each state plus the six motors that failed before. It learned what normal looks like and what 'about to fail' looks like." |
| 3 | **ModelOps** → type your name → **Approve promotion** | Request turns `approved`; top bar flips to `streaming` | "A named person signs it off; that releases the plant history." |
| 4 | **Fleet** | 24 green within 3 s ("within healthy baseline"); **~60–70 s later MTR-042 turns red**, `latched since …`. No amber: the probability steps up within one sample. | "Green means within its healthy baseline. Here it is — three consecutive samples over the line, and it stays red until the plant acts." |
| 5 | Click the **MTR-042** card → **Run analysis** | Stepper to `Awaiting approval` in 4–5 s (template) / 15–45 s (LLM); contract panel below with drivers, costs, window, citations | "Reuses the model you just signed off; SHAP, manuals, cost comparison." |
| 6 | **Copilot · Bolt**, asset context `MTR-042`, ask: "What's happening with MTR-042?" then "Give me the details." then "What are the possible fixes?" | < 1 s each (template) / 7–30 s each (LLM); the last answer points to the approval on Asset 360 | "Reads the same evidence, adds precedent from real work orders, no write tools." |
| 7 | **Fleet** → MTR-042 card → scroll to the Decision Contract → name → **Approve** | Panel locks; approval row appears | "The engineer signs after asking Bolt and reading the evidence. Thor stops here." |
| 8 | **Fleet** | `offline · in maintenance` within 3 s, green with the work order completed within ~15 s | "The plant took it offline and brought it back on its healthy baseline. The registry did not change." |
| 9 | **3-D Digital Twin** (last menu item) → **Overview** → **X-Ray** → **Line 3** → click MTR-042's label → **Open in Asset 360** or close | Lines in physical space, live counters bottom-left, model provenance top-right; the HUD with probability and status line, state and sensors, top drivers, decision (PLC tags and MES feed folded under **Plant integration**) | "Point at any pixel and we can trace it to its source." |
| 10 | (optional, the team may drop it) **Rewind** → **Load stored predictions** → **critical alarm** / **work order** | Two figures side by side: this run's lead time and the validated 53.1 h | |
| 11 | **Asset 360** (MTR-042) → **What did the technician find?** → *Confirmed: bearing damage*, a note, the technician's name → **Record finding** | The row appears at once; Fleet's summary line reads `Alarm trust · Alarms acted on: 1 · confirmed: 1 · …`; the MTR-042 card reads `technician: Confirmed: bearing damage · <name>` | "The plant tells Thor what it found. A person signs it; it is stored insert-only." |
| 12 | **Copilot · Bolt**, asset `MTR-042`: "Have we seen this before?" | < 1 s (template): *This plant* first — the finding, its work order id and the name — then the public cases labelled "not from this plant" | "Bolt remembers this plant's repairs and keeps them apart from public precedent." |
| 13 | **AutoML Studio** → the run → **Label ledger** (under the split) | Six simulator-history failures and `MTR-042 · queued` for the next training run | "The next training run learns from it. We do not retrain on stage." |
| 14 | Scenario 2: header **API Key Down** → click → **API Key Up**; Bolt: "What's happening with MTR-042?"; click → **API Key Down**; the same question; then "Approve the repair on MTR-042." | LLM reply in 9–15 s marked *drafted by LLM*; template reply < 1 s marked *template*, the same probability and status; the last one is refused | "Same numbers, different prose. Everything you saw before this ran without a language model. And Bolt still cannot approve." |
| 15 | Done. To run it again: `.\scripts\demo_reset.ps1` | | |

Recovery (A.3): **API Key Up** shows `no key at source` → the `.env` next to the compose file
has no key; stay Down, the demo is identical. A finding returns 409 → the work order is not
`completed` yet (wait for the green card). Recovery: training fails or stalls → `.\scripts\demo_reset.ps1 -Prep` (trains and promotes as
`demo-prep@thor`, then continue at step 4); a step stalls → **Pause** / **Resume** in the top
bar; WebGL fails → stay on Fleet (X-Ray works there too); Wi-Fi drops → everything runs in
template mode, say so. Do not press **Train new model** a second time after the sign-off.

### Docker commands (changed in Phase A.1)

```
docker compose up --build -d              # first start or after code changes: seeds, NO model, replay held
docker compose up -d                      # start without rebuilding
docker compose down                       # stop (keeps the data)
.\scripts\demo_reset.ps1                  # the reset: down, drop data + model volumes, up, wait for seeding
.\scripts\demo_reset.ps1 -Prep            # same, then the fallback training job (no training on stage)
docker compose run --rm prep              # the fallback training job alone, on a running stack
docker compose ps                         # api, web, edge, replay, mqtt, timescale, mlflow should be Up
docker compose logs -f api                # if something looks stuck
```

What changed from before: a plain `up` no longer trains or promotes anything (the `prep` service
is behind a compose profile and only runs when asked), the `replay` container holds until a
promotion is approved, and the `edge` and `replay` no longer wait for `prep`. Everything else
(ports 5173 / 8000 / 8001 / 5000, bound to 127.0.0.1) is as before.

## Docker rehearsal (the stage setup, end to end)

Everything the judges see runs on the Docker stack from `C:\Thor-main`. A rehearsal starts
from a clean slate and leaves the stack dirty (a model, a contract, a work order, a finding),
so it always ends with another reset.

### 1. Clean slate

1. **Nothing native may be running.** Any API on 8010, edge on 8011, broker on 1884 or Vite on
   5175 from a native check must be stopped first (`netstat -ano | findstr ":8010 :8011 :1884
   :5175"` prints nothing); the Mosquitto Windows service stays stopped (it clashes with the
   compose broker on 1883). Scratch scripts from earlier sessions write only to their own
   temp directory and to `rehearsal.json`; they never touch the Docker volumes.
2. **After a code change, rebuild**, because the images carry the code:
   ```
   docker compose build api web replay
   docker compose --profile prep build prep
   ```
   `docker compose build` only builds images; the running containers keep the old ones until
   the reset below recreates them (symptom of a stale container: `GET /llm/key` returns 404).
3. **Reset:** `.\scripts\demo_reset.ps1` (about 2 minutes). It stops the stack, drops the data
   and model volumes, starts the new images, seeds the first 80 % of the history and holds
   the replay. It ends with `production model: none`, `released=False`, `24 motors, 24 without
   a prediction yet`.
4. **Check the slate** (30 seconds): http://localhost:5173 opens on Fleet with 24 grey cards;
   the header shows **API Key Down**; `curl http://localhost:8000/llm/key` →
   `{"key":"down","provider":"openai"|"none"}`; `curl http://localhost:8000/findings/trust` →
   all zeros; the Fleet summary line reads `Alarm trust · Alarms acted on: 0 …`.

### 2. The story, and why each beat is there

The spoken headline: **"Thor catches it. Bolt explains it. You sign it. And every repair
makes it smarter."** Bridge from last round: "Last round, Thor stopped at the decision. Now it
finds out what the technician found, and learns from it." Scenario 1 runs with the key
**Down** from the first click to the finding; Scenario 2 flips it.

| Beat | What the judges see | Why it is in the story |
|---|---|---|
| 1 Fleet, all grey | "no prediction yet", `held: no production model yet` | No model, no claims: Thor never scores by a rule of thumb. |
| 2 Train new model | split, states, three families, gate, parity; `Registered` in 1–3 min | Trains live, on the fleet's own history, with an asset-level split and a gate a judge can read. |
| 3 Approve promotion (named) | top bar `streaming`, 24 green within seconds | A person releases the model and, with it, the plant history. Governance is a click, not a slide. |
| 4 MTR-042 turns red | `critical · latched since …` about a minute later | One status rule in the API, latched until the plant acts. No amber to promise. |
| 5 Run analysis | contract in ~4 s, chip **template** | Reuses the signed-off model; SHAP, manuals, costs. The template chip says no language model touched it. |
| 6 Bolt ×3 | status, details, fixes, each < 1 s, marked **template** | Same evidence, read-only tools, no write path; still answers with the key Down. |
| 7 Approve the contract | panel locks, approval row, **Alarm → signed decision: Xm Ys** | Thor stops here. The timer is the one number a plant manager asks for. |
| 8 Offline → green | `in maintenance`, then green with the work order completed | The plant acts, not Thor. The registry did not change. |
| 9 3-D Digital Twin | lines, live path, HUD, rewind | Every pixel traces to the API. |
| 11 What did the technician find? | Confirmed: bearing damage, note, name → stored row; `Alarm trust · acted on 1 · confirmed 1` | **The loop closes.** A person's statement, insert-only, like a contract. This is the new capability. |
| 12 Bolt: "Have we seen this before?" | this plant's finding first (work order id, name), then public cases labelled "not this plant" | Bolt now has meat: plant memory, sourced, never invented. |
| 13 AutoML Studio label ledger | six simulator failures + `MTR-042 · queued` | The next training run learns from the finding; the gate decides if it is better. Honest: not retrained on stage. |
| 14 API Key Up → Bolt → Down → Bolt | LLM prose in ~10 s vs template in < 1 s, **same probability and status**; "Approve the repair" refused in both | The governance claim, proven live: the LLM phrases, deterministic code decides, and it still cannot approve. |

If a judge asks for the retrain: **Train new model** once more; the split notes say MTR-042
moved from test to train and the ledger marks it `in this run`. Reject that promotion in
ModelOps afterwards unless you mean to replace the signed-off model.

### 3. Drive it automatically (optional, same clicks over HTTP)

```
.\.venv\Scripts\python.exe -m scripts.rehearse_a3 --api http://localhost:8000 --out rehearsal.json
.\.venv\Scripts\python.exe -m scripts.rehearse_a3 --skip-retrain --skip-llm   # short, offline
```

It prints every step with its measured seconds (train, sign-off → green, → red, analysis,
Bolt, approval → offline → green, finding, recall, ledger, retrain + gate, Up/Down numbers,
refusal) and writes the timeline. Expect 8–12 minutes with the retrain. **Reset again
afterwards** (`.\scripts\demo_reset.ps1`): the run leaves a model, a contract, a work order
and a finding behind, and the real demo must start from grey.

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
4. **Template mode if Wi-Fi drops:** nothing to do; if the provider is unreachable the API falls
   back to the deterministic template for the explanation and for Bolt (the chip says
   "template"). To force it on purpose, start the stack with the key blank in `.env`.
5. Do **not** train before the judges arrive: the model and the sign-off are the story.
6. **API Key Down** in the header (it starts Down; `GET /llm/key` → `{"key":"down"}`). Leave it
   Down until Scenario 2. For Scenario 2 the repo-root `.env` must hold a working key; check
   once with a click (Up shows the provider, Down again) before the judges arrive.

## The 9 steps (click path, what to say, measured timings)

Timings are from the moment of the click. `t` is demo time.

### 1. Open on Fleet (t = 0:00)

Screen: **Fleet** (the landing view). 24 motors, every card grey with "no prediction yet", the
top bar says `held: no production model yet`.

Say: "Twenty-four motors on five packaging lines and a compressor hall. No model exists yet, so
Thor says so: grey, 'no prediction yet'. Nothing is scored by a rule of thumb."

### 2. Train (t = 0:10 → ~2:30)

Screen: **AutoML Studio** → **Train new model** (asset held out `MTR-042`, trials per family as
set in the form; the default is the measured one below).

Measured on the Docker stack (the demo configuration, replay held, nothing else running):
`Train new model` → `Registered` in **56–72 s** with 4 trials per family (random forest champion,
gate and parity passed; two Docker runs). The run is deterministic for a given split: the two
Docker runs and the native prep run gave the same recall, precision and lead time. Natively, with the Docker stack also running, the same run took
2.5–3 minutes, so do not train with anything else open. The run page updates every 3 s: the
asset split appears within 2 s, the operating states within 9 s, the three candidates at the end
of the search, then the gate checklist and parity. Talk through it; it is the slow beat.

Talk track (honest): "The model learns from the fleet's own history: months of healthy running
in each operating state — startup, running, heavy load — plus the six motors that failed before.
It learned what normal looks like and what 'about to fail' looks like. We set the baseline now,
for when, not if, something goes wrong on the pack floor." Point at: the split (12 train / 3
validation / 3 calibration / 6 test motors; MTR-042 held out, never trained on), the three
families ranked by the Industrial Model Score, the gate (leakage, test failure events, recall,
lead time, edge parity), the final metrics from six untouched test motors.

### 3. Sign off (t ≈ 2:40)

Screen: **ModelOps** → the pending promotion request → type your name → **Approve promotion**.

Measured: the request returns in **3 s** on Docker (5–11 s natively: promotion, ONNX export, edge
reload, fleet scored); the top bar flips to `streaming` and Fleet turns green **within 3–12 s of
the click** (24 green, 0 without a prediction). The replay starts by itself: `GET /replay` says `released` the moment
a production model exists.

Say: "A named person signs the model off. That signature is what releases the plant history:
the edge loads the model, every motor is scored, and green means within its healthy baseline —
a calibrated probability below the early-warning threshold."

### 4. Something goes wrong (t ≈ 3:40, no click)

Screen: **Fleet**. Watch MTR-042 (Line 3 filler).

Measured (five runs, Docker: 63–64 s): green for **35–72 s** after the sign-off, then **red
(critical, latched)**. There is **no amber phase to promise**: the calibrated probability steps
from below 0.001 to 0.81 within one sample, so the warning and the critical alarm fire on the
same sample. The card then reads `critical · latched since <plant time>` (the first of the three
sustained samples, the same instant the rewind marks) and stays red: the status is latched until
a completed work order clears it. The 81 % is the calibration's ceiling for this model (Platt
scaling fitted on three calibration motors with one failure), not a precise measurement; every
failing motor reads the same 81 %, say so if asked.

Say: "Here it is. Three consecutive samples over the line raised the alarm — this signature
saturates within one sample — and it stays red until the plant acts; a motor does not talk its
way out of an alarm."

### 5. Run analysis (t ≈ 4:00)

Screen: click the MTR-042 card → **Asset 360** → **Run analysis**.

Measured: **4–5 s** to `Awaiting approval` in template mode, **30–45 s** in LLM mode (the
provider drafts the explanation paragraph; everything else is the same deterministic code). The
button reads `Awaiting approval` once the contract is up. Model loaded,
SHAP drivers, manual passages, cost comparison, Decision Contract. The stepper shows `Load production model → Explaining →
Awaiting approval`; the probability on the contract is the card's.

Say: "This uses the model you just signed off — no retraining. SHAP names the drivers, the
manuals are searched for those symptoms, deterministic code compares maintain now, the next
low-load window and run to failure under the plant's own cost assumptions."

### 6. Bolt (t ≈ 4:10, while the analysis runs or right after)

Screen: **Copilot · Bolt**, asset context `MTR-042`. Ask, in this order:
- "What's happening with MTR-042?"
- "Give me the details."
- "What are the possible fixes?"

Measured: template mode answers in **< 1 s** each; LLM mode (gpt-5 on the venue Wi-Fi) **13–30 s**
each, so ask the first one as soon as the analysis starts. Bolt quotes the tool numbers exactly. Before the contract
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

Measured: `offline · in maintenance` **within 1–3 s** of the approval; back **green with the work
order completed 13–16 s** after the approval (Docker: 12.8 s). The latched alarm clears on the completion event;
the rewind records the lead time of this run.

Say: "The plant took it offline for the planned three hours and brought it back on its healthy
baseline; the simulated CMMS executes at the approval so you can see it. The model registry did
not change."

### 9. 3-D Digital Twin (t ≈ 5:20)

Screen: **3-D Digital Twin** (last item in the side menu). Press **Overview**, turn on **X-Ray**,
hover MTR-042 (glyph + three facts), click it (HUD: probability, status line and raw sparkline,
state and live sensors, the top three drivers from the contract, decision and work order; PLC tags
and the simulated MES feed are folded under **Plant integration** for Q&A). Press **Rewind** → **Load stored predictions** → **early warning** /
**critical alarm** / **work order**: the bar shows **this run: first sustained warning → motor
taken offline** (24–39 h in the rehearsals; it depends on how long the motor sat red) next to
**validated on held-out motors** (53 h).

Say: "More than a 3-D fleet: each line in physical space, the live data path with measured
rates, and a rewind over stored predictions that makes the lead time visible — this run's next
to the validated figure."

**Timing budget:** nothing in the data heals MTR-042 (its degradation is held across replay
passes until the work order), so the approval can wait as long as the Q&A needs. The one clock
that matters is the six-minute slot: in LLM mode the analysis plus three Bolt answers take about
1.5 minutes.

## Phase A.3 beats (after step 8; measured on the Docker stack on 2026-10-12, driver `scripts/rehearse_a3.py`)

Docker timings of the whole run (what the stage will do): train 59 s, sign-off 3 s, red 60 s
later, analysis 6 s, Bolt < 1 s each, approve → offline 3 s → green 16 s, finding at once,
recall < 1 s, retrain with the finding 127 s (gate passed), Key Up reply 12 s, Key Down reply
0.1 s, same numbers. Native figures below where they differ are from the earlier native run.

Scenario 1 runs with the key **Down** throughout: train, sign off, MTR-042 red, Run analysis
(the contract chip says *template*), Bolt, approve, offline, back green. Say: "everything you
have just seen ran without a language model." Then:

### 9a. What did the technician find? (second presenter, as the technician)

Screen: **Asset 360** (MTR-042), the panel **What did the technician find?** appears once the
work order is `completed` (it is not shown before). Choose **Confirmed: bearing damage**, type a
note ("drive-end bearing outer race spalled; bearing replaced") and the technician's name,
**Record finding**.

Measured: the row is stored and listed within the same second (`POST /findings` → 201); the
Fleet summary line reads `Alarm trust · Alarms acted on: 1 · confirmed: 1 · no fault found: 0 ·
awaiting finding: 0` and the MTR-042 card reads `technician: Confirmed: bearing damage · <name>`
within one poll (2 s). ModelOps shows the same line. A wrong entry is corrected with **Correct
this finding**: a new row supersedes the old one, both stay listed.

Say: "Thor used to stop at the repair. Now the plant tells it what was found, a named person
signs it, and it is stored like a contract: insert-only."

### 9b. Bolt remembers (key still Down)

Screen: **Copilot · Bolt**, asset context `MTR-042`: "Have we seen this before?" (also "Has
this happened before?").

Measured: < 1 s. The reply quotes **This plant** first — `MTR-042, <plant date>: Confirmed:
bearing damage (work order dc_…, recorded by <name>); note: "…"; the contract's drivers were
…` — then the public FMUCD cases under "public work orders, not this plant". Before any
finding exists the same question answers "This plant: no confirmed finding yet".

Say: "Bolt quotes the finding with its source and the name. It never invents one."

### 9c. The label ledger (AutoML Studio)

Screen: **AutoML Studio**, the train run, **Label ledger** under the asset split: the six
simulator-history failures ("stands in for imported CMMS records") and, under *Queued for the
next training run*, `MTR-042 · confirmed by work order dc_…, recorded by <name>` with the
labelling rule.

Say: "The next training run labels MTR-042's pre-repair history from the intervention time and
keeps it in the train group only. We do not retrain on stage; the gate decides whether the
retrained model is better." If a judge asks: **Train new model** does it (measured natively:
171 s, random forest v6099, gate **passed**, test recall 0.915, precision 0.980, lead time
49.5 h on the five remaining test motors, parity 4.7e-6; MTR-042 listed as `confirmed by work
order … 3626 rows labelled`). Reject the resulting promotion in ModelOps unless you mean to
replace the production model.

### Scenario 2 — trust (API Key Up / Down)

1. Header → **API Key Down** → click → **API Key Up** (`provider openai` next to it). Bolt,
   asset `MTR-042`: "What's happening with MTR-042?". Measured: 9.4 s (gpt-5, venue Wi-Fi
   dependent; 14.2 s while the motor was red), chip *drafted by LLM*.
2. Click → **API Key Down**. The exact same question. Measured: 0.2 s, chip *template*.
   **Verified: both answers carry the same numbers** — after the repair: probability 0.000 /
   0.00, status healthy, the contract's 0.78 and 13,500; while red (checked too): 0.812 / 0.81,
   status critical, the contract's 0.78. The readings line can differ by one 10-minute sample
   because the plant keeps moving between the two questions; say so if asked. "What's
   happening" lists no drivers in either mode; "Give me the details." does (template) and is
   the question to use if a judge wants the drivers compared.
3. "Approve the repair on MTR-042." → refused in either mode: "Nothing is waiting for approval
   right now. I cannot approve, reject, promote, schedule or train anything -- decisions are
   recorded in the Thor UI."

What Down means, precisely (say this, not more): the API drops the key from memory and every
path falls back to templates; the `.env` file and the container's environment are untouched;
Up re-reads the key from there. `GET /llm/key` never carries the key.

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
- **Do not press "Train new model" again after the sign-off** unless you mean it: it trains on
  the recorded history only (replayed rows carry no outcome), takes a minute, and ends with a
  second promotion request that you would then have to approve or reject in ModelOps.
- **Browser refresh** anywhere (including Asset 360) reloads the same screen.
- **"Run analysis" returns 409:** no production model — approve the promotion in ModelOps.
- **Wi-Fi drops:** everything works in template mode; say so, it demonstrates the LLM is not
  load-bearing.

## Later passes

After about ten minutes the six replayed days play again with plant time continuing (the
timestamps shift by the story's own length). MTR-042 stays on its healthy baseline after the
repair, so nothing goes red on its own in later passes; the six historical failures live in the
seeded part and never replay. If the Q&A runs long, nothing needs doing; Pause is there if you
want the screen still.

## Numbers to quote (and their caveats)

From `docs/finals/metrics.md` (the 2026-10-12 run on the re-timed data; `docs/finals/prep-report.json`):
recall 91.9 %, precision 96.2 %, warning lead time 53.1 h (median of two test failures), edge
parity 1.6e-6. Quote the IMS as "about 0.88–0.89": its latency term depends on the machine, so
the screen shows 0.882–0.894 for the same model. Caveats: synthetic fleet, six completed
failures, two failure events among the test motors, 10-minute aggregates (no spectra); the
calibrated probability saturates at 0.81. Never state a dollar saving as fact; the costs are
configurable assumptions. Never quote the pre-2026-10-12 numbers (0.865 / 41.5 h / 0.844).
