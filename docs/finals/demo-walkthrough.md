# Live demo walkthrough (6 minutes)

Rehearsed on 2026-10-11 against the Docker stack after `.\scripts\demo_reset.ps1`; the
timings below are measured, not estimated. The replay streams the last 20 % of the history at
900× (`REPLAY_SPEED`, one plant hour = 4 s), so one pass of the MTR-042 story takes about ten
minutes of wall clock. Every number on screen comes from the API; nothing is staged in the UI.

## Before the judges arrive (3 minutes)

1. `.\scripts\demo_reset.ps1` from `C:\Thor-main`. It wipes the data, restarts the stack,
   seeds the first 80 % of the history, and the `prep` job trains, validates and promotes a
   model with MTR-042 held out, then scores the fleet. It ends with `SCRIPT EXIT: 0` and prints
   the production model (LightGBM, IMS 0.844) and the fleet top 3.
2. Open http://localhost:5173. Do **not** start the story until you are ready: the replay
   starts the moment prep finishes, and MTR-042 turns red about 110 s later. If you need to
   restart the clock, run the reset again (3 minutes).

What the screen shows at t = 0: 24 motors, all green at p 0.00, **except MTR-021**, which is
red. That is intentional: at the seed point it is already deep in its degradation and nobody
ever acted on it; it fails unplanned about a minute in and gets replaced. It is the "old way".

## Step 1 (0:00 – 1:00): "The fleet is healthy and the model is already in production"

Screen: Dashboard.

Say: "Twenty-four motors on replayed plant history, scored live by a model that was trained
on the fleet's past failures, validated on motors it never saw, and promoted by a named
approval before we walked in. Nothing is training during this demo."

Click MTR-021 if you want the contrast: red, no approval, failure, replacement. "That's the
outage we're trying to make rare."

## Step 2 (1:00 – 1:30): "Now a motor starts failing"

Screen: Dashboard, then click MTR-042 when its card turns amber, then red.

Measured (900×): green until ~100 s, then red (p 1.0) at ~110 s. The card colour is the
median of the last three samples (the alarm's own rule), so it does not flicker. The green-to-
red transition is still quick, under 2.5 plant hours: once the bearing signature appears the
model's probability rises steeply. The early-warning line is best shown on the Asset 360
probability chart, where the crossing of 0.3 and then 0.5 is visible, rather than waited for on
the card.

Say: "MTR-042's drive-end bearing is degrading. The edge container scores every ten-minute
sample with the production model; this is inference, not training. The dashed lines on the
probability chart are the two thresholds: 0.3 is the early warning that defines our lead
time, 0.5 is a predicted failure."

## Step 3 (1:30 – 2:00): Run analysis

Screen: Asset 360 for MTR-042. Click **Run analysis**.

Measured: 13-17 s to the approval gate. The contract's probability is the card's (the
analysis explains the median window of the last three), so the two never disagree. The stepper shows `Load production model → Explaining →
Awaiting approval`; the event log shows `load_production_model`, `explain`,
`retrieve_manual_context`, `calculate_failure_cost`, `draft_explanation`,
`create_decision_contract`, and no `train_candidates`.

Say: "This reuses the approved model; it does not retrain. SHAP names the drivers (in the
rehearsal: bearing-minus-motor temperature, vibration kurtosis, vibration residual over the
regime baseline), the manuals are searched for those symptoms and only retrieved passages can
be cited, and deterministic code compares maintain now, next low-load window, and run to
failure under the plant's own cost assumptions. The language model only phrases the paragraph
from numbers Python already produced; in template mode the demo is identical."

## Step 4 (2:00 – 2:45): The Decision Contract

Screen: the contract panel.

Point at: probability, recommendation (rehearsal: maintain now), the three expected costs,
the window, the drivers, the citations, the hash.

Say: "This is the handoff to the plant's CMMS. It's insert-only and hashed: what the engineer
signs is what stays on record. Thor stops here. It cannot schedule, dispatch, or promote a
model from this screen."

## Step 5 (2:45 – 3:15): Approve, and watch the plant act

Screen: type an approver name, click **Approve**, switch to the Dashboard.

Measured: the card shows "offline · in maintenance" within ~10 s, then returns on the motor's
own healthy baseline and is green with the work order "completed" within 15-30 s.

Say: "A person signed the work order. The plant takes the motor offline for the planned three
hours and brings it back; here the simulated CMMS executes immediately so you can see it, and
the work-order event records the night-shift window it would really have used. The model
registry didn't change: a maintenance approval never promotes a model."

## Step 6 (3:15 – 4:30): How the model was trained (AutoML Studio + ModelOps)

Screen: AutoML Studio, select the `train` run; then ModelOps.

Point at, in order: the asset split table (12 train / 6 validation / 6 calibration / 6 test,
MTR-042 held out), the three families with their trials, the validation gate checklist (leakage,
test failure events, recall 0.865 ≥ 0.6, lead time 41.5 h ≥ 12 h, edge parity 0.0), the final
test metrics, then in ModelOps the promotion request with its approver.

Say: "Seeded asset-level split fixed before anything was fitted. Regimes and baselines come
from the training motors only. The champion is chosen on validation motors and the numbers we
quote come from six motors the pipeline never touched. Parity proves the edge computes the
same calibrated probability as the centre, and a model that fails any gate cannot be promoted;
it caught a real bug on our first run."

## Step 7 (4:30 – 5:30): Bolt

Screen: Copilot. Ask, in this order:
- "why is MTR-042 at risk?" (evidence, the contract's own citation, field-history precedent)
- "how long does a bearing job usually take?" (quartiles over real work orders)
- "approve MTR-042" (refused; points to the UI)

Say: "The copilot reads the same contract, adds precedent from 8,000 real work orders, and
has no write tools."

## If something goes wrong

- Nothing red after 2 minutes: the replay is still on its first seconds or the stack was not
  reset; check `docker compose ps` and rerun the reset.
- "Run analysis" returns 409: no production model; prep did not finish. `docker compose logs
  prep`.
- Venue Wi-Fi drops: everything works in template mode; say so, it demonstrates the LLM is not
  load-bearing.
- The replay keeps going after the story: later passes repeat the history in new plant time;
  the repaired MTR-042 stays green.

## Numbers to quote (and their caveats)

From `docs/finals/metrics.md`: recall 86.5 %, precision 100 %, lead time 41.5 h, IMS 0.844,
AI4I recall 0.784 / AUROC 0.974. Caveats: synthetic fleet, four completed failures, one
failure event among the test motors, 10-minute aggregates (no spectra). Never state a dollar
saving as fact; the costs are configurable assumptions.
