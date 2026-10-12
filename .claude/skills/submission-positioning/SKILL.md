---
name: submission-positioning
description: Use when drafting hackathon submission text, a pitch, a README intro, or a demo script for Thor (ABB Theme 1). Holds the verbatim one-liner, differentiators, rubric mapping, and tech-stack name-drops.
---

# Submission positioning — Theme 1 (Agentic Predictive Maintenance Studio)

We submit under **Theme 1 only**. Use this language verbatim when drafting submission text;
don't invent new claims.

- **One-liner:** "An agentic AutoML + MLOps platform that turns raw industrial sensor data into
  a validated, explainable, human-approved predictive-maintenance decision."
- **Differentiators to lead with** (most teams stop before any of these):
  1. **State-aware fault detection** — clusters operating state (RPM/load/startup; called
     regimes in the literature) first, so a load change is never mistaken for a failure. Say
     "state" everywhere else, never "regime".
  2. **Industrial Model Score** — champion selection weighted by recall, warning lead time,
     calibration, and inference latency, not raw accuracy.
  3. **Decision Contract** — an immutable, evidence-linked audit record for every recommendation
     (provenance + explainability + auditability).
  4. **Prescriptive, not just predictive** — compares cost of maintain-now / maintain-later /
     run-to-failure before recommending a window.
  5. **Governed agent architecture** — the LLM only plans and explains; every number comes from
     deterministic code; a human approves every action. State this as architecture, not a slogan.
- **Rubric mapping** (one line each, no invented point totals):
  - *Innovation:* state-awareness + prescriptive layer, not another anomaly-threshold demo.
  - *Technical excellence:* time/asset-aware validation, calibration, drift detection, MLflow
    registry, edge deployment.
  - *Problem-solution fit:* directly implements every bullet in ABB's Theme 1 feature list.
  - *Scalability:* asset → site → enterprise story via the Decision Contract as an MES/CMMS
    integration point (say this explicitly — it's a real scalability argument, not filler).
  - *UX / Presentation:* one motor, one emerging fault, one decision — no feature-tour demo.
- **Finals headline (spoken, 2026-10-12; the formal one-liner above stays unchanged):**
  "Thor catches it. Bolt explains it. You sign it. And every repair makes it smarter."
  And the bridge from last round: "Last round, Thor stopped at the decision. Now it finds out
  what the technician found, and learns from it." Say what it means literally: a technician's
  finding on a completed work order is a person's statement, stored insert-only; a confirmed
  bearing damage labels that motor's pre-repair history for the next training run; the gate, not
  the finding, decides whether the retrained model is better. Never claim a finding improves the
  model.
- **Tech stack to name-drop:** Python, FastAPI, MLflow, LangGraph, Docker, SHAP, LightGBM/XGBoost,
  PostgreSQL — these are ABB's own suggested Theme 1 technologies, so use them by name.
- **Never:** claim a specific dollar savings as fact (say "under these configurable plant-cost
  assumptions..."); cite unverified competitor claims; call any part of this an MES.
