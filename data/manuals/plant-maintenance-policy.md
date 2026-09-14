# Plant Maintenance Policy

Document ID: PMP-2025
Applies to: All production assets at the site, including the induction-motor fleet
Revision: 2025 edition (replaces PMP-2023)

## 1. Policy statement

### 1.1 Purpose

This policy sets out how maintenance decisions are made, costed, approved, and recorded at
the plant. Its aim is to minimise the total cost of asset ownership, which includes the cost
of maintenance work, the cost of lost production, and the cost of safety and environmental
consequences, while keeping every consequential decision in the hands of an accountable
person.

### 1.2 Principles

1. Planned work is cheaper than unplanned work. Every unplanned stoppage is investigated
   to find out whether it could have been planned.
2. Condition drives timing. Time-based intervals are the fallback where condition data does
   not exist.
3. Decisions are made by people. Monitoring systems, analysis software, and automated
   recommendation tools inform a decision; they do not make it.
4. Every decision that changes the state of an asset or commits plant resources is recorded
   with the evidence that supported it, so that it can be audited afterward.

### 1.3 Scope

Applies to mechanical, electrical, and instrumentation maintenance on all production and
utility assets. Building maintenance and IT are covered by separate policies.

## 2. Maintenance strategies

### 2.1 Run-to-failure

Permitted only for assets classified as low criticality (section 5.2) where the failure
consequence is limited to the asset itself, spares are held, and the repair can be done
inside a normal shift. Run-to-failure is a deliberate, documented choice per asset, not a
default.

### 2.2 Time-based preventive maintenance

Fixed-interval tasks (lubrication, filter changes, inspections) set from OEM guidance and
adjusted from site history. Intervals are reviewed annually against the failure records.

### 2.3 Condition-based maintenance

Work is triggered by a measured change in condition (vibration, temperature, current,
oil analysis, thermography). This is the default strategy for medium and high criticality
rotating equipment, including all motors above 30 kW. Condition-based maintenance requires
a baseline, a documented alert limit, and a named person responsible for reviewing alerts.

### 2.4 Predictive maintenance

An extension of condition-based maintenance in which a model estimates the probability of
failure within a defined horizon and the remaining useful life, and that estimate is used
to choose between acting now, acting at the next planned window, or deferring. Predictive
recommendations are subject to the approval rules in section 6; the model does not
schedule work.

## 3. Cost model for maintenance decisions

### 3.1 Purpose of the cost model

To compare intervention options on a common basis, every recommendation to maintain now,
maintain later, or continue running is expressed as an expected cost over a fixed decision
horizon. The expected cost is the sum of the planned cost of the intervention and the
probability-weighted cost of a failure before that intervention happens.

### 3.2 Cost components

The site holds the following reference figures for the 75 kW motor class. They are
planning assumptions reviewed each budget cycle, not measured costs for any single event,
and any report that quotes them must say so.

- Planned bearing replacement (parts, labour, planning, standby switch-over): 4,200 currency
  units per event.
- Unplanned bearing failure repair (recovery, journal repair or motor swap, expedited parts,
  overtime): 18,500 currency units per event.
- Lost production during downtime: 3,100 currency units per hour of line stoppage.
- Planned stoppage duration for a bearing change: 3 hours.
- Unplanned stoppage duration for a seized bearing: 14 hours on average, with a range of
  8 to 36 hours depending on collateral damage.

### 3.3 Expected-cost formula

For each option:

- Maintain now: planned cost plus planned downtime cost. The probability of failure before
  the intervention is the current instantaneous probability, which is small if action is
  taken immediately.
- Maintain later at the next planned window: planned cost plus planned downtime cost, plus
  the probability of failure before that window multiplied by the unplanned cost (repair
  plus unplanned downtime).
- Run to failure over the decision horizon: the probability of failure within the horizon
  multiplied by the unplanned cost (repair plus unplanned downtime), with no planned cost.

The option with the lowest expected cost is the recommended one, subject to the safety
overrides in section 3.5 and the approval rules in section 6.

### 3.4 Failure probability inputs

The failure probability comes from the validated predictive model for the asset class,
expressed as the calibrated probability of failure within the model horizon. The
probability of failure before a future time is derived from a monotone hazard curve
anchored on the current probability and the model's demonstrated warning lead time. The
curve, its parameters, and the model version must be recorded with the decision.

### 3.5 Safety and criticality overrides

The cost comparison never overrides a safety condition. If a trip limit is reached (for
example, bearing temperature above 95 C) the asset is stopped regardless of cost. For high
criticality assets, run-to-failure is not an admissible option even if it is the cheapest,
and the comparison must present only the two planned options.

### 3.6 Worked example

A motor shows a calibrated 62 percent probability of drive-end bearing failure within 48
hours. The next planned low-load window is 60 hours away. Using the section 3.2 figures:

- Maintain now: 4,200 plus 3 hours times 3,100, equal to 13,500.
- Maintain later: the same 13,500 plus roughly 0.75 (probability of failing before the
  window) times (18,500 plus 14 hours times 3,100), which adds about 46,400.
- Run to failure over one week: roughly 0.9 times 61,900, about 55,700.

Maintain now is recommended and the difference is large enough that the decision is not
sensitive to the exact cost figures.

## 4. Alert handling

### 4.1 Response times

- Stage 1 (early indicator): review within one working day.
- Stage 2 (sustained trend): review within 4 hours; raise a planned work order.
- Stage 3 (severity limit): review within 1 hour; schedule within 48 hours.
- Stage 4 (trip): immediate stop; operations notified within 15 minutes.

### 4.2 Trend triggers

A change of more than 25 percent in a monitored indicator within one week triggers a review
even when no absolute limit has been reached.

### 4.3 False alarm handling

Every alert is closed with a disposition: confirmed fault, regime change, sensor fault,
lubrication transient, or external source. Alerts with more than 30 percent false
dispositions over a quarter have their limits reviewed.

## 5. Asset criticality

### 5.1 Criticality classes

- High: failure stops the main production line or creates a safety or environmental risk.
- Medium: failure stops a secondary line or reduces throughput, with a standby available.
- Low: failure affects only the asset; repair is inside a shift with spares held.

### 5.2 Consequences of classification

Criticality sets the admissible strategies (section 2), the approval level (section 6),
and the response times (section 4). Medium criticality is the default for motors above
30 kW; the conveyor drive motors on the main line are high criticality.

## 6. Approval rules for maintenance actions

### 6.1 Who may approve

- Planned maintenance inside the current week's schedule: shift maintenance supervisor.
- Planned maintenance requiring a line stoppage outside the schedule: maintenance manager
  and production manager jointly.
- Run-to-failure decisions on medium criticality assets: reliability engineer, recorded
  with the cost comparison.
- Any stoppage of a high criticality asset: plant manager or delegate.

### 6.2 What an approval must record

Every approval, whether granted or refused, is recorded as an immutable entry containing:

1. The asset and the recommendation being approved or refused.
2. The evidence supporting the recommendation: the failure probability and the model
   version that produced it, the top contributing indicators, the cost comparison with the
   assumptions used, and any manual references cited.
3. The proposed maintenance window.
4. The identity of the approver, the decision, the time, and any note.

Records are never edited after creation. A change of decision is a new record that
references the earlier one.

### 6.3 Automated recommendations

Software that generates maintenance recommendations, including predictive models and
decision-support agents, must:

- Present the recommendation with its evidence in the form described in section 6.2.
- Never execute a maintenance action, issue a work order, or change a model in production
  without a recorded human approval.
- Cite only reference documents that actually exist and only the sections that contain the
  cited content.
- State clearly that cost figures are planning assumptions.

The purpose of these rules is to keep the decision accountable to a person and to make the
reasoning auditable afterward.

### 6.4 Model changes

A predictive model may be moved to production use only after a validation report showing
it was tested on assets it was not trained on, with a documented recall, warning lead time,
and calibration. Promotion of a model is itself an approval under section 6.2, recorded by
the reliability engineer or maintenance manager.

## 7. Maintenance windows

### 7.1 Standard low-load windows

The plant's standard low-load window is 02:00 to 05:00 site time each day, during the
night-shift changeover when the main line runs at reduced rate. Planned work on medium
criticality rotating equipment is scheduled into this window by default. The extended
window on Sunday, 00:00 to 08:00, is used for work requiring more than three hours.

### 7.2 Choosing a window

When a condition-based or predictive recommendation proposes maintenance later, the
window chosen must be the earliest standard window that ends before the estimated
probability of failure exceeds 30 percent. If no such window exists the recommendation
becomes maintain now.

### 7.3 Load reduction pending maintenance

Where production allows, an asset with a stage-2 or stage-3 bearing alert is run in the
nominal regime or below until the maintenance window. This roughly doubles the expected
remaining life of a degrading bearing.

## 8. Records and audit

### 8.1 Work orders

Every intervention is executed against a work order in the plant maintenance system,
which holds the approved recommendation, the parts consumed, the labour, and the
post-repair verification readings.

### 8.2 Decision records

Decision records under section 6.2 are retained for the life of the asset plus five years
and are made available to internal audit and to the annual reliability review.

### 8.3 Annual review

Each year the reliability team reviews all unplanned stoppages, all refused
recommendations, and all false alarms, and proposes changes to alert limits, cost
assumptions, criticality classes, and this policy.
