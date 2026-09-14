# Three-Phase Induction Motor Maintenance Manual

Document ID: MM-IM-075-R4, revision 4 (section 4 rewritten after the 2024 bearing-failure
review). Applies to low-voltage squirrel-cage induction motors, 37 kW to 160 kW, IEC frame 225 to 315.

## 1. Scope and purpose

### 1.1 Scope

This manual covers routine inspection, condition monitoring, lubrication, and corrective
maintenance for the plant's fleet of 75 kW class three-phase induction motors driving pumps,
fans, conveyors, and compressors. It applies to totally enclosed fan-cooled (TEFC) machines
with deep-groove ball bearings on the drive end (DE) and non-drive end (NDE).

### 1.2 Related documents

Section 4 lists the failure modes seen in service with the sensor signatures that precede
each one; VA-GUIDE-R2 (Vibration Analysis Guide) and PMP-2025 (Plant Maintenance Policy,
cost model and approval rules) are the companion documents.

## 2. Machine description

### 2.1 Construction

The motor is a 4-pole TEFC machine with a cast-iron frame and a die-cast aluminium rotor.
The drive-end bearing is a 6313-C3 deep-groove ball bearing; the non-drive-end bearing is a
6311-C3. Both are grease-lubricated through external nipples with a relief valve on the DE shield.

### 2.2 Nameplate ratings

Rated power 75 kW; rated speed 1480 rpm at 50 Hz (synchronous 1500 rpm, nominal slip 1.3
percent); rated current 138 A at 400 V; service factor 1.0; insulation class F with
temperature rise class B (80 K); bearing design life (L10) 40,000 hours at rated radial load.

### 2.3 Normal operating regimes

Plant motors run in three recognisable regimes; condition-monitoring limits in section 7 are
set per regime because a load change raises vibration and temperature with no fault present.

- R1 idle and startup: 500 to 700 rpm equivalent loading during soft-start, load below 15 percent.
- R2 nominal: 1470 to 1490 rpm, load 45 to 65 percent. This is the regime the motor spends
  most of its life in.
- R3 high load: 1490 to 1505 rpm, load 75 to 95 percent, typically during peak production shifts.

Compare a reading only against the baseline for the same regime. A 30 percent rise in
vibration RMS during a transition from R2 to R3 is normal; the same rise inside R2 is not.

## 3. Routine inspection

### 3.1 Daily walk-down

Listen for a change in bearing noise (a rumble or a periodic click); check the DE bearing
housing with an infrared thermometer and log any reading noticeably hotter than the frame;
confirm the fan cowl is unobstructed; note grease purge from the DE relief valve (a small
purge after re-greasing is normal, continuous purge is not).

### 3.2 Weekly and quarterly checks

Weekly: record the online sensor readings (section 7) per motor per regime and inspect the
coupling guard. Quarterly: insulation resistance test (1 kV megger, minimum 100 megohm at
40 C), soft-foot and alignment check, and a spectrum reading at the DE and NDE bearings.

## 4. Failure modes and their signatures

### 4.1 Failure-mode ranking

From the 2019 to 2024 failure records for this fleet, root causes rank as follows:

1. Drive-end bearing wear and fatigue: 51 percent of all unplanned stoppages.
2. Non-drive-end bearing wear: 14 percent.
3. Stator winding insulation breakdown: 12 percent.
4. Rotor bar cracking: 8 percent.
5. Misalignment and coupling faults: 9 percent.
6. Other (cooling, terminal box, cable): 6 percent.

Bearings account for two thirds of unplanned stoppages, and the drive end fails almost four
times as often as the non-drive end because it carries the radial coupling load. This is the
failure mode the online monitoring package is primarily designed to catch early.

### 4.2 Drive-end bearing wear

#### 4.2.1 Mechanism

Drive-end bearing wear begins as sub-surface fatigue on the outer race in the load zone. A
small spall forms, then each ball passing the spall produces an impact. Over days to weeks
the spall grows, adjacent spalls join, and the rolling surfaces roughen. Friction and heat
rise, the grease degrades, and the bearing enters a run-away phase that ends in cage
fracture or seizure. Contributing causes on this fleet were: over-greasing (grease churning),
contaminated grease, coupling misalignment above 0.08 mm, and electrical fluting from
variable-speed drives without a shaft grounding ring.

#### 4.2.2 Vibration RMS signature

Overall vibration velocity RMS at the DE bearing housing is the first broadband indicator.
In a healthy R2 regime the fleet baseline is 1.2 to 1.8 mm/s RMS. A sustained rise of more
than 2 standard deviations above the regime baseline, held for three or more consecutive
10-minute samples, is the earliest reliable warning and typically appears 5 to 10 days before
functional failure. A reading above 4.5 mm/s RMS in R2 corresponds to ISO 10816-3 zone C
(unsatisfactory for long-term operation) for this machine class; above 7.1 mm/s RMS is zone
D and the motor should be scheduled for immediate removal from service.

#### 4.2.3 Kurtosis signature

Kurtosis of the raw acceleration signal measures the impulsiveness of the waveform. A
healthy bearing produces a near-Gaussian signal with kurtosis close to 3.0. The first spall
produces sharp periodic impacts and raises kurtosis to 4 to 6 before the overall RMS has
moved appreciably. Kurtosis is therefore the most sensitive early-stage indicator, but it is
also the noisiest: treat a single high reading as a prompt to look, not as a fault. Kurtosis
that rises and then falls back toward 3 while RMS keeps rising is the classic signature of a
bearing moving from the early spall stage to the advanced roughened stage.

#### 4.2.4 Crest factor signature

Crest factor (peak divided by RMS) behaves like kurtosis: a healthy bearing sits between
2.5 and 3.5, early spalling lifts it to 4 or above, and late-stage generalised roughening
brings it back down as RMS catches up with the peaks. A crest factor above 4.0 sustained for
one hour in a stable regime should be logged as a stage-1 bearing alert.

#### 4.2.5 Bearing temperature signature

The DE bearing housing temperature normally runs 10 to 20 K above ambient in R2 and 15 to
25 K above ambient in R3. A rise in the difference between DE bearing temperature and motor
frame temperature (the temperature delta) of more than 5 K against the regime baseline
indicates friction heating from a degraded bearing or degraded grease. Temperature is a
late indicator: it usually moves 2 to 4 days after vibration RMS has already crossed the
warning level. A DE bearing temperature above 95 C is an immediate stop condition.

#### 4.2.6 Recommended actions by stage

Stage 1 (kurtosis or crest factor elevated, RMS within 2 standard deviations of baseline):

- Confirm with a spectrum reading; look for outer-race defect frequency (BPFO) sidebands.
- Check grease condition and quantity; re-grease only if the last interval has elapsed.
- Increase monitoring to hourly review and set a stage-2 trigger on RMS.
- No planned stoppage required yet; typical remaining life is 10 to 20 days.

Stage 2 (RMS more than 2 standard deviations above regime baseline, sustained):

- Raise a planned bearing replacement work order for the next low-load window.
- Expected lead time to functional failure at this stage is 3 to 7 days in R2, shorter in R3.
- Reduce load to R2 or below if production allows; this roughly doubles remaining life.
- Prepare parts: one 6313-C3 bearing, one 6311-C3 bearing, DE and NDE seals, grease charge.

Stage 3 (RMS above 4.5 mm/s or bearing temperature delta above 8 K):

- Replace within 48 hours. Continued operation risks shaft journal damage and a rotor
  rub, which converts a 4-hour bearing change into a 3-day rewind or motor replacement.
- Do not attempt to run to the next scheduled outage.

Stage 4 (RMS above 7.1 mm/s, temperature above 95 C, or audible knocking):

- Stop the motor now and switch to the standby unit where one exists.

#### 4.2.7 Regime confounders

Before acting on any DE bearing indicator, confirm the motor has not changed regime. A
transition from R2 to R3 raises RMS by 20 to 40 percent and bearing temperature by 5 to 8 K
with no fault present. Startup transients in R1 produce short kurtosis spikes above 6 that
disappear within 30 minutes. The monitoring system's regime-normalised z-score removes this
effect; the raw values do not.

### 4.3 Non-drive-end bearing wear

The NDE bearing shows the same signatures as section 4.2 but with lower amplitudes because
it carries only the rotor weight and the fan load. Warning limits are 70 percent of the DE
limits. NDE failures on this fleet are usually caused by electrical fluting on drive-fed
motors; if the spectrum shows fluting, fit a shaft grounding ring at the same time as the
bearing change.

### 4.4 Stator winding insulation

Insulation degradation shows as a falling megger reading over successive quarterly tests
and a rising motor frame temperature not explained by load. Motor temperature rising while
the bearing temperature delta stays flat points to the winding rather than the bearing.

### 4.5 Misalignment

Misalignment raises vibration at twice running speed in the axial direction, with RMS that
is high immediately after a coupling change and does not trend upward over time. Kurtosis
stays near 3. A flat, elevated RMS with normal kurtosis after recent maintenance is
misalignment until proven otherwise.

## 5. Lubrication

### 5.1 Grease specification

Lithium-complex grease, NLGI grade 2, base oil viscosity 100 cSt at 40 C, with a service
temperature range of minus 20 C to plus 140 C. Do not mix with polyurea grease from the
previous specification; a motor changed to the new grease must be purged first.

### 5.2 Re-lubrication intervals

At 1480 rpm and a bearing temperature below 70 C: DE bearing 6313-C3, 45 grams every 4,000
running hours; NDE bearing 6311-C3, 35 grams every 4,000 running hours. Halve the interval for every 15 K the bearing runs above 70 C. Halve it again for motors in
the wash-down area. Under-greasing shows as a slow RMS rise with a flat temperature; over-
greasing shows as a temperature rise of 5 to 10 K within hours of servicing, which then
settles after the excess purges.

Procedure: clean the nipple and relief valve, remove the relief plug, add the specified
quantity slowly with the motor running, refit the plug after one hour, and log the quantity
and the bearing temperature before and 4 hours after.

## 6. Corrective maintenance

### 6.1 Drive-end bearing replacement

Planned duration: 3 hours with a standby unit available, 4 hours without. Crew: one
mechanical fitter and one electrician for isolation. Parts: 6313-C3 bearing, DE seal,
grease charge, coupling element if worn. Always replace the NDE bearing at the same time;
the additional 30 minutes avoids a second stoppage within the year.

Procedure summary: isolate and lock out, uncouple, remove fan and shields, pull and refit
bearings (induction heater), pack to one third free space, realign to within 0.05 mm.

### 6.2 Unplanned failure recovery

An unplanned bearing seizure typically takes 12 to 16 hours to recover because the shaft
journal must be inspected and often built up, and a rotor rub sends the motor to the rewind
shop. PMP-2025 section 3 gives the cost basis for comparing planned and unplanned work.

Post-repair acceptance after a 30-minute loaded run in R2: vibration RMS below 1.5 mm/s,
kurtosis below 3.5, crest factor below 3.5, DE bearing temperature delta below 12 K.

## 7. Online condition-monitoring package

### 7.1 Sensors fitted

Each motor carries an accelerometer on the DE bearing housing (radial, horizontal), RTDs in
the DE bearing shield and on the stator frame, a current transformer on one phase, and a
speed reference. Values are sampled once every 10 minutes and published to the historian.

### 7.2 Derived quantities

From the accelerometer the monitoring system computes vibration velocity RMS, kurtosis, and
crest factor over each 10-minute sample. From the RTDs it computes the DE bearing
temperature and the bearing-minus-frame temperature delta. Rolling means and slopes are
computed over a 2-hour (12-sample) window per motor, and each value is expressed as a
z-score against that motor's healthy baseline for the current regime.

### 7.3 Alarm limits

- Vibration RMS z-score above 2.0 sustained for 3 samples: stage-2 bearing alert.
- Kurtosis above 4.5 or crest factor above 4.0 sustained for 6 samples: stage-1 bearing alert.
- Bearing temperature delta z-score above 2.0: stage-2 bearing alert.
- Vibration RMS above 4.5 mm/s in any regime: stage-3 bearing alert.
- Bearing temperature above 95 C: stage-4, trip.

The online package detects broadband degradation only; it does not distinguish an outer-race
defect from an inner-race or ball defect. A stage-1 or stage-2 alert should always be
followed by a spectrum reading before the work order is finalised.
