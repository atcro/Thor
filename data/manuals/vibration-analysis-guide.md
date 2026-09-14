# Vibration Analysis Guide for Rotating Machinery

Document ID: VA-GUIDE-R2
Applies to: Plant rotating equipment monitored by the online condition-monitoring package
Revision: 2

## 1. Purpose

### 1.1 What this guide covers

This guide explains the vibration quantities used by the plant's condition-monitoring
system, how they are measured, what the ISO 10816 severity zones mean for our machine
classes, and how to read the broadband indicators (RMS, kurtosis, crest factor) together to
stage a developing fault. It is written for maintenance engineers and reliability analysts
who need to interpret an alert, not for vibration specialists writing spectra reports.

### 1.2 What this guide does not cover

Detailed spectrum diagnosis (defect-frequency calculation, envelope analysis, phase
analysis) is covered by the analyst training course and the OEM analyser manual. This guide
tells you when to call for a spectrum, not how to interpret one.

## 2. Measurement basics

### 2.1 Sensor and mounting

Vibration is measured with a piezoelectric accelerometer stud-mounted or adhesive-mounted
on the bearing housing, as close to the load zone as possible. On motors this is the
drive-end bearing housing in the horizontal radial direction. Magnet mounts are acceptable
for walk-around readings but reduce the usable frequency range to about 2 kHz; permanent
online sensors are stud-mounted and usable to 10 kHz.

### 2.2 Units

- Acceleration: m/s squared, or g (1 g = 9.81 m/s squared). Sensitive to high-frequency
  bearing and gear faults.
- Velocity: mm/s. The standard severity unit for machine condition in the 10 Hz to 1 kHz
  band. All ISO 10816 limits are in velocity RMS.
- Displacement: micrometres peak-to-peak. Used for low-speed machines and shaft-relative
  proximity probes; not used on this fleet.

### 2.3 Sampling

The online package captures a 4-second time waveform at 25.6 kHz every 10 minutes and
computes the derived quantities in section 3 from that waveform. Walk-around readings use a
1-second capture at the same rate.

## 3. Broadband indicators

### 3.1 Overall RMS

The root-mean-square of the velocity waveform in the 10 Hz to 1 kHz band. RMS represents
the energy of the vibration and is the quantity compared against ISO 10816 zone limits. It
responds to every fault that adds energy, so it is the least specific indicator, but it is
stable and repeatable and is the basis of every severity decision.

RMS rises late in the life of a rolling-element bearing. By the time RMS has doubled, the
bearing has usually been spalling for days. This is why the monitoring package also tracks
kurtosis and crest factor.

### 3.2 Kurtosis

Kurtosis is the fourth statistical moment of the acceleration waveform, normalised so that
a Gaussian (random, healthy) signal has a value of 3.0. Kurtosis measures how impulsive the
signal is: a waveform with a few sharp spikes riding on a low background has high
kurtosis, while a waveform that is uniformly rough has kurtosis near 3 again even if its
RMS is high.

Interpretation for rolling-element bearings:

- 2.5 to 3.5: healthy, or advanced generalised wear (check RMS to tell them apart).
- 3.5 to 4.5: possible early defect; confirm with a second reading and a spectrum.
- 4.5 to 8: localised spall producing periodic impacts; stage-1 alert.
- Above 8: severe localised damage or a transient (startup, impact from outside). Re-read
  after 30 minutes before acting.

Kurtosis typically leads RMS by 3 to 10 days on this fleet's bearings.

### 3.3 Crest factor

Crest factor is the ratio of the peak absolute value of the waveform to its RMS. A pure
sine wave has a crest factor of 1.41; a Gaussian random signal sits between 3 and 4. Like
kurtosis, it rises when impacts are present and falls back as the damage spreads and RMS
catches up.

Interpretation:

- Below 3.5: no impulsive content.
- 3.5 to 4.0: watch; re-read.
- Above 4.0 sustained for one hour: impulsive content present, stage-1 alert.
- Above 6: severe impacts; check for looseness or a broken cage.

Crest factor is cheaper to compute than kurtosis and less sensitive to isolated outliers,
so the two are used together: both elevated is a strong early signal, only one elevated is
a prompt to look again.

### 3.4 Reading the three indicators together

The staging logic used by the online package and by the reliability workflow is:

1. Kurtosis and crest factor rise, RMS flat: stage 1, early localised defect. Days to weeks
   of life remain. Confirm and plan.
2. RMS starts rising while kurtosis and crest factor stay high: stage 2, defect growing.
   Schedule replacement in the next planned window.
3. RMS high and climbing, kurtosis and crest factor falling back toward baseline: stage 3,
   damage has generalised. Replace within 48 hours.
4. RMS in zone D, temperature rising fast, audible noise: stage 4, stop.

A rising RMS with flat kurtosis and crest factor after maintenance is usually misalignment
or unbalance, not a bearing defect. Do not order bearings on RMS alone.

### 3.5 Regime normalisation

Every indicator depends on speed and load. A motor moving from 55 percent to 85 percent
load will show 20 to 40 percent higher RMS with no fault. The monitoring package therefore
keeps a separate healthy baseline (mean and standard deviation) per motor per operating
regime, and reports each indicator as a z-score: (reading minus regime baseline mean)
divided by regime baseline standard deviation. Alert limits in section 6 are expressed in
z-score for the trend indicators and in absolute mm/s for the ISO severity limits.

## 4. ISO 10816-3 severity zones

### 4.1 Machine classification

ISO 10816-3 applies to industrial machines above 15 kW between 120 and 15,000 rpm measured
on non-rotating parts. Our motors fall in Group 2 (medium machines, 15 kW to 300 kW) on
rigid foundations. The zone boundaries for that group are given in section 4.2.

### 4.2 Zone boundaries, Group 2, rigid foundation

- Zone A, newly commissioned: below 1.4 mm/s RMS.
- Zone B, unrestricted long-term operation: 1.4 to 2.8 mm/s RMS.
- Zone C, restricted operation, plan corrective action: 2.8 to 4.5 mm/s RMS.
- Zone D, damage may occur, remove from service: above 4.5 mm/s RMS.

For flexible foundations the boundaries are 2.3, 4.5, and 7.1 mm/s. Motors on the elevated
conveyor gallery are treated as flexible; everything on the ground floor is treated as
rigid. When in doubt use the rigid limits, which are more conservative.

### 4.3 Using the zones

The zones describe absolute severity, not trend. A motor that has always run at 2.6 mm/s
in zone B is acceptable; a motor that has moved from 1.3 to 2.6 mm/s over a week has
doubled its vibration energy and is developing a fault even though it is still in zone B.
Both the zone and the trend must be considered. The plant policy (PMP-2025 section 4) is
that a change of more than 25 percent in RMS within one week triggers an investigation
regardless of zone.

### 4.4 Commissioning acceptance

New or overhauled motors are accepted in zone A (below 1.4 mm/s rigid) after a 30-minute
loaded run. A motor returned from repair that reads in zone B should be re-checked for
alignment and soft foot before acceptance.

## 5. When to take a spectrum

### 5.1 Trigger conditions

Request a spectrum reading from the analyst when any of the following occurs:

- Any stage-1 or stage-2 alert from the online package.
- RMS increases by more than 25 percent within a week in a stable regime.
- A machine has been re-coupled, realigned, or has had a bearing changed.
- A walk-down reveals a change in noise.

### 5.2 What the spectrum adds

The spectrum separates the broadband energy into its frequency components. For a bearing
fault the analyst looks for the outer-race (BPFO), inner-race (BPFI), ball (BSF), and cage
(FTF) defect frequencies and their harmonics in the envelope spectrum. For misalignment the
signature is at 2x running speed, for unbalance at 1x, for looseness at many harmonics. The
spectrum converts "the bearing is degrading" into "the outer race is spalled", which
determines the parts and the urgency.

### 5.3 Bearing defect frequencies for this fleet

For the 6313-C3 drive-end bearing at 1480 rpm (24.67 Hz shaft speed), 8 balls, 16.7 mm
ball diameter, 105 mm pitch diameter, zero contact angle:

- BPFO: 82.8 Hz
- BPFI: 114.6 Hz
- BSF: 73.5 Hz
- FTF: 10.35 Hz

Frequencies scale linearly with shaft speed; recompute for R1 and R3.

## 6. Alert limits used by the monitoring package

### 6.1 Trend limits (z-score against regime baseline)

- Vibration RMS z-score above 2.0 sustained for 3 consecutive 10-minute samples: stage 2.
- Vibration RMS z-score above 3.0: stage 3 regardless of absolute value.
- Bearing temperature z-score above 2.0: stage 2.
- Kurtosis above 4.5 or crest factor above 4.0 for 6 consecutive samples: stage 1.

### 6.2 Absolute limits (ISO)

- RMS above 4.5 mm/s (rigid) or 7.1 mm/s (flexible): stage 3.
- Bearing temperature above 95 C: stage 4, trip.

### 6.3 Suppression rules

Alerts are suppressed for 30 minutes after a regime change and for 60 minutes after a
start, because startup transients in R1 routinely push kurtosis above 6 and crest factor
above 5 with no fault present.

## 7. Common misreadings

### 7.1 Load change mistaken for a fault

The most common false alarm. Always check rpm and load before acting on an RMS or
temperature change. The z-score handles this automatically; the raw value does not.

### 7.2 Sensor fault mistaken for machine fault

A sudden step to a constant value, a reading of exactly zero, or a value that does not
change for six or more samples usually means a failed cable or a loose sensor, not a
machine fault. Check the sensor before raising a work order.

### 7.3 Grease-related transients

Re-greasing raises temperature by 5 to 10 K for a few hours and can briefly raise RMS.
Do not stage a bearing alert within 8 hours of a lubrication event.

### 7.4 Neighbouring machine

On shared foundations, vibration from an adjacent machine can appear on a healthy motor.
If two machines alarm together, look at both before ordering parts for either.
