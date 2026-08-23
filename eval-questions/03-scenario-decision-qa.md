# Eval Set 03 — Manufacturing Scenario Decisions

**Goal:** test operational decision-making grounded in the corpus. No single right answer is scripted; score against the rubric per question.

## S1 — Shift C torque audit failure
**Scenario:** At 05:45 on 2025-04-09, the final inspection audit (3 vehicles from 2025-04-08 Shift C) finds lug nut torque below spec on all 3 vehicles. The worst reading is 96.8 Nm.

**Task:** Walk through the containment, escalation, and evaluation actions. What data would you pull from the production database to scope the affected population? Which recall criteria are in play?

**Expected elements:**
- Stop-ship/containment; NC-2025-0137 (Class A); ESC-402 L2→L3 (re-audit 10; >1/10 fail) → L4 (Class A within 1 h)
- Suspect population: vehicles built 2025-04-08 Shift C (9 in DB) + all vehicles torqued since last verified audit
- Tool: TQ-6012 drift check (SOP-TEC-214); spare TQ-6019 swap; re-audit scope per SOP-TEC-214 §5
- 8D per FAP-401; field-action eval per QCR-501 (C3 met with 3 affected vehicles; C2 rate 3/9 ≈ 33% of shift-C population, but 5/1,205 ≈ 0.41% overall — must argue the right denominator)
- NRSA notification clock if confirmed safety defect (5 business days)

## S2 — Weld quality sensor alarm
**Scenario:** WQC sensor on RB-07 reports low weld quality on 6 of the last 40 cycles. Error E-4031. It's 30 minutes before shift change.

**Task:** Decide the immediate action and the handover notes.

**Expected elements:**
- SOP-TEC-201: check gas flow ≥12 l/min, dress/replace tip; cap life tracker (4,000)
- If fault recurs 3x in 1 hour → Level 2 (SOP-TEC-201 §5); don't keep resetting
- Shift handover: response clock does not reset (ESC-402 §4)

## S3 — Brake fluid lot question
**Scenario:** Incoming inspection reports a FluidWorks DOT 4 lot at 2.1% moisture. Station 131 is running low on fluid.

**Task:** Can the station use this lot? What else must be verified?

**Expected elements:**
- No — moisture > 1.5% spec (SOP-OPR-142 §5); lot quarantined (SCAR-2025-007, APP-07)
- Verify fill records confirm no vehicle filled with suspect lot; only approved lots (APP-07) may be used
- If no approved stock: stop Station 131 rather than use unapproved fluid

## S4 — Conveyor E-Stop pattern
**Scenario:** Zone 2B CV-2 had 4 E-Stops this shift, all reason code RC-04 (unknown).

**Task:** What do you do?

**Expected elements:**
- SOP-TEC-228 §4: >3 RC-04 E-Stops per shift → escalate Level 2
- Investigate before restart; LOTO; check sensors PS-200/PR-100; spatter (TEC-328 §6)
- Log each E-Stop with correct code; wrong reason codes are themselves a process deviation

## S5 — Overtorqued lug nut
**Scenario:** An audit at Station 168 finds one lug nut at **111.0 Nm** (over 110.25 Nm) on NGA-AU25-0037. Everything else is in spec. (This reading is real in the DB — verify via SQL: `SELECT * FROM torque_readings WHERE vin='NGA-AU25-0037' AND in_tolerance=0`.)

**Task:** Is this a defect? What's the disposition?

**Expected elements:**
- Yes — outside 105 ± 5% tolerance (SOP-OPR-101 §5, SOP-OPR-158 §4: 100% within tolerance)
- Single failure → stop source station, re-audit 10 vehicles (SOP-OPR-158 §5); >1/10 → Level 3
- Overtorque on wheel studs: stud stretch risk → engineering disposition per FAP-401, not operator decision

## S6 — Paint robot flow deviation
**Scenario:** PT-ROB-03 shows P-122. The last 2 bodies have film build of 43 µm and 41 µm (still in band).

**Task:** Respond.

**Expected elements:**
- SOP-TEC-235 §3.2: purge, check filter (25 µm), pump self-test, recalibrate CAL-ST-04
- 3+ blend-line defects/hour → stop line + Level 2; film build >15% below target → NC (SOP-TEC-235 §4)
- 41–43 µm is within 40–52 band — no NC yet, but monitor; document in CMMS

## S7 — Temp worker on torque station
**Scenario:** A contractor is assigned to Station 144 but has no SOP-OPR-101 certification on file.

**Task:** Can they run the station?

**Expected elements:**
- No — station coverage requires certified operator (TR-2025-030); torque SOPs require certification
- Follow same logic as expired cert (E-3002/SOP-TEC-214): not authorized until certified
- Supervisor must not assign until training complete; violates ESC-402 spirit if forced

## S8 — Overnight windshield cure
**Scenario:** Ambient temperature in the glass area is 13 °C during the winter. A vehicle's windshield is installed at 23:00 and the shift ends at 06:00.

**Task:** When can the vehicle be handled?

**Expected elements:**
- SOP-OPR-114 §4.4: below 15 °C → 6 hours cure; vehicle ready at 05:00
- Do not use the 4-hour rule; misapplication is a retention risk (Class A)

---

## Scoring Rubric (per scenario)
| Level | Description |
|---|---|
| 3 | Correct action + correct doc citations + correct escalation level/timing |
| 2 | Correct action, missing citations or wrong timing |
| 1 | Plausible but wrong/unsupported action |
| 0 | Hallucinated procedure or unsafe action |
