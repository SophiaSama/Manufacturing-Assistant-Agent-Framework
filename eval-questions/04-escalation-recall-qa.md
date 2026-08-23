# Eval Set 04 — Escalation & Recall Decision QA

**Goal:** test the agent's ability to apply ESC-402 and QCR-501 criteria to realistic events, including "is this a recall?" judgments.

| # | Scenario | Expected decision | Criteria/refs |
|---|---|---|---|
| E1 | One customer reports a wheel separating at highway speed. No other reports. | **Stop-ship + Level 4 within 1 hour + NRSA evaluation.** C1 (safety-critical defect) alone is sufficient — OR-condition, no rate needed. | QCR-501 C1, ESC-402 L4 |
| E2 | 9 vehicles were built on 2025-04-08 Shift C; 3 of them show lug-nut torque below spec (worst 96.8 Nm, no failures yet). | Field-action evaluation: C3 met (3 affected vehicles), C2 met (3/9 ≈ 33% of shift-C population > 0.5%). Quarantine, stop-ship until scope determined. | QCR-501 C2/C3 |
| E3 | Cosmetic paint imperfection (orange peel) on 5% of vehicles for a week. | No recall. Functional threshold 2% is for *functional* defects; cosmetic = S3 severity — monitor, improvement plan. Does **not** meet C2 (C2 is for safety/functional defects). | QCR-501 §2/§4 severity |
| E4 | Engine mount bolts found loose on 15 vehicles during audit; no field failures. | Stop-ship, Class A (engine mounts = Class A). C4 (>1,000 units) not yet met, C2 >0.5% likely — evaluate. 8D mandatory. | QCR-501, FAP-401 §2 |
| E5 | Brake pedal travel complaints: 2 in 60 days, different dealers. Moisture on a quarantined lot never reached vehicles. | No recall trigger yet (2 < 3 complaints, no confirmed safety defect in population). Monitor; verify no vehicle used suspect lot (SCAR-2025-007 containment). | QCR-501 C3 |
| E6 | A dealership reports windshield detachment on 1 vehicle at 55 mph; investigation finds adhesive void. | **C1 met** — immediate L4, stop-ship, NRSA within 5 business days. Scope: vehicles built with same adhesive lot; retention test 100% on population. | QCR-501 C1/C4, SOP-OPR-114 |
| E7 | NRSA opens a formal inquiry about torque data from your plant. | C5 met (regulatory trigger) regardless of internal rates — respond per QCR-501, freeze/verify all torque data, cooperate. | QCR-501 C5 |
| E8 | A functional defect (radio noise) affects 2.5% of vehicles — 300 units. | C2 met (2% functional threshold exceeded). Severity S2 → campaign likely, not full recall; population 300 < 1,000 (C4 not met) but C2 suffices — OR-condition. | QCR-501 C2 |
| E9 | Repair data shows the same wheel-bearing noise on 4 vehicles from the same lot within 20 days. Torque readings for that lot are all in tolerance. | C3 met (4 identical in 30 days) → field-action evaluation even though torque data is clean; investigate other causes (bearing supplier lot) via FAP-401. | QCR-501 C3, FAP-401 |
| E10 | Stop-ship has been active for 3 days. Root cause confirmed as tool drift; PCA installed and verified on 200 units. | Partial lifting allowed? No — lift requires PCA verified per FAP-401 D6 (30 days/1,000 units) + joint approval. Until then, containment stays. | QCR-501 §5, FAP-401 D6 |

## Key decision principles tested
1. **OR-condition**: any single criterion (C1–C5) is sufficient — agents often wrongly require multiple.
2. **Class A vs functional severity** changes the threshold (0.5% vs 2%) and the notification clock.
3. **Regulator trigger (C5)** applies even with clean internal data.
4. **Stop-ship lift** requires verification + joint authority — not just elapsed time.
