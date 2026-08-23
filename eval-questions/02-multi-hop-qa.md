# Eval Set 02 — Multi-Hop QA (cross-document reasoning)

**Goal:** answer requires reading ≥2 documents and connecting facts.

| # | Question | Reasoning chain | Expected answer |
|---|---|---|---|
| M1 | A torque audit at Station 168 fails on 2 of 10 re-audited vehicles. What happens next and who is involved? | SOP-OPR-158 §5 (stop source station, re-audit 10, >1 fail → escalate) + ESC-402 Level 3 | Stop Stations 144/118, re-audit, escalate to Process Engineering (Level 3, ≤8 h); open NC per FAP-401 |
| M2 | TQ-6012 reports a drift flag. Which documents govern what happens to the tool, and which station is affected? | SOP-TEC-214 (quarantine, recalibrate) + SOP-OPR-101 §4.1 (pre-op cal sticker) + TEC-314 (specs) + SOP-OPR-158 (re-audit) | Tool auto-quarantined in CMMS; re-calibrate on CR-500; station 144 halts; re-audit completed vehicles; NC if below-spec found |
| M3 | Why would a CV-2 false jam alarm cause a weld quality problem at Station 51? | TEC-328 §6 (spatter on PS-200-1) + SOP-TEC-228 + TEC-328 (pin wear → misalignment → E-5023) + SOP-TEC-201 | Spatter → jam; repeated jams cause pallet datum pin wear >1 mm → body misalignment → RB-12 TCP deviation E-5023 |
| M4 | A weld robot hits its cap life limit mid-shift. What stops production and what must be done? | TEC-301 (4,000-weld hard limit) + SOP-TEC-201 §4.3 (tip dress, replace cap) | Cell halts automatically; replace electrode cap CP-RX-7712, log in tool-life tracker; TCP re-check if needed |
| M5 | Vehicles built with TQ-6012 before the drift was caught — what determines whether they are affected? | SOP-TEC-214 §5 (re-audit since last verified audit) + SOP-OPR-158 (audit records) + QCR-501 (C3/C2) | Re-audit vehicles since last verified audit; below-spec fasteners → quarantine + NC; 3+ vehicles → field-action evaluation |
| M6 | Windshield retention fails the destructive lab test. What is the escalation and what recall criteria apply? | SOP-OPR-114 §5 (retention audit) + QCR-501 (windshield = Class A) + ESC-402 (Class A → L4, 1 h) + FAP-401 (8D mandatory) | Class A: immediate L4 notification (≤1 h), stop-ship, 8D per FAP-401, NRSA notification within 5 business days, field-action eval per C1/C2 |
| M7 | The fill machine at Station 131 has a moisture reading of 1.8% in the reservoir. What actions are required? | SOP-OPR-142 §5 (replace fluid if >1.5%) + SOP-TEC-214 (sensor cal) | Maintenance drains/replaces fluid; verify with test kit; log in QDS |
| M8 | Film build on clear coat drops to 38 µm for 3 consecutive bodies. What's the process? | TEC-342 (40–52 µm) + SOP-TEC-235 §4 (3/hour → stop line, Level 2) + FAP-401 + QCR-501 | Stop line, inform Paint Shop supervisor, escalate L2; open NC, 8D; PT-ROB-03 diagnosis per P-122 |
| M9 | A second servo amplifier fails on RB-07 within 7 days of the first. What is special about this? | SOP-TEC-201 §5 (batch analysis) + FAP-401 (D4 supplier parts) | Level 3 + parts-quality notification; supplier batch analysis per FAP-401; check parts lot traceability |
| M10 | You find 3 identical field complaints of wheel noise. Torque data shows 5 of 445 readings out of tolerance on Shift C (2025-04-08), all below spec; 2 more on Shift B (overtorque, 110.3/111.0 Nm). Which recall criteria apply? | QCR-501 C3 (3 complaints/30 days) + C2 (Shift C 5/445 = 1.12% > 0.5%) | C3 met (3 complaints) → field-action evaluation. C2 met: Shift C rate 1.12% > 0.5% safety threshold. Shift B at 0.53% is overtorque (different mechanism) — separate evaluation |
| M11 | Who must be involved in the decision to lift a stop-ship? | ESC-402 §3.5 + QCR-501 §5 | Plant Manager + Quality Director jointly; only after root cause confirmed + PCA verified (FAP-401 D6) |
| M12 | An operator reports the audit wrench TQ-6018 reads 104 Nm on a nut that TQ-6012 installed at 103 Nm. Both claim ±1.5%. Is this a conflict? | TEC-314 (±1.5% accuracy) — 104 vs 103 is within overlapping tolerances | No conflict: each tool has ±1.5% uncertainty; both readings consistent within measurement error. Flag only if drift >2% on self-checks |

## Scoring
- Full credit: correct chain + correct final action + doc IDs cited.
- Watch for: agents stopping at the first relevant doc, or inventing contacts/values not in the corpus.
