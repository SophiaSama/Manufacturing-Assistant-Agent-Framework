# Eval Set 01 — Retrieval QA (single-document fact lookup)

**Corpus:** main corpus (README + operator-sops/ + technician-sops/ + machine-details/ + failure-analysis/ + recall-quality/).
**Goal:** test accurate fact retrieval. Expected answers cite the source doc.

| # | Question | Expected answer | Source |
|---|---|---|---|
| R1 | What is the wheel lug nut torque target at Station 144? | 105 Nm ± 5% (99.75–110.25 Nm) | SOP-OPR-101 |
| R2 | In what sequence are lug nuts tightened? | Star: 1→3→5→2→4 | SOP-OPR-101 |
| R3 | What is the torque audit frequency at final inspection? | Every 2 hours, 3 consecutive vehicles | SOP-OPR-158 |
| R4 | What tool is used for the torque audit? | TQ-6018 (TorqMaster TF-6000, calibrated weekly) | SOP-OPR-158 |
| R5 | What is the windshield adhesive bead dimension? | 8 mm ± 1 mm base, 12 mm ± 1 mm height | SOP-OPR-114 |
| R6 | What is the minimum windshield adhesive cure time? | 4 hours at 20–25 °C (6 h below 15 °C) | SOP-OPR-114 |
| R7 | What is the windshield retention test threshold? | > 1,200 N pull strength at 7 days cure | SOP-OPR-114 |
| R8 | What is the engine mount bolt tightening spec? | 62 Nm ± 2 Nm + 90° ± 5° (torque-to-yield) | SOP-OPR-127 |
| R9 | What is the rear transmission mount torque? | 45 Nm ± 3 Nm | SOP-OPR-127 |
| R10 | What brake fluid is used and at what fill pressure? | DOT 4 (ISO 4925 Class 6), 0.5–1.0 bar | SOP-OPR-142 |
| R11 | What is the brake bleed sequence? | RR → RL → FR → FL | SOP-OPR-142 |
| R12 | What does error code E-4012 on RB-07 mean? | Servo overcurrent, axis 1 | SOP-TEC-201 |
| R13 | What does error code E-5023 on RB-07 mean? | TCP deviation > 0.5 mm | SOP-TEC-201 |
| R14 | What is the electrode cap life limit? | 4,000 welds | TEC-301 / SOP-TEC-201 |
| R15 | What is the TorqMaster TF-6000 torque range and accuracy? | 5–150 Nm, ±1.5% of reading | TEC-314 |
| R16 | What is the torque tool calibration interval? | 3 months or 100,000 cycles | SOP-TEC-214 / TEC-314 |
| R17 | What drift threshold flags a torque tool? | > 2% between weekly self-checks | SOP-TEC-214 |
| R18 | What is CV-2's maximum speed and capacity? | 12 m/min, 36 pallets per lane | TEC-328 |
| R19 | What is the most common CV-2 jam cause? | Weld spatter on photoelectric sensor PS-200-1 | TEC-328 / SOP-TEC-228 |
| R20 | What is the clear coat film build target? | 45 µm (acceptance 40–52 µm) | TEC-342 |
| R21 | What is the paint robot atomizer speed? | Up to 40,000 rpm (65 mm bell) | TEC-342 |
| R22 | What is the 8D verification period before closure? | 30 days or 1,000 units, whichever longer | FAP-401 |
| R23 | Who must approve 8D closure? | Plant Manager | FAP-401 |
| R24 | What are the four escalation response-time targets? | L1 ≤30 min, L2 ≤2 h, L3 ≤8 h, L4 ≤24 h | ESC-402 |
| R25 | Within what time must the Quality Director be notified of a Class A defect? | 1 hour | ESC-402 / QCR-501 |
| R26 | What are the recall trigger defect-rate thresholds? | >0.5% safety-related, >2% functional | QCR-501 |
| R27 | How many identical complaints in 30 days trigger evaluation? | 3 | QCR-501 |
| R28 | What affected-population size triggers evaluation? | > 1,000 vehicles | QCR-501 |
| R29 | How long do you have to notify NRSA? | 5 business days | QCR-501 |
| R30 | Which systems are Class A (safety-critical)? | Brakes, steering, airbags, seat belts, fuel, wheel retention, engine mounts, windshield retention | QCR-501 |

## Scoring
- Full credit: exact value + correct source doc ID.
- Partial: correct value, no source.
- Note: R3/R6/R9/R17/R28 test attention to *current* revision values (revision history contains older numbers).
