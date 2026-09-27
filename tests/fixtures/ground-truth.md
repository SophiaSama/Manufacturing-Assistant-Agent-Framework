# Ground Truth — Northgate Assembly Plant Corpus (Eval Answer Key)

> **Important:** Exclude this file from the agent's retrieval corpus when running RAG/QA evals. It exists to score answers.

## 1. Fastener / Torque Facts

| Question | Answer | Source |
|---|---|---|
| Wheel lug nut torque spec (Aurora/Solstice) | 105 Nm ± 5% (99.75–110.25 Nm) | SOP-OPR-101 §4.3 |
| Wheel tightening sequence | Star: 1→3→5→2→4 | SOP-OPR-101 §4.3 |
| Wheel station / tool | Station 144, tool TQ-6012 (TorqMaster TF-6000) | SOP-OPR-101 |
| Engine mount bolt torque | 62 Nm ± 2 Nm then +90° ± 5° (torque-to-yield), M12 | SOP-OPR-127 §4.2 |
| Rear transmission mount torque | 45 Nm ± 3 Nm, M10 (increased from 40 Nm in Rev 5) | SOP-OPR-127 §4.3 |
| Engine mount station / tool | Station 118, tool TQ-6015 | SOP-OPR-127 |
| Torque audit frequency | Every 2 hours, 3 consecutive vehicles | SOP-OPR-158 §4 |
| Audit failure response | Stop source station, re-audit 10 vehicles; >1 of 10 fails → ESC-402 Level 3 | SOP-OPR-158 §5 |
| Torque tool accuracy / calibration | ±1.5% of reading; every 3 months or 100,000 cycles | TEC-314, SOP-TEC-214 |
| Drift warning threshold | >2% between weekly self-checks (was 3% before Rev 3/Rev 4) | SOP-TEC-214, TEC-314 |

## 2. Process Facts

| Question | Answer | Source |
|---|---|---|
| Windshield adhesive bead | 8 mm base ± 1 mm, 12 mm height ± 1 mm | SOP-OPR-114 §4.2 |
| Max open time adhesive→glass | 10 minutes | SOP-OPR-114 §4.2 |
| Minimum cure before handling | 4 hours at 20–25 °C (6 h below 15 °C) | SOP-OPR-114 §4.4 |
| Windshield retention audit threshold | >1,200 N pull strength at 7 days (was 1,000 N) | SOP-OPR-114 §5 |
| Leak test frequency | Every 10th vehicle, 2 bar / 3 min | SOP-OPR-114 §5 |
| Brake fluid | DOT 4 (ISO 4925 Class 6), fill pressure 0.5–1.0 bar | SOP-OPR-142 |
| Brake bleed sequence | RR → RL → FR → FL | SOP-OPR-142 §4.2 |
| Fill machine pressure hold | 0.8 bar, 45 seconds | SOP-OPR-142 §4.2 |
| Clear coat film build target | 45 µm (acceptance 40–52 µm); deviation >15% below target → NC | TEC-342, SOP-TEC-235 |
| Paint booth conditions | 22 °C ± 2 °C, RH 55–65%, air 0.35 MPa | TEC-342 |

## 3. Equipment Facts

| Question | Answer | Source |
|---|---|---|
| Welding robot model / station | RoboTech RX-700, RB-07 @ Sta. 42 (also RB-12 @ Sta. 51) | TEC-301 |
| Robot payload / reach / repeatability | 150 kg / 2.7 m / ±0.05 mm | TEC-301 |
| Electrode cap life | 4,000 welds hard limit (was 5,000 before Rev 10) | TEC-301, SOP-TEC-201 |
| RB-07 error codes | E-4012 servo overcurrent, E-5023 TCP deviation >0.5 mm, E-5041 wire feed, E-4055 axis 5 limit, E-4031 low weld quality | SOP-TEC-201 |
| Torque tool range / drive | 5–150 Nm, 1/2" drive | TEC-314 |
| Conveyor CV-2 max speed / capacity | 12 m/min, 36 pallets/lane | TEC-328 |
| CV-2 sensors | PS-200 photoelectric (entry/exit), PR-100 proximity (lift) | TEC-328 |
| Most common CV-2 jam cause | Weld spatter on PS-200-1 | TEC-328 §6, SOP-TEC-228 |
| Paint robot atomizer | 65 mm rotary bell, 40,000 rpm, 150–350 ml/min, HV ≤60 kV operational | TEC-342 |
| PT-ROB-03 error codes | P-110 atomizer speed, P-122 flow deviation >10%, P-134 HV fault, P-145 air pressure | SOP-TEC-235 |

## 4. Escalation Facts

| Question | Answer | Source |
|---|---|---|
| Level 1 (line) | Shift supervisor; resolve ≤ 30 min | ESC-402 |
| Level 2 (maintenance) | Maintenance supervisor; ≤ 2 h | ESC-402 |
| Level 3 (engineering) | Maintenance Mgr / Process Eng / Quality Eng; ≤ 8 h | ESC-402 |
| Level 4 (plant/corporate) | Plant Manager + Quality Director; ≤ 24 h | ESC-402 |
| Class A safety event notification | Quality Director within 1 hour, direct to Level 4 | ESC-402, QCR-501 |
| Escalation triggers (examples) | 3x fault in 1 h → L2; down >2 h → L3; audit >1/10 fail → L3; 3 identical NCs in 30 days → L3 | ESC-402, SOP-TEC-201, SOP-OPR-158 |
| Stop-ship authority | Plant Manager + Quality Director jointly; both must confirm to lift | ESC-402 §3.5 |

## 5. Failure Analysis (8D) Facts

| Question | Answer | Source |
|---|---|---|
| When 8D is mandatory | Class A NC, 3 identical NCs/30 days, field complaint, >2 h production stop, audit failure | FAP-401 §2 |
| Containment start target | D1–D3 within 24 hours | FAP-401 §6 |
| PCA verification period | 30 days or 1,000 units, whichever longer | FAP-401 D6 |
| Closure target | 60 days; approved by Plant Manager | FAP-401 D8 |
| Escalation from 8D to field action | Root cause affecting >500 units OR defect rate >0.5% | FAP-401 §5 |

## 6. Recall Criteria Facts

| Question | Answer | Source |
|---|---|---|
| C1 safety defect | Immediate stop-ship, Level 4 within 1 h, NRSA notify | QCR-501 §3, §6 |
| C2 defect rate thresholds | >0.5% (5,000 ppm) safety-related; >2% (20,000 ppm) functional | QCR-501 §3 |
| C3 complaint pattern | 3+ identical complaints in rolling 30 days | QCR-501 §3 |
| C4 affected population | >1,000 vehicles | QCR-501 §3 |
| C5 regulatory | NRSA inquiry / GVSR non-compliance / authority investigation | QCR-501 §3 |
| Criteria relationship | OR — any single criterion suffices | QCR-501 §3 note |
| NRSA notification timeline | Within 5 business days of confirming safety defect | QCR-501 §6 |
| Owner notification | Within 60 days of NRSA approval | QCR-501 §4 |
| Recall approval >10,000 vehicles | Corporate Quality VP required | QCR-501 §4 |
| Class A systems list | Brakes, steering, airbags, seat belts, fuel, wheel retention, engine mounts, windshield retention | QCR-501 §2 |

## 7. People

| Role | Name |
|---|---|
| Plant Manager | Dana Whitfield |
| Quality Director | Priya Raman |
| Maintenance Manager | Tom Okafor |
| Process Engineering Lead | Miguel Santos |
| Safety Officer | Karen Lindqvist |
| Shift A / B / C Supervisors | Elena Novak / Marcus Reed / Sofia Chen |

## 8. Deliberate Consistency Checks (for cross-doc reasoning evals)

1. TQ-6018 (audit wrench) appears in SOP-OPR-158 §4 and SOP-TEC-214 §2 — calibration interval must be consistent (3 months/100k).
2. Engine mount 62 Nm + 90° appears in SOP-OPR-127, SOP-TEC-214 (T+Y mode), TEC-314 (T+Y mode) — identical.
3. Cap life 4,000 welds appears in TEC-301 §4 and SOP-TEC-201 §4.3 — identical.
4. Film build 45 µm target appears in TEC-342 §2 and SOP-TEC-235 §4 — identical.
5. Wheel torque audit interval 2 h appears in SOP-OPR-101 §5 and SOP-OPR-158 §4 — identical.
