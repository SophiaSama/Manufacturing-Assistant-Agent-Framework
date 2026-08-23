# Machine Technical Specification

| Field | Value |
|---|---|
| **Document ID** | TEC-301 |
| **Title** | RoboTech RX-700 Welding Robot – Technical Specification |
| **Equipment** | WELD-ROBOT-07 (RB-07), Station 42; WELD-ROBOT-12 (RB-12), Station 51 |
| **Revision** | Rev 11 |
| **Effective Date** | 2025-02-01 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. General Description

Six-axis electric servo welding robot for resistance spot welding of body-in-white panels. RB-07 welds the LH side-panel assembly (28 welds per cycle); RB-12 welds the roof-to-body flange (16 welds per cycle).

## 2. Mechanical Specifications

| Parameter | Value |
|---|---|
| Axes | 6 (A1–A6) |
| Payload (at wrist) | 150 kg |
| Reach (max) | 2.7 m |
| Repeatability | ± 0.05 mm |
| Wrist torque (A6) | 45 Nm |
| Weight (robot only) | 1,150 kg |
| Mounting | Floor, bolted (4 x M30) |

## 3. Electrical Specifications

| Parameter | Value |
|---|---|
| Supply voltage | 400 V AC, 3-phase, 50 Hz |
| Controller | RoboTech RC-800 |
| Servo amps | 6 x RB-AMP-400 |
| Weld controller | MFDC, 1,000 Hz, 20 kA max |
| Communication | Profinet to line PLC (Nexus NX-3000) |
| High-voltage discharge time | 5 minutes after power-off (servo bus) |

## 4. Process Parameters (Spot Welding)

| Parameter | Value |
|---|---|
| Weld current | 8.5 – 9.5 kA (program-dependent) |
| Weld time | 180 ms |
| Electrode force | 3.2 kN |
| Electrode cap | CP-RX-7712, CuCrZr |
| Cap life | 4,000 welds (hard limit in tracker) |
| Weld quality sensor | WQC-100 (monitors current/force) |
| Cooling water flow | ≥ 6 l/min through weld gun |

## 5. Maintenance Intervals

| Task | Interval |
|---|---|
| Grease axes A1–A6 | Every 2,000 operating hours |
| Gearbox oil check (A1–A3) | Every 6 months |
| TCP verification | Every shift start and after any collision |
| Cap dressing | Every 500 welds |
| Servo amp diagnostics | Weekly (auto-report to CMMS) |
| Full calibration | Annually |

## 6. Error Code Reference (Summary)

See SOP-TEC-201 for the fault-response procedure.

| Code | Meaning |
|---|---|
| E-4012 | Servo overcurrent (axis-dependent in code sub-field) |
| E-5023 | TCP deviation > 0.5 mm |
| E-5041 | Wire feed fault |
| E-4055 | Axis 5 joint limit |
| E-4031 | Weld quality low (WQC sensor) |

## 7. Spare Parts

| Spare | Part No. |
|---|---|
| Servo amplifier (A1) | SP-RX-04012 |
| Electrode cap | CP-RX-7712 |
| Weld gun transformer | SP-RX-03110 |
| Teach pendant cable | SP-RX-08900 |

## 8. Cross-References

- SOP-TEC-201 — Welding Robot RB-07 Fault Diagnosis and Recovery
- TEC-328 — Conveyor Line 2 (CV-2) Technical Specification
- ESC-402 — Escalation Workflow Matrix

## 9. Revision History

| Rev | Date | Change |
|---|---|---|
| 9 | 2024-07-01 | Initial release |
| 10 | 2024-11-15 | Cap life reduced from 5,000 to 4,000 welds |
| 11 | 2025-02-01 | Added Profinet comms and WQC-100 sensor reference |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
