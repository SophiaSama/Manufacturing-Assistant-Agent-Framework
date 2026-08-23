# Standard Operating Procedure

| Field | Value |
|---|---|
| **Document ID** | SOP-OPR-101 |
| **Title** | Wheel Installation and Lug Nut Torque |
| **Department** | General Assembly – Chassis Line |
| **Station** | Station 144 (Wheel Install) |
| **Revision** | Rev 8 |
| **Effective Date** | 2025-05-01 |
| **Owner** | Process Engineering Lead, Miguel Santos |
| **Applies To** | Aurora (AU-2025), Solstice (SO-2025) |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the standardized procedure for installing road wheels and applying the correct lug nut torque on the Aurora and Solstice models at Station 144. This procedure ensures wheel retention meets the design requirement of 105 Nm ± 5% and prevents lug nut fatigue failures.

## 2. Scope

Applies to all operators assigned to Station 144 on Shifts A, B, and C, and to quality auditors performing torque audits. Does not apply to spare-wheel or aftermarket operations.

## 3. Required Equipment

| Item | Spec | Tool ID |
|---|---|---|
| Torque wrench (battery-operated) | TorqMaster TF-6000, 5–150 Nm range | TQ-6012 |
| Torque reaction arm | Passive, model RA-200 | RA-201 |
| Lug nuts | M12 x 1.5, 5 per wheel, part no. AU-44102 (Aurora) / SO-44102 (Solstice) | — |
| Wheels | 18" alloy (AU), 17" steel (SO) | — |
| Air duster | 6 bar max | — |

## 4. Procedure

### 4.1 Pre-Operation Checks
1. Verify the torque tool TQ-6012 displays a valid calibration sticker dated within the last 3 months (see SOP-TEC-214 for calibration policy).
2. Confirm the tool battery level is ≥ 50%. Do not start the station with a low battery.
3. Inspect the wheel mounting face for dirt, burrs, or corrosion. Reject and quarantine the wheel if the mounting face has scratches deeper than 0.3 mm.
4. Verify the hub studs are clean and free of thread damage. Thread-damaged studs must be reported to the shift supervisor immediately.

### 4.2 Wheel Installation
1. Lift the wheel onto the hub using two hands. Use the positioning aid at the station to align the 5 stud holes.
2. Hand-start all 5 lug nuts. Never use the power tool to start a nut.
3. Run the tool in tightening mode.

### 4.3 Torque Application
1. Apply torque in a **star (criss-cross) sequence**: 1 → 3 → 5 → 2 → 4 (see the diagram on the station board).
2. Target torque: **108 Nm ± 5% (102.6 Nm – 113.4 Nm)**.
3. The tool TQ-6012 logs each cycle to the station PLC. If the tool reports an out-of-tolerance reading, the station halts and a red light illuminates. Do not release the vehicle until a maintenance technician clears the fault (see SOP-TEC-214).
4. Confirm the green station light before releasing the vehicle.

### 4.4 Post-Operation
1. Visually confirm all 5 lug nuts are seated flush with the wheel face.
2. Stamp the wheel with the operator number and shift code.

## 5. Quality Requirements

- Torque audit: QC performs a re-torque audit every **2 hours** on 3 consecutive vehicles (see SOP-OPR-158).
- Acceptance: 100% of audited nuts within 105 Nm ± 5%.
- Recorded torque readings are stored in the Quality Data System (QDS) under station ID 144.

## 6. Safety Requirements

- Wear approved PPE: safety glasses, steel-toe boots, cut-resistant gloves.
- Keep hands clear of the reaction arm travel path while the tool is cycling.
- Report any tool that makes abnormal noise or vibration to maintenance before the next vehicle.

## 7. Cross-References

- SOP-TEC-214 — TorqMaster TF-6000 Calibration and Drift Troubleshooting
- SOP-OPR-158 — Final Inspection and Torque Audit
- TEC-314 — TorqMaster TF-6000 Technical Specification
- QCR-501 — Recall Trigger Criteria (wheel retention is a safety-critical system)

## 8. Revision History

| Rev | Date | Change |
|---|---|---|
| 5 | 2024-08-02 | Initial release |
| 6 | 2024-11-19 | Updated torque audit frequency from 4h to 2h |
| 7 | 2025-03-15 | Added tool TQ-6012 to required equipment; updated calibration reference |
| 8 | 2025-05-01 | Torque spec updated to 108 Nm per latest validation study |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
