# Standard Operating Procedure

| Field | Value |
|---|---|
| **Document ID** | SOP-OPR-127 |
| **Title** | Engine Mount Installation |
| **Department** | General Assembly – Engine Dress Line |
| **Station** | Station 118 (Engine Dress) |
| **Revision** | Rev 6 |
| **Effective Date** | 2025-05-05 |
| **Owner** | Process Engineering Lead, Miguel Santos |
| **Applies To** | Aurora (AU-2025, 2.0L turbo); Solstice (SO-2025, 2.5L NA) |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the installation procedure for the two engine mounts (right-hand and left-hand) and the rear transmission mount at Station 118, including the torque-to-yield fastening of the M12 mounting bolts.

## 2. Scope

Applies to Station 118 operators on all shifts. Engine mount installation is a **safety-critical** operation; deviation triggers the escalation process in ESC-402.

## 3. Components and Fasteners

| Item | Spec | Part No. |
|---|---|---|
| RH engine mount | Hydraulic, AU-2025 | AU-51040 |
| LH engine mount | Hydraulic, AU-2025 | AU-51041 |
| Rear transmission mount | Rubber-isolated | AU-51050 |
| Mount bolts (RH/LH) | M12 x 1.25, grade 10.9 | B-99122 |
| Rear mount bolts | M10 x 1.5, grade 10.9 | B-99130 |

## 4. Procedure

### 4.1 Pre-Operation Checks
1. Confirm the engine is centered on the dress line carrier within ± 2 mm of the datum (checked by the laser at Station 116).
2. Verify part numbers on the mount labels match the vehicle build sheet (VCATS).
3. Inspect the hydraulic mount for fluid leaks. Any leak = reject the mount.

### 4.2 RH / LH Mount Installation
1. Position the mount on the engine bracket; the locating dowel must seat fully.
2. Install 4 mounting bolts M12 x 1.25 by hand, 2 turns each, alternating.
3. Tighten with tool **TQ-6015** (TorqMaster TF-6000) using torque-to-yield mode:
   - First stage: **62 Nm ± 2 Nm**
   - Second stage: **rotate an additional 90° ± 5°**
4. The tool records angle and torque to the QDS. Out-of-spec readings halt the station.

### 4.3 Rear Transmission Mount
1. Install 2 bolts M10 x 1.5. Tighten to **50 Nm ± 3 Nm** with tool TQ-6015 in standard mode.
2. Confirm the mount isolator is not pre-loaded (visual gap check per WIS-118-02).

### 4.4 Post-Installation
1. Lower the engine onto the chassis (Station 119) only after Station 118 green light confirms all 10 fasteners within spec.
2. Record operator stamp in VCATS.

## 5. Quality Requirements

- Torque-to-yield joints are single-use; a bolt removed after torquing must be replaced.
- QC audit: 2 vehicles per shift at Station 168 re-check RH mount bolt angle on the audit rig.
- Acceptance: first-stage torque within 60–64 Nm; final angle within 85–95°.

## 6. Safety Requirements

- PPE: safety glasses, steel-toe boots, arm protection.
- Never place hands between the engine and the carrier while the carrier is moving.
- Hydraulic mount fluid is skin irritant; wash affected area immediately.

## 7. Cross-References

- SOP-TEC-214 — Torque Tool Calibration and Drift Troubleshooting
- SOP-OPR-158 — Final Inspection and Torque Audit
- ESC-402 — Escalation Workflow Matrix
- QCR-501 — Recall Trigger Criteria (engine mount failures are safety-critical)

## 8. Revision History

| Rev | Date | Change |
|---|---|---|
| 3 | 2024-06-15 | Initial release |
| 4 | 2024-10-22 | Added torque-to-yield angle recording to QDS |
| 5 | 2025-02-10 | Rear mount torque increased from 40 Nm to 45 Nm |
| 6 | 2025-05-05 | Rear mount torque increased to 50 Nm after durability review |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
