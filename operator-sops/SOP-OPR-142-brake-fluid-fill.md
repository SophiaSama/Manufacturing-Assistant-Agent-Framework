# Standard Operating Procedure

| Field | Value |
|---|---|
| **Document ID** | SOP-OPR-142 |
| **Title** | Brake Fluid Fill and Bleed |
| **Department** | General Assembly – Chassis Line |
| **Station** | Station 131 (Brake Fill) |
| **Revision** | Rev 3 |
| **Effective Date** | 2024-12-05 |
| **Owner** | Process Engineering Lead, Miguel Santos |
| **Applies To** | Aurora (AU-2025), Solstice (SO-2025) |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the pressure-fill and bleeding procedure for the hydraulic brake system at Station 131 to guarantee a fluid-free-air system per requirement ENGR-3415.

## 2. Scope

Applies to Station 131 operators and to maintenance personnel performing fill-machine calibration.

## 3. Equipment and Fluid

| Item | Spec |
|---|---|
| Fill machine | Nexus BF-900, pressure fill |
| Brake fluid | DOT 4, part no. FL-88200 (ISO 4925 Class 6) |
| Fluid test kit | Checks boiling point (wet) ≥ 155 °C |

## 4. Procedure

### 4.1 Pre-Operation Checks
1. Confirm fill machine Nexus BF-900 pressure is set to **0.5 – 1.0 bar**.
2. Verify the fluid reservoir is connected to the correct model color-coded coupling (Aurora = blue, Solstice = green).
3. Check fluid level in the machine tank; top up with DOT 4 only. Never mix DOT 3 and DOT 4.

### 4.2 Fill Sequence
1. Connect the fill nozzle to the reservoir. The machine vents the ABS module automatically.
2. Run the automatic fill cycle: pressure hold at 0.8 bar for 45 seconds.
3. Bleed sequence (manual verification): **RR → RL → FR → FL**.
4. Verify the reservoir level settles at the MAX mark ± 2 mm.

### 4.3 Post-Fill Verification
1. Depress the brake pedal 5 times; pedal must hold firm with less than 10 mm travel at the 5th application.
2. Inspect all 4 calipers and the ABS block for leaks with a torch. Any drip = quarantine vehicle, tag for repair line.
3. Record fill pressure and fluid lot number in QDS.

## 5. Quality Requirements

- Fluid lot must match the approved supplier lot list (see APP-07). Unlisted lots are quarantined pending chemical lab approval.
- Moisture check: machine drains and replaces fluid if reservoir moisture content exceeds 1.5% (checked weekly by maintenance).
- Acceptance: zero air bubbles at caliper bleed screws; pedal travel within spec.

## 6. Safety Requirements

- PPE: safety glasses, chemical-resistant gloves. DOT 4 damages paint and is an eye hazard.
- Wipe any spills immediately; brake fluid is hygroscopic and corrosive.
- Ensure the vehicle is on level ground before filling.

## 7. Cross-References

- FAP-401 — Failure Analysis Procedure
- ESC-402 — Escalation Workflow Matrix
- QCR-501 — Recall Trigger Criteria (brake system is safety-critical)
- SOP-TEC-214 — Calibration of fill machine sensors

## 8. Revision History

| Rev | Date | Change |
|---|---|---|
| 1 | 2024-07-08 | Initial release |
| 2 | 2024-10-01 | Bleed sequence clarified (RR→RL→FR→FL) |
| 3 | 2024-12-05 | Added moisture check requirement for machine reservoir |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
