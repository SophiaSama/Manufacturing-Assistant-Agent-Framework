# Standard Operating Procedure

| Field | Value |
|---|---|
| **Document ID** | SOP-TEC-228 |
| **Title** | Conveyor CV-2 Jam Recovery and E-Stop Reset |
| **Department** | Maintenance – Material Handling |
| **Equipment** | Body-in-White Conveyor Line 2 (CV-2), Zone 2B |
| **Revision** | Rev 6 |
| **Effective Date** | 2025-02-18 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the safe recovery procedure for conveyor line CV-2 when a pallet jam occurs or an emergency stop is triggered. CV-2 carries body-in-white shells between Station 42 and Station 51 at up to 12 m/min.

## 2. Jam Causes (Most Common First)

1. Pallet misalignment at the Zone 2B transfer point (photoelectric sensor PS-200 blocked by weld spatter).
2. Obstructed proximity sensor PR-100 (metal chips) on the lift section.
3. Oversized weld spatter on the pallet datum pins.
4. PLC program fault in the Nexus NX-3000 controller (rare, log required).

## 3. Recovery Procedure

### 3.1 Confirm the Fault
1. Read the fault message on the CV-2 HMI. Note the zone and sensor ID.
2. Check sensor status: PS-200 (Zone 2B entry) and PR-100 (lift) must both be clear before restart.

### 3.2 Clear the Jam (line stopped, lockout required)
1. Apply lockout/tagout (LOTO) to the CV-2 main disconnect.
2. Enter the conveyor pit wearing a hi-vis vest and hard hat.
3. Remove spatter or debris. Inspect the pallet datum pins for damage (worn pins > 1 mm wear = replace).
4. Confirm no bodies are contacting the side rails.
5. Remove LOTO, restore power.

### 3.3 Restart Sequence
1. On the HMI: press **RESET**, then start the conveyor in **jog mode** at 20% speed.
2. Verify the pallet passes the PS-200 and PR-100 sensors and the transfer completes.
3. Return to full speed only after 3 successful transfers.

## 4. E-Stop Reset Rules

- E-Stop resets are performed **only** by maintenance personnel or the shift supervisor.
- After an E-Stop, walk the full conveyor zone and confirm no person is in the envelope.
- Record every E-Stop in the CMMS with the reason code (RC-01 debris, RC-02 operator, RC-03 sensor, RC-04 unknown).
- More than 3 E-Stops per shift with reason code RC-04 → escalate per ESC-402 Level 2.

## 5. Safety Requirements

- Two-person rule when working in the conveyor pit.
- Never reach over a moving conveyor section.
- Lockout/tagout is mandatory — no exceptions for "quick clears."

## 6. Cross-References

- TEC-328 — Conveyor Line 2 (CV-2) Technical Specification
- ESC-402 — Escalation Workflow Matrix
- FAP-401 — Failure Analysis Procedure (recurring jams)

## 7. Revision History

| Rev | Date | Change |
|---|---|---|
| 4 | 2024-07-11 | Initial release |
| 5 | 2024-12-02 | Added E-Stop reason codes RC-01..RC-04 |
| 6 | 2025-02-18 | Added 3-successful-transfers rule before full speed |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
