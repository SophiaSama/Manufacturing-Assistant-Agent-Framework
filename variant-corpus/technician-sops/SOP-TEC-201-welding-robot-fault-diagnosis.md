# Standard Operating Procedure

| Field | Value |
|---|---|
| **Document ID** | SOP-TEC-201 |
| **Title** | Welding Robot RB-07 Fault Diagnosis and Recovery |
| **Department** | Maintenance – Body Shop |
| **Equipment** | WELD-ROBOT-07 (RoboTech RX-700), Station 42 |
| **Revision** | Rev 8 |
| **Effective Date** | 2025-03-01 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the diagnosis and recovery procedure for the most common fault codes on welding robot RB-07 at Station 42 (LH side-panel weld). This document is the first line of defense before a full robot teardown (see TEC-301 for technical specifications).

## 2. General Safety Rules

1. Always place the robot in **teach mode** and lock out the cell with the safety gate key before entering the robot envelope.
2. Verify the robot has come to a full stop before any manual work.
3. High-voltage circuits (servo amps) hold charge for up to 5 minutes after power-off. Wait 5 minutes before touching electrical cabinets.

## 3. Fault Code Table

| Code | Description | Likely Cause | Action |
|---|---|---|---|
| E-4012 | Servo overcurrent, axis 1 | Servo amplifier fault or cable chafing | Check cable harness; if OK, replace servo amp (spare: SP-RX-04012) |
| E-5023 | TCP deviation > 0.5 mm | Worn electrode cap or tool offset drift | Replace electrode cap, re-teach TCP (see 4.2) |
| E-5041 | Wire feed fault | Kink in wire liner, wrong wire tension | Clear liner, set tension 2.2–2.6 N |
| E-4055 | Axis 5 joint limit reached | Program path drift | Re-zero axis 5, verify path against WIS-42-03 |
| E-4031 | Weld quality low (from WQC sensor) | Tip wear or gas flow < 12 l/min | Check gas flow; dress or replace tip |

## 4. Recovery Procedures

### 4.1 General Reset
1. Clear the fault in the controller HMI (user level).
2. Press reset, then jog the robot to a safe position at 10% speed.
3. If the fault recurs 3 times within 1 hour, do **not** keep resetting. Escalate per ESC-402 Level 2.

### 4.2 TCP Re-Teach (E-5023)
1. Remove the electrode cap and replace with a new one (part no. CP-RX-7712).
2. Run the automated TCP calibration cycle (TCPCAL.MAC) with the calibration pin.
3. Accept the result only if deviation is < 0.2 mm. Otherwise re-run.

### 4.3 Tip Change (E-4031)
1. Use the tip dresser cycle D-42 before changing tips.
2. If dressing does not restore weld quality, replace the electrode cap and log the change in the tool life tracker.
3. Cap life limit: **4,000 welds**. The tracker halts the cell when the limit is reached.

## 5. Escalation Triggers

- Same fault code more than 3 times in 1 hour → Level 2 (Maintenance Supervisor).
- Robot down longer than 2 hours → Level 3 (Process Engineering + Maintenance Manager).
- Suspected servo amp failure on a second robot within 7 days → Level 3 with parts-quality notification (batch analysis per FAP-401).

## 6. Cross-References

- TEC-301 — RoboTech RX-700 Technical Specification
- ESC-402 — Escalation Workflow Matrix
- FAP-401 — Failure Analysis Procedure
- SOP-OPR-158 — Final Inspection (weld audit reference)

## 7. Revision History

| Rev | Date | Change |
|---|---|---|
| 6 | 2024-07-25 | Initial release |
| 7 | 2024-11-08 | Added WQC sensor fault E-4031 |
| 8 | 2025-03-01 | Wire tension spec updated to 2.2–2.6 N |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
