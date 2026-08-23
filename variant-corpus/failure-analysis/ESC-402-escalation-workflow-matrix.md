# Escalation Workflow Matrix

| Field | Value |
|---|---|
| **Document ID** | ESC-402 |
| **Title** | Escalation Workflow and Response Matrix |
| **Department** | Plant Operations |
| **Revision** | Rev 8 |
| **Effective Date** | 2025-05-15 |
| **Owner** | Plant Manager, Dana Whitfield |
| **Applies To** | All stations and shifts, Northgate Assembly Plant |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the escalation levels, trigger conditions, responsible roles, and response time targets for operational, quality, and safety events. Escalation is the decision spine for everything from a station fault to a stop-ship decision.

## 2. Escalation Levels

### Level 1 – Line Response
| Field | Value |
|---|---|
| **Triggers** | Station fault, minor quality deviation, tool drift flag, sensor fault |
| **Responsible** | Operator → Shift Supervisor |
| **Response time** | Immediate; resolve within **30 minutes** |
| **Example** | Wheel torque tool out-of-tolerance at Station 144 (SOP-OPR-101 §4.3) |

### Level 2 – Maintenance / Supervision
| Field | Value |
|---|---|
| **Triggers** | Fault recurring 3x in 1 hour, equipment down < 2 h, drift not cleared by re-calibration, E-Stop RC-04 pattern |
| **Responsible** | Shift Supervisor → Maintenance Supervisor / Maintenance Technician |
| **Response time** | **2 hours** |
| **Example** | RB-07 fault E-4012 recurring (SOP-TEC-201 §5) |

### Level 3 – Engineering / Management
| Field | Value |
|---|---|
| **Triggers** | Equipment down > 2 h, audit failure > 1 of 10 vehicles, recurring defect pattern (3 identical NCs in 30 days), second servo amp failure in 7 days |
| **Responsible** | Maintenance Manager / Process Engineering Lead / Quality Engineer |
| **Response time** | **8 hours** |
| **Example** | Torque audit failure at Station 168 (SOP-OPR-158 §5) |

### Level 4 – Plant / Corporate
| Field | Value |
|---|---|
| **Triggers** | Any safety-critical (Class A) defect, stop-ship decision, potential field action, regulator notification, fire/serious injury |
| **Responsible** | Plant Manager + Quality Director (+ Corporate Quality VP) |
| **Response time** | **24 hours** (Class A: notify Quality Director within **4 hours**) |
| **Example** | Suspected wheel retention defect in field (QCR-501) |

## 3. Escalation Rules

1. Every escalation must be logged in the QDS with a timestamp and the initiating person's name.
2. The responding role may close the escalation only at their own level; cross-level resolution is documented by the higher level.
3. **Class A (safety) events jump directly to Level 4** — do not wait for lower-level closure.
4. Escalations not acknowledged within the response time automatically page the next level up (auto-flag in QDS).
5. Stop-ship authority resides with the Plant Manager (Level 4) and the Quality Director (Level 4) jointly; either may initiate, both must confirm to lift.

## 4. Response Time Clock

- The response clock starts when the event is logged in QDS with the correct event type.
- Shift handover does not reset the clock; the incoming supervisor inherits open escalations.

## 5. Key Contacts (Northgate Assembly Plant)

| Role | Name |
|---|---|
| Plant Manager | Dana Whitfield |
| Quality Director | Priya Raman |
| Maintenance Manager | Tom Okafor |
| Process Engineering Lead | Miguel Santos |
| Safety Officer | Karen Lindqvist |
| Shift A Supervisor | Elena Novak |
| Shift B Supervisor | Marcus Reed |
| Shift C Supervisor | Sofia Chen |

## 6. Cross-References

- FAP-401 — Failure Analysis Procedure
- QCR-501 — Recall Trigger Criteria
- SOP-OPR-158 — Final Inspection and Torque Audit
- SOP-TEC-201 / 214 / 228 / 235 — Equipment troubleshooting SOPs

## 7. Revision History

| Rev | Date | Change |
|---|---|---|
| 5 | 2024-06-20 | Initial release |
| 6 | 2024-10-05 | Added auto-flag rule for unacknowledged escalations |
| 7 | 2025-01-15 | Class A notification time tightened to 1 hour |
| 8 | 2025-05-15 | Class A notification time revised to 4 hours to reduce false alarms |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
