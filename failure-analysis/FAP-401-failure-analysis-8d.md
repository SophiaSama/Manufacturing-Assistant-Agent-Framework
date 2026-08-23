# Failure Analysis Procedure

| Field | Value |
|---|---|
| **Document ID** | FAP-401 |
| **Title** | Failure Analysis Procedure (8D Methodology) |
| **Department** | Quality – Failure Analysis Team |
| **Revision** | Rev 4 |
| **Effective Date** | 2025-02-25 |
| **Owner** | Quality Director, Priya Raman |
| **Applies To** | All NC records, field complaints, and internal process failures |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the structured failure analysis process (8D methodology) for all product and process failures at Northgate Assembly Plant, from initial containment to closure. This procedure is the foundation for deciding whether a quality issue rises to a recall-level action (see QCR-501).

## 2. When to Use

Failure analysis is mandatory when any of the following occurs:

1. Any **Class A (safety-related)** non-conformance — brake, steering, airbag, seat belt, fuel system, wheel retention, engine mounts, windshield retention.
2. Any repeated defect: **3 or more identical NCs within 30 days**.
3. Field complaint received from a dealership or regulator.
4. Production stop of more than 2 hours caused by a process failure.
5. Audit failure per SOP-OPR-158 §5.

## 3. Roles and Responsibilities

| Role | Responsibility |
|---|---|
| Failure Analysis Lead (FAL) | Owns the 8D, coordinates the team |
| Quality Engineer | Data collection, containment verification |
| Process Engineer | Process root-cause analysis |
| Maintenance Manager | Equipment root-cause analysis |
| Supplier Quality Engineer (SQE) | Supplier component analysis |
| Plant Manager | Approves closure and field actions |

## 4. 8D Process

### D1 – Team Formation
- FAL assigned within **4 hours** of the failure being logged.
- Team includes Quality, Process, Maintenance, and SQE (if supplier parts involved).

### D2 – Problem Description
- Document the failure in measurable terms: what, where, when, how many, defect rate, photos, and VINs affected.
- Log in the QDS with NC number.

### D3 – Containment
- **Immediate:** quarantine suspect inventory, stop-ship per ESC-402.
- **Short-term:** 100% inspection or 100% re-verification at the source station.
- Containment remains active until D5 verification is complete.

### D4 – Root Cause Analysis
- Use 5-Why and Ishikawa. Equipment faults additionally require the machine log review (CMMS) and error-code timeline.
- **Do not** close D4 on a single hypothesis; verify each cause with data.
- For equipment root causes, confirm against the machine technical specs (TEC-301, TEC-314, TEC-328, TEC-342).

### D5 – Corrective Action Plan
- Define permanent corrective actions (PCA) with owners and due dates.
- Each PCA must be verifiable (KPI defined).

### D6 – Verification
- Run the PCA for a minimum of **30 days** or **1,000 units**, whichever is longer.
- Demonstrate defect rate at or below the pre-failure baseline.

### D7 – Prevention
- Update SOPs, work instructions, FMEAs, and control plans.
- Training records must be updated for affected operators/technicians.

### D8 – Closure
- FAL presents the complete 8D to the Plant Manager for approval.
- Closure is documented in QDS; the NC is closed only after approval.

## 5. Escalation Interface

- If the failure is a safety-critical Class A, notify the Quality Director **immediately** (within 1 hour) regardless of 8D stage — see ESC-402 Level 4.
- If the confirmed root cause affects more than **500 units** or defect rate exceeds **0.5%**, the FAL must initiate the field-action evaluation per QCR-501.

## 6. Time Targets

| Stage | Target |
|---|---|
| D1–D3 complete | 24 hours |
| D4 root cause confirmed | 7 days |
| D5–D6 complete | 30 days |
| D7–D8 closure | 60 days |

## 7. Cross-References

- ESC-402 — Escalation Workflow Matrix
- QCR-501 — Recall Trigger Criteria
- SOP-OPR-158 — Final Inspection and Torque Audit
- SOP-TEC-201 / 214 / 228 / 235 — Equipment troubleshooting SOPs

## 8. Revision History

| Rev | Date | Change |
|---|---|---|
| 2 | 2024-07-15 | Initial release |
| 3 | 2024-11-10 | D6 verification period extended to 30 days / 1,000 units |
| 4 | 2025-02-25 | Added escalation interface thresholds (500 units / 0.5%) |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
