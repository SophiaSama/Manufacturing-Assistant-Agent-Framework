# Quality Policy

| Field | Value |
|---|---|
| **Document ID** | QCR-501 |
| **Title** | Recall Trigger Criteria and Field-Action Decision Procedure |
| **Department** | Quality – Field Quality |
| **Revision** | Rev 3 |
| **Effective Date** | 2025-04-10 |
| **Owner** | Quality Director, Priya Raman |
| **Applies To** | Aurora (AU-2025), Solstice (SO-2025), all Northgate-built vehicles |
| **Classification** | Confidential – Controlled |

---

## 1. Purpose

Define the objective criteria that trigger a recall or field action when a quality issue is identified, and the decision procedure for initiating, scoping, and executing the action. This document operationalizes the company's safety-first recall policy.

## 2. Definitions

| Term | Definition |
|---|---|
| **Class A (safety-critical) system** | Brakes, steering, airbags, seat belts, fuel system, wheel retention, engine mounts, windshield retention |
| **Field action** | Recall, service campaign, or stop-sale |
| **Affected population** | Vehicles sharing the same suspected root cause (VIN range, build date, part lot) |
| **NRSA** | National Road Safety Administration (fictional national regulator) |
| **GVSR** | Global Vehicle Safety Regulation (applicable safety standards) |

## 3. Recall Trigger Criteria

A recall/field action **must be initiated** when ANY of the following is met:

### C1 – Safety-Critical Defect
- Any confirmed or suspected defect in a Class A system that could cause **injury or death** (e.g., wheel separation, brake failure, engine mount detachment, windshield detachment while driving).
- **Action:** immediate stop-ship, escalate to Level 4 within **1 hour** (ESC-402), notify NRSA per §6.

### C2 – Defect Rate Threshold
- Confirmed defect rate exceeding **0.5% (5,000 ppm)** for safety-related defects, or **2% (20,000 ppm)** for functional (non-safety) defects in the affected population.

### C3 – Complaint Pattern
- **3 or more identical field complaints** within a rolling **30-day** window (verified, not anecdotal).

### C4 – Affected Population Size
- More than **1,000 vehicles** affected by a confirmed or strongly suspected root cause, even if the defect rate is below C2.

### C5 – Regulatory or Legal Trigger
- NRSA inquiry, GVSR non-compliance, or any safety investigation opened by an authority.

> **Note:** The criteria are OR conditions. Meeting any single criterion is sufficient to initiate the field-action evaluation. When in doubt, the Quality Director decides in favor of safety.

## 4. Decision Procedure

### Step 1 – Triage (within 4 hours of detection)
1. Failure Analysis Lead (FAL) confirms the defect per FAP-401 D1–D3.
2. Determine which criteria (C1–C5) are met.
3. If C1 is met → stop-ship immediately and notify Level 4.

### Step 2 – Field-Action Evaluation (within 5 business days)
1. Estimate the affected population using QDS, VCATS build data, and part-lot traceability.
2. Classify severity: **S1** (injury possible, recall), **S2** (functional, likely campaign), **S3** (cosmetic, monitor).
3. Document the recommended action in the Field Action Proposal (FAP-501 form).

### Step 3 – Decision
- Decision authority: Quality Director + Plant Manager (joint).
- Corporate Quality VP must approve all recalls involving > 10,000 vehicles.
- Record the decision rationale in QDS.

### Step 4 – Execution
- Recall execution includes: NRSA notification (§6), dealer hold/stop-sale, parts allocation, repair instruction (TSB), and owner notification letters.
- Owner notification within **60 days** of NRSA approval (legal requirement).

## 5. Stop-Ship and Containment

- Stop-ship is triggered by: C1 (immediate), C3 confirmed, or C4 reached.
- Containment actions: quarantine suspect VIN range, 100% inspection at source, hold at port/dealer.
- Stop-ship is lifted only after: (a) root cause confirmed, (b) permanent corrective action verified per FAP-401 D6, and (c) joint approval of Plant Manager + Quality Director.

## 6. Regulatory Notification

- Notify NRSA within **5 business days** of confirming a safety-related defect.
- NRSA report includes: defect description, affected population (VINs), root cause status, and planned corrective action.
- Failure to notify within 5 business days is a compliance violation.

## 7. Monitoring and Escalation

- Field quality monitors dealer repair data and customer complaints daily.
- A second defect pattern in the same system within 6 months of a completed recall re-opens the evaluation automatically.

## 8. Cross-References

- FAP-401 — Failure Analysis Procedure (8D)
- ESC-402 — Escalation Workflow Matrix
- SOP-OPR-101 — Wheel Installation and Torque
- SOP-OPR-127 — Engine Mount Installation
- SOP-OPR-114 — Windshield Installation

## 9. Revision History

| Rev | Date | Change |
|---|---|---|
| 1 | 2024-09-01 | Initial release |
| 2 | 2025-01-30 | Added C4 population threshold (1,000 vehicles) |
| 3 | 2025-04-10 | Added §6 regulatory notification timeline (5 business days) |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
