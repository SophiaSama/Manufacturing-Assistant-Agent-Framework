# Standard Operating Procedure

| Field | Value |
|---|---|
| **Document ID** | SOP-TEC-235 |
| **Title** | Paint Robot PT-ROB-03 Fault Diagnosis (SprayTech ST-450) |
| **Department** | Maintenance – Paint Shop |
| **Equipment** | PT-ROB-03 (SprayTech ST-450), Station 74 (clear coat) |
| **Revision** | Rev 3 |
| **Effective Date** | 2025-03-20 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define diagnosis and recovery for paint-application faults on robot PT-ROB-03 in the clear-coat booth. Paint defects found here can cascade into rework and, if systemic, into quality actions per QCR-501.

## 2. Fault Code Table

| Code | Description | Likely Cause | Action |
|---|---|---|---|
| P-110 | Atomizer speed low | Bell cup worn or drive belt slip | Inspect bell cup; replace if worn (spare SP-ST-3401) |
| P-122 | Paint flow deviation > 10% | Pump calibration drift or clogged filter | Check flow meter, clean filter, re-calibrate pump |
| P-134 | Electrostatic voltage fault | HV cable arcing or grounding issue | Inspect cable, verify ground; do not operate above 60 kV |
| P-145 | Air pressure low in booth supply | Compressor issue or filter clog | Verify 0.35 MPa at the booth manifold |

## 3. Recovery Procedures

### 3.1 Atomizer (P-110)
1. Purge the paint line with solvent (2 minutes).
2. Remove and inspect the bell cup (65 mm). Worn edge or imbalance = replace.
3. Verify atomizer speed reaches **40,000 rpm** at idle before resuming production.

### 3.2 Paint Flow (P-122)
1. Run the paint pump self-test; acceptable flow band is 150–350 ml/min.
2. Replace the inlet filter (25 µm) and re-test.
3. If flow is still off, re-calibrate the pump per calibration procedure CAL-ST-04.

### 3.3 HV Fault (P-134)
1. De-energize the electrostatic unit and ground the applicator.
2. Inspect the HV cable for carbon tracking. Replace if any tracking marks are visible.
3. Verify booth grounding resistance < 1 Ω before re-energizing.

## 4. Quality Implications

- Any robot fault that stops clear-coat application mid-body produces a **blend line defect**. The body is routed to the rework line automatically.
- 3 or more blend-line defects per hour → stop the line and inform the Paint Shop supervisor; escalate per ESC-402 Level 2.
- Recurring film-build deviation (> 15% below target 45 µm) → open an NC and analyze per FAP-401.

## 5. Cross-References

- TEC-342 — SprayTech ST-450 Technical Specification
- ESC-402 — Escalation Workflow Matrix
- FAP-401 — Failure Analysis Procedure
- QCR-501 — Recall Trigger Criteria

## 6. Revision History

| Rev | Date | Change |
|---|---|---|
| 1 | 2024-08-15 | Initial release |
| 2 | 2024-12-19 | Added film-build deviation trigger (45 µm target) |
| 3 | 2025-03-20 | HV fault grounding check requirement added |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
