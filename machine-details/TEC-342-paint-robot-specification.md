# Machine Technical Specification

| Field | Value |
|---|---|
| **Document ID** | TEC-342 |
| **Title** | SprayTech ST-450 Paint Robot – Technical Specification |
| **Equipment** | PT-ROB-03 (clear coat, Station 74); PT-ROB-01/02 (primer/base, Stations 61/68) |
| **Revision** | Rev 6 |
| **Effective Date** | 2025-03-20 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. General Description

Electrostatic paint application robot with rotary atomizer for automotive clear coat. PT-ROB-03 applies clear coat to the Aurora/Solstice body exterior.

## 2. Specifications

| Parameter | Value |
|---|---|
| Axes | 6 |
| Atomizer | Rotary bell, 65 mm diameter |
| Bell speed (max) | 40,000 rpm |
| Paint flow range | 150 – 350 ml/min |
| Electrostatic voltage | Up to 80 kV (operational limit 60 kV) |
| Booth air supply | 0.35 MPa at manifold |
| Film build target (clear coat) | 45 µm (acceptance 40 – 52 µm) |
| Paint | Solvent-borne 2K clear coat (part no. PC-99010) |
| Booth temperature | 22 °C ± 2 °C |
| Relative humidity | 55 – 65% |

## 3. Process Notes

- Electrostatic charging is disabled below 30 kV to avoid arcing at low flow.
- Color change purge cycle: 90 seconds with solvent flush.
- Atomizer bell replaced at 10,000 operating hours or on visual wear.

## 4. Maintenance Intervals

| Task | Interval |
|---|---|
| Bell cup inspection | Daily |
| HV cable inspection | Weekly |
| Pump calibration (CAL-ST-04) | Monthly |
| Inlet filter (25 µm) replacement | Weekly |
| Booth grounding check (< 1 Ω) | Weekly |
| Full robot calibration | Annually |

## 5. Error Code Reference (Summary)

See SOP-TEC-235 for the fault-response procedure.

| Code | Meaning |
|---|---|
| P-110 | Atomizer speed low |
| P-122 | Paint flow deviation > 10% |
| P-134 | Electrostatic voltage fault |
| P-145 | Booth air pressure low |

## 6. Cross-References

- SOP-TEC-235 — Paint Robot PT-ROB-03 Fault Diagnosis
- FAP-401 — Failure Analysis Procedure
- QCR-501 — Recall Trigger Criteria

## 7. Revision History

| Rev | Date | Change |
|---|---|---|
| 4 | 2024-07-22 | Initial release |
| 5 | 2024-12-19 | Film build target set at 45 µm (40–52 µm band) |
| 6 | 2025-03-20 | Operational HV limit clarified at 60 kV |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
