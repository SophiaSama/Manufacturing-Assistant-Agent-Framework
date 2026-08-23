# Machine Technical Specification

| Field | Value |
|---|---|
| **Document ID** | TEC-314 |
| **Title** | TorqMaster TF-6000 Torque Tool – Technical Specification |
| **Equipment** | TQ-6012 (Sta. 144), TQ-6015 (Sta. 118), TQ-6018 (Sta. 168), TQ-6019 (spare) |
| **Revision** | Rev 5 |
| **Effective Date** | 2025-01-20 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. General Description

Battery-operated, transducerized torque wrench with angle control and data logging. Used for safety-critical fastening operations in General Assembly.

## 2. Specifications

| Parameter | Value |
|---|---|
| Torque range | 5 – 150 Nm |
| Accuracy | ± 1.5% of reading |
| Angle accuracy | ± 2° |
| Drive | 1/2" square |
| Battery | 18 V Li-ion, 4 Ah (approx. 8,000 cycles per charge) |
| Communication | Wireless (2.4 GHz) to station PLC / QDS |
| Data logging | Every cycle: torque, angle, time, tool ID |
| Calibration interval | 3 months or 100,000 cycles (whichever first) |
| Drift warning threshold | 2% between weekly self-checks |
| Reaction arm | Passive RA-200 (Station 144 only) |

## 3. Operating Modes

| Mode | Use |
|---|---|
| Standard (torque) | Wheel lug nuts (105 Nm), rear mount (45 Nm) |
| Torque-to-yield (T+Y) | Engine mount bolts (62 Nm + 90°) |
| Audit | Peak torque verification with ± 1% reporting |

## 4. Communication and Data

- Each completed cycle transmits a data record to QDS; a missing record for 3 consecutive cycles triggers a station warning.
- Tool clock must be synced weekly; timestamp accuracy is critical for audit traceability.

## 5. Maintenance

- Weekly self-check on calibration rig Nexus CR-500 (see SOP-TEC-214).
- Battery cells replaced at 500 charge cycles.
- Transducer replacement: SP-TF-2200.

## 6. Cross-References

- SOP-TEC-214 — Calibration Policy and Drift Troubleshooting
- SOP-OPR-101 — Wheel Installation and Torque
- SOP-OPR-127 — Engine Mount Installation
- SOP-OPR-158 — Final Inspection and Torque Audit

## 7. Revision History

| Rev | Date | Change |
|---|---|---|
| 3 | 2024-07-01 | Initial release |
| 4 | 2024-11-20 | Added angle accuracy and torque-to-yield mode detail |
| 5 | 2025-01-20 | Drift warning threshold updated to 2% (was 3%) |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
