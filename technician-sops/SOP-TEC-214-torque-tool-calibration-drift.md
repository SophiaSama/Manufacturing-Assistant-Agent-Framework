# Standard Operating Procedure

| Field | Value |
|---|---|
| **Document ID** | SOP-TEC-214 |
| **Title** | TorqMaster TF-6000 – Calibration Policy and Drift Troubleshooting |
| **Department** | Maintenance – Tooling |
| **Equipment** | TorqMaster TF-6000 torque tools (TQ-6012, TQ-6015, TQ-6018) |
| **Revision** | Rev 4 |
| **Effective Date** | 2025-01-20 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. Purpose

Define the calibration schedule, drift thresholds, and troubleshooting steps for all TorqMaster TF-6000 torque tools used in production. Incorrect torque application is a safety risk and a recall trigger (see QCR-501).

## 2. Tool Inventory

| Tool ID | Station | Application | Nominal Torque |
|---|---|---|---|
| TQ-6012 | 144 | Wheel lug nuts | 105 Nm ± 5% |
| TQ-6015 | 118 | Engine mount bolts | 62 Nm + 90° |
| TQ-6018 | 168 | Audit wrench | 105 Nm (audit) |

## 3. Calibration Schedule

- **Interval:** every **3 months** OR every **100,000 cycles**, whichever comes first.
- Performed on the calibration rig (Nexus CR-500) with certified master transducers (± 0.5% accuracy).
- Acceptance: measured torque within **± 1.5%** of target at 20%, 60%, and 100% of tool range.
- A calibration certificate is attached to each tool and recorded in the CMMS (tool ID-based).

## 4. Drift Monitoring

- The tool's internal diagnostics flag **drift > 2%** between consecutive weekly self-checks.
- A drift flag quarantines the tool automatically in the CMMS: it cannot be logged out for production use.

### 4.1 Troubleshooting a Drift Flag
1. Verify the tool battery is > 50%. Low battery causes false drift flags.
2. Re-run the weekly self-check on the calibration rig.
3. If drift persists > 2%, perform a full re-calibration.
4. If re-calibration cannot bring the tool within ± 1.5%, replace the transducer (spare SP-TF-2200) or send to factory service.

## 5. Production Tool Failure at Station

When a station tool (e.g., TQ-6012) reports out-of-tolerance:
1. The station halts automatically (red light).
2. Maintenance swaps in the spare tool (TQ-6019) and verifies 5 test cycles on the rig.
3. Vehicles completed since the last verified audit are re-audited (see SOP-OPR-158 §4).
4. If re-audit finds any fastener below spec, the affected vehicles are quarantined and dispositioned per FAP-401.

## 6. Cross-References

- TEC-314 — TorqMaster TF-6000 Technical Specification
- SOP-OPR-101 — Wheel Installation and Torque
- SOP-OPR-127 — Engine Mount Installation
- SOP-OPR-158 — Final Inspection and Torque Audit
- QCR-501 — Recall Trigger Criteria

## 7. Revision History

| Rev | Date | Change |
|---|---|---|
| 2 | 2024-06-30 | Initial release |
| 3 | 2024-10-12 | Drift threshold reduced from 3% to 2% |
| 4 | 2025-01-20 | Added spare tool TQ-6019 and swap verification step |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
