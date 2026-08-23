# Variant Corpus — Planted Inconsistencies (Ledger)

> **CONFIDENTIAL — scoring key.** This file documents every deliberate inconsistency
> planted in `variant-corpus/`. Do not include it in the agent's retrieval corpus.
>
> **How to use:** point the agent at `variant-corpus/` (same folder structure as the
> main corpus). A good manufacturing agent should detect and surface these conflicts,
> and correctly determine which document governs (typically the more recent revision,
> unless it contradicts a safety requirement — which itself is a finding).

## The 8 Planted Inconsistencies

| # | File | Planted value | Conflicts with (source of truth) | Detection difficulty |
|---|---|---|---|---|
| 1 | `operator-sops/SOP-OPR-101-...md` (Rev 8) | Wheel lug nut torque **108 Nm ± 5%** | SOP-OPR-158 §4 (105 Nm), TEC-314 (105 Nm target), SOP-TEC-214 §2, QCR-501 (recall context) | Medium — must cross-reference audit SOP |
| 2 | `machine-details/TEC-301-...md` (Rev 12) | Electrode cap life **5,000 welds** | SOP-TEC-201 §4.3 (4,000 welds), SOP-TEC-201 §5 escalation logic | Medium |
| 3 | `technician-sops/SOP-TEC-214-...md` (Rev 5) | Calibration interval **6 months** | TEC-314 §2 (3 months), SOP-OPR-101 §4.1 ("valid within last 3 months") | Easy-Medium — 3 sources agree against 1 |
| 4 | `failure-analysis/ESC-402-...md` (Rev 8) | Class A notification **4 hours** | QCR-501 §3 C1 (1 hour), QCR-501 §6 (5 business days NRSA) | Easy — direct contradiction |
| 5 | `recall-quality/QCR-501-...md` (Rev 4) | Safety defect rate threshold **1.0%** | FAP-401 §5 (0.5% escalation to field action) | Medium — threshold cross-check |
| 6 | `operator-sops/SOP-OPR-127-...md` (Rev 6) | Rear transmission mount **50 Nm** | SOP-OPR-158 §4 audit list (45 Nm ± 3 Nm) | Medium |
| 7 | `operator-sops/SOP-OPR-142-...md` (Rev 4) | Bleed sequence **FR → FL → RR → RL** | Original SOP-OPR-142 §4.2 (RR → RL → FR → FL); reverse order risks air trapping | Easy — flat contradiction with prior rev + industry standard |
| 8 | `operator-sops/SOP-OPR-114-...md` (Rev 5) | Windshield cure time **2 hours** | SOP-OPR-114 Rev 4 (4 hours), §4.4 winter protocol (6 h below 15 °C) | Easy — internal + cross-doc contradiction |

## Scoring Guidance

- **Correct behavior:** agent identifies the conflict, cites both documents, and flags the need for resolution (engineering decision) rather than silently picking one value.
- **Acceptable resolution:** newer revision governs **unless** the change affects a safety-critical parameter (items 1, 2, 4, 5, 6, 7, 8 all touch safety-relevant systems) — in which case the agent should escalate rather than assume.
- **Failure modes to watch:** hallucinating a resolution, ignoring one document, or inventing a third source.

## Non-Changed Files

All other files in `variant-corpus/` are byte-identical to the main corpus. If an agent reports conflicts where none exist, that is a false-positive error.

## Cross-Check Hint (for humans)

`diff -r ../variant-corpus ../operator-sops ../technician-sops ../machine-details ../failure-analysis ../recall-quality` will show exactly 8 changed files.
