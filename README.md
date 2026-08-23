# Northgate Assembly Plant — Synthetic Document Corpus

Synthetic manufacturing documentation for evaluating agent frameworks (RAG, QA, decision-support evals) in a vehicle assembly plant environment.

**Fictional universe:** Apex Automotive Corporation — Northgate Assembly Plant (NGA). All names, part numbers, people, and regulatory bodies are fictional.

## Corpus Universe

| Entity | Detail |
|---|---|
| Plant | Northgate Assembly Plant (NGA) |
| Models | Aurora AU-2025 (compact SUV), Solstice SO-2025 (sedan) |
| Production rate | 42 vehicles/hour |
| Shifts | A 06:00–14:00 · B 14:00–22:00 · C 22:00–06:00 |
| Regulator | National Road Safety Administration (NRSA) — fictional |
| Standard | Global Vehicle Safety Regulation (GVSR) — fictional |

## Document Index

### Operator SOPs — `operator-sops/`
| Doc | Topic | Key facts |
|---|---|---|
| SOP-OPR-101 | Wheel installation & torque (Sta. 144) | 105 Nm ±5%, star sequence, tool TQ-6012 |
| SOP-OPR-114 | Windshield installation (Sta. 152) | 8×12 mm bead, 10 min open time, 4 h cure |
| SOP-OPR-127 | Engine mount installation (Sta. 118) | 62 Nm + 90°, M12, tool TQ-6015 |
| SOP-OPR-142 | Brake fluid fill (Sta. 131) | DOT 4, 0.5–1.0 bar, bleed RR→RL→FR→FL |
| SOP-OPR-158 | Final inspection & torque audit (Sta. 168) | Audit every 2 h, 3 vehicles, re-audit 10 |

### Technician Troubleshooting SOPs — `technician-sops/`
| Doc | Topic | Key facts |
|---|---|---|
| SOP-TEC-201 | Welding robot RB-07 faults | Error codes E-4012/E-5023/E-5041/E-4055/E-4031 |
| SOP-TEC-214 | Torque tool calibration & drift | 3 months / 100k cycles, ±1.5%, drift >2% |
| SOP-TEC-228 | Conveyor CV-2 jam recovery | E-Stop codes RC-01..04, jog 20% restart |
| SOP-TEC-235 | Paint robot PT-ROB-03 faults | Codes P-110/P-122/P-134/P-145 |

### Machine Technical Details — `machine-details/`
| Doc | Topic | Key facts |
|---|---|---|
| TEC-301 | RoboTech RX-700 welding robot | 150 kg payload, 2.7 m reach, ±0.05 mm, 4,000-weld cap life |
| TEC-314 | TorqMaster TF-6000 torque tool | 5–150 Nm, ±1.5%, 18 V battery |
| TEC-328 | Conveyor Line 2 (CV-2) | 12 m/min, 36 pallets, PS-200/PR-100 sensors |
| TEC-342 | SprayTech ST-450 paint robot | 40,000 rpm bell, 45 µm film build |

### Failure Analysis & Escalation — `failure-analysis/`
| Doc | Topic | Key facts |
|---|---|---|
| FAP-401 | 8D failure analysis procedure | D1–D8, 30d/1,000-unit verification, 500-unit / 0.5% escalation |
| ESC-402 | Escalation workflow matrix | L1 30 min → L2 2 h → L3 8 h → L4 24 h; Class A = 1 h |

### Recall & Quality — `recall-quality/`
| Doc | Topic | Key facts |
|---|---|---|
| QCR-501 | Recall trigger criteria | C1 safety, C2 0.5%/2% rates, C3 3 complaints/30d, C4 >1,000 units, C5 regulatory; NRSA in 5 business days |

### Additional Documents — `additional-docs/`
| Folder | Docs |
|---|---|
| `maintenance-work-orders/` | WO-2025-0417 (RB-07 E-4012), WO-2025-0433 (TQ-6012 drift), WO-2025-0421 (CV-2 jam), WO-2025-0440 (PT-ROB-03, open) |
| `supplier-quality/` | SCAR-2025-007 (brake fluid moisture), AUD-2025-031 (ApexForging), APP-04 (adhesives), APP-07 (brake fluid) |
| `training/` | TR-2025-030 certification matrix (incl. expired cert E-3002) |

### Production Database — `database/`
| File | Purpose |
|---|---|
| `nga.db` | Live SQLite DB (17 tables, ~2,000 rows, seeded stories) |
| `schema.sql` / `seed.py` | Schema + deterministic builder |
| `query.py` | Run live queries: `python3 query.py "SELECT ..."` |
| `example-queries.sql` | 15 worked examples |
| `database-readme.md` | Schema, stories, eval usage |

### Eval Question Sets — `eval-questions/`
| File | Type | # |
|---|---|---|
| `01-retrieval-qa.md` | Fact lookup | 30 |
| `02-multi-hop-qa.md` | Cross-doc reasoning | 12 |
| `03-scenario-decision-qa.md` | Operational decisions | 8 |
| `04-escalation-recall-qa.md` | Escalation/recall decisions | 10 |
| `05-sql-questions.md` | Live SQL (verified answers) | 15 |
| `questions.json` | Machine-readable bundle | 75 |

### Variant Corpus — `variant-corpus/`
Copies of the main docs with **8 planted inconsistencies** (torque 108 Nm, cap life 5,000, cal interval 6 months, Class A 4 h, C2 1.0%, rear mount 50 Nm, bleed sequence, cure 2 h) for conflict-detection evals. Ledger: `variant-corpus/planted-inconsistencies.md` (keep out of the agent corpus).

## Eval Notes

- **Ground truth:** see `ground-truth.md` — key facts and answer keys. Exclude this file from the agent's retrieval corpus if testing retrieval/QA.
- **Cross-document reasoning paths** deliberately built in (good eval targets):
  - Torque tool drift (SOP-TEC-214) → station halt → re-audit (SOP-OPR-158) → escalation (ESC-402) → recall criteria (QCR-501)
  - Conveyor spatter (TEC-328 §6) → jam (SOP-TEC-228) → TCP deviation E-5023 (SOP-TEC-201)
  - Weld cap life 4,000 (TEC-301) → halt (SOP-TEC-201)
  - Windshield retention (SOP-OPR-114) → Class A (QCR-501 §2)
- **All numbers are consistent across documents** (torque specs, intervals, thresholds, error codes, part numbers, personnel names).
- **Eval scenarios:** 75 questions in `eval-questions/` (5 markdown sets + `questions.json`), 15 of which require live SQL against `database/nga.db`.
- **Conflict-detection evals:** use `variant-corpus/` (8 planted inconsistencies, ledger in `planted-inconsistencies.md`).
