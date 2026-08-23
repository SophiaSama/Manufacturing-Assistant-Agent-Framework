# NGA Production Database (nga.db) — Eval Data Store

Synthetic SQLite database of production-line test data for the Northgate Assembly Plant. Designed for eval scenarios that require **live SQL queries** against manufacturing data.

## Files

| File | Purpose |
|---|---|
| `nga.db` | The SQLite database (built). Use directly with `sqlite3` or `query.py`. |
| `schema.sql` | Schema DDL (17 tables). |
| `seed.py` | Deterministic builder (`python3 seed.py` to rebuild). Seeded RNG → reproducible. |
| `query.py` | Query helper: `python3 query.py "SELECT ..."` or `-f file.sql` |
| `example-queries.sql` | 15 worked example queries (run `python3 query.py -f example-queries.sql`) |

## Schema Overview

| Table | Content |
|---|---|
| `lines`, `stations`, `machines`, `shifts` | Production topology (BODY/PAINT/GA/FINAL; stations 42…168; RB-07, TQ-6012, PT-ROB-03…) |
| `vehicles`, `build_records` | 60 vehicles (40 Aurora AU-2025, 20 Solstice SO-2025), 2025-04-07…09, 3 shifts, PASS/FAIL per station |
| `torque_readings` | 1,205 lug-nut readings (20 per vehicle, tool TQ-6012, target 105 Nm ±5%) |
| `quality_checks` | TORQUE_AUDIT, LEAK_TEST, RETENTION, FILM_BUILD results with spec limits |
| `defects`, `nc_records` | 5 defects / 3 NCs with classes (A = safety) |
| `work_orders`, `maintenance_logs` | WO-2025-0417…0440; drift flags, repairs, calibrations |
| `escalations` | 5 escalations across levels 1–4 (ESC-402 semantics) |
| `field_actions` | FA-2025-001 (campaign, evaluation, criterion C3) |
| `suppliers`, `parts` | 5 suppliers, 11 part-lots (one quarantined brake-fluid lot) |
| `training_records` | 28 certifications; one expired (E-3002 / SOP-TEC-214) |

## Embedded Stories (consistent with the corpus docs)

1. **TORQUE-LOW (Class A)** — TQ-6012 drifts on 2025-04-08 Shift C → NGA-AU25-0015 (3 nuts: 97.2/96.8/97.9 Nm), 0016, 0017 (1 nut each) fail at final inspection → NC-2025-0137 → escalations L2→L3→**L4 open** → field action FA-2025-001 (CAMPAIGN, 42 units, criterion C3). Vehicles QUARANTINED.
2. **E-5023 (Class B)** — RB-07 TCP deviation 0.7 mm on NGA-AU25-0007 (04-07, Shift B) → NC-2025-0142 (closed after re-teach).
3. **P-122 (Class B)** — PT-ROB-03 flow deviation, clear-coat film build 38 µm on NGA-SO25-0016 (04-09, Shift A) → NC-2025-0150, WO-2025-0440 **open**. Vehicle HELD.
4. **Quarantined lot** — brake fluid FW-2025-03-30 (moisture concern) from FluidWorks.
5. **Expired certification** — James Park (E-3002), SOP-TEC-214 expired 2025-03-15.

## Eval Usage Notes

- Give the agent `nga.db` + schema, and let it write/run SQL to answer questions (e.g., via `query.py`).
- Questions in `../eval-questions/05-sql-questions.md` require live queries; answers were verified against this exact DB build.
- **Expected answers depend on the seeded data** — do not regenerate the DB with a different seed mid-eval.
- Ground-truth answers for data questions: see `../ground-truth.md` (docs) — the SQL answers are embedded in the eval question file.
