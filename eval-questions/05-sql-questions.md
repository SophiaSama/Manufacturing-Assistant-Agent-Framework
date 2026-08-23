# Eval Set 05 — Live SQL Queries

**Environment:** `database/nga.db` (SQLite). The agent must write and execute SQL (e.g., `python3 query.py "SELECT ..."`) to answer. Schema: `database/schema.sql`.

All answers below were **verified** against the current build of `nga.db`.

| # | Question | Expected answer | Notes |
|---|---|---|---|
| SQL1 | How many vehicles were built on 2025-04-08 Shift C? | **9** | `vehicles` where build_date='2025-04-08' AND build_shift='C' |
| SQL2 | How many lug-nut readings are out of tolerance overall? | **7** | `torque_readings` where in_tolerance=0 |
| SQL3 | Which vehicle(s) have 2 or more out-of-tolerance lug nuts? | **NGA-AU25-0015 (3 nuts)** | GROUP BY vin HAVING COUNT(*)>=2 |
| SQL4 | Compute the out-of-tolerance rate by shift. | A: 0%, B: 0.53%, C: 1.12% | JOIN vehicles; note C = 5/445 (22 vehicles, incl. the 5 planted low readings) |
| SQL5 | Which vehicles are NOT in SHIPPED status, and why? | AU-0015/0016/0017 QUARANTINED (torque), SO-0016 HELD (film build) | `vehicles` + `defects` |
| SQL6 | Which part lots are quarantined? | FL-88200 (brake fluid DOT 4), lot FW-2025-03-30, qty 150 | `parts` status |
| SQL7 | List open/contained NC records and their highest escalation level. | NC-2025-0137 (TORQUE-LOW, escalation L4 OPEN), NC-2025-0150 (P-122, escalation L2 closed — NC itself still OPEN) | `nc_records` LEFT JOIN `escalations` MAX(level) |
| SQL8 | Which open work orders exist? | WO-2025-0440 (PT-ROB-03, P-122, P2, Ana Souza) | `work_orders` status='OPEN' |
| SQL9 | Which vehicles failed final inspection (station 168)? | NGA-AU25-0015, 0016, 0017 | `build_records` where station_id='168' AND result='FAIL' |
| SQL10 | What is the torque-defect rate within the 2025-04-08 Shift C population (criterion C2 check)? | 3 affected of 9 built = **33.3%** (vs 0.5% safety threshold → far exceeds) | `torque_readings` + `vehicles`; denominator argument matters |
| SQL11 | How long did Level 2 escalations take to resolve? | NC-2025-0137: 75 min; NC-2025-0142: 125 min; NC-2025-0150: 165 min | julianday diff × 1440 |
| SQL12 | Which employees have expired certifications? | E-3002 James Park (SOP-TEC-214, expired 2025-03-15) | `training_records` status='EXPIRED' |
| SQL13 | Which station has the most open Class A defects? | Station 168 (3 TORQUE-LOW, Class A) | `defects` GROUP BY station_id, defect_class |
| SQL14 | Give the defect story: for NC-2025-0137 list its escalation chain. | L2 (Marcus Reed) → L3 (Miguel Santos) → L4 (Dana Whitfield, OPEN) | `escalations` ordered by level |
| SQL15 | Combine data + docs: does the torque issue meet QCR-501 criteria, and what field action exists? | C3 met (3 vehicles), C2 met within shift-C population; FA-2025-001 CAMPAIGN, EVALUATION, 42 units, criterion C3 | `field_actions` + QCR-501 logic |

## Grading
- Correct result + correct SQL: full credit.
- Correct result via wrong query (e.g., hardcoded): 50% — the point is the agent can *query live data*.
- SQL with wrong result: 0, but record the SQL for analysis.
- Bonus: agent explains *why* the number matters (e.g., SQL10 vs QCR-501 C2).
