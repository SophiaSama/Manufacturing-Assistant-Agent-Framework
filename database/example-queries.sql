-- ============================================================================
-- Example SQL queries against nga.db
-- Run:  python3 query.py -f example-queries.sql
-- ============================================================================

-- 1. All out-of-tolerance torque readings (simple filter)
SELECT vin, nut_position, torque_nm, target_nm, ts
FROM torque_readings
WHERE in_tolerance = 0
ORDER BY torque_nm ASC;

-- 2. Vehicles with 2+ out-of-tolerance nuts (aggregation)
SELECT vin, COUNT(*) AS bad_nuts, MIN(torque_nm) AS min_torque
FROM torque_readings
WHERE in_tolerance = 0
GROUP BY vin
HAVING COUNT(*) >= 2;

-- 3. Out-of-tolerance rate by shift (join)
SELECT v.build_shift, COUNT(*) FILTER (WHERE t.in_tolerance = 0) AS bad,
       COUNT(*) AS total,
       ROUND(100.0 * COUNT(*) FILTER (WHERE t.in_tolerance = 0) / COUNT(*), 2) AS pct
FROM torque_readings t
JOIN vehicles v ON v.vin = t.vin
GROUP BY v.build_shift;

-- 4. Defect count by station and class
SELECT station_id, defect_class, COUNT(*) AS n
FROM defects
GROUP BY station_id, defect_class
ORDER BY n DESC;

-- 5. Open NC records with their latest escalation level
SELECT n.nc_id, n.defect_code, n.defect_class, n.status,
       MAX(e.level) AS max_escalation_level
FROM nc_records n
LEFT JOIN escalations e ON e.nc_id = n.nc_id
GROUP BY n.nc_id
ORDER BY max_escalation_level DESC NULLS LAST;

-- 6. Machines with open work orders
SELECT wo.wo_id, m.machine_id, m.model, m.station_id, wo.fault_code,
       wo.priority, wo.opened_at, wo.technician
FROM work_orders wo
JOIN machines m ON m.machine_id = wo.machine_id
WHERE wo.status = 'OPEN';

-- 7. Quarantined parts / lots
SELECT part_no, part_name, lot, supplier_id, qty, received_at
FROM parts
WHERE status != 'APPROVED';

-- 8. Vehicles with a FAIL build record at final inspection (station 168)
SELECT b.vin, v.model, v.build_date, v.build_shift, v.status
FROM build_records b
JOIN vehicles v ON v.vin = b.vin
WHERE b.station_id = '168' AND b.result = 'FAIL';

-- 9. Film-build results below spec
SELECT vin, measured_value, spec_min, spec_max, result
FROM quality_checks
WHERE check_type = 'FILM_BUILD' AND result = 'FAIL';

-- 10. Full torque incident chain: NC -> escalations -> field action
SELECT e.level, e.triggered_at, e.responder, e.status AS esc_status
FROM escalations e
WHERE e.nc_id = 'NC-2025-0137'
ORDER BY e.level;

SELECT action_id, defect_code, action_type, status, affected_units, criteria_met
FROM field_actions;

-- 11. Operators with expired certifications
SELECT emp_id, employee_name, certification, certified_at, expires_at
FROM training_records
WHERE status = 'EXPIRED';

-- 12. Defects by shift for the last 7 days (join)
SELECT v.build_shift, d.defect_code, COUNT(*) AS n
FROM defects d
JOIN vehicles v ON v.vin = d.vin
WHERE d.detected_at >= '2025-04-03'
GROUP BY v.build_shift, d.defect_code;

-- 13. Vehicles built on 2025-04-08 Shift C (suspect population for torque issue)
SELECT vin, model, lot, status
FROM vehicles
WHERE build_date = '2025-04-08' AND build_shift = 'C';

-- 14. Maintenance: repairs per machine in April
SELECT machine_id, log_type, COUNT(*) AS n
FROM maintenance_logs
WHERE performed_at >= '2025-04-01'
GROUP BY machine_id, log_type
ORDER BY n DESC;

-- 15. Time-to-resolve escalations (minutes) for closed L2 escalations
SELECT nc_id, level,
       ROUND((julianday(resolved_at) - julianday(triggered_at)) * 1440, 0) AS minutes
FROM escalations
WHERE status = 'CLOSED' AND level = 2;
