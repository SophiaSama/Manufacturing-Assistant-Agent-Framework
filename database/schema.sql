-- ============================================================================
-- Northgate Assembly Plant (NGA) — Synthetic Production Database Schema
-- Database: nga.db  |  Builder: seed.py
-- Purpose: Eval data store for manufacturing agent framework testing.
-- All data is synthetic and fictional.
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Production lines and stations
-- ---------------------------------------------------------------------------
CREATE TABLE lines (
    line_id   TEXT PRIMARY KEY,        -- BODY, PAINT, GA, FINAL
    line_name TEXT NOT NULL,
    max_speed REAL NOT NULL            -- vehicles per hour
);

CREATE TABLE stations (
    station_id   TEXT PRIMARY KEY,     -- 42, 51, 61, 74, 118, 131, 144, 152, 168
    line_id      TEXT NOT NULL REFERENCES lines(line_id),
    station_name TEXT NOT NULL,
    zone         TEXT
);

CREATE TABLE machines (
    machine_id   TEXT PRIMARY KEY,     -- RB-07, TQ-6012, PT-ROB-03 ...
    machine_type TEXT NOT NULL,        -- WELD_ROBOT, TORQUE_TOOL, PAINT_ROBOT, CONVEYOR, FILL_MACHINE
    model        TEXT NOT NULL,
    station_id   TEXT REFERENCES stations(station_id),
    install_date TEXT NOT NULL,        -- ISO date
    status       TEXT NOT NULL DEFAULT 'ACTIVE'   -- ACTIVE, DOWN, QUARANTINED, RETIRED
);

CREATE TABLE shifts (
    shift_id    TEXT PRIMARY KEY,      -- A, B, C
    start_time  TEXT NOT NULL,
    end_time    TEXT NOT NULL,
    supervisor  TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Vehicles and build tracking
-- ---------------------------------------------------------------------------
CREATE TABLE vehicles (
    vin        TEXT PRIMARY KEY,       -- NGA-AU25-0001 style
    model      TEXT NOT NULL,          -- AU-2025 / SO-2025
    build_date TEXT NOT NULL,
    build_shift TEXT NOT NULL REFERENCES shifts(shift_id),
    lot        TEXT NOT NULL,          -- production lot
    status     TEXT NOT NULL DEFAULT 'SHIPPED'  -- SHIPPED, QUARANTINED, HELD, SCRAPPED
);

CREATE TABLE build_records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    vin        TEXT NOT NULL REFERENCES vehicles(vin),
    station_id TEXT NOT NULL REFERENCES stations(station_id),
    ts         TEXT NOT NULL,          -- ISO datetime
    result     TEXT NOT NULL           -- PASS / FAIL
);

-- ---------------------------------------------------------------------------
-- Quality data
-- ---------------------------------------------------------------------------
CREATE TABLE torque_readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    vin         TEXT NOT NULL REFERENCES vehicles(vin),
    tool_id     TEXT NOT NULL REFERENCES machines(machine_id),
    nut_position TEXT NOT NULL,        -- LF1..LF5, RF1..RF5, LR1..LR5, RR1..RR5
    ts          TEXT NOT NULL,
    torque_nm   REAL NOT NULL,
    target_nm   REAL NOT NULL,         -- 105 for wheel studs
    in_tolerance INTEGER NOT NULL      -- 1 = ok (within target +/- 5%)
);

CREATE TABLE quality_checks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    vin          TEXT NOT NULL REFERENCES vehicles(vin),
    check_type   TEXT NOT NULL,        -- TORQUE_AUDIT, LEAK_TEST, RETENTION, FILM_BUILD
    station_id   TEXT NOT NULL REFERENCES stations(station_id),
    result       TEXT NOT NULL,        -- PASS / FAIL
    measured_value REAL,
    spec_min     REAL,
    spec_max     REAL,
    ts           TEXT NOT NULL
);

CREATE TABLE defects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    vin         TEXT NOT NULL REFERENCES vehicles(vin),
    station_id  TEXT NOT NULL REFERENCES stations(station_id),
    defect_code TEXT NOT NULL,         -- E-5023, P-122, TORQUE-LOW, ...
    defect_class TEXT NOT NULL,        -- A = safety-critical, B = functional, C = cosmetic
    description TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'OPEN'   -- OPEN, CONTAINED, CLOSED
);

CREATE TABLE nc_records (
    nc_id      TEXT PRIMARY KEY,       -- NC-2025-0137
    vin        TEXT REFERENCES vehicles(vin),
    defect_code TEXT NOT NULL,
    defect_class TEXT NOT NULL,
    opened_at  TEXT NOT NULL,
    opened_by  TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'OPEN'   -- OPEN, CONTAINED, CLOSED
);

-- ---------------------------------------------------------------------------
-- Maintenance
-- ---------------------------------------------------------------------------
CREATE TABLE work_orders (
    wo_id        TEXT PRIMARY KEY,     -- WO-2025-0417
    machine_id   TEXT NOT NULL REFERENCES machines(machine_id),
    fault_code   TEXT,
    priority     TEXT NOT NULL,        -- P1, P2, P3
    opened_at    TEXT NOT NULL,
    completed_at TEXT,
    status       TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN, IN_PROGRESS, COMPLETED
    technician   TEXT
);

CREATE TABLE maintenance_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id   TEXT NOT NULL REFERENCES machines(machine_id),
    log_type     TEXT NOT NULL,        -- CALIBRATION, INSPECTION, REPAIR, DRIFT_FLAG
    performed_at TEXT NOT NULL,
    result       TEXT NOT NULL,
    note         TEXT
);

-- ---------------------------------------------------------------------------
-- Escalations & field actions
-- ---------------------------------------------------------------------------
CREATE TABLE escalations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    nc_id          TEXT NOT NULL REFERENCES nc_records(nc_id),
    level          INTEGER NOT NULL,   -- 1..4 per ESC-402
    triggered_at   TEXT NOT NULL,
    acknowledged_at TEXT,
    resolved_at    TEXT,
    responder      TEXT,
    status         TEXT NOT NULL DEFAULT 'OPEN'
);

CREATE TABLE field_actions (
    action_id      TEXT PRIMARY KEY,   -- FA-2025-001
    defect_code    TEXT NOT NULL,
    action_type    TEXT NOT NULL,      -- RECALL / CAMPAIGN / STOP_SHIP
    status         TEXT NOT NULL DEFAULT 'EVALUATION',
    opened_at      TEXT NOT NULL,
    affected_units INTEGER,
    criteria_met   TEXT                -- C1, C2, C3, C4, C5 (see QCR-501)
);

-- ---------------------------------------------------------------------------
-- Supply chain & training
-- ---------------------------------------------------------------------------
CREATE TABLE suppliers (
    supplier_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    part_family TEXT NOT NULL
);

CREATE TABLE parts (
    part_no     TEXT NOT NULL,         -- AU-44102, GL-77100 ...
    part_name   TEXT NOT NULL,
    supplier_id TEXT NOT NULL REFERENCES suppliers(supplier_id),
    lot         TEXT NOT NULL,
    received_at TEXT NOT NULL,
    qty         INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'APPROVED',  -- APPROVED / QUARANTINED / REJECTED
    PRIMARY KEY (part_no, lot)
);

CREATE TABLE training_records (
    emp_id         TEXT NOT NULL,
    employee_name  TEXT NOT NULL,
    certification  TEXT NOT NULL,      -- SOP-OPR-101 etc
    certified_at   TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'CERTIFIED',
    PRIMARY KEY (emp_id, certification)
);
