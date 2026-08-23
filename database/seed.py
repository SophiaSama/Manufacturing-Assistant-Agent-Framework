#!/usr/bin/env python3
"""Seed the Northgate Assembly Plant (NGA) synthetic production database.

Generates nga.db from schema.sql with deterministic, story-consistent data.
The stories planted here mirror the corpus documents (SOP/TEC/FAP/ESC/QCR).

Storylines embedded:
  1. TORQUE-LOW  : TQ-6012 drift on 2025-04-08 Shift C -> AU-0015/16/17 low
                    lug nut torque -> NC-2025-0137 (Class A) -> escalation
                    L2/L3 -> L4 open field-action evaluation (FA-2025-001).
  2. E-5023      : RB-07 TCP deviation on AU-0007 -> NC-2025-0142 (Class B).
  3. P-122       : PT-ROB-03 flow deviation, film build 38 um on SO-0006
                    -> NC-2025-0150 (Class B), WO-2025-0440 open.
  4. Quarantined brake fluid lot FW-2025-03-30 (moisture concern).
  5. One expired training certification (E-3002 / SOP-TEC-214).
"""
import os
import random
import sqlite3
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "nga.db")
random.seed(42)

WHEEL_TARGET = 105.0
WHEEL_MIN, WHEEL_MAX = 99.75, 110.25  # 105 Nm +/- 5%
NUTS = [f"{wheel}{i}" for wheel in ("LF", "RF", "LR", "RR") for i in range(1, 6)]


def dt(iso):
    return datetime.fromisoformat(iso)


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    with open(os.path.join(HERE, "schema.sql")) as f:
        cur.executescript(f.read())

    # ------------------------------------------------------------------ lines
    lines = [
        ("BODY", "Body Shop", 42.0),
        ("PAINT", "Paint Shop", 42.0),
        ("GA", "General Assembly", 42.0),
        ("FINAL", "Final & Quality", 42.0),
    ]
    cur.executemany("INSERT INTO lines VALUES (?,?,?)", lines)

    stations = [
        ("42", "BODY", "Weld LH side panel", "Zone 2B"),
        ("51", "BODY", "Roof weld", "Zone 2C"),
        ("61", "PAINT", "Primer", "Booth 1"),
        ("68", "PAINT", "Base coat", "Booth 2"),
        ("74", "PAINT", "Clear coat", "Booth 3"),
        ("118", "GA", "Engine dress", "Engine line"),
        ("131", "GA", "Brake fill", "Chassis line"),
        ("144", "GA", "Wheel install", "Chassis line"),
        ("152", "GA", "Glass", "Trim line"),
        ("168", "FINAL", "Final inspection / torque audit", "Final line"),
    ]
    cur.executemany("INSERT INTO stations VALUES (?,?,?,?)", stations)

    machines = [
        ("RB-07", "WELD_ROBOT", "RoboTech RX-700", "42", "2024-03-01", "ACTIVE"),
        ("RB-12", "WELD_ROBOT", "RoboTech RX-700", "51", "2024-03-01", "ACTIVE"),
        ("CV-2", "CONVEYOR", "Nexus CV-2", "42", "2024-03-01", "ACTIVE"),
        ("PT-ROB-01", "PAINT_ROBOT", "SprayTech ST-450", "61", "2024-04-15", "ACTIVE"),
        ("PT-ROB-02", "PAINT_ROBOT", "SprayTech ST-450", "68", "2024-04-15", "ACTIVE"),
        ("PT-ROB-03", "PAINT_ROBOT", "SprayTech ST-450", "74", "2024-04-15", "ACTIVE"),
        ("BF-900", "FILL_MACHINE", "Nexus BF-900", "131", "2024-05-01", "ACTIVE"),
        ("TQ-6012", "TORQUE_TOOL", "TorqMaster TF-6000", "144", "2024-11-01", "ACTIVE"),
        ("TQ-6015", "TORQUE_TOOL", "TorqMaster TF-6000", "118", "2024-11-01", "ACTIVE"),
        ("TQ-6018", "TORQUE_TOOL", "TorqMaster TF-6000", "168", "2025-01-15", "ACTIVE"),
        ("TQ-6019", "TORQUE_TOOL", "TorqMaster TF-6000", "144", "2025-01-20", "ACTIVE"),
    ]
    cur.executemany("INSERT INTO machines VALUES (?,?,?,?,?,?)", machines)

    shifts = [
        ("A", "06:00", "14:00", "Elena Novak"),
        ("B", "14:00", "22:00", "Marcus Reed"),
        ("C", "22:00", "06:00", "Sofia Chen"),
    ]
    cur.executemany("INSERT INTO shifts VALUES (?,?,?,?)", shifts)

    # ------------------------------------------------------------- vehicles
    # Aurora: 40 units (2025-04-07..09), Solstice: 20 units (same window)
    vehicles, build_ts = [], {}
    aurora_dates = ["2025-04-07"] * 14 + ["2025-04-08"] * 14 + ["2025-04-09"] * 12
    solstice_dates = ["2025-04-07"] * 7 + ["2025-04-08"] * 7 + ["2025-04-09"] * 6

    def assign(idx, model, dates, default_shift="C"):
        """idx is 1-based. Return (date, shift)."""
        date = dates[idx - 1]
        shifts_cycle = ["A", "B", "C"]
        shift = STORY_SHIFT[(model, idx)] if (model, idx) in STORY_SHIFT else shifts_cycle[idx % 3]
        return date, shift

    # Story assignments: torque issue vehicles on 04-08 Shift C,
    # E-5023 on 04-07 Shift B, film-build on 04-09 Shift A.
    STORY_SHIFT = {
        ("AU-2025", 15): "C", ("AU-2025", 16): "C", ("AU-2025", 17): "C",
        ("AU-2025", 7): "B",
        ("SO-2025", 16): "A",
    }
    lot_counter = {}

    for model, prefix, n, dates in (("AU-2025", "NGA-AU25", 40, aurora_dates),
                                    ("SO-2025", "NGA-SO25", 20, solstice_dates)):
        for i in range(1, n + 1):
            vin = f"{prefix}-{i:04d}"
            date, shift = assign(i, model, dates)
            lot = f"L-{date}-{shift}"
            lot_counter[lot] = lot_counter.get(lot, 0) + 1
            # build start times within shift (hh:mm)
            if shift == "A":
                start = dt(f"{date} 07:{10 + (i % 40):02d}:00")
            elif shift == "B":
                start = dt(f"{date} 15:{10 + (i % 40):02d}:00")
            else:
                start = dt(f"{date} 22:{10 + (i % 40):02d}:00")
            vehicles.append((vin, model, date, shift, lot, "SHIPPED"))
            build_ts[vin] = start

    # Story statuses: torque suspects quarantined, film-build unit held
    status_override = {
        "NGA-AU25-0015": "QUARANTINED",
        "NGA-AU25-0016": "QUARANTINED",
        "NGA-AU25-0017": "QUARANTINED",
        "NGA-SO25-0016": "HELD",
    }
    vehicles = [(vin, model, date, shift, lot, status_override.get(vin, st))
                for vin, model, date, shift, lot, st in vehicles]

    cur.executemany("INSERT INTO vehicles VALUES (?,?,?,?,?,?)", vehicles)

    # ----------------------------------------------------------------- build
    build_rows = []
    for vin, model, date, shift, lot, _ in vehicles:
        base = build_ts[vin]
        for k, sid in enumerate(("42", "51", "61", "68", "74", "118", "131",
                                 "144", "152", "168")):
            result = "PASS"
            if vin == "NGA-AU25-0015" and sid == "168":
                result = "FAIL"
            elif vin in ("NGA-AU25-0016", "NGA-AU25-0017") and sid == "168":
                result = "FAIL"
            elif vin == "NGA-AU25-0007" and sid == "42":
                result = "FAIL"
            elif vin == "NGA-SO25-0016" and sid == "74":
                result = "FAIL"
            build_rows.append((vin, sid, (base + timedelta(minutes=3 + k * 2)).isoformat(sep=" "), result))
    cur.executemany("INSERT INTO build_records (vin, station_id, ts, result) VALUES (?,?,?,?)", build_rows)

    # ------------------------------------------------------ torque readings
    torque_rows = []
    for vin, model, *_ in vehicles:
        if model == "AU-2025":
            ts = build_ts[vin] + timedelta(minutes=25)  # reaches sta 144
        else:
            ts = build_ts[vin] + timedelta(minutes=25)
        for nut in NUTS:
            t = random.gauss(WHEEL_TARGET, 1.6)
            in_tol = 1 if WHEEL_MIN <= t <= WHEEL_MAX else 0
            torque_rows.append((vin, "TQ-6012", nut, ts.isoformat(sep=" "),
                                round(t, 1), WHEEL_TARGET, in_tol))
    # Planted low-torque story (Shift C 04-08, TQ-6012 drift)
    for vin, nuts in {
        "NGA-AU25-0015": {"LF2": 97.2, "LF4": 96.8, "RF1": 97.9},
        "NGA-AU25-0016": {"LR3": 98.9},
        "NGA-AU25-0017": {"RR2": 98.4},
    }.items():
        ts = build_ts[vin] + timedelta(minutes=25)
        for k, (nut, val) in enumerate(nuts.items()):
            torque_rows.append((vin, "TQ-6012", nut, ts.isoformat(sep=" "),
                                val, WHEEL_TARGET, 0))
    cur.executemany("INSERT INTO torque_readings (vin, tool_id, nut_position, ts, torque_nm, target_nm, in_tolerance) "
                    "VALUES (?,?,?,?,?,?,?)", torque_rows)

    # -------------------------------------------------------- quality checks
    qc = []
    # TORQUE_AUDIT at station 168 (per SOP-OPR-158: every 2h, 3 vehicles)
    audits = [
        ("2025-04-07 08:00", ["NGA-AU25-0001", "NGA-AU25-0002", "NGA-AU25-0003"], "PASS", 104.8),
        ("2025-04-07 16:00", ["NGA-AU25-0009", "NGA-AU25-0010", "NGA-AU25-0011"], "PASS", 105.2),
        ("2025-04-08 16:00", ["NGA-AU25-0021", "NGA-AU25-0022", "NGA-AU25-0023"], "PASS", 104.9),
        ("2025-04-09 05:45", ["NGA-AU25-0015", "NGA-AU25-0016", "NGA-AU25-0017"], "FAIL", 96.8),
    ]
    for ts, vins, res, val in audits:
        for vin in vins:
            qc.append((vin, "TORQUE_AUDIT", "168", res, val, WHEEL_MIN, WHEEL_MAX, ts))
    # LEAK_TEST every 10th vehicle (SOP-OPR-114)
    for vin, *_ in vehicles:
        if vin.endswith(("001", "011", "021", "031")):
            qc.append((vin, "LEAK_TEST", "152", "PASS", None, None, None, "2025-04-08 12:00"))
    # RETENTION 1/shift (SOP-OPR-114): pull strength > 1200 N
    for vin in ("NGA-AU25-0003", "NGA-AU25-0010", "NGA-AU25-0020",
                "NGA-AU25-0030", "NGA-AU25-0040", "NGA-SO25-0005", "NGA-SO25-0015"):
        qc.append((vin, "RETENTION", "152", "PASS", round(random.uniform(1240, 1360), 1), 1200.0, None,
                   "2025-04-09 13:00"))
    # FILM_BUILD at station 74 (TEC-342: 40-52 um)
    for vin, val, res in [
        ("NGA-AU25-0001", 46.0, "PASS"), ("NGA-AU25-0010", 45.0, "PASS"),
        ("NGA-AU25-0025", 44.0, "PASS"), ("NGA-AU25-0035", 47.0, "PASS"),
        ("NGA-SO25-0002", 45.0, "PASS"), ("NGA-SO25-0016", 38.0, "FAIL"),
        ("NGA-SO25-0012", 46.0, "PASS"), ("NGA-AU25-0038", 41.0, "PASS"),
    ]:
        qc.append((vin, "FILM_BUILD", "74", res, val, 40.0, 52.0, "2025-04-09 09:10"))
    cur.executemany("INSERT INTO quality_checks (vin, check_type, station_id, result, measured_value, spec_min, spec_max, ts) "
                    "VALUES (?,?,?,?,?,?,?,?)", qc)

    # --------------------------------------------------------------- defects
    defects = [
        ("NGA-AU25-0015", "168", "TORQUE-LOW", "A",
         "3 lug nuts below 99.75 Nm (LF2 97.2, LF4 96.8, RF1 97.9); wheel retention risk", "2025-04-09 05:40", "CONTAINED"),
        ("NGA-AU25-0016", "168", "TORQUE-LOW", "A", "LR3 at 98.9 Nm (below spec)", "2025-04-09 05:42", "CONTAINED"),
        ("NGA-AU25-0017", "168", "TORQUE-LOW", "A", "RR2 at 98.4 Nm (below spec)", "2025-04-09 05:43", "CONTAINED"),
        ("NGA-AU25-0007", "42", "E-5023", "B", "RB-07 TCP deviation 0.7 mm at LH side panel", "2025-04-07 15:20", "CLOSED"),
        ("NGA-SO25-0016", "74", "P-122", "B", "Clear coat flow deviation; film build 38 um (spec 40-52)", "2025-04-09 09:10", "OPEN"),
    ]
    cur.executemany("INSERT INTO defects (vin, station_id, defect_code, defect_class, description, detected_at, status) "
                    "VALUES (?,?,?,?,?,?,?)", defects)

    nc_records = [
        ("NC-2025-0137", "NGA-AU25-0015", "TORQUE-LOW", "A", "2025-04-09 05:45", "Priya Raman", "CONTAINED"),
        ("NC-2025-0142", "NGA-AU25-0007", "E-5023", "B", "2025-04-07 15:25", "Miguel Santos", "CLOSED"),
        ("NC-2025-0150", "NGA-SO25-0016", "P-122", "B", "2025-04-09 09:15", "Miguel Santos", "OPEN"),
    ]
    cur.executemany("INSERT INTO nc_records VALUES (?,?,?,?,?,?,?)", nc_records)

    # ------------------------------------------------------------- work orders
    wos = [
        ("WO-2025-0417", "RB-07", "E-4012", "P1", "2025-04-07 08:00", "2025-04-07 09:15", "COMPLETED", "Tom Okafor"),
        ("WO-2025-0419", "RB-07", "E-5023", "P2", "2025-04-07 15:30", "2025-04-07 17:00", "COMPLETED", "Lena Fischer"),
        ("WO-2025-0421", "CV-2", "RC-01", "P3", "2025-04-08 10:30", "2025-04-08 11:00", "COMPLETED", "Raj Patel"),
        ("WO-2025-0433", "TQ-6012", "DRIFT", "P1", "2025-04-09 07:15", "2025-04-09 09:30", "COMPLETED", "Chen Wei"),
        ("WO-2025-0440", "PT-ROB-03", "P-122", "P2", "2025-04-09 09:30", None, "OPEN", "Ana Souza"),
    ]
    cur.executemany("INSERT INTO work_orders VALUES (?,?,?,?,?,?,?,?)", wos)

    mlogs = [
        ("TQ-6012", "DRIFT_FLAG", "2025-04-09 06:30", "FAIL", "weekly self-check drift 2.4% > 2% threshold"),
        ("TQ-6012", "CALIBRATION", "2025-04-09 09:30", "PASS", "recalibrated on Nexus CR-500, within +/-1.5%"),
        ("TQ-6018", "CALIBRATION", "2025-04-05 08:00", "PASS", "weekly audit wrench calibration"),
        ("RB-07", "REPAIR", "2025-04-07 09:15", "PASS", "servo amp replaced (SP-RX-04012)"),
        ("RB-07", "REPAIR", "2025-04-07 17:00", "PASS", "TCP re-taught, deviation 0.15 mm"),
        ("CV-2", "INSPECTION", "2025-04-08 11:00", "PASS", "cleared weld spatter from PS-200-1 (RC-01)"),
        ("PT-ROB-03", "INSPECTION", "2025-04-09 10:30", "PASS", "replaced bell cup (SP-ST-3401)"),
    ]
    cur.executemany("INSERT INTO maintenance_logs (machine_id, log_type, performed_at, result, note) "
                    "VALUES (?,?,?,?,?)", mlogs)

    # ------------------------------------------------------------ escalations
    esc = [
        ("NC-2025-0137", 2, "2025-04-09 05:45", "2025-04-09 06:00", "2025-04-09 07:00", "Marcus Reed", "CLOSED"),
        ("NC-2025-0137", 3, "2025-04-09 07:00", "2025-04-09 07:30", "2025-04-10 10:00", "Miguel Santos", "CLOSED"),
        ("NC-2025-0137", 4, "2025-04-10 10:00", "2025-04-10 10:30", None, "Dana Whitfield", "OPEN"),
        ("NC-2025-0142", 2, "2025-04-07 15:25", "2025-04-07 15:40", "2025-04-07 17:30", "Tom Okafor", "CLOSED"),
        ("NC-2025-0150", 2, "2025-04-09 09:15", "2025-04-09 09:25", "2025-04-09 12:00", "Tom Okafor", "CLOSED"),
    ]
    cur.executemany("INSERT INTO escalations (nc_id, level, triggered_at, acknowledged_at, resolved_at, responder, status) "
                    "VALUES (?,?,?,?,?,?,?)", esc)

    field_actions = [
        ("FA-2025-001", "TORQUE-LOW", "CAMPAIGN", "EVALUATION", "2025-04-10 10:00", 42, "C3"),
    ]
    cur.executemany("INSERT INTO field_actions VALUES (?,?,?,?,?,?,?)", field_actions)

    # ----------------------------------------------------------- suppliers/parts
    suppliers = [
        ("SUP-001", "ApexForging", "Fasteners & lug nuts"),
        ("SUP-002", "ChemBond", "Adhesives & primers"),
        ("SUP-003", "FluidWorks", "Brake fluids"),
        ("SUP-004", "ApexPaint", "Coatings"),
        ("SUP-005", "EliteRubber", "Engine & transmission mounts"),
    ]
    cur.executemany("INSERT INTO suppliers VALUES (?,?,?)", suppliers)

    parts = [
        ("AU-44102", "Lug nut M12x1.5 (Aurora)", "SUP-001", "LF-2025-03-17", "2025-03-18", 12000, "APPROVED"),
        ("SO-44102", "Lug nut M12x1.5 (Solstice)", "SUP-001", "LF-2025-03-17", "2025-03-18", 8000, "APPROVED"),
        ("B-99122", "Engine mount bolt M12x1.25", "SUP-001", "LF-2025-03-02", "2025-03-03", 6000, "APPROVED"),
        ("GL-77100", "Windshield adhesive PU", "SUP-002", "CB-2025-03-02", "2025-03-03", 400, "APPROVED"),
        ("GL-77105", "Glass primer black", "SUP-002", "CB-2025-03-02", "2025-03-03", 200, "APPROVED"),
        ("FL-88200", "Brake fluid DOT 4", "SUP-003", "FW-2025-02-10", "2025-02-12", 600, "APPROVED"),
        ("FL-88200", "Brake fluid DOT 4", "SUP-003", "FW-2025-03-30", "2025-04-01", 150, "QUARANTINED"),
        ("PC-99010", "Clear coat 2K solvent", "SUP-004", "AP-2025-03-25", "2025-03-26", 900, "APPROVED"),
        ("AU-51040", "RH engine mount (hydraulic)", "SUP-005", "ER-2025-02-28", "2025-03-01", 500, "APPROVED"),
        ("AU-51041", "LH engine mount (hydraulic)", "SUP-005", "ER-2025-02-28", "2025-03-01", 500, "APPROVED"),
        ("AU-51050", "Rear transmission mount", "SUP-005", "ER-2025-02-28", "2025-03-01", 500, "APPROVED"),
    ]
    cur.executemany("INSERT INTO parts VALUES (?,?,?,?,?,?,?)", parts)

    # ------------------------------------------------------------ training records
    emp = [
        ("E-1001", "Elena Novak", ["SOP-OPR-101", "SOP-OPR-114", "SOP-OPR-127", "SOP-OPR-142", "SOP-OPR-158"]),
        ("E-1002", "Marcus Reed", ["SOP-OPR-101", "SOP-OPR-127", "SOP-OPR-158", "SOP-TEC-214"]),
        ("E-1003", "Sofia Chen", ["SOP-OPR-101", "SOP-OPR-114", "SOP-OPR-142", "SOP-OPR-158"]),
        ("E-2001", "Raj Patel", ["SOP-TEC-201", "SOP-TEC-214", "SOP-TEC-228"]),
        ("E-2002", "Lena Fischer", ["SOP-TEC-201", "SOP-TEC-214", "SOP-TEC-228", "SOP-TEC-235"]),
        ("E-2003", "Chen Wei", ["SOP-TEC-201", "SOP-TEC-214", "SOP-TEC-228", "SOP-TEC-235"]),
        ("E-3001", "Ana Souza", ["SOP-TEC-214", "SOP-TEC-235"]),
        ("E-3002", "James Park", ["SOP-TEC-214", "SOP-OPR-101"]),
    ]
    training = []
    base = dt("2024-06-01")
    for emp_id, name, certs in emp:
        for i, cert in enumerate(certs):
            cert_date = base + timedelta(days=30 * i)
            expires = cert_date.replace(year=cert_date.year + 1)
            status = "CERTIFIED"
            if emp_id == "E-3002" and cert == "SOP-TEC-214":
                expires = dt("2025-03-15")
                status = "EXPIRED"
            training.append((emp_id, name, cert, cert_date.date().isoformat(),
                             expires.date().isoformat(), status))
    cur.executemany("INSERT INTO training_records VALUES (?,?,?,?,?,?)", training)

    con.commit()
    con.close()
    print("nga.db created successfully.")


if __name__ == "__main__":
    main()
