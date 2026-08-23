#!/usr/bin/env python3
"""Run a live SQL query against the NGA database.

Usage:
    python3 query.py "SELECT * FROM defects;"
    python3 query.py -f example-queries.sql --limit 20
    echo "SELECT COUNT(*) FROM vehicles;" | python3 query.py
"""
import argparse
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).parent / "nga.db"


def run(sql: str, limit: int = 50) -> str:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    out = []
    try:
        for row in con.execute(sql).fetchmany(limit):
            out.append(" | ".join(f"{k}={v}" for k, v in dict(row).items()))
    finally:
        con.close()
    return "\n".join(out) if out else "(no rows)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sql", nargs="?", help="SQL to run")
    ap.add_argument("-f", "--file", help="file containing SQL (split on ';')")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if args.file:
        text = Path(args.file).read_text()
        statements = []
        for s in text.split(";"):
            lines = [ln for ln in s.splitlines() if not ln.strip().startswith("--")]
            stmt = "\n".join(lines).strip()
            if stmt:
                statements.append(stmt)
        for i, s in enumerate(statements, 1):
            print(f"=== Statement {i} ===")
            print(s)
            print(run(s, args.limit))
    elif args.sql:
        print(run(args.sql, args.limit))
    else:
        sql = sys.stdin.read().strip()
        print(run(sql, args.limit))


if __name__ == "__main__":
    main()
