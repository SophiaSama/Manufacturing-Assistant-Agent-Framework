"""Initialize synthetic core banking SQLite database."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "banking.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

schema = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    customer_type TEXT NOT NULL, -- Retail / Corporate
    risk_rating TEXT NOT NULL,   -- Low / Medium / High
    country_of_incorporation TEXT,
    pep_status INTEGER DEFAULT 0 -- Politically Exposed Person (0/1)
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    account_type TEXT NOT NULL, -- Checking / Savings / Escrow
    balance REAL NOT NULL,
    currency TEXT DEFAULT 'USD',
    status TEXT DEFAULT 'ACTIVE', -- ACTIVE / FROZEN / RESTRICTED
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    amount REAL NOT NULL,
    direction TEXT NOT NULL, -- INBOUND / OUTBOUND
    counterparty_name TEXT,
    counterparty_country TEXT,
    memo TEXT,
    FOREIGN KEY(account_id) REFERENCES accounts(account_id)
);

CREATE TABLE IF NOT EXISTS sanctions_list (
    entity_id TEXT PRIMARY KEY,
    entity_name TEXT NOT NULL,
    program TEXT NOT NULL,
    match_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_alerts (
    alert_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT DEFAULT 'OPEN',
    FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
);
"""

data = """
INSERT OR REPLACE INTO customers VALUES ('CUST-001', 'Acme Trading Corp', 'Corporate', 'Medium', 'US', 0);
INSERT OR REPLACE INTO customers VALUES ('CUST-002', 'Vostok Trading Group', 'Corporate', 'High', 'Cyprus', 1);
INSERT OR REPLACE INTO customers VALUES ('CUST-003', 'Jane Doe', 'Retail', 'Low', 'US', 0);

INSERT OR REPLACE INTO accounts VALUES ('ACC-101', 'CUST-001', 'Checking', 450000.0, 'USD', 'ACTIVE');
INSERT OR REPLACE INTO accounts VALUES ('ACC-202', 'CUST-002', 'Checking', 1850000.0, 'USD', 'RESTRICTED');
INSERT OR REPLACE INTO accounts VALUES ('ACC-303', 'CUST-003', 'Savings', 12500.0, 'USD', 'ACTIVE');

INSERT OR REPLACE INTO transactions VALUES ('TX-9001', 'ACC-101', '2026-03-01 10:15:00', 45000.0, 'OUTBOUND', 'Global Supply LLC', 'DE', 'Invoice payment 4402');
INSERT OR REPLACE INTO transactions VALUES ('TX-9002', 'ACC-202', '2026-03-02 14:20:00', 1200000.0, 'OUTBOUND', 'Orion Minerals International', 'Panama', 'Consulting fee');
INSERT OR REPLACE INTO transactions VALUES ('TX-9003', 'ACC-303', '2026-03-03 09:00:00', 9500.0, 'INBOUND', 'Cash Deposit', 'US', 'ATM Deposit');

INSERT OR REPLACE INTO sanctions_list VALUES ('SAN-1001', 'Al-Baraka Logistics Ltd', 'SDGT', 'CONFIRMED_MATCH');
INSERT OR REPLACE INTO sanctions_list VALUES ('SAN-1002', 'Vostok Trading Group', 'UKRAINE-EO14024', 'CONFIRMED_MATCH');
INSERT OR REPLACE INTO sanctions_list VALUES ('SAN-1003', 'Orion Minerals International', 'BURMA', 'CONFIRMED_MATCH');

INSERT OR REPLACE INTO audit_alerts VALUES ('ALT-501', 'TX-9002', 'OFAC_SANCTION_HIT', 'CRITICAL', 'OPEN');
INSERT OR REPLACE INTO audit_alerts VALUES ('ALT-502', 'TX-9003', 'STRUCTURING_WARNING', 'MEDIUM', 'OPEN');
"""

def init():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema)
    conn.executescript(data)
    conn.commit()
    conn.close()
    print(f"Initialized banking database at {DB_PATH}")

if __name__ == "__main__":
    init()
