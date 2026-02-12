-- Maryland Campaign Finance Database Schema

CREATE TABLE IF NOT EXISTS committees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    committee_id TEXT UNIQUE,
    committee_name TEXT,
    jurisdiction TEXT,
    office_sought TEXT,
    filing_period TEXT,
    raw_data TEXT
);

CREATE TABLE IF NOT EXISTS contributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    committee_id TEXT,
    committee_name TEXT,
    contributor_name TEXT,
    contributor_name_norm TEXT,
    contributor_type TEXT,
    contributor_address TEXT,
    employer TEXT,
    occupation TEXT,
    amount REAL,
    date TEXT,
    filing_period TEXT,
    raw_data TEXT
);

CREATE TABLE IF NOT EXISTS expenditures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    committee_id TEXT,
    committee_name TEXT,
    payee_name TEXT,
    payee_name_norm TEXT,
    amount REAL,
    date TEXT,
    category TEXT,
    purpose TEXT,
    filing_period TEXT,
    raw_data TEXT
);

CREATE TABLE IF NOT EXISTS donor_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    donor_name_norm TEXT,
    committee_id_1 TEXT,
    committee_name_1 TEXT,
    committee_id_2 TEXT,
    committee_name_2 TEXT,
    total_to_1 REAL,
    total_to_2 REAL,
    count_to_1 INTEGER,
    count_to_2 INTEGER
);

CREATE TABLE IF NOT EXISTS vendor_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_name_norm TEXT,
    committee_id_1 TEXT,
    committee_name_1 TEXT,
    committee_id_2 TEXT,
    committee_name_2 TEXT,
    total_from_1 REAL,
    total_from_2 REAL,
    count_from_1 INTEGER,
    count_from_2 INTEGER
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
    name, type, entity_id, extra
);

CREATE TABLE IF NOT EXISTS data_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Indices for fast queries
CREATE INDEX IF NOT EXISTS idx_contrib_committee ON contributions(committee_id);
CREATE INDEX IF NOT EXISTS idx_contrib_name_norm ON contributions(contributor_name_norm);
CREATE INDEX IF NOT EXISTS idx_contrib_date ON contributions(date);
CREATE INDEX IF NOT EXISTS idx_contrib_amount ON contributions(amount);
CREATE INDEX IF NOT EXISTS idx_expend_committee ON expenditures(committee_id);
CREATE INDEX IF NOT EXISTS idx_expend_payee_norm ON expenditures(payee_name_norm);
CREATE INDEX IF NOT EXISTS idx_expend_date ON expenditures(date);
CREATE INDEX IF NOT EXISTS idx_expend_amount ON expenditures(amount);
CREATE INDEX IF NOT EXISTS idx_donor_links_name ON donor_links(donor_name_norm);
CREATE INDEX IF NOT EXISTS idx_vendor_links_name ON vendor_links(vendor_name_norm);
CREATE INDEX IF NOT EXISTS idx_committees_jurisdiction ON committees(jurisdiction);
