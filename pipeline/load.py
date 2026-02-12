"""Data loading pipeline â CSV to SQLite."""
import os
import sys
import sqlite3
import shutil
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, BACKUP_DIR
from pipeline.clean import (
    load_committees_raw, load_contributions_raw, load_expenditures_raw,
    prepare_committees, prepare_contributions, prepare_expenditures,
)
from pipeline.networks import compute_donor_links, compute_vendor_links


def init_db():
    """Create tables from schema."""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'db', 'schema.sql')
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.close()


def backup_db():
    """Backup existing database before reload."""
    if os.path.exists(DB_PATH):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = os.path.join(BACKUP_DIR, f'campaign_finance_{ts}.db')
        shutil.copy2(DB_PATH, backup)
        # Keep only last 3 backups
        backups = sorted([
            os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) if f.endswith('.db')
        ])
        for old in backups[:-3]:
            os.remove(old)


def clear_tables():
    """Clear all data tables."""
    conn = sqlite3.connect(DB_PATH)
    for table in ['committees', 'contributions', 'expenditures', 'donor_links', 'vendor_links']:
        conn.execute(f"DELETE FROM {table}")
    # Clear FTS index
    try:
        conn.execute("DELETE FROM search_index")
    except Exception:
        pass
    conn.commit()
    conn.close()


def bulk_insert(conn, table, records, columns):
    """Insert records in chunks of 10000."""
    if not records:
        return 0
    placeholders = ', '.join(['?'] * len(columns))
    col_str = ', '.join(columns)
    sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"

    count = 0
    chunk_size = 10000
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        rows = [tuple(r.get(c, '') for c in columns) for r in chunk]
        conn.executemany(sql, rows)
        count += len(chunk)
    return count


def build_search_index(conn):
    """Build FTS5 search index from loaded data."""
    try:
        conn.execute("DELETE FROM search_index")
    except Exception:
        pass

    # Index committees
    conn.execute("""
        INSERT INTO search_index (name, type, entity_id, extra)
        SELECT committee_name, 'committee', committee_id, jurisdiction
        FROM committees WHERE committee_name != ''
    """)
    # Index donors (unique normalized names)
    conn.execute("""
        INSERT INTO search_index (name, type, entity_id, extra)
        SELECT DISTINCT contributor_name_norm, 'donor', '', ''
        FROM contributions
        WHERE contributor_name_norm != '' AND contributor_name_norm IS NOT NULL
    """)
    # Index vendors
    conn.execute("""
        INSERT INTO search_index (name, type, entity_id, extra)
        SELECT DISTINCT payee_name_norm, 'vendor', '', ''
        FROM expenditures
        WHERE payee_name_norm != '' AND payee_name_norm IS NOT NULL
    """)
    conn.commit()


def run_full_load(progress_callback=None):
    """Full pipeline: backup -> init -> load CSVs -> compute networks -> index."""
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    log("Backing up existing database...")
    backup_db()

    log("Initializing database schema...")
    init_db()

    log("Clearing existing data...")
    clear_tables()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Load committees
    log("Loading committees CSV...")
    df = load_committees_raw()
    if df is not None:
        records = prepare_committees(df)
        n = bulk_insert(conn, 'committees', records,
                       ['committee_id', 'committee_name', 'jurisdiction', 'office_sought', 'filing_period'])
        conn.commit()
        log(f"  Loaded {n} committees")
    else:
        log("  No committees.csv found, skipping")

    # Load contributions
    log("Loading contributions CSV (this may take a minute for large files)...")
    df = load_contributions_raw()
    if df is not None:
        log(f"  Read {len(df)} raw rows, preparing...")
        records = prepare_contributions(df)
        n = bulk_insert(conn, 'contributions', records,
                       ['committee_id', 'committee_name', 'contributor_name', 'contributor_name_norm',
                        'contributor_type', 'contributor_address', 'employer', 'occupation',
                        'amount', 'date', 'filing_period'])
        conn.commit()
        log(f"  Loaded {n} contributions")
    else:
        log("  No contributions.csv found, skipping")

    # Load expenditures
    log("Loading expenditures CSV...")
    df = load_expenditures_raw()
    if df is not None:
        log(f"  Read {len(df)} raw rows, preparing...")
        records = prepare_expenditures(df)
        n = bulk_insert(conn, 'expenditures', records,
                       ['committee_id', 'committee_name', 'payee_name', 'payee_name_norm',
                        'amount', 'date', 'category', 'purpose', 'filing_period'])
        conn.commit()
        log(f"  Loaded {n} expenditures")
    else:
        log("  No expenditures.csv found, skipping")

    # Build search index
    log("Building search index...")
    build_search_index(conn)

    # Update metadata
    conn.execute("DELETE FROM data_meta")
    conn.execute("INSERT INTO data_meta (key, value) VALUES (?, ?)",
                ('last_load', datetime.now().isoformat()))
    conn.execute("INSERT INTO data_meta (key, value) VALUES (?, ?)",
                ('committees_count', str(conn.execute("SELECT COUNT(*) FROM committees").fetchone()[0])))
    conn.execute("INSERT INTO data_meta (key, value) VALUES (?, ?)",
                ('contributions_count', str(conn.execute("SELECT COUNT(*) FROM contributions").fetchone()[0])))
    conn.execute("INSERT INTO data_meta (key, value) VALUES (?, ?)",
                ('expenditures_count', str(conn.execute("SELECT COUNT(*) FROM expenditures").fetchone()[0])))
    conn.commit()
    conn.close()

    # Compute networks
    log("Computing donor network links...")
    donor_count = compute_donor_links(DB_PATH)
    log(f"  Found {donor_count} donor links")

    log("Computing vendor network links...")
    vendor_count = compute_vendor_links(DB_PATH)
    log(f"  Found {vendor_count} vendor links")

    log("Pipeline complete!")
    return {
        'committees': conn.execute("SELECT COUNT(*) FROM committees").fetchone()[0] if False else 0,
        'donor_links': donor_count,
        'vendor_links': vendor_count,
    }


if __name__ == '__main__':
    run_full_load()
