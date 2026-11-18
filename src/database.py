"""
RevGuard AI — SQLite persistence layer
DB: data/revguard.db  (relative to this file's location)
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "revguard.db",
)


def init_db() -> None:
    """Create tables if they do not already exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_runs (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp      TEXT,
                num_accounts       INTEGER,
                total_leaked_usd   REAL,
                precision          REAL,
                recall             REAL,
                f1_score           REAL,
                ml_corroborated    INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leak_records (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id             INTEGER REFERENCES audit_runs(id),
                customer_id        TEXT,
                invoice_number     TEXT,
                service            TEXT,
                leak_type          TEXT,
                expected_usd       REAL,
                billed_usd         REAL,
                leaked_usd         REAL,
                priority           TEXT,
                ml_anomaly_score   REAL,
                ml_flagged         INTEGER,
                detection_engine   TEXT,
                evidence           TEXT,
                resolution         TEXT DEFAULT 'OPEN',
                jira_ticket        TEXT,
                created_at         TEXT
            )
            """
        )
        conn.commit()


def save_audit_run(metrics: dict, leaks: list) -> int:
    """
    Persist a full audit run.

    Parameters
    ----------
    metrics : dict
        Keys: num_accounts, total_leaked_usd, precision, recall, f1_score,
              ml_corroborated
    leaks : list of dict
        Each dict maps to leak_records columns (minus id, run_id, created_at).

    Returns
    -------
    int
        The new audit_runs.id (run_id).
    """
    init_db()
    run_timestamp = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.execute(
            """
            INSERT INTO audit_runs
                (run_timestamp, num_accounts, total_leaked_usd,
                 precision, recall, f1_score, ml_corroborated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_timestamp,
                metrics.get("num_accounts"),
                metrics.get("total_leaked_usd"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                metrics.get("ml_corroborated"),
            ),
        )
        run_id = cur.lastrowid
        created_at = datetime.utcnow().isoformat()
        for leak in leaks:
            conn.execute(
                """
                INSERT INTO leak_records
                    (run_id, customer_id, invoice_number, service, leak_type,
                     expected_usd, billed_usd, leaked_usd, priority,
                     ml_anomaly_score, ml_flagged, detection_engine, evidence,
                     resolution, jira_ticket, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    leak.get("customer_id"),
                    leak.get("invoice_number"),
                    leak.get("service"),
                    leak.get("leak_type"),
                    leak.get("expected_usd"),
                    leak.get("billed_usd"),
                    leak.get("leaked_usd"),
                    leak.get("priority"),
                    leak.get("ml_anomaly_score"),
                    leak.get("ml_flagged"),
                    leak.get("detection_engine"),
                    leak.get("evidence"),
                    leak.get("resolution", "OPEN"),
                    leak.get("jira_ticket"),
                    created_at,
                ),
            )
        conn.commit()
    return run_id


def get_all_leaks(resolution_filter: str | None = None) -> list[dict]:
    """
    Return all leak_records as a list of dicts.

    Parameters
    ----------
    resolution_filter : str, optional
        If provided, only rows whose `resolution` matches this value are
        returned (e.g. 'OPEN', 'RECOVERED').
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if resolution_filter is not None:
            cur = conn.execute(
                "SELECT * FROM leak_records WHERE resolution = ?",
                (resolution_filter,),
            )
        else:
            cur = conn.execute("SELECT * FROM leak_records")
        return [dict(row) for row in cur.fetchall()]


def get_leak_by_id(leak_id: int) -> dict | None:
    """Return a single leak_record dict, or None if not found."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM leak_records WHERE id = ?", (leak_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None


def mark_recovered(leak_id: int, jira_ticket: str = "") -> bool:
    """
    Set resolution='RECOVERED' and update jira_ticket for the given leak.

    Returns True if a row was updated, False if the id was not found.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            UPDATE leak_records
               SET resolution = 'RECOVERED',
                   jira_ticket = ?
             WHERE id = ?
            """,
            (jira_ticket, leak_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_audit_runs() -> list[dict]:
    """Return all audit_run rows ordered by run_timestamp DESC."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM audit_runs ORDER BY run_timestamp DESC"
        )
        return [dict(row) for row in cur.fetchall()]


def get_kpi_summary() -> dict:
    """
    Return aggregate KPIs across all leak_records.

    Keys
    ----
    total_open_usd      : sum of leaked_usd where resolution='OPEN'
    total_recovered_usd : sum of leaked_usd where resolution='RECOVERED'
    open_count          : number of OPEN leaks
    recovered_count     : number of RECOVERED leaks
    total_count         : total number of leak records
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN resolution = 'OPEN'      THEN leaked_usd ELSE 0 END), 0) AS total_open_usd,
                COALESCE(SUM(CASE WHEN resolution = 'RECOVERED' THEN leaked_usd ELSE 0 END), 0) AS total_recovered_usd,
                COUNT(CASE WHEN resolution = 'OPEN'      THEN 1 END)                            AS open_count,
                COUNT(CASE WHEN resolution = 'RECOVERED' THEN 1 END)                            AS recovered_count,
                COUNT(*)                                                                         AS total_count
            FROM leak_records
            """
        )
        row = cur.fetchone()
        return {
            "total_open_usd":      row[0],
            "total_recovered_usd": row[1],
            "open_count":          row[2],
            "recovered_count":     row[3],
            "total_count":         row[4],
        }
