"""
Local SQLite cache for miner-side data.

Miners use this to cache scraped filings and track which URLs
have already been processed, avoiding redundant work.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


class MinerCache:
    """Simple SQLite cache for a miner's scraped data."""

    def __init__(self, db_path: str = "miner_cache.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                filing_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
            CREATE INDEX IF NOT EXISTS idx_filings_url ON filings(url);

            CREATE TABLE IF NOT EXISTS processed_urls (
                url TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        conn.commit()

    def store_filing(
        self,
        ticker: str,
        filing_type: str,
        content: str,
        url: str,
        filing_date: datetime,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO filings (ticker, filing_type, content, url, filing_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, filing_type, content, url, filing_date.isoformat()),
        )
        conn.commit()

    def get_recent_filings(self, ticker: str, limit: int = 10) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT ticker, filing_type, content, url, filing_date
            FROM filings
            WHERE ticker = ?
            ORDER BY filing_date DESC
            LIMIT ?
            """,
            (ticker, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def has_been_processed(self, url: str) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM processed_urls WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    def mark_processed(self, url: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO processed_urls (url) VALUES (?)", (url,)
        )
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def clear(self) -> None:
        """Clear all cached data (useful for tests)."""
        conn = self._get_conn()
        conn.executescript(
            """
            DELETE FROM filings;
            DELETE FROM processed_urls;
            """
        )
        conn.commit()
