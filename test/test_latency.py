import sqlite3
from pathlib import Path

import src.database as database


def test_old_latency_is_ignored(tmp_path, monkeypatch):
    db_file = tmp_path / "latency.db"
    schema = Path("data/schema.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(db_file)
    conn.executescript(schema)
    conn.executemany(
        """
        INSERT INTO query_logs
            (raw_query, cleaned_query, predicted_route, latency_ms, created_at)
        VALUES ('q', 'q', 'pro', 9000, datetime('now', '-1 hour'))
        """,
        [()] * 5,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(database, "db_path", db_file)

    assert database.get_estimated_latency("pro") == (None, 0)