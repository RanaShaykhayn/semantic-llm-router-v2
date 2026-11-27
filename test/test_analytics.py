import sqlite3
from data.analytics import get_route_metrics, sanitize_query_for_logging

def test_sanitize_query_for_logging():
    raw_text = "طلب حساس جداً"
    assert sanitize_query_for_logging(raw_text, store_raw_text=True) == raw_text
    assert "REDACTED" in sanitize_query_for_logging(raw_text, store_raw_text=False)


def test_get_route_metrics_with_dummy_db(tmp_path):
    db_file = str(tmp_path / "test_router.db")
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE query_logs (
            predicted_route TEXT,
            latency_ms REAL,
            real_cost REAL,
            was_cache_hit BOOLEAN
        )
    """)
    conn.executemany(
        "INSERT INTO query_logs VALUES (?, ?, ?, ?)",
        [
            ("gemini-2.5-flash", 100.0, 0.0001, 0),
            ("gemini-2.5-flash", 200.0, 0.0001, 1),
            ("RAG (Flash)", 500.0, 0.0005, 0),
        ],
    )
    conn.commit()
    conn.close()

    metrics = get_route_metrics(db_file)
    assert len(metrics) == 2
    assert metrics[0]["route"] == "gemini-2.5-flash"
    assert metrics[0]["total_queries"] == 2
    assert metrics[0]["cache_hit_rate_pct"] == 50.0