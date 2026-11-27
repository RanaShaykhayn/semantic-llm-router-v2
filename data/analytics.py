import sqlite3
from typing import Any


def get_route_metrics(db_path: str) -> list[dict[str, Any]]:
    query = """
    WITH ordered AS (
        SELECT
            predicted_route AS route,
            latency_ms,
            real_cost,
            was_cache_hit,
            ROW_NUMBER() OVER (
                PARTITION BY predicted_route
                ORDER BY latency_ms
            ) AS rn,
            COUNT(*) OVER (PARTITION BY predicted_route) AS cnt
        FROM query_logs
        WHERE latency_ms IS NOT NULL
    )
    SELECT
        route,
        COUNT(*) AS total_queries,
        ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
        ROUND(
            MIN(
                CASE
                    WHEN rn * 100 >= 95 * cnt THEN latency_ms
                END
            ),
            2
        ) AS p95_latency_ms,
        ROUND(COALESCE(SUM(real_cost), 0.0), 6) AS total_cost_usd,
        ROUND(
            AVG(
                CASE
                    WHEN was_cache_hit THEN 1.0
                    ELSE 0.0
                END
            ) * 100,
            1
        ) AS cache_hit_rate_pct
    FROM ordered
    GROUP BY route
    ORDER BY total_queries DESC;
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def sanitize_query_for_logging(query: str, store_raw_text: bool = True) -> str:
    if store_raw_text:
        return query
    return f"[REDACTED len={len(query)}]"