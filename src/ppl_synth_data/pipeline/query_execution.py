from __future__ import annotations

import json
from typing import Any


def run_queries(env: Any, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_text = json.dumps(query, ensure_ascii=False)
        result, success = env.run_experiment(query_text)
        rows.append(
            {
                "query": query,
                "query_text": query_text,
                "result": result,
                "success": bool(success),
            }
        )
    return rows
