from __future__ import annotations

import json
from typing import Any

import pytest

from ppl_synth_data.pipeline.query_execution import run_queries

pytestmark = pytest.mark.narrow_unit_not_acceptance


class _StubEnv:
    def __init__(self, results: list[tuple[Any, bool]]) -> None:
        self._results = iter(results)

    def run_experiment(self, input_str: str) -> tuple[Any, bool]:
        return next(self._results)


def test_empty_queries():
    env = _StubEnv([])
    assert run_queries(env, []) == []


def test_single_query():
    env = _StubEnv([("ok", True)])
    rows = run_queries(env, [{"a": 1}])
    assert len(rows) == 1
    assert rows[0]["query"] == {"a": 1}
    assert rows[0]["query_text"] == json.dumps({"a": 1})
    assert rows[0]["result"] == "ok"
    assert rows[0]["success"] is True


def test_failure_coerced_to_bool():
    env = _StubEnv([("err", 0)])
    rows = run_queries(env, [{"x": 1}])
    assert rows[0]["success"] is False


def test_multiple_queries():
    env = _StubEnv([("r1", True), ("r2", False)])
    rows = run_queries(env, [{"q": 1}, {"q": 2}])
    assert len(rows) == 2
    assert rows[0]["success"] is True
    assert rows[1]["success"] is False
