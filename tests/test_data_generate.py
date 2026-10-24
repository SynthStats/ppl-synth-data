from __future__ import annotations

import json
from typing import Any

import pytest

from ppl_synth_data.pipeline.query_execution import run_queries

pytestmark = pytest.mark.narrow_unit_not_acceptance


class _FakeEnv:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def run_experiment(self, input_str: str) -> tuple[Any, bool]:
        self.inputs.append(input_str)
        payload = json.loads(input_str)
        return payload["x"] * 2, True


def test_run_queries_serializes_dict_queries_to_text() -> None:
    env = _FakeEnv()
    queries = [{"x": 1.5}, {"x": -2.0}]

    rows = run_queries(env, queries)

    assert len(env.inputs) == 2
    assert all(isinstance(item, str) for item in env.inputs)
    assert [json.loads(item) for item in env.inputs] == queries
    assert rows == [
        {
            "query": {"x": 1.5},
            "query_text": env.inputs[0],
            "result": 3.0,
            "success": True,
        },
        {
            "query": {"x": -2.0},
            "query_text": env.inputs[1],
            "result": -4.0,
            "success": True,
        },
    ]
