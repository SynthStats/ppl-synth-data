from __future__ import annotations

import json
from typing import Any

import pytest

from ppl_synth_data.envs.base import BaseEnv, InputValidationError


class _TestEnv(BaseEnv):
    env_name = "test"
    response_law = {"family": "hand_built"}
    observed_data: list[tuple[Any, ...]]
    include_prior = False
    _feature_names = ("x",)
    _feature_domains = {"x": (0.0, 10.0)}
    _multi_feature_input = False

    def __init__(self):
        self.observed_data = []

    def reset(self, seed=None):
        self.observed_data = []

    def step(self, x_values):
        return x_values[0] * 2

    def get_ordered_column_names(self):
        return ["x", "y"]


class _MultiFeatureEnv(_TestEnv):
    _feature_names = ("x", "y")
    _feature_domains = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
    _multi_feature_input = True

    def step(self, x_values):
        return x_values[0] + x_values[1]

    def get_ordered_column_names(self):
        return ["x", "y", "z"]


def test_validate_input_valid():
    env = _TestEnv()
    assert env.validate_input("5.0") == (5.0,)


def test_validate_input_out_of_bounds():
    env = _TestEnv()
    with pytest.raises(InputValidationError, match="hand_built.*x.*between"):
        env.validate_input("15.0")


def test_validate_input_non_numeric():
    env = _TestEnv()
    with pytest.raises(InputValidationError, match="hand_built.*single float.*x"):
        env.validate_input("abc")


def test_validate_input_infinity():
    env = _TestEnv()
    with pytest.raises(InputValidationError, match="hand_built.*x.*finite"):
        env.validate_input("inf")


def test_run_experiment_success():
    env = _TestEnv()
    result, ok = env.run_experiment("3.0")
    assert ok is True
    assert result == 6.0
    assert len(env.observed_data) == 1
    assert env.observed_data[0] == (3.0, 6.0)


def test_run_experiment_invalid_input():
    env = _TestEnv()
    result, ok = env.run_experiment("abc")
    assert ok is False
    assert isinstance(result, str)
    assert "hand_built" in result and "x" in result
    assert len(env.observed_data) == 0


def test_get_data_returns_copy():
    env = _TestEnv()
    env.run_experiment("1.0")
    data = env.get_data()
    data.clear()
    assert len(env.observed_data) == 1


def test_get_df():
    env = _TestEnv()
    env.run_experiment("2.0")
    df = env.get_df()
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 1
    assert df.iloc[0]["x"] == 2.0
    assert df.iloc[0]["y"] == 4.0


def test_get_ordered_features():
    env = _TestEnv()
    assert env.get_ordered_features() == ["x"]


def test_multi_feature_valid():
    env = _MultiFeatureEnv()
    result = env.validate_input(json.dumps({"x": 1.0, "y": 2.0}))
    assert result == (1.0, 2.0)


def test_multi_feature_missing_key():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*y"):
        env.validate_input(json.dumps({"x": 1.0}))


def test_multi_feature_run_experiment():
    env = _MultiFeatureEnv()
    result, ok = env.run_experiment(json.dumps({"x": 3.0, "y": 4.0}))
    assert ok is True
    assert result == 7.0
    assert env.observed_data[0] == (3.0, 4.0, 7.0)


def test_multi_feature_invalid_json():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*input.*valid JSON string"):
        env.validate_input("not json")


def test_multi_feature_boolean_rejected():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*x.*numeric"):
        env.validate_input(json.dumps({"x": True, "y": 2.0}))


def test_reset_clears_data():
    env = _TestEnv()
    env.run_experiment("1.0")
    env.reset()
    assert len(env.observed_data) == 0


def test_multi_feature_non_numeric_string():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*y.*numeric"):
        env.validate_input(json.dumps({"x": 1.0, "y": "abc"}))


def test_multi_feature_infinity():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*y.*finite"):
        env.validate_input(json.dumps({"x": 1.0, "y": float("inf")}))


def test_multi_feature_out_of_bounds():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*y.*between"):
        env.validate_input(json.dumps({"x": 1.0, "y": 999.0}))


def test_multi_feature_negative_out_of_bounds():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*x.*between"):
        env.validate_input(json.dumps({"x": -1.0, "y": 2.0}))


def test_validate_feature_payload_non_dict():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*input.*JSON object"):
        env._validate_feature_payload([1, 2])


def test_validate_feature_payload_none():
    env = _MultiFeatureEnv()
    with pytest.raises(InputValidationError, match="hand_built.*input.*JSON object"):
        env._validate_feature_payload(None)


def test_single_feature_boundary_values():
    env = _TestEnv()
    assert env.validate_input("0.0") == (0.0,)
    assert env.validate_input("10.0") == (10.0,)


def test_multi_feature_boundary_values():
    env = _MultiFeatureEnv()
    result = env.validate_input(json.dumps({"x": 0.0, "y": 10.0}))
    assert result == (0.0, 10.0)


@pytest.mark.parametrize("domain", [(5.0, 5.0), (10.0, 0.0)])
def test_hand_built_environment_refuses_invalid_domain_at_standardization(domain):
    class DomainPositionEnv(_TestEnv):
        _feature_domains = {"x": domain}

        def step(self, x_values):
            return self._standardize_features(x_values)[0]

    env = DomainPositionEnv()
    with pytest.raises(ValueError, match="hand_built.*x"):
        env.step((5.0,))
