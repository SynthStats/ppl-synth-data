"""Contract behavior for numeric and string fixed-count designs."""

import ast
import json

import numpy as np
import pytest
from pydantic import ValidationError

from ppl_synth_data.envs.base import InputValidationError
from ppl_synth_data.envs.declared_response import DeclaredResponseEnv
from ppl_synth_data.harbor.declared_gold import render_declared_gold
from ppl_synth_data.specs.models import FourFamilySpec


def _declaration(levels, coefficients):
    return {
        "env_id": "fixed-counts-coding-test",
        "env_name": "fixed_counts_coding_test",
        "domain": "synthetic",
        "scenario": {"intent": "Check declared fixed-count coding", "source_row_sha256": "a" * 64},
        "generative_family": "regression_continuous",
        "features": [
            {
                "name": "dose",
                "law": {
                    "dist": "fixed_counts",
                    "levels": levels,
                    "counts": [10] * len(levels),
                },
            }
        ],
        "response": {
            "family": "regression_continuous",
            "intercept": 1.25,
            "coefficients": coefficients,
            "noise_sigma": 0.75,
        },
        "data_plan": {"n_observations": 10 * len(levels)},
    }


@pytest.mark.parametrize("levels", [[0, 50, 100], [0.5, 2.5, 5.5], [0, 2.5, 5]])
def test_numeric_levels_use_one_raw_scale_coefficient(levels):
    declaration = _declaration(levels, {"dose": 0.4})
    spec = FourFamilySpec.model_validate(declaration)
    assert spec.response.coefficients == {"dose": 0.4}
    env = DeclaredResponseEnv(spec.model_dump(mode="json"), seed=17)
    env.reset(seed=17)
    first = env.step(env.validate_input(json.dumps({"dose": levels[0]})))
    env.reset(seed=17)
    last = env.step(env.validate_input(json.dumps({"dose": levels[-1]})))
    # Identical Gaussian innovations cancel, leaving the declared raw dose contrast.
    assert last - first == pytest.approx(0.4 * (levels[-1] - levels[0]), abs=1e-12)


def test_string_levels_keep_reference_coding():
    spec = FourFamilySpec.model_validate(
        _declaration(["control", "low", "high"], {"dose=low": 0.4, "dose=high": 1.7})
    )
    env = DeclaredResponseEnv(spec.model_dump(mode="json"), seed=19)
    values = []
    for level in ["control", "low", "high"]:
        env.reset(seed=19)
        values.append(env.step(env.validate_input(json.dumps({"dose": level}))))
    assert values[1] - values[0] == pytest.approx(0.4, abs=1e-12)
    assert values[2] - values[0] == pytest.approx(1.7, abs=1e-12)
    visible = {
        "n": 6,
        "dose": np.array(["control", "low", "high"] * 2),
        "y": np.array([1.0, 1.6, 2.9, 1.2, 1.7, 3.1]),
    }
    code, metadata = render_declared_gold(spec, visible)
    assert metadata["coefficient_keys"] == ["dose=low", "dose=high"]
    designs = _design_expressions(code)
    assert len(designs) == 2
    assert all(isinstance(expression, ast.Compare) for expression in designs)
    assert [expression.comparators[0].value for expression in designs] == ["low", "high"]


@pytest.mark.parametrize("levels", [[0, "high"], [0.5, "high"], [1, "1"]])
def test_mixed_numeric_string_levels_refuse_at_levels_pointer(levels):
    with pytest.raises(ValidationError, match=r"/features/0/law/levels"):
        FourFamilySpec.model_validate(_declaration(levels, {"dose=high": 0.4}))


@pytest.mark.parametrize("levels,query", [([0, 50], 50.0), ([0.0, 50.0], 50), ([0.5, 2.5], 2.5)])
def test_numeric_queries_accept_equal_numeric_values(levels, query):
    env = DeclaredResponseEnv(_declaration(levels, {"dose": 0.4}), seed=23)
    assert env.validate_input(json.dumps({"dose": query})) == (query,)


@pytest.mark.parametrize("query", [True, False, "50", 51, 0.5, None])
def test_numeric_queries_refuse_bool_string_and_undeclared_values(query):
    env = DeclaredResponseEnv(_declaration([0, 50], {"dose": 0.4}), seed=23)
    with pytest.raises(InputValidationError, match=r"/features/0/law"):
        env.validate_input(json.dumps({"dose": query}))


def _design_expressions(code):
    tree = ast.parse(code)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    return [
        node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("x_") for target in node.targets
        )
    ]


@pytest.mark.parametrize("levels", [[0, 50, 100], [0.5, 2.5, 5.5], [0, 2.5, 5]])
def test_numeric_gold_has_one_untransformed_data_column(levels):
    spec = FourFamilySpec.model_validate(_declaration(levels, {"dose": 0.4}))
    visible = {"n": 6, "dose": np.array(levels * 2), "y": np.array([1.0, 2.0, 3.5, 1.2, 2.2, 3.3])}
    code, metadata = render_declared_gold(spec, visible)
    assert metadata["coefficient_keys"] == ["dose"]
    designs = _design_expressions(code)
    assert len(designs) == 1
    expected = ast.parse("np.asarray(data['dose'])", mode="eval").body
    assert ast.dump(designs[0]) == ast.dump(expected)


@pytest.mark.parametrize(
    "levels,coefficients,phrase",
    [
        ([0, 50], {"dose": 0.4}, "raw numeric value"),
        (["control", "treated"], {"dose=treated": 0.4}, "reference level"),
    ],
)
def test_prior_prompt_explains_fixed_count_coding(levels, coefficients, phrase):
    env = DeclaredResponseEnv(_declaration(levels, coefficients), seed=29)
    message = env.generate_system_message(include_prior=True)
    assert phrase in message
    if isinstance(levels[0], str):
        assert "control" in message
