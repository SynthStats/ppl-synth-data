import pytest

from ppl_synth_data.harbor.declared_gold import render_declared_gold
from ppl_synth_data.specs.models import EnvSpec


def spec(family="regression_continuous", **params):
    return EnvSpec.model_validate(
        {
            "env_id": "temperature_yield",
            "env_name": "temperature_yield",
            "domain": "engineering",
            "complexity": "simple",
            "model_type": "inferential",
            "key_concepts": ["regression"],
            "io_schema": {
                "input_type": "float",
                "input_description": "temperature",
                "output_type": "float",
                "target_name": "yield_value",
                "feature_names": ["temperature"],
            },
            "data_plan": {"n_observations": 100, "covariates": ["temperature"]},
            "generative_family": family,
            "generative_params": {
                "intercept": 1.5,
                "slope": 0.3,
                "lower_bound": 0.0,
                "upper_bound": 10.0,
                **params,
            },
        }
    )


@pytest.mark.parametrize(
    "family",
    [
        "regression_continuous",
        "regression_count",
        "zero_inflated_count",
        "binary_choice",
    ],
)
@pytest.mark.parametrize("visible", [None, {"temperature": [0.0, 1.0], "yield_value": [1.0, 2.0]}])
def test_template_spec_is_refused(family, visible):
    with pytest.raises(TypeError, match="FourFamilySpec"):
        render_declared_gold(spec(family, noise_sigma=0.6), visible)


def test_silent_drop_fields_are_rejected():
    with pytest.raises(TypeError, match="FourFamilySpec"):
        render_declared_gold(spec("regression_count", dispersion=14))


def test_code_shaped_numeric_field_is_rejected():
    with pytest.raises(TypeError, match="FourFamilySpec"):
        render_declared_gold(spec(slope="__import__('os').system('false')"))
