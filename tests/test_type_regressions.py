import numpy as np
import pytest

from ppl_synth_data.envs.count_kernel import sample_count
from ppl_synth_data.gates.likelihood_screens import flags
from ppl_synth_data.specs.models import CountDeclaration, ZeroInflatedResponse


@pytest.mark.parametrize(
    "model,family",
    [(CountDeclaration, "regression_count"), (ZeroInflatedResponse, "zero_inflated_count")],
)
@pytest.mark.parametrize("distribution", ["poisson", "nb2"])
def test_validated_count_response_preserves_fields_and_sampling(model, family, distribution):
    payload = {"coefficients": {"x": 0.5}, "family": family, "intercept_log_rate": 0.0,
               "count_distribution": distribution}
    if distribution == "nb2":
        payload["nb2_alpha"] = 2.0
    if family == "zero_inflated_count":
        payload["zero_inflation"] = 0.2
    response = model.model_validate(payload)
    assert response.model_dump(mode="json") == payload
    assert list(response.model_dump()) == list(payload)
    if distribution == "nb2":
        assert response.nb2_alpha is not None
    draws = sample_count(response, {"x": 1.0}, np.random.default_rng(1), size=10)
    assert draws.shape == (10,)
    assert np.all(draws >= 0)
    assert np.all(draws == np.floor(draws))


@pytest.mark.parametrize("missing", ["screen_floor_entry_share", "screen_max_log_lik"])
def test_invalid_missing_screen_numbers_keep_their_type_error(missing):
    metrics = {"screen_lpd_sd": None, "screen_floor_entry_share": 0.0, "screen_max_log_lik": 0.0}
    metrics[missing] = None
    with pytest.raises(TypeError):
        flags(metrics, outcome=[1])


def test_continuous_screen_does_not_read_an_unused_maximum():
    assert flags({"screen_lpd_sd": None, "screen_floor_entry_share": 0.0}, outcome=[1.5]) == []
