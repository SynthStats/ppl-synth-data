"""Declared-gold priors use rstanarm's visible-data autoscaling.

For a Gaussian identity model, the centred intercept prior is Normal(mean(y), 2.5 * sd(y));
otherwise its mean is zero and the response scale is one. Each coefficient uses Normal(0, 2.5 *
sy / sx), and Gaussian sigma uses Exponential(1 / sy). Standard deviations use ddof=1, matching
R's sd(). The renderer writes these constants from the visible rows, so held-out builds keep the
same prior. The raw-scale intercept is a Deterministic transformation of the free centred
intercept.

Source: https://mc-stan.org/rstanarm/articles/priors.html.
"""

import ast
import json
import os
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
from scipy import stats

from ppl_synth_data.harbor.declared_gold import render_declared_gold
from ppl_synth_data.pipeline.validation.data_loading import load_sample_data
from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec

REAL_SPECS = Path(
    os.environ.get(
        "GOLD_REAL_SPECS", Path(__file__).parent / "fixtures/declared_specs/selected-120.jsonl"
    )
)
REAL_DATA = Path(
    os.environ.get(
        "GOLD_REAL_DATA",
        str(Path(__file__).resolve().parents[1] / "artifacts/reference/stage_C_data/data"),
    )
)

VISIBLE = {
    "n": 6,
    "load": np.array([450.0, 480.0, 500.0, 520.0, 530.0, 560.0]),
    "site": np.array(["a", "b", "a", "b", "b", "a"]),
    "y": np.array([7200.0, 8100.0, 7800.0, 8900.0, 8700.0, 8600.0]),
}
SITE_B = (VISIBLE["site"] == "b").astype(float)


def spec_for(family, *, distribution=None, levels=("a", "b"), **response_values):
    response = {"family": family, "coefficients": {"load": 12.0, "site=b": 600.0}}
    if family == "regression_continuous":
        response.update(intercept=1800.0, noise_sigma=300.0)
    elif family == "binary_choice":
        response.update(intercept=0.4, link="logit")
    else:
        response.update(intercept_log_rate=0.4, count_distribution=distribution)
        if distribution == "nb2":
            response["nb2_alpha"] = 2.0
        if family == "zero_inflated_count":
            response["zero_inflation"] = 0.25
    response.update(response_values)
    return FourFamilySpec.model_validate(
        {
            "env_id": "autoscale-check",
            "env_name": "autoscale_check",
            "domain": "engineering",
            "scenario": {"intent": "Response to measured inputs", "source_row_sha256": "1" * 64},
            "generative_family": family,
            "features": [
                {"name": "load", "law": {"dist": "normal", "mean": 500.0, "sd": 50.0}},
                {
                    "name": "site",
                    "law": {
                        "dist": "categorical",
                        "levels": list(levels),
                        "probs": [1.0 / len(levels)] * len(levels),
                    },
                },
            ],
            "response": response,
            "data_plan": {"n_observations": 30},
        }
    )


def visible_for(family):
    if family == "regression_continuous":
        return dict(VISIBLE)
    if family == "binary_choice":
        return {**VISIBLE, "y": np.array([0, 1, 0, 1, 1, 0])}
    return {**VISIBLE, "y": np.array([0, 3, 1, 7, 0, 2])}


def prior_calls(code):
    """Map each named PyMC call to (distribution, keyword values); literals are evaluated."""
    calls = {}
    for node in ast.walk(ast.parse(code)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pm"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        keywords = {}
        for keyword in node.keywords:
            try:
                keywords[keyword.arg] = ast.literal_eval(keyword.value)
            except ValueError:
                keywords[keyword.arg] = ast.unparse(keyword.value)
        calls[node.args[0].value] = (node.func.attr, keywords)
    return calls


def build(code, data):
    namespace = {"np": np, "pm": pm}
    exec(compile(code, "gold", "exec"), namespace)
    return namespace["model"](data)


def test_four_family_gold_refuses_to_render_without_visible_data():
    with pytest.raises(ValueError, match="visible data"):
        render_declared_gold(spec_for("regression_continuous"))


def test_template_gold_refuses_visible_data():
    from tests.test_declared_gold import spec as legacy_spec

    legacy = legacy_spec(noise_sigma=0.6)
    assert isinstance(legacy, EnvSpec)
    with pytest.raises(TypeError, match="FourFamilySpec"):
        render_declared_gold(legacy, dict(VISIBLE))


def test_continuous_priors_follow_rstanarm_autoscaling():
    code, audit = render_declared_gold(spec_for("regression_continuous"), dict(VISIBLE))
    calls = prior_calls(code)
    sy = np.std(VISIBLE["y"], ddof=1)
    sx = [np.std(VISIBLE["load"], ddof=1), np.std(SITE_B, ddof=1)]
    assert calls["intercept_centered"][0] == "Normal"
    assert calls["intercept_centered"][1] == pytest.approx(
        {"mu": float(np.mean(VISIBLE["y"])), "sigma": 2.5 * sy}, rel=1e-12
    )
    for index, scale in enumerate(sx):
        assert calls[f"coef_{index}"] == (
            "Normal",
            pytest.approx({"mu": 0.0, "sigma": 2.5 * sy / scale}, rel=1e-12),
        )
    assert calls["sigma"] == ("Exponential", pytest.approx({"lam": 1.0 / sy}, rel=1e-12))
    assert calls["intercept"][0] == "Deterministic"
    autoscale = audit["prior_autoscale"]
    assert autoscale["centers"] == pytest.approx(
        [float(np.mean(VISIBLE["load"])), float(np.mean(SITE_B))], rel=1e-12
    )
    assert autoscale["x_scales"] == pytest.approx(sx, rel=1e-12)
    assert autoscale["y_location"] == pytest.approx(float(np.mean(VISIBLE["y"])), rel=1e-12)
    assert autoscale["y_scale"] == pytest.approx(sy, rel=1e-12)
    assert autoscale["constant_in_visible"] == []
    assert autoscale["visible_rows"] == 6
    assert "mc-stan.org/rstanarm/articles/priors.html" in autoscale["rule"]
    assert "autoscal" in audit["parameter_treatment"]


@pytest.mark.parametrize(
    "family,distribution",
    [
        ("binary_choice", None),
        ("regression_count", "poisson"),
        ("regression_count", "nb2"),
        ("zero_inflated_count", "poisson"),
        ("zero_inflated_count", "nb2"),
    ],
)
def test_non_gaussian_priors_use_unit_response_scale(family, distribution):
    code, audit = render_declared_gold(
        spec_for(family, distribution=distribution), visible_for(family)
    )
    calls = prior_calls(code)
    sx = [np.std(VISIBLE["load"], ddof=1), np.std(SITE_B, ddof=1)]
    expected = {"mu": 0.0, "sigma": 2.5}
    if family != "binary_choice":
        # A count intercept is centered at log(mean visible count), as rstanarm centers a Gaussian
        # intercept at mean(y). The model's starting point (the prior mean) is then mu = mean count;
        # at mu = 1, PyMC 5.28's zero-inflated NB density underflows to -inf for counts in the
        # hundreds (4f-v01-fwd-425b, 436b). An initval would also give that start, but the validator
        # cannot score a model that has one in the scoring runtime.
        expected["mu"] = float(np.log(np.mean(visible_for(family)["y"])))
    assert calls["intercept_centered"] == ("Normal", pytest.approx(expected, rel=1e-12))
    assert "initval" not in code
    for index, scale in enumerate(sx):
        assert calls[f"coef_{index}"] == (
            "Normal",
            pytest.approx({"mu": 0.0, "sigma": 2.5 / scale}, rel=1e-12),
        )
    assert "sigma" not in calls
    assert calls["intercept"][0] == "Deterministic"
    if distribution == "nb2":
        assert calls["alpha"] == ("HalfNormal", {"sigma": 10.0})
    if family == "zero_inflated_count":
        assert calls["zero_inflation"] == ("Beta", {"alpha": 1.0, "beta": 1.0})
    assert audit["prior_autoscale"]["y_location"] == pytest.approx(expected["mu"], rel=1e-12)
    assert audit["prior_autoscale"]["y_scale"] == 1.0


@pytest.mark.parametrize(
    "family,distribution",
    [
        ("regression_continuous", None),
        ("binary_choice", None),
        ("regression_count", "nb2"),
        ("zero_inflated_count", "poisson"),
    ],
)
def test_priors_never_read_the_declared_parameter_values(family, distribution):
    first, _ = render_declared_gold(
        spec_for(family, distribution=distribution), visible_for(family)
    )
    changed = {"coefficients": {"load": -3.0, "site=b": 0.25}}
    if family == "regression_continuous":
        changed.update(intercept=-50.0, noise_sigma=9.0)
    elif family == "binary_choice":
        changed.update(intercept=-2.0)
    else:
        changed.update(intercept_log_rate=3.0)
        if distribution == "nb2":
            changed["nb2_alpha"] = 7.0
        if family == "zero_inflated_count":
            changed["zero_inflation"] = 0.6
    second, _ = render_declared_gold(
        spec_for(family, distribution=distribution, **changed), visible_for(family)
    )
    assert first == second


def test_prior_constants_are_literals_so_other_rows_keep_the_same_prior():
    code, _ = render_declared_gold(spec_for("regression_continuous"), dict(VISIBLE))
    for name, (_, keywords) in prior_calls(code).items():
        if name in {"intercept_centered", "coef_0", "coef_1", "sigma"}:
            assert all(isinstance(value, float) for value in keywords.values()), (name, keywords)
    other = {
        "n": 2,
        "load": np.array([100.0, 900.0]),
        "site": np.array(["b", "b"]),
        "y": np.array([1.0, 2.0]),
    }
    one, two = build(code, dict(VISIBLE)), build(code, other)
    point = one.initial_point()
    assert point.keys() == two.initial_point().keys()
    prior_names = ["intercept_centered", "coef_0", "coef_1", "sigma"]
    logp_one = one.compile_logp(vars=[one[name] for name in prior_names])(point)
    logp_two = two.compile_logp(vars=[two[name] for name in prior_names])(point)
    assert logp_one == pytest.approx(logp_two, rel=1e-12)


@pytest.mark.parametrize(
    "family,distribution",
    [
        ("regression_continuous", None),
        ("binary_choice", None),
        ("regression_count", "nb2"),
        ("zero_inflated_count", "nb2"),
    ],
)
def test_declared_raw_scale_point_reproduces_the_declared_law(family, distribution):
    spec = spec_for(family, distribution=distribution)
    data = visible_for(family)
    code, audit = render_declared_gold(spec, data)
    model = build(code, data)
    centers = audit["prior_autoscale"]["centers"]
    intercept, slopes = -0.2, (0.004, 0.3)
    point = model.initial_point()
    point.update(
        intercept_centered=np.array(intercept + slopes[0] * centers[0] + slopes[1] * centers[1]),
        coef_0=np.array(slopes[0]),
        coef_1=np.array(slopes[1]),
    )
    eta = intercept + slopes[0] * data["load"] + slopes[1] * SITE_B
    y = data["y"]
    if family == "regression_continuous":
        point["sigma_log__"] = np.array(np.log(0.6))
        expected = stats.norm.logpdf(y, eta, 0.6)
    elif family == "binary_choice":
        expected = stats.bernoulli.logpmf(y, 1.0 / (1.0 + np.exp(-eta)))
    else:
        point["alpha_log__"] = np.array(np.log(2.0))
        base = stats.nbinom.pmf(y, 2.0, 2.0 / (2.0 + np.exp(eta)))
        if family == "zero_inflated_count":
            point["zero_inflation_logodds__"] = np.array(np.log(0.25 / 0.75))
            expected = np.log(0.75 * base + 0.25 * (y == 0))
        else:
            expected = np.log(base)
    actual = model.compile_logp(vars=model.observed_RVs, sum=False)(point)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    raw_intercept = model.compile_fn(
        model.replace_rvs_by_values([model["intercept"]]),
        inputs=model.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )(point)[0]
    assert float(raw_intercept) == pytest.approx(intercept, rel=1e-9, abs=1e-9)


def test_visible_constant_column_uses_unit_x_scale_and_is_recorded():
    spec = spec_for(
        "regression_continuous",
        levels=("a", "b", "c"),
        coefficients={"load": 12.0, "site=b": 600.0, "site=c": -40.0},
    )
    code, audit = render_declared_gold(spec, dict(VISIBLE))
    calls = prior_calls(code)
    sy = np.std(VISIBLE["y"], ddof=1)
    # Level "c" never occurs in the visible rows, so its indicator column is constant.
    assert calls["coef_2"] == ("Normal", pytest.approx({"mu": 0.0, "sigma": 2.5 * sy}, rel=1e-12))
    assert audit["prior_autoscale"]["constant_in_visible"] == ["site=c"]
    assert audit["prior_autoscale"]["x_scales"][2] == 1.0


@pytest.mark.parametrize("family", ["regression_count", "zero_inflated_count"])
def test_all_zero_visible_counts_refuse(family):
    data = {**visible_for(family), "y": np.zeros(6, dtype=int)}
    with pytest.raises(ValueError, match="zero"):
        render_declared_gold(spec_for(family, distribution="nb2"), data)


def test_constant_visible_gaussian_response_refuses():
    data = {**VISIBLE, "y": np.full(6, 5.0)}
    with pytest.raises(ValueError, match="constant"):
        render_declared_gold(spec_for("regression_continuous"), data)


@pytest.mark.parametrize("column", ["load", "y"])
def test_nonfinite_visible_values_refuse(column):
    data = dict(VISIBLE)
    data[column] = data[column].astype(float).copy()
    data[column][2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        render_declared_gold(spec_for("regression_continuous"), data)


def test_missing_visible_column_refuses():
    data = {key: value for key, value in VISIBLE.items() if key != "site"}
    with pytest.raises(ValueError, match="visible data missing"):
        render_declared_gold(spec_for("regression_continuous"), data)


def _real_specs():
    wanted = {"000", "088", "152", "005", "009", "006", "018"}
    specs, binary = [], None
    for line in REAL_SPECS.read_text().splitlines():
        spec = json.loads(line)
        if spec["env_id"][-3:] in wanted:
            specs.append(spec)
        elif binary is None and spec["response"]["family"] == "binary_choice":
            binary = spec
    assert len(specs) == len(wanted) and binary is not None
    return [*specs, binary]


@pytest.mark.parametrize("payload", _real_specs(), ids=lambda spec: spec["env_id"])
@pytest.mark.heavy_lane(reason="Requires visible Stage C parquets via GOLD_REAL_DATA")
def test_real_gold_prior_predictive_is_finite(payload):
    spec = FourFamilySpec.model_validate(payload)
    data = load_sample_data(REAL_DATA / f"{spec.env_id}.parquet")
    code, _ = render_declared_gold(spec, data)
    model = build(code, data)
    with model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=11)
    draws = np.asarray(prior.prior_predictive["y"])
    assert np.isfinite(draws).all()


REAL_B_SPECS = Path(
    os.environ.get(
        "GOLD_REAL_B_SPECS",
        str(Path(__file__).resolve().parents[1] / "artifacts/reference/envspecs.jsonl"),
    )
)
REAL_B_DATA = Path(
    os.environ.get(
        "GOLD_REAL_B_DATA",
        str(Path(__file__).resolve().parents[1] / "artifacts/reference/stage_C_data/data"),
    )
)


@pytest.mark.parametrize("env_id", ["4f-v01-fwd-425b", "4f-v01-fwd-436b"])
@pytest.mark.heavy_lane(
    reason=(
        "Requires specs and parquets via GO"
        "LD_REAL_B_SPECS and GOLD_REAL_B_DATA"
    )
)
def test_real_zero_inflated_nb_gold_has_finite_initial_log_density(env_id):
    payload = next(
        json.loads(line)
        for line in REAL_B_SPECS.read_text().splitlines()
        if json.loads(line)["env_id"] == env_id
    )
    spec = FourFamilySpec.model_validate(payload)
    data = load_sample_data(REAL_B_DATA / f"{env_id}.parquet")
    code, _ = render_declared_gold(spec, data)
    model = build(code, data)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize(
    "family,distribution",
    [
        ("regression_continuous", None),
        ("binary_choice", None),
        ("regression_count", "poisson"),
        ("regression_count", "nb2"),
        ("zero_inflated_count", "poisson"),
        ("zero_inflated_count", "nb2"),
    ],
)
def test_gold_model_converts_the_way_the_validator_needs(family, distribution):
    # The validator samples with idata_kwargs={"log_likelihood": True} and
    # scores held-out
    # rows with pm.compute_log_likelihood; PyMC refuses both on a model with a non-default initval
    # ("NotImplementedError: Cannot convert models with non-default initial_values").
    from pymc.model.fgraph import clone_model, fgraph_from_model

    data = visible_for(family)
    code, _ = render_declared_gold(spec_for(family, distribution=distribution), data)
    model = build(code, data)
    fgraph_from_model(model)
    clone_model(model)
    assert all(value is None for value in model.rvs_to_initial_values.values())
