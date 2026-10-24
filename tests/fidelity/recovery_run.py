"""Real sampler adapter and the pre-recorded simultaneous Wald policy."""

import copy
import math
import zlib

import numpy as np
from input_recovery import check_input_laws, input_check_count
from recovery_mle import (
    UndefinedCoding,
    design_columns,
    design_matrix,
    fit,
    parameter_names,
    targets,
)
from scipy.stats import norm

FALSE_ALARM_BUDGET = 0.01
MIN_OBSERVATIONS = 20000


def parameter_budget(specs):
    # Allocate to the entire corpus, even on the pilot; undefined coding reserves
    # its declared coefficient count, so refusal cannot make a band narrower.
    total = 0
    for spec in specs:
        r = spec["response"]
        total += (
            1
            + len(r["coefficients"])
            + (r["family"] == "regression_continuous")
            + (r.get("count_distribution") == "nb2")
            + (r["family"] == "zero_inflated_count")
        )
        total += input_check_count(spec)
    return total


def sample(spec, mutation=None):
    from ppl_synth_data.envs.declared_inputs import sample_inputs
    from ppl_synth_data.envs.declared_response import DeclaredResponseEnv
    from ppl_synth_data.specs.models import FourFamilySpec

    typed = FourFamilySpec.model_validate(spec)
    seed = 260930 + zlib.crc32(spec["env_id"].encode())
    episodes = math.ceil(MIN_OBSERVATIONS / spec["data_plan"]["n_observations"])
    rows = []
    for episode in range(episodes):
        rows.extend(sample_inputs(typed, seed=seed + episode))
    altered = copy.deepcopy(spec)
    response = altered["response"]
    numeric = (
        next(
            f["name"]
            for f in spec["features"]
            if f["law"]["dist"] in {"normal", "truncated_normal", "discrete_uniform"}
        )
        if mutation in {"minmax", "drop", "reverse"}
        else None
    )
    if mutation == "drop":
        response["coefficients"][numeric] = 0.0
    if mutation == "reverse":
        response["coefficients"][numeric] *= -1
    if mutation == "poisson_for_nb2":
        response["count_distribution"] = "poisson"
        del response["nb2_alpha"]
    if mutation == "ignore_zero_inflation":
        altered["generative_family"] = response["family"] = "regression_count"
        del response["zero_inflation"]
    env = DeclaredResponseEnv(altered, seed=seed + 100000)
    response_rows = copy.deepcopy(rows)
    if mutation == "minmax":
        values = np.array([row[numeric] for row in rows])
        values = (values - values.min()) / np.ptp(values)
        for row, value in zip(response_rows, values, strict=True):
            row[numeric] = float(value)
    names = [feature["name"] for feature in spec["features"]]
    outcomes = np.array([env.step(tuple(row[name] for name in names)) for row in response_rows])
    return rows, outcomes, seed, episodes


def check_spec(spec, budget, mutation=None):
    base = {"env_id": spec["env_id"], "family": spec["generative_family"], "mutation": mutation}
    try:
        design_columns(spec)
    except UndefinedCoding as exc:
        return {**base, "status": "REFUSED", "reason": str(exc)}
    rows, y, seed, episodes = sample(spec, mutation)
    episode_size = spec["data_plan"]["n_observations"]
    inputs = check_input_laws(
        spec,
        [rows[i : i + episode_size] for i in range(0, len(rows), episode_size)],
        FALSE_ALARM_BUDGET / budget,
    )
    fitted = fit(spec, design_matrix(spec, rows), y)
    truth = targets(spec)
    critical = float(norm.isf(FALSE_ALARM_BUDGET / (2 * budget)))
    zscores = np.abs(fitted.estimates - truth) / fitted.standard_errors
    if not np.isfinite(zscores).all() or not (fitted.standard_errors > 0).all():
        raise ValueError("non-finite recovery evidence")
    parameters = [
        dict(name=name, truth=float(value), estimate=float(est), se=float(se), z=float(z))
        for name, value, est, se, z in zip(
            parameter_names(spec),
            truth,
            fitted.estimates,
            fitted.standard_errors,
            zscores,
            strict=True,
        )
    ]
    violations = [p["name"] for p in parameters if p["z"] > critical]
    violations += ["input:" + item["name"] for item in inputs["checks"] if not item["passed"]]
    return {
        **base,
        "status": "REJECT" if violations else "PASS",
        "n": len(rows),
        "episodes": episodes,
        "input_seed_first": seed,
        "input_seed_last": seed + episodes - 1,
        "response_seed": seed + 100000,
        "budget_parameters": budget,
        "suite_alpha": FALSE_ALARM_BUDGET,
        "z_critical": critical,
        "parameters": parameters,
        "inputs": inputs,
        "violations": violations,
        "optimizer": fitted.optimizer,
    }
