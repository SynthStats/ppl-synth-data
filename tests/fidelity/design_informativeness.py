"""descriptive design information at the declared n, never an admission gate.

Run directly with --specs, --output-dir and --replications. The policy is written before
sampling. Existing outputs are refused, and each completed spec is flushed. The independent MLE
implementation is unchanged; only visible rows reach it.
"""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from recovery_mle import design_matrix, fit, parameter_names, targets
from scipy.stats import binom, binomtest, norm


def cp_interval(k, n, alpha=0.05):
    """Two-sided exact binomial interval; no observations means no interval."""
    if not 0 < alpha < 1 or not 0 <= k <= n:
        raise ValueError("require 0 < alpha < 1 and 0 <= k <= n")
    if n == 0:
        return None
    interval = binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")
    return [float(interval.low), float(interval.high)]


def make_policy(specs, replications, base_seed=260930):
    if not specs or replications < 1 or base_seed < 0:
        raise ValueError("require nonempty specs, positive replications and nonnegative seed")
    count = sum(1 + len(spec["response"]["coefficients"]) for spec in specs)
    alpha = 0.01 / count
    # Precision and power are design calculations, performed before any draws.
    expected_covered = int(round(0.95 * replications))
    nominal_interval = cp_interval(expected_covered, replications)
    threshold = [k for k in range(replications + 1) if binom.cdf(k, replications, 0.95) < alpha / 2]
    cutoff = max(threshold) if threshold else -1
    return {
        "ticket": "design-informativeness-at-declared-n",
        "admission_decision": "not_applicable_descriptive_measurement",
        "replications": replications,
        "base_seed": base_seed,
        "suite_alpha": 0.01,
        "coefficient_count": count,
        "coefficient_alpha": alpha,
        "budget_spec_count": len(specs),
        "wald_confidence": 0.95,
        "visible_ratio": 0.8,
        "split_policy": "random_row",
        ("seed_rule"): (
            "first 64 bits SHA256(base_seed:env_id:replication:stream), three tagged streams"
        ),
        ("coverage_denominator"): (
            "successful fits; all-trial interval-delivery fractions also reported"
        ),
        "undercoverage_flag": "two-sided exact suite CP upper endpoint < 0.95",
        "low_sign_flag": "nonzero truth and fewer than half successful intervals exclude zero",
        "zero_success": "conditional rates, intervals and flags are null",
        "nominal_cp95_at_rounded_95_percent": nominal_interval,
        "nominal_cp95_width": nominal_interval[1] - nominal_interval[0],
        "undercoverage_count_cutoff": cutoff,
        "power_at_85_percent_coverage_if_all_fits_succeed": float(
            binom.cdf(cutoff, replications, 0.85)
        )
        if cutoff >= 0
        else 0.0,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }


def sample_visible(spec, replication, base_seed):
    from ppl_synth_data.envs.declared_inputs import sample_inputs
    from ppl_synth_data.envs.declared_response import DeclaredResponseEnv
    from ppl_synth_data.pipeline.split_policy import SplitPolicy, split_visible_hidden
    from ppl_synth_data.specs.models import FourFamilySpec

    if replication < 0 or base_seed < 0:
        raise ValueError("negative replication or seed")
    typed = FourFamilySpec.model_validate(spec)
    seeds = {
        f"{stream}_seed": int.from_bytes(
            hashlib.sha256(
                f"{base_seed}:{spec['env_id']}:{replication}:{stream}".encode()
            ).digest()[:8],
            "big",
        )
        for stream in ("input", "response", "split")
    }
    rows = sample_inputs(typed, seed=seeds["input_seed"])
    env = DeclaredResponseEnv(spec, seed=seeds["response_seed"])
    names = [feature["name"] for feature in spec["features"]]
    outcomes = [env.step(tuple(row[name] for name in names)) for row in rows]
    frame = pd.DataFrame(rows)
    frame["y"] = outcomes
    split = split_visible_hidden(
        frame,
        typed,
        split_policy=SplitPolicy.RANDOM_ROW,
        seed=seeds["split_seed"],
        visible_ratio=0.8,
    )
    audit = {
        **seeds,
        "n_total": len(frame),
        "n_visible": len(split.visible),
        "n_hidden": len(split.hidden),
        "visible_positions": split.visible.index.tolist(),
        "hidden_positions": split.hidden.index.tolist(),
    }
    return (
        split.visible[names].to_dict("records"),
        split.visible["y"].to_numpy(dtype=float),
        audit,
    )


def summarize_trials(spec, trials, policy):
    if len(trials) != policy["replications"]:
        raise ValueError("trial count differs from the recorded replication policy")
    successes = [trial for trial in trials if trial["failure"] is None]
    failures = [trial for trial in trials if trial["failure"] is not None]
    n_success, n_trials = len(successes), len(trials)
    names, truth = parameter_names(spec), targets(spec)
    width = 1 + len(spec["response"]["coefficients"])
    critical = float(norm.ppf(0.975))
    coefficients = []
    if successes:
        estimates = np.asarray([trial["estimates"] for trial in successes], dtype=float)
        errors = np.asarray([trial["standard_errors"] for trial in successes], dtype=float)
        if (
            estimates.shape != (n_success, len(names))
            or errors.shape != estimates.shape
            or not np.isfinite(estimates).all()
            or not np.isfinite(errors).all()
            or not (errors > 0).all()
        ):
            raise ValueError("successful fit contains invalid estimates or standard errors")
        low, high = estimates - critical * errors, estimates + critical * errors
    for index, name in enumerate(names[:width]):
        covered = (
            int(np.sum((low[:, index] <= truth[index]) & (truth[index] <= high[:, index])))
            if successes
            else 0
        )
        excluded = int(np.sum((low[:, index] > 0) | (high[:, index] < 0))) if successes else 0
        correct = (
            int(np.sum(low[:, index] > 0 if truth[index] > 0 else high[:, index] < 0))
            if successes and truth[index] != 0
            else None
        )
        cp95 = cp_interval(covered, n_success)
        cp_suite = cp_interval(covered, n_success, policy["coefficient_alpha"])
        coefficients.append(
            {
                "name": name,
                "truth": float(truth[index]),
                "n_success": n_success,
                "count_covered": covered,
                "coverage": covered / n_success if successes else None,
                "coverage_cp95": cp95,
                "coverage_cp_suite": cp_suite,
                "count_excludes_zero": excluded,
                "sign_detectability": excluded / n_success if successes else None,
                "sign_detectability_cp95": cp_interval(excluded, n_success),
                "count_correct_sign": correct,
                "correct_sign_detectability": correct / n_success if correct is not None else None,
                "all_trials_coverage": covered / n_trials,
                "all_trials_sign_detectability": excluded / n_trials,
                "flag_undercoverage": cp_suite[1] < 0.95 if cp_suite is not None else None,
                "flag_low_sign": bool(truth[index] != 0 and excluded / n_success < 0.5)
                if successes
                else None,
                "mean_estimate": float(estimates[:, index].mean()) if successes else None,
                "bias": float(estimates[:, index].mean() - truth[index]) if successes else None,
            }
        )
    nonconverged = sum("MLE did not converge" in trial["failure"] for trial in failures)
    return {
        "env_id": spec["env_id"],
        "family": spec["generative_family"],
        "n_declared": spec["data_plan"]["n_observations"],
        "n_visible": int(0.8 * spec["data_plan"]["n_observations"]),
        "n_trials": n_trials,
        "n_success": n_success,
        "n_fit_failures": len(failures),
        "fit_failure_share": len(failures) / n_trials,
        "fit_failure_cp95": cp_interval(len(failures), n_trials),
        "optimizer_nonconvergence_share": nonconverged / n_trials,
        "failure_reasons": dict(Counter(trial["failure"] for trial in failures)),
        "failure_kinds": dict(Counter(trial.get("failure_kind", "fit") for trial in failures)),
        "coefficients": coefficients,
        "coefficient_alpha": policy["coefficient_alpha"],
        "admission_decision": "not_applicable_descriptive_measurement",
    }


def run_spec(spec, policy):
    trials = []
    for replication in range(policy["replications"]):
        # Sampling/schema failures are not inference failures: they propagate.
        rows, outcomes, audit = sample_visible(spec, replication, policy["base_seed"])
        trial = {"replication": replication, "audit": audit}
        try:
            fitted = fit(spec, design_matrix(spec, rows), outcomes)
            if (
                not np.isfinite(fitted.estimates).all()
                or not np.isfinite(fitted.standard_errors).all()
                or not (fitted.standard_errors > 0).all()
            ):
                raise ValueError("invalid finite estimates or positive standard errors")
            trial.update(
                estimates=fitted.estimates.tolist(),
                standard_errors=fitted.standard_errors.tolist(),
                optimizer=fitted.optimizer,
                failure=None,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            message = str(exc)
            kind = (
                "optimizer_nonconvergence"
                if "MLE did not converge" in message
                else "unidentified_design"
                if "predictor" in message or "rank-deficient" in message
                else "invalid_information"
                if "information" in message
                else "invalid_fit"
            )
            trial.update(failure=message, failure_kind=kind)
        trials.append(trial)
    result = summarize_trials(spec, trials, policy)
    result["trials"] = trials
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=400)
    parser.add_argument("--base-seed", type=int, default=260930)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    raw = args.specs.read_bytes()
    specs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(specs) != 120 or len({s["env_id"] for s in specs}) != 120:
        raise ValueError("require the complete selected 120-spec corpus for budget allocation")
    policy = make_policy(specs, args.replications, args.base_seed)
    selected = specs
    if args.pilot:
        selected = [
            next(s for s in specs if s["generative_family"] == family)
            for family in (
                "binary_choice",
                "regression_continuous",
                "regression_count",
                "zero_inflated_count",
            )
        ]
    policy.update(
        source=str(args.specs.resolve()),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        selected_ids=[s["env_id"] for s in selected],
        pilot=args.pilot,
        recorded_at=datetime.now(UTC).isoformat(),
        source_files_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("recovery_mle.py"))
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "policy.json").write_text(
        json.dumps(policy, indent=2, allow_nan=False) + "\n"
    )
    columns = [
        "env_id",
        "family",
        "n_visible",
        "n_trials",
        "n_success",
        "fit_failure_share",
        "coefficient",
        "truth",
        "coverage",
        "coverage_cp95",
        "coverage_cp_suite",
        "sign_detectability",
        "flag_undercoverage",
        "flag_low_sign",
    ]
    with (
        (args.output_dir / "results.jsonl").open("x") as output,
        (args.output_dir / "summary.tsv").open("x", newline="") as table,
    ):
        writer = csv.DictWriter(table, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for spec in selected:
            result = run_spec(spec, policy)
            output.write(json.dumps(result, allow_nan=False) + "\n")
            output.flush()
            for coefficient in result["coefficients"]:
                row = {key: result[key] for key in columns[:6]}
                row["coefficient"] = coefficient["name"]
                row.update({key: coefficient[key] for key in columns[7:]})
                writer.writerow(row)
            table.flush()
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in ("env_id", "n_visible", "n_success", "n_fit_failures")
                    }
                ),
                flush=True,
            )
    (args.output_dir / "complete.json").write_text(
        json.dumps(
            {
                "spec_count": len(selected),
                "replications_per_spec": args.replications,
                "status": "measurement_complete",
                "admission_decision": "none",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
