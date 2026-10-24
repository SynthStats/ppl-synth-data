"""Independent input-law recovery; NumPy/SciPy only, no production imports.

Each stochastic check receives one Bonferroni share from the response+input
suite. Structural support and exact episode allocations are equality checks.
"""

from bisect import bisect_right
from collections import Counter

import numpy as np
from scipy.stats import binomtest, chi2, chisquare, kstest, truncnorm, ttest_1samp


def input_check_count(spec):
    counts = {
        "normal": 2,
        "truncated_normal": 3,
        "discrete_uniform": 1,
        "bernoulli": 1,
        "categorical": 1,
        "fixed_counts": 0,
    }
    return sum(counts[f["law"]["dist"]] for f in spec["features"])


def check_input_laws(spec, episodes, alpha):
    """Recover laws from the same complete episodes used by response recovery."""
    if not 0 < alpha < 1:
        raise ValueError("input-law alpha must be in (0,1)")
    if not episodes or any(not episode for episode in episodes):
        raise ValueError("input recovery requires nonempty whole episodes")
    checks = []
    expected_n = spec["data_plan"]["n_observations"]
    bad_sizes = [i for i, episode in enumerate(episodes) if len(episode) != expected_n]
    checks.append(
        dict(
            name="episode_sizes", passed=not bad_sizes, expected=expected_n, bad_episodes=bad_sizes
        )
    )

    def test(name, pvalue, **evidence):
        pvalue = float(pvalue)
        if not np.isfinite(pvalue) or not 0 <= pvalue <= 1:
            raise ValueError(f"invalid input-law pvalue for {name}: {pvalue}")
        checks.append(
            dict(name=name, passed=pvalue >= alpha, pvalue=pvalue, alpha=alpha, **evidence)
        )

    for feature in spec["features"]:
        name, law = feature["name"], feature["law"]
        dist = law["dist"]
        values = [row[name] for episode in episodes for row in episode]
        n = len(values)
        if dist == "fixed_counts":
            expected = dict(zip(law["levels"], law["counts"], strict=True))
            failures = [
                dict(episode=i, counts=dict(Counter(row[name] for row in episode)))
                for i, episode in enumerate(episodes)
                if Counter(row[name] for row in episode) != expected
            ]
            checks.append(
                dict(
                    name=f"{name}:fixed_counts",
                    passed=not failures,
                    expected=expected,
                    episodes_checked=len(episodes),
                    failures=failures,
                )
            )
        elif dist == "normal":
            x = np.asarray(values, dtype=float)
            if n < 2 or not np.isfinite(x).all():
                checks.append(dict(name=f"{name}:support", passed=False))
                continue
            mean, sd = float(x.mean()), float(x.std(ddof=1))
            test(
                f"{name}:mean",
                ttest_1samp(x, law["mean"]).pvalue,
                estimate=mean,
                truth=law["mean"],
                n=n,
                method="two-sided Student t",
            )
            statistic = (n - 1) * sd**2 / law["sd"] ** 2
            test(
                f"{name}:sd",
                min(1.0, 2 * min(chi2.cdf(statistic, n - 1), chi2.sf(statistic, n - 1))),
                estimate=sd,
                truth=law["sd"],
                n=n,
                method="two-sided chi-square variance",
            )
        elif dist == "truncated_normal":
            x = np.asarray(values, dtype=float)
            low, high = law.get("low"), law.get("high")
            valid = (
                n >= 2
                and np.isfinite(x).all()
                and (low is None or np.all(x >= low))
                and (high is None or np.all(x <= high))
            )
            checks.append(dict(name=f"{name}:support", passed=bool(valid), low=low, high=high))
            if not valid:
                continue
            a = -np.inf if low is None else (low - law["mean"]) / law["sd"]
            b = np.inf if high is None else (high - law["mean"]) / law["sd"]
            reference = truncnorm(a, b, loc=law["mean"], scale=law["sd"])
            mean, variance, _, kurtosis = map(float, reference.stats(moments="mvsk"))
            fourth = (kurtosis + 3) * variance**2
            variance_of_variance = (fourth - (n - 3) / (n - 1) * variance**2) / n
            if (
                not np.isfinite([mean, variance, variance_of_variance]).all()
                or variance <= 0
                or variance_of_variance <= 0
            ):
                raise ValueError(f"{name}: truncated moments exceed reliable numeric range")
            sample_mean, sample_variance = float(x.mean()), float(x.var(ddof=1))

            def moment_pvalue(delta, sampling_variance):
                # Chebyshev gives a conservative finite-sample two-sided p-value.
                # Unlike a Normal variance test, this holds for a truncated law.
                return min(1.0, sampling_variance / delta**2) if delta else 1.0

            test(
                f"{name}:mean",
                moment_pvalue(sample_mean - mean, variance / n),
                estimate=sample_mean,
                truth=mean,
                n=n,
                method="Chebyshev truncated mean",
            )
            test(
                f"{name}:sd",
                moment_pvalue(sample_variance - variance, variance_of_variance),
                estimate=float(np.sqrt(sample_variance)),
                truth=float(np.sqrt(variance)),
                n=n,
                method="Chebyshev unbiased variance with truncated fourth moment",
            )
            test(
                f"{name}:cdf",
                kstest(x, reference.cdf, method="exact").pvalue,
                n=n,
                method="two-sided one-sample Kolmogorov-Smirnov; declared parameters",
            )
        elif dist == "bernoulli":
            valid = all(type(v) in (int, float) and v in (0, 1) for v in values)
            checks.append(dict(name=f"{name}:support", passed=valid))
            if valid:
                successes = int(sum(values))
                test(
                    f"{name}:p",
                    binomtest(successes, n, law["p"]).pvalue,
                    estimate=successes / n,
                    truth=law["p"],
                    n=n,
                    method="exact two-sided binomial",
                )
        elif dist in {"categorical", "discrete_uniform"}:
            if dist == "discrete_uniform":
                low, high = law["low"], law["high"]
                valid = all(type(v) is int and low <= v <= high for v in values)
            else:
                levels = law["levels"]
                valid = all(v in levels for v in values)
            support = dict(name=f"{name}:support", passed=valid)
            if dist == "discrete_uniform" and valid:
                support.update(
                    observed_min=min(values), observed_max=max(values), low=low, high=high
                )
            checks.append(support)
            if not valid:
                continue
            counts = Counter(values)
            if dist == "discrete_uniform":
                support_size = high - low + 1
                # Integer cells cannot always realize n//5 bins with E >= 5.
                # Fix the largest admissible partition from (low, high, n),
                # without inspecting observed frequencies or sample extrema.
                min_width = (5 * support_size + n - 1) // n
                bin_count = min(support_size, n // 5, support_size // min_width)
                if bin_count < 2:
                    raise ValueError(f"{name}: insufficient sample size for chi-square inference")
                edges = [low + i * support_size // bin_count for i in range(bin_count + 1)]
                widths = [right - left for left, right in zip(edges[:-1], edges[1:], strict=True)]
                binned = Counter(bisect_right(edges, value) - 1 for value in values)
                observed = np.array([binned[i] for i in range(bin_count)])
                probs = np.asarray([width / support_size for width in widths])
                expected = np.asarray([n * width / support_size for width in widths])
                evidence = dict(
                    bins=[
                        [left, right - 1] for left, right in zip(edges[:-1], edges[1:], strict=True)
                    ],
                    support_size=support_size,
                    expected=expected.tolist(),
                )
            else:
                observed = np.array([counts[level] for level in levels])
                probs = np.asarray(law["probs"])
                expected = n * probs
                evidence = dict(levels=levels)
            # The declared pooled design must support Pearson's approximation.
            if np.any(expected < 5):
                raise ValueError(
                    f"{name}: insufficient expected frequencies for chi-square inference"
                )
            pvalue = float(chisquare(observed, expected).pvalue)
            evidence.update(
                observed=observed.tolist(),
                estimates=(observed / n).tolist(),
                probabilities=probs.tolist(),
                n=n,
                method="Pearson multinomial goodness of fit",
            )
            if dist == "discrete_uniform":
                endpoints = counts[low] > 0 and counts[high] > 0
                # Missing either endpoint has probability at most 2(1-1/k)^n.
                # Combine this valid event p-value and Pearson with Bonferroni,
                # retaining a single preallocated omnibus input-law test.
                endpoint_p = min(1.0, 2 * (1 - 1 / support_size) ** n) if not endpoints else 1.0
                evidence.update(
                    endpoints_observed=endpoints,
                    frequency_pvalue=pvalue,
                    endpoint_pvalue=endpoint_p,
                )
                pvalue = min(1.0, 2 * min(pvalue, endpoint_p))
            test(f"{name}:frequencies", pvalue, **evidence)
        else:
            raise ValueError(f"unsupported input law {dist!r}")
    return dict(passed=all(c["passed"] for c in checks), checks=checks, alpha=alpha)
