from __future__ import annotations

L0_PARSE_TIMEOUT_S = 2
L1_AST_SAFETY_TIMEOUT_S = 2
L2_MODEL_BUILD_TIMEOUT_S = 30
L2_RUNTIME_PRELOAD_TIMEOUT_S = 90
L3_HARBOR_PROFILE_TIMEOUT_S = 5
L4_EXPLOIT_HEURISTICS_TIMEOUT_S = 2
L5_DATA_PERTURBATION_TIMEOUT_S = 45
# Keep this L6 timeout equal to the standalone validator's timeout.
L6_HOLDOUT_ELPD_TIMEOUT_S = 600
PYMC_HARBOR_TIMEOUT_MARGIN_S = 30


def cascade_timeout_budget_s() -> int:
    """Worst-case time spent inside the PyMC Harbor L0-L5 structural cascade."""
    return (
        L0_PARSE_TIMEOUT_S
        + L1_AST_SAFETY_TIMEOUT_S
        + L2_RUNTIME_PRELOAD_TIMEOUT_S
        + L2_MODEL_BUILD_TIMEOUT_S
        + L3_HARBOR_PROFILE_TIMEOUT_S
        + L4_EXPLOIT_HEURISTICS_TIMEOUT_S
        + L5_DATA_PERTURBATION_TIMEOUT_S
    )


def full_cascade_timeout_budget_s() -> int:
    """Worst-case time spent inside the full L0-L6 cascade, held-out ELPD included."""
    return cascade_timeout_budget_s() + L6_HOLDOUT_ELPD_TIMEOUT_S


def pymc_harbor_verifier_timeout_sec() -> float:
    """Timeout for the full Harbor verifier process, not just the cascade."""
    budget = full_cascade_timeout_budget_s() + PYMC_HARBOR_TIMEOUT_MARGIN_S
    return float(((budget + 9) // 10) * 10)
