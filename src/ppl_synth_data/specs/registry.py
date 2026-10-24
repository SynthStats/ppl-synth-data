from __future__ import annotations

SUPPORTED_FAMILIES: tuple[str, ...] = (
    "regression_continuous",
    "regression_count",
    "zero_inflated_count",
    "binary_choice",
    "categorical_choice",
    "mixture_continuous",
    "hierarchical_grouped",
    "time_series_level_seasonality",
    "survival_with_censoring",
    "ode_predator_prey",
    "irt_binary_response",
    "infection_process",
    "spatial_signal_source",
    "choice_with_discounting",
)


def validate_family(family: str) -> None:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported generative_family={family!r}; expected one of {SUPPORTED_FAMILIES}"
        )
