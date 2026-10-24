from __future__ import annotations

import json
import math
from typing import Any, Generic, TypeVar

import pandas as pd


class InputValidationError(ValueError):
    """Raised when an environment query does not satisfy the input contract."""


InputValue = TypeVar("InputValue")


class BaseEnv(Generic[InputValue]):
    env_name: str
    response_law: dict[str, Any]
    observed_data: list[tuple[Any, ...]]
    include_prior: bool
    _feature_names: list[str]
    _feature_domains: dict[str, tuple[float, float]]
    _multi_feature_input: bool
    _integer_features: frozenset[str] = frozenset()

    def reset(self, seed: int | None = None) -> None:
        raise NotImplementedError

    def step(self, x_values: tuple[float | InputValue, ...]) -> Any:
        raise NotImplementedError

    def _standardize_features(self, x_values: tuple[float, ...]) -> tuple[float, ...]:
        """Map validated raw inputs to their positions in the declared domains."""
        standardized = []
        for name, value in zip(self._feature_names, x_values, strict=True):
            lower, upper = self._feature_domains[name]
            if not (
                math.isfinite(lower)
                and math.isfinite(upper)
                and lower < upper
                and math.isfinite(upper - lower)
            ):
                raise ValueError(
                    f"{self.response_law['family']}: feature_domains[{name!r}] "
                    f"requires finite bounds with lower < upper; got ({lower}, {upper})"
                )
            standardized.append((value - lower) / (upper - lower))
        return tuple(standardized)

    def _validate_feature_payload(self, payload: Any) -> tuple[float, ...]:
        family = self.response_law["family"]
        if not isinstance(payload, dict):
            raise InputValidationError(
                f"{family}: input for features {self._feature_names!r} must be a JSON object."
            )

        values: list[float] = []
        for name in self._feature_names:
            if name not in payload:
                raise InputValidationError(f"{family}: input must include feature '{name}'.")
            raw_value = payload[name]
            lower, upper = self._feature_domains[name]
            domain_context = f" between {lower} and {upper}; got {raw_value!r}."
            if isinstance(raw_value, bool):
                raise InputValidationError(
                    f"{family}: feature '{name}' must be numeric" + domain_context
                )
            try:
                value = float(raw_value)
            except (ValueError, TypeError, OverflowError) as exc:
                raise InputValidationError(
                    f"{family}: feature '{name}' must be a finite numeric value" + domain_context
                ) from exc
            if not math.isfinite(value):
                raise InputValidationError(
                    f"{family}: feature '{name}' must be finite" + domain_context
                )
            if value < lower or value > upper:
                raise InputValidationError(
                    f"{family}: feature '{name}' must be between {lower} and {upper}; got {value}."
                )
            if name in self._integer_features:
                if not value.is_integer():
                    raise InputValidationError(
                        f"{family}: feature '{name}' must be an integer" + domain_context
                    )
                value = int(value)
            values.append(value)
        return tuple(values)

    def validate_input(self, input_str: str) -> tuple[float | InputValue, ...]:
        if self._multi_feature_input:
            try:
                parsed = json.loads(input_str)
            except (json.JSONDecodeError, TypeError) as exc:
                raise InputValidationError(
                    f"{self.response_law['family']}: input for features {self._feature_names!r} "
                    "must be a valid JSON string."
                ) from exc
            return self._validate_feature_payload(parsed)

        try:
            x = float(input_str.strip())
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            name = self._feature_names[0]
            lower, upper = self._feature_domains[name]
            raise InputValidationError(
                f"{self.response_law['family']}: input must be a single float for feature '{name}' "
                f"between {lower} and {upper}; got {input_str!r}."
            ) from exc
        return self._validate_feature_payload({self._feature_names[0]: x})

    def run_experiment(self, input_str: str) -> tuple[Any, bool]:
        try:
            x_values = self.validate_input(input_str)
        except InputValidationError as exc:
            return str(exc), False
        y = self.step(x_values)
        self.observed_data.append(self._observation_row(x_values, y))
        return y, True

    def _observation_row(
        self, x_values: tuple[float | InputValue, ...], result: Any
    ) -> tuple[Any, ...]:
        response = result if isinstance(result, tuple) else (result,)
        return (*x_values, *response)

    def get_data(self) -> list[tuple[Any, ...]]:
        return list(self.observed_data)

    def get_df(self) -> pd.DataFrame:
        cols = self.get_ordered_column_names()
        return pd.DataFrame(self.observed_data, columns=cols)

    def get_ordered_column_names(self) -> list[str]:
        raise NotImplementedError

    def get_ordered_features(self) -> list[str]:
        return list(self._feature_names)

    def _input_domain_message(self) -> str:
        domains = "; ".join(
            f"{name}: [{self._feature_domains[name][0]}, {self._feature_domains[name][1]}]"
            for name in self._feature_names
        )
        return f" Input domains (inclusive): {domains}."

    def generate_system_message(self, include_prior: bool = False, goal: str | None = None) -> str:
        raise NotImplementedError
