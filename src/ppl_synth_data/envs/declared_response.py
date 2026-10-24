"""Raw-scale environments compiled from the closed four-family contract."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.special import expit

from ppl_synth_data.envs.base import BaseEnv, InputValidationError
from ppl_synth_data.envs.count_kernel import count_mean, sample_count
from ppl_synth_data.specs.models import CategoricalInput, FixedCountsInput, FourFamilySpec


class DeclaredResponseEnv(BaseEnv[float | int | str]):
    def __init__(self, declaration: dict, *, seed: int | None):
        self._spec = FourFamilySpec.model_validate(declaration)
        self._response = self._spec.response
        self.env_name = self._spec.env_name
        self._feature_names = [feature.name for feature in self._spec.features]
        self._multi_feature_input = len(self._feature_names) > 1
        self.response_law = {"family": self._spec.generative_family, "standardization": "none"}
        self.include_prior = False
        self._seed = 0 if seed is None else seed
        self.reset(seed=self._seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self.observed_data = []

    def validate_input(self, input_str: str) -> tuple[float | int | str, ...]:
        try:
            payload = json.loads(input_str)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "/features: require a numeric scalar or JSON object"
            ) from exc
        if not isinstance(payload, dict) and len(self._feature_names) == 1:
            payload = {self._feature_names[0]: payload}
        if not isinstance(payload, dict) or set(payload) != set(self._feature_names):
            raise InputValidationError("/features: query keys must equal declared feature names")
        values = []
        for index, feature in enumerate(self._spec.features):
            value = payload[feature.name]
            pointer = f"/features/{index}/law"
            law = feature.law
            if feature.uses_reference_coding and isinstance(
                law, (CategoricalInput, FixedCountsInput)
            ):
                if not any(type(value) is type(level) and value == level for level in law.levels):
                    raise InputValidationError(
                        pointer + ": query must be a declared level with its exact type"
                    )
                values.append(value)
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise InputValidationError(pointer + ": query must be finite numeric")
            if law.dist == "fixed_counts" and value not in law.levels:
                raise InputValidationError(pointer + ": query must be a declared numeric level")
            if law.dist == "truncated_normal" and (
                (law.low is not None and value < law.low)
                or (law.high is not None and value > law.high)
            ):
                raise InputValidationError(pointer + ": query must lie within the declared bounds")
            if law.dist == "discrete_uniform" and (
                not isinstance(value, int) or not law.low <= value <= law.high
            ):
                raise InputValidationError(
                    pointer + ": query must be an integer within inclusive bounds"
                )
            if law.dist == "bernoulli" and (not isinstance(value, int) or value not in (0, 1)):
                raise InputValidationError(pointer + ": query must be integer 0 or 1")
            values.append(value)
        return tuple(values)

    def step(self, x_values: tuple[float | int | str, ...]) -> int | float:
        design: dict[str, float] = {}
        for feature, value in zip(self._spec.features, x_values, strict=True):
            if feature.uses_reference_coding and isinstance(
                feature.law, (CategoricalInput, FixedCountsInput)
            ):
                for level in feature.law.levels[1:]:
                    design[f"{feature.name}={level}"] = int(value == level)
            else:
                assert isinstance(value, (int, float))
                design[feature.name] = value
        response = self._response
        if response.family == "regression_count":
            return int(sample_count(response, design, self._rng))
        if response.family == "zero_inflated_count":
            # Validate the count law even on an observation that becomes structural zero.
            count_mean(response, design)
            if self._rng.random() < response.zero_inflation:
                return 0
            return int(sample_count(response, design, self._rng))
        mean = response.intercept + math.fsum(
            coefficient * design[name] for name, coefficient in response.coefficients.items()
        )
        if not math.isfinite(mean):
            raise ValueError("/response: linear predictor must be finite")
        if response.family == "regression_continuous":
            value = float(self._rng.normal(mean, response.noise_sigma))
            if not math.isfinite(value):
                raise ValueError("/response: Gaussian draw exceeds finite numeric range")
            return value
        if response.family == "binary_choice":
            return int(self._rng.binomial(1, expit(mean)))
        raise ValueError("/response/family: response law not implemented: " + response.family)

    def get_ordered_column_names(self) -> list[str]:
        return [*self._feature_names, "y"]

    def generate_system_message(self, include_prior: bool = False, goal: str | None = None) -> str:
        if self._response.family == "regression_continuous":
            message = "This environment returns continuous measurements."
            law = (
                " The conditional mean is linear in the declared feature"
                " effects with Gaussian noise."
            )
        elif self._response.family == "binary_choice":
            message = "This environment returns binary outcomes (0 or 1)."
            law = " The log odds are linear in the declared feature effects."
        elif self._response.family == "zero_inflated_count":
            message = "This environment returns integer counts, including structural zeros."
            law = " The count component has a log mean linear in the declared feature effects."
        else:
            message = "This environment returns non-negative integer counts."
            law = " The log expected count is linear in the declared feature effects."
        message += " Inputs: " + ", ".join(self._feature_names) + "."
        if include_prior:
            message += law
            for feature in self._spec.features:
                if feature.uses_reference_coding and isinstance(
                    feature.law, (CategoricalInput, FixedCountsInput)
                ):
                    message += (
                        f" {feature.name} is reference-coded with reference level "
                        f"{feature.law.levels[0]!r}."
                    )
                else:
                    message += f" {feature.name} uses its raw numeric value."
        if goal:
            message += f" Goal: {goal}"
        return message
