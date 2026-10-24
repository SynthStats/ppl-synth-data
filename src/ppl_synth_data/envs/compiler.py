from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ppl_synth_data.pipeline.env_loader import env_module_filename, snake_to_camel
from ppl_synth_data.pipeline.io import atomic_write_text
from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec
from ppl_synth_data.specs.registry import SUPPORTED_FAMILIES

_SAFE_ENV_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# derived from the single source of truth in specs/registry.py
_FAMILY_TEMPLATE: dict[str, str] = {fam: f"{fam}.j2" for fam in SUPPORTED_FAMILIES}


def _pointer(parent: str, key: str) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _leaf_paths(value: Any, pointer: str) -> set[str]:
    if isinstance(value, dict) and value:
        return set().union(*(_leaf_paths(v, _pointer(pointer, k)) for k, v in value.items()))
    if isinstance(value, list) and value:
        return set().union(
            *(_leaf_paths(v, _pointer(pointer, str(i))) for i, v in enumerate(value))
        )
    return {pointer}


class _ParameterReads(dict):
    """Track template reads; validation enumeration alone does not consume a value."""

    def __init__(self, raw: dict[str, Any], pointer: str, reads: set[str]) -> None:
        super().__init__(raw)
        self.raw = raw
        self.pointer = pointer
        self.reads = reads

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        pointer = _pointer(self.pointer, key)
        if isinstance(value, dict) and value:
            return _ParameterReads(value, pointer, self.reads)
        self.reads.update(_leaf_paths(value, pointer))
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default

    def items(self):
        return ((key, self[key]) for key in self)

    def values(self):
        return (self[key] for key in self)


def _require_parameters(spec: dict[str, Any], parameter_units: dict[str, str]) -> dict[str, Any]:
    """Validate the complete parameter declaration supplied by the family template."""
    family = spec["generative_family"]
    supplied = spec["generative_params"]
    params = supplied.raw if isinstance(supplied, _ParameterReads) else supplied
    if "coefficients" in parameter_units:
        _require_feature_domains(spec, spec["io_schema"]["feature_names"])
    missing = set(parameter_units) - set(params)
    if missing:
        raise ValueError(f"{family}: missing required response parameters {sorted(missing)!r}")
    unsupported = set(params) - set(parameter_units)
    if unsupported:
        pointers = sorted(_pointer("/generative_params", name) for name in unsupported)
        raise ValueError(
            f"{family}: unsupported response parameters {pointers!r}; missing implemented semantics"
        )
    for name, value in params.items():
        if name == "coefficients":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{family}: response parameter {name!r} must be a finite number")
        if name in {"noise_sigma", "rate_sigma", "k_sigma", "ability_sigma", "difficulty_sigma"}:
            if value < 0:
                raise ValueError(f"{family}: response parameter {name!r} must be nonnegative")
        if name in {
            "n_categories",
            "n_components",
            "n_subjects",
            "population_n",
            "n_students",
            "n_questions",
            "n_sources",
            "dim",
        } and (value <= 0 or int(value) != value):
            raise ValueError(f"{family}: response parameter {name!r} must be a positive integer")
        if name in {"zero_inflation", "epsilon"} and not 0 <= value <= 1:
            raise ValueError(f"{family}: response parameter {name!r} must be in [0, 1]")
        if (
            name in {"season_period", "baseline_rate", "regularization_m", "choice_scale"}
            and value <= 0
        ):
            raise ValueError(f"{family}: response parameter {name!r} must be positive")
    if family == "spatial_signal_source" and params["dim"] != 2:
        raise ValueError(f"{family}: response parameter 'dim' must be 2 for x,y queries")
    if (
        family == "mixture_continuous"
        and params["component_mean_lower"] >= params["component_mean_upper"]
    ):
        raise ValueError(f"{family}: component_mean_lower must be less than component_mean_upper")
    return supplied


def _require_coefficients(spec: dict[str, Any], features: list[str]) -> list[float]:
    family = spec["generative_family"]
    coefficients = spec["generative_params"].get("coefficients")
    if not isinstance(coefficients, dict) or set(coefficients) != set(features):
        raise ValueError(
            f"{family}: generative_params.coefficients must map every declared feature name "
            "exactly once to its effect per unit of (x - lower) / (upper - lower); "
            f"expected {features!r}, got {coefficients!r}"
        )
    values = []
    for name in features:
        value = coefficients[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{family}: coefficient for feature {name!r} must be a finite number")
        values.append(float(value))
    return values


def _require_feature_domains(
    spec: dict[str, Any], features: list[str]
) -> dict[str, tuple[float, float]]:
    family = spec["generative_family"]
    if not features:
        raise ValueError(f"{family}: IOSchema.feature_names must be non-empty")
    if len(set(features)) != len(features):
        raise ValueError(f"{family}: IOSchema.feature_names must contain unique names")
    declared = spec["io_schema"]["feature_names"]
    if declared != features:
        raise ValueError(f"{family}: IOSchema.feature_names must be {features!r}; got {declared!r}")
    domains = spec["io_schema"]["feature_domains"]
    unused = set(domains) - set(features)
    if unused:
        raise ValueError(f"{family}: feature_domains contains unused features {sorted(unused)!r}")
    index_counts = {
        "irt_binary_response": {"student_id": "n_students", "question_id": "n_questions"},
        "survival_with_censoring": {"subject_id": "n_subjects"},
    }.get(family, {})
    integer_features = set(index_counts)
    if family == "hierarchical_grouped":
        integer_features.update(spec["data_plan"]["grouping"])
    result: dict[str, tuple[float, float]] = {}
    for name in features:
        if name not in domains:
            raise ValueError(
                f"{family}: IOSchema.feature_domains is missing a domain for feature {name!r}"
            )
        lower, upper = domains[name]
        singleton_identifier = name in integer_features and lower == upper
        if not (
            math.isfinite(lower)
            and math.isfinite(upper)
            and (lower < upper or singleton_identifier)
            and math.isfinite(upper - lower)
        ):
            raise ValueError(
                f"{family}: feature_domains[{name!r}] requires finite bounds with lower < upper; "
                f"got ({lower}, {upper})"
            )
        result[name] = (lower, upper)
        if name in integer_features and (int(lower) != lower or int(upper) != upper):
            raise ValueError(f"{family}: feature_domains[{name!r}] requires integer index bounds")
        if name in index_counts:
            count = spec["generative_params"][index_counts[name]]
            if lower < 0 or upper > count - 1:
                raise ValueError(
                    f"{family}: feature_domains[{name!r}] ({lower}, {upper}) must fit "
                    f"{index_counts[name]} indices [0, {count - 1}]"
                )
    if spec["generative_family"] == "choice_with_discounting" and result["delay_days"][0] < 0:
        raise ValueError(
            "choice_with_discounting: feature 'delay_days' requires nonnegative bounds"
        )
    if family == "ode_predator_prey" and result["t"][0] < 0:
        raise ValueError("ode_predator_prey: feature_domains['t'] requires nonnegative bounds")
    return result


class TemplateCompiler:
    def __init__(self, template_dir: Path | None = None) -> None:
        base = Path(__file__).resolve().parent
        self._template_dir = template_dir or (base / "templates")
        self._jinja = Environment(  # noqa: S701 - template output is Python source, not HTML
            loader=FileSystemLoader(str(self._template_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._jinja.globals["require_feature_domains"] = _require_feature_domains
        self._jinja.globals["require_coefficients"] = _require_coefficients
        self._jinja.globals["require_parameters"] = _require_parameters

    def compile_envspec(self, spec: EnvSpec | FourFamilySpec, out_dir: Path) -> Path:
        family = spec.generative_family
        if family not in _FAMILY_TEMPLATE:
            raise ValueError(
                f"Unsupported generative_family={family!r}; "
                f"expected one of {sorted(_FAMILY_TEMPLATE)}"
            )
        if not isinstance(spec, FourFamilySpec) and not _SAFE_ENV_NAME.fullmatch(spec.env_name):
            raise ValueError(
                f"{family}: env_name must match ^[a-z][a-z0-9_]*$ to prevent path traversal"
            )

        if isinstance(spec, FourFamilySpec):
            template = self._jinja.get_template("declared_response.j2")
            code = template.render(
                spec=spec.model_dump(mode="json"),
                class_name=f"{snake_to_camel(spec.env_name)}Env",
            )
            out_path = out_dir.resolve() / env_module_filename(spec)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(out_path, code)
            return out_path

        target_name = spec.io_schema.target_name
        if not isinstance(target_name, str) or not _SAFE_FIELD_NAME.fullmatch(target_name):
            raise ValueError(f"{family}: target_name is required and must be a safe slug")

        render_context = {}
        if family == "hierarchical_grouped":
            if len(spec.data_plan.grouping) != 1:
                raise ValueError(f"{family}: grouping must declare exactly one group field")
            render_context["group_field"] = spec.data_plan.grouping[0]
        for group_field in spec.data_plan.grouping:
            if not _SAFE_FIELD_NAME.fullmatch(group_field):
                raise ValueError(f"{family}: grouping field {group_field!r} must be a safe slug")

        class_name = f"{snake_to_camel(spec.env_name)}Env"
        if not class_name.isidentifier():
            raise ValueError(f"{family}: env_name produces an invalid class name: {class_name}")

        template = self._jinja.get_template(_FAMILY_TEMPLATE[family])
        payload = spec.model_dump(mode="json")
        parameters = payload["generative_params"]
        declared = set().union(
            *(
                _leaf_paths(value, _pointer("/generative_params", name))
                for name, value in parameters.items()
            )
        )
        reads: set[str] = set()
        payload["generative_params"] = _ParameterReads(parameters, "/generative_params", reads)
        code = template.render(
            spec=payload,
            class_name=class_name,
            target_name=target_name,
            **render_context,
        )
        unread = declared - reads
        if unread:
            raise ValueError(
                f"{family}: unread declarations {sorted(unread)!r}; missing template consumer"
            )

        base_dir = out_dir.resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = (base_dir / f"{spec.env_name}.py").resolve()
        if base_dir not in out_path.parents:
            raise ValueError(f"{family}: env_name would write outside out_dir")
        atomic_write_text(out_path, code)
        return out_path
