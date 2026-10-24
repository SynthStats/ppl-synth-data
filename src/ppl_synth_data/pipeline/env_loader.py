from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec, InputType


def load_env_instance(
    env_module_path: Path,
    *,
    env_name: str,
    seed: int,
    trusted_root: Path | None = None,
) -> Any:
    if not env_module_path.exists():
        raise FileNotFoundError(f"compiled env module missing: {env_module_path}")
    if env_module_path.is_symlink():
        raise RuntimeError(f"refusing to execute symlinked env module: {env_module_path}")
    resolved_module_path = env_module_path.resolve()
    if trusted_root is not None:
        resolved_root = trusted_root.resolve()
        if not resolved_module_path.is_relative_to(resolved_root):
            raise RuntimeError(
                f"env module path escapes trusted root: module={resolved_module_path}, "
                f"trusted_root={resolved_root}"
            )

    spec = importlib.util.spec_from_file_location(f"generated_env_{env_name}", resolved_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated module: {resolved_module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_name = f"{snake_to_camel(env_name)}Env"
    try:
        env_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"generated module missing expected class {class_name}: {resolved_module_path}"
        ) from exc
    try:
        return env_cls(seed=seed)
    except Exception as exc:
        raise RuntimeError(
            f"failed to instantiate {class_name} from {resolved_module_path}: {exc}"
        ) from exc


def snake_to_camel(value: str) -> str:
    chunks = re.split(r"[_\-\s]+", value)
    name = "".join(chunk[:1].upper() + chunk[1:] for chunk in chunks if chunk)
    if name and (not name.isascii() or not name.isidentifier()):
        return "Display" + hashlib.sha256(value.encode()).hexdigest()
    return name


def env_module_filename(spec: EnvSpec | FourFamilySpec) -> str:
    """Typed display names are metadata; encode the unique ID as a safe filename."""
    if isinstance(spec, FourFamilySpec):
        return "declared_" + spec.env_id.encode().hex() + ".py"
    return spec.env_name + ".py"


def seed_for_env(base_seed: int, env_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{env_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def validate_feature_types(spec: EnvSpec) -> None:
    """Reject feature declarations that the stage C design cannot generate."""
    schema = spec.io_schema
    for name, feature_type in schema.feature_types.items():
        if feature_type != "continuous":
            raise RuntimeError(
                f"{spec.generative_family}: stage C env_id={spec.env_id!r}: "
                f"feature {name!r} declares unsupported "
                f"type {feature_type!r}; only independent 'continuous' features with "
                "explicit feature_domains bounds are supported"
            )

    if schema.input_type in (InputType.float, InputType.json) or len(schema.feature_names) > 1:
        for name in schema.feature_names:
            if name not in schema.feature_types:
                raise RuntimeError(
                    f"stage C env_id={spec.env_id!r}: IOSchema.feature_types is missing "
                    f"a declaration for feature {name!r}; declare 'continuous' with "
                    "explicit feature_domains bounds; no feature type is inferred"
                )

    if schema.feature_types and schema.input_type not in (InputType.float, InputType.json):
        name = next(iter(schema.feature_types))
        raise RuntimeError(
            f"stage C env_id={spec.env_id!r}: feature {name!r} declares 'continuous' "
            f"with incompatible input_type {schema.input_type.value!r}; "
            "continuous features require 'float' or 'json' input encoding"
        )


def _draw_from_declared_domain(spec: EnvSpec, name: str, rng: np.random.Generator) -> float:
    """Draw one value for a declared covariate from its IOSchema domain.

    Every name in ``feature_names`` that reaches a numeric query branch must
    have a matching ``(low, high)`` entry in ``feature_domains``; there is no
    default range to fall back to; a spec that omits it fails loudly here,
    at the point stage C actually needs the value, rather than emitting a
    made-up number.
    """
    domain = spec.io_schema.feature_domains.get(name)
    if domain is None:
        raise RuntimeError(
            f"env_id={spec.env_id!r}: IOSchema.feature_domains is missing an entry for "
            f"declared feature_name {name!r}; every feature_name needs a numeric "
            "(low, high) domain before stage C can draw a query for it"
        )
    low, high = domain
    return float(rng.uniform(low, high))


def build_query(spec: EnvSpec, idx: int, *, rng: np.random.Generator) -> str:
    validate_feature_types(spec)
    it = spec.io_schema.input_type
    if it is InputType.float:
        feature_names = list(spec.io_schema.feature_names)
        if len(feature_names) > 1:
            feature_payload: dict[str, float] = {}
            for name in feature_names:
                val = _draw_from_declared_domain(spec, name, rng)
                feature_payload[name] = val
            return json.dumps(feature_payload, ensure_ascii=False)

        if not feature_names:
            raise RuntimeError(
                f"env_id={spec.env_id!r}: InputType.float query has no declared "
                "feature_names; a scalar query needs exactly one covariate name to "
                "look up its domain"
            )
        val = _draw_from_declared_domain(spec, feature_names[0], rng)
        return repr(val)

    if it is InputType.int:
        return str(idx % 10)

    if it is InputType.json:
        json_payload: dict[str, Any] = {}
        feature_names = list(spec.io_schema.feature_names)
        if feature_names:
            for name in feature_names:
                val = _draw_from_declared_domain(spec, name, rng)
                json_payload[name] = val
            return json.dumps(json_payload, ensure_ascii=False)

        fields = list(spec.data_plan.grouping) + list(spec.data_plan.covariates)
        if not fields:
            fields = ["group"]
        for name in fields:
            json_payload[name] = f"{name}_{idx % 4}"
        return json.dumps(json_payload, ensure_ascii=False)

    if it is InputType.none:
        return ""

    if it is InputType.string:
        return str(idx)

    raise ValueError(f"unsupported input_type: {it!r}")
