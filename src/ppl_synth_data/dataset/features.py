from __future__ import annotations

from collections import OrderedDict

from datasets import Features, Sequence, Value


def _base_fields(runtime_name: str = "pymc") -> OrderedDict[str, Value | Sequence]:
    return OrderedDict(
        {
            "id": Value("string"),
            "prompt_id": Value("string"),
            "env_id": Value("string"),
            "split": Value("string"),
            "prompt": Value("string"),
            "completion": Value("string"),
            "model_type": Value("string"),
            "domain": Value("string"),
            "complexity": Value("string"),
            "key_concepts": Sequence(Value("string")),
            "has_observed_data": Value("bool"),
            f"{runtime_name}_version": Value("string"),
            "trace_id": Value("string"),
            "provenance_json": Value("string"),
            "candidate_outputs_json": Value("string"),
        }
    )


def sft_features(runtime_name: str = "pymc") -> Features:
    return Features(_base_fields(runtime_name))


def with_data_features(runtime_name: str = "pymc") -> Features:
    fields = _base_fields(runtime_name)
    fields.update(
        {
            "env_name": Value("string"),
            "env_spec_json": Value("string"),
            "data_ref": Value("string"),
            "parsed_program": Value("string"),
            "seed": Value("int64"),
            "generator_version": Value("string"),
            "trace_json": Value("string"),
            "validation_json": Value("string"),
        }
    )
    return Features(fields)


def pymc_sft_features() -> Features:
    return sft_features("pymc")


def pymc_with_data_features() -> Features:
    return with_data_features("pymc")
