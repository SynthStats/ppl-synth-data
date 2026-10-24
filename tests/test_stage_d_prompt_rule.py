"""Request-boundary checks only; captured replies never become training data."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ppl_synth_data.llm.backend import LLMRequest
from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.pipeline.artifacts import resolve_run_paths
from ppl_synth_data.pipeline.config_defaults import (
    effective_reasoning_effort,
    load_default_pipeline_defaults,
)
from ppl_synth_data.pipeline.io import read_jsonl, write_jsonl
from ppl_synth_data.pipeline.stages.stage_d_prompts import run_stage_d
from ppl_synth_data.runtimes import get_runtime
from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec, PromptRecord
from tests.pipeline.helpers import CaptureBackend, minimal_envspec_record

RULE = "no-numeric-summaries-v1"
MODEL = "request-capture-only"
CAPTURE_REPLY = "Request capture only; never use this reply as training data."


def declared_spec():
    return FourFamilySpec.model_validate(
        {
            "env_id": "request-capture",
            "env_name": "Request capture",
            "domain": "ecology",
            "scenario": {
                "intent": "Study how cover varies with exposure.",
                "source_row_sha256": "1234567890abcdef" * 4,
            },
            "generative_family": "regression_continuous",
            "features": [
                {
                    "name": "exposure",
                    "units": "hours",
                    "law": {
                        "dist": "truncated_normal",
                        "mean": 123.456,
                        "sd": 78.9,
                        "low": 0.12,
                        "high": 987.654,
                    },
                },
                {"name": "treated", "law": {"dist": "bernoulli", "p": 0.35}},
            ],
            "response": {
                "family": "regression_continuous",
                "intercept": 246.8,
                "coefficients": {"exposure": -135.79, "treated": 975.31},
                "noise_sigma": 864.2,
            },
            "data_plan": {"n_observations": 4321},
            "units": {"y": "percent"},
            "allowed_hints": ["A numerical hint: 765.43"],
            "forbidden_hints": ["Parameter value 654.32"],
        }
    )


def prepare_run(tmp_path, spec):
    paths = resolve_run_paths(tmp_path, "capture")
    paths.stage_dirs["B"].mkdir(parents=True)
    paths.stage_dirs["C"].mkdir()
    summary = {"env_id": spec.env_id, "n": 4321, "mean": 567.89}
    write_jsonl(paths.stage_dirs["B"] / "envspecs.jsonl", [spec.model_dump(mode="json")])
    write_jsonl(paths.stage_dirs["C"] / "summaries.jsonl", [summary])
    return paths, summary


def capture(paths, **kwargs):
    backend = CaptureBackend([CAPTURE_REPLY])
    run_stage_d(
        paths,
        backend=backend,
        backend_name="openai",
        model=MODEL,
        runtime=get_runtime("pymc"),
        use_batch=False,
        **kwargs,
    )
    return backend


def old_request(spec, summary):
    """The request construction from Stage D before the rule keyword existed."""
    defaults = load_default_pipeline_defaults().stages.prompts
    spec_json = json.dumps(spec.model_dump(mode="json"), ensure_ascii=False)
    user = (
        f"Environment specification JSON:\n{spec_json}\n\n"
        f"Data summary JSON:\n{json.dumps(summary, ensure_ascii=False)}\n"
    )
    return LLMRequest(
        model=MODEL,
        system=load_prompt_text("prompt_gen.md"),
        user=user,
        reasoning_effort=effective_reasoning_effort(defaults.reasoning_effort),
        max_output_tokens=defaults.max_output_tokens,
    )


@pytest.mark.parametrize("typed", [True, False])
def test_default_request_is_identical_to_old_code_path(tmp_path, typed):
    spec = (
        declared_spec()
        if typed
        else EnvSpec.model_validate(
            minimal_envspec_record(env_id="request-capture", env_name="Request capture")
        )
    )
    paths, summary = prepare_run(tmp_path, spec)
    expected = old_request(spec, summary)
    assert capture(paths, prompt_rule="default").requests == [expected]
    assert capture(paths).requests == [expected]


def test_no_numeric_rule_sends_only_allowed_spec_fields(tmp_path):
    spec = declared_spec()
    paths, _ = prepare_run(tmp_path, spec)
    backend = capture(paths, prompt_rule=RULE)
    user = backend.requests[0].user
    assert "Data summary" not in user
    # The fixture uses digit-free allowed names and intent. Every parameter digit
    # (including those in the digest and hints) must disappear from this request.
    assert not any(character.isdigit() for character in user)
    assert json.loads(user.split("Environment specification JSON:\n", 1)[1]) == {
        "env_id": "request-capture",
        "env_name": "Request capture",
        "domain": "ecology",
        "scenario": {"intent": "Study how cover varies with exposure."},
        "generative_family": "regression_continuous",
        "features": [
            {"name": "exposure", "units": "hours", "law": {"dist": "truncated_normal"}},
            {"name": "treated", "law": {"dist": "bernoulli"}},
        ],
        "response": {"family": "regression_continuous"},
    }


@pytest.mark.parametrize(
    "family,extra,allowed",
    [
        (
            "regression_count",
            {"intercept_log_rate": 1.23, "count_distribution": "nb2", "nb2_alpha": 4.56},
            {"count_distribution": "nb2"},
        ),
        (
            "zero_inflated_count",
            {"intercept_log_rate": 1.23, "count_distribution": "poisson", "zero_inflation": 0.67},
            {"count_distribution": "poisson"},
        ),
        ("binary_choice", {"intercept": 1.23, "link": "logit"}, {"link": "logit"}),
    ],
)
def test_redaction_preserves_declared_response_kind(tmp_path, family, extra, allowed):
    raw = declared_spec().model_dump(mode="json")
    raw["generative_family"] = family
    raw["response"] = {
        "family": family,
        "coefficients": {"exposure": 2.34, "treated": 5.67},
        **extra,
    }
    paths, _ = prepare_run(tmp_path, FourFamilySpec.model_validate(raw))
    user = capture(paths, prompt_rule=RULE).requests[0].user
    sent = json.loads(user.split("Environment specification JSON:\n", 1)[1])
    assert sent["response"] == {"family": family, **allowed}


def test_no_numeric_rule_uses_new_system_prompt(tmp_path):
    paths, _ = prepare_run(tmp_path, declared_spec())
    system = capture(paths, prompt_rule=RULE).requests[0].system
    assert system != load_prompt_text("prompt_gen.md")
    assert system == load_prompt_text("prompt_gen_no_numbers.md")


def test_no_numeric_rule_requires_declared_spec_before_request(tmp_path):
    spec = EnvSpec.model_validate(
        minimal_envspec_record(env_id="request-capture", env_name="Capture")
    )
    paths, _ = prepare_run(tmp_path, spec)
    backend = CaptureBackend([CAPTURE_REPLY])
    with pytest.raises(ValueError, match="requires FourFamilySpec"):
        run_stage_d(
            paths,
            backend=backend,
            backend_name="openai",
            model=MODEL,
            runtime=get_runtime("pymc"),
            use_batch=False,
            prompt_rule=RULE,
        )
    assert backend.requests == []


def test_saved_prompt_record_carries_rule_in_metadata(tmp_path):
    spec = declared_spec()
    paths, _ = prepare_run(tmp_path, spec)
    capture(paths, prompt_rule=RULE)
    rows = read_jsonl(paths.stage_dirs["D"] / "prompts.jsonl")
    assert len(rows) == 1
    record = PromptRecord.model_validate(rows[0])
    assert record.metadata["prompt_rule"] == RULE
    assert record.metadata["scenario"] == spec.scenario.model_dump(mode="json")
    assert record.metadata["data_origin"] == "regenerated"


@pytest.mark.parametrize("rule", ["unknown", "", "NO-NUMERIC-SUMMARIES-V1", None])
def test_unknown_rule_raises_before_any_request_or_output(tmp_path, rule):
    paths, _ = prepare_run(tmp_path, declared_spec())
    backend = CaptureBackend([CAPTURE_REPLY])
    with pytest.raises(ValueError, match="prompt_rule"):
        run_stage_d(
            paths,
            backend=backend,
            backend_name="openai",
            model=MODEL,
            runtime=get_runtime("pymc"),
            use_batch=False,
            prompt_rule=rule,
        )
    assert backend.requests == []
    assert not paths.stage_dirs["D"].exists()


@pytest.mark.heavy_lane(
    reason="Requires prompts, receipts, and Stage B specs in artifacts/reference"
)
def test_resume_does_not_reuse_default_prose_for_new_rule(tmp_path):
    source = Path(__file__).resolve().parents[1] / "artifacts" / "reference"
    previous = read_jsonl(source / "stage_D_prompts/prompts.jsonl")[0]
    spec_row = next(
        row
        for row in read_jsonl(source / "stage_B_envs/envspecs.jsonl")
        if row["env_id"] == previous["env_id"]
    )
    paths, _ = prepare_run(tmp_path, FourFamilySpec.model_validate(spec_row))
    paths.stage_dirs["D"].mkdir()
    write_jsonl(paths.stage_dirs["D"] / "prompts.jsonl", [previous])
    receipts = [
        row
        for row in read_jsonl(source / "stage_D_prompts/receipts.jsonl")
        if row["env_id"] == previous["env_id"]
    ]
    assert receipts
    write_jsonl(paths.stage_dirs["D"] / "receipts.jsonl", receipts)
    # A positive control proves that the old prompt really is resumable.
    assert capture(paths, prompt_rule="default", resume=True).requests == []
    backend = capture(paths, prompt_rule=RULE, resume=True)
    assert len(backend.requests) == 1
    saved = read_jsonl(paths.stage_dirs["D"] / "prompts.jsonl")
    assert len(saved) == 1
    assert saved[0]["metadata"]["prompt_rule"] == RULE


def load_resume_cli():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "tools" / "resume_stage_d.py"
    module_spec = importlib.util.spec_from_file_location("resume_cli", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rule_args", [[], ["--prompt-rule", "unknown"]])
@pytest.mark.heavy_lane(reason="Requires the Stage D resume CLI wrapper")
def test_resume_cli_requires_valid_prompt_rule(tmp_path, monkeypatch, rule_args):
    cli = load_resume_cli()

    def refuse_call(*args, **kwargs):
        pytest.fail("CLI must reject arguments before running Stage D")

    monkeypatch.setattr(cli, "run_stage_d", refuse_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(cli.__file__),
            "--run-root",
            str(tmp_path),
            "--run-id",
            "capture",
            "--backend",
            "bedrock_converse",
            "--model",
            MODEL,
            "--runtime",
            "pymc",
            *rule_args,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("rule", ["default", RULE])
@pytest.mark.heavy_lane(reason="Requires the Stage D resume CLI wrapper")
def test_resume_cli_passes_prompt_rule(tmp_path, monkeypatch, rule):
    paths, _ = prepare_run(tmp_path, declared_spec())
    cli = load_resume_cli()
    backend = CaptureBackend([CAPTURE_REPLY])

    def local_stage(paths, **kwargs):
        assert kwargs["prompt_rule"] == rule
        assert kwargs["resume"] is True
        assert kwargs["use_batch"] is False
        assert kwargs["backend_name"] == "bedrock_converse"
        # Exercise the real stage with the recording backend, without provider
        # receipts or a provider call. The captured reply stays in tmp_path.
        kwargs["backend_name"] = "openai"
        return run_stage_d(paths, backend=backend, **kwargs)

    monkeypatch.setattr(cli, "run_stage_d", local_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(cli.__file__),
            "--run-root",
            str(paths.run_root),
            "--run-id",
            paths.run_id,
            "--backend",
            "bedrock_converse",
            "--model",
            MODEL,
            "--runtime",
            "pymc",
            "--prompt-rule",
            rule,
        ],
    )
    cli.main()
    assert len(backend.requests) == 1
