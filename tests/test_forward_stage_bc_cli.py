from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ppl_synth_data.cli import app


def _spec(env_id: str, title: str) -> dict:
    return {
        "env_id": env_id,
        "env_name": title,
        "domain": "synthetic",
        "scenario": {
            "intent": "Observe a binary response to raw indicator and continuous inputs",
            "source_row_sha256": "a" * 64,
        },
        "generative_family": "binary_choice",
        "features": [
            {"name": "indicator", "law": {"dist": "bernoulli", "p": 0.4}},
            {"name": "signal", "law": {"dist": "normal", "mean": 40.0, "sd": 2.0}},
        ],
        "response": {
            "family": "binary_choice",
            "intercept": -1.0,
            "coefficients": {"indicator": 0.5, "signal": 0.02},
            "link": "logit",
        },
        "data_plan": {"n_observations": 20},
        "units": {"signal": "raw units", "y": "indicator"},
        "allowed_hints": ["response family"],
        "forbidden_hints": ["coefficient values"],
    }


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prepare(root: Path, specs: list[dict]) -> tuple[Path, Path, bytes]:
    run_dir = root / "forward-cli"
    source = run_dir / "stage_A_envspec" / "envspecs.jsonl"
    source.parent.mkdir(parents=True)
    payload = "".join(json.dumps(spec) + "\n" for spec in specs).encode()
    source.write_bytes(payload)
    return run_dir, source, payload


def _stage(root: Path, stage: str) -> dict:
    args = ["stage", stage, "--run-id", "forward-cli", "--run-root", str(root)]
    if stage == "c":
        args += ["--split-policy", "random_row", "--visible-ratio", "0.8", "--seed", "137"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    return json.loads(result.stdout)


def _check_modules(run_dir: Path, manifest: dict, count: int) -> None:
    modules = [Path(path) for path in manifest["compiled_modules"]]
    assert len(modules) == count
    assert len(set(modules)) == count
    for module in modules:
        assert module.resolve().parent == (run_dir / "stage_B_envs" / "envs").resolve()
        assert re.fullmatch(r"[a-z][a-z0-9_]*\.py", module.name)
        assert module.is_file()
        compile(module.read_text(), str(module), "exec")


def _check_data(root: Path, run_dir: Path, specs: list[dict]) -> None:
    manifest = _stage(root, "c")
    assert manifest["materialized_env_count"] == len(specs)
    assert manifest["skipped_env_count"] == 0
    refs = _rows(run_dir / "stage_C_data" / "data_refs.jsonl")
    assert {ref["env_id"] for ref in refs} == {spec["env_id"] for spec in specs}
    by_id = {spec["env_id"]: spec for spec in specs}
    for ref in refs:
        assert ref["env_name"] == by_id[ref["env_id"]]["env_name"]
        visible = pd.read_parquet(run_dir / ref["data_ref"])
        hidden = pd.read_parquet(run_dir / ref["hidden_data_ref"])
        assert (len(visible), len(hidden)) == (16, 4)
        combined = pd.concat([visible, hidden], ignore_index=True)
        assert set(combined.columns) == {"indicator", "signal", "y"}
        assert set(combined["indicator"]).issubset({0, 1})
        assert set(combined["y"]).issubset({0, 1})
        assert (combined["signal"] > 9.0).all()


def test_cli_preserves_human_title_and_writes_safe_module(tmp_path: Path) -> None:
    specs = [_spec("forward-title", "Customer choice: A/B (raw units)?")]
    run_dir, source, original = _prepare(tmp_path, specs)

    _manifest = _stage(tmp_path, "b")

    assert source.read_bytes() == original


def test_cli_preserves_unicode_compatibility_title(tmp_path: Path) -> None:
    specs = [_spec("forward-unicode", "K")]
    run_dir, source, original = _prepare(tmp_path, specs)
    manifest = _stage(tmp_path, "b")
    _check_modules(run_dir, manifest, 1)
    _check_data(tmp_path, run_dir, specs)
    assert source.read_bytes() == original
    assert _rows(run_dir / "stage_B_envs" / "envspecs.jsonl") == specs
    _check_modules(run_dir, manifest, 1)
    _check_data(tmp_path, run_dir, specs)
    assert source.read_bytes() == original


def test_cli_preserves_equal_display_names_for_distinct_ids(tmp_path: Path) -> None:
    specs = [_spec("forward-first", "shared_title"), _spec("forward-second", "shared_title")]
    run_dir, source, original = _prepare(tmp_path, specs)

    manifest = _stage(tmp_path, "b")

    assert source.read_bytes() == original
    assert _rows(run_dir / "stage_B_envs" / "envspecs.jsonl") == specs
    _check_modules(run_dir, manifest, 2)
    _check_data(tmp_path, run_dir, specs)
    assert source.read_bytes() == original


def test_cli_records_invalid_coefficients_and_samples_only_compiled_specs(tmp_path: Path) -> None:
    valid = [_spec("forward-before", "valid_before"), _spec("forward-after", "valid_after")]
    invalid = _spec("forward-invalid", "invalid_coefficients")
    invalid["response"]["coefficients"]["unsupported_effect"] = 0.7
    run_dir, source, original = _prepare(tmp_path, [valid[0], invalid, valid[1]])

    manifest = _stage(tmp_path, "b")

    assert source.read_bytes() == original
    assert _rows(run_dir / "stage_B_envs" / "envspecs.jsonl") == valid
    _check_modules(run_dir, manifest, 2)
    refusals = _rows(run_dir / "stage_B_envs" / "refusals.jsonl")
    assert len(refusals) == 1
    refusal = refusals[0]
    assert refusal["env_id"] == "forward-invalid"
    assert {
        "source_row",
        "source_row_sha256",
        "scenario_intent",
        "extractor",
        "refusals",
    } <= refusal.keys()
    assert refusal["source_row_sha256"] == invalid["scenario"]["source_row_sha256"]
    assert refusal["scenario_intent"] == invalid["scenario"]["intent"]
    reasons = refusal["refusals"]
    assert reasons
    assert any(reason["json_pointer"] == "/response/coefficients" for reason in reasons)
    for reason in reasons:
        assert reason["missing_semantics"]
        assert reason["intended_mechanism"]
    assert "unsupported_effect" in json.dumps(reasons)
    _check_data(tmp_path, run_dir, valid)
    assert source.read_bytes() == original
