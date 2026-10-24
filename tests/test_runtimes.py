from __future__ import annotations

from unittest.mock import patch

import pytest

from ppl_synth_data import runtime_registry
from ppl_synth_data.runtimes import get_runtime, pymc_completion, pymc_runtime
from ppl_synth_data.runtimes.base import TargetRuntime
from ppl_synth_data.runtimes.lazyppl_runtime import (
    LazyPPLRuntime,
    _check_dangerous_pragmas,
)
from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime


@pytest.mark.parametrize(
    ("runtime_name", "alias", "artifact_key"),
    [
        ("pymc", "pymc", "elpd"),
        ("lazyppl", "haskell", "log_lik_matrix"),
    ],
)
def test_runtime_registry_matches_runtime_implementation(
    runtime_name: str,
    alias: str,
    artifact_key: str,
) -> None:
    runtime = get_runtime(alias)
    runtime_spec = runtime_registry.get_runtime_spec(alias)

    assert runtime_registry.normalize_runtime_name(alias) == runtime_name
    assert runtime_spec.name == runtime_name
    assert runtime.name == runtime_name
    assert runtime.metadata_version_key() == runtime_spec.metadata_version_key
    assert runtime.runtime_version_tag() == runtime_spec.default_version
    assert runtime.harbor_task_id_prefix() == runtime_spec.harbor_task_id_prefix
    assert (
        runtime_registry.score_artifact_key_from_payload(
            {runtime_spec.metadata_version_key: runtime.runtime_version_tag()}
        )
        == artifact_key
    )
    assert (
        runtime_registry.runtime_name_from_payload(
            {"shared_trial_record": {"runtime": {"name": alias}}}
        )
        == runtime_name
    )


def test_get_runtime_pymc() -> None:
    rt = get_runtime("pymc")
    assert isinstance(rt, PyMCRuntime)
    assert rt.name == "pymc"


def test_get_runtime_lazyppl() -> None:
    rt = get_runtime("lazyppl")
    assert isinstance(rt, LazyPPLRuntime)
    assert rt.name == "lazyppl"


def test_get_runtime_haskell_alias_maps_to_lazyppl() -> None:
    rt = get_runtime("haskell")
    assert isinstance(rt, LazyPPLRuntime)
    assert rt.name == "lazyppl"


def test_get_runtime_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown target runtime"):
        get_runtime("fortran")


def test_get_runtime_returns_fresh_instances() -> None:
    assert get_runtime("pymc") is not get_runtime("pymc")
    assert get_runtime("lazyppl") is not get_runtime("lazyppl")


def test_pymc_runtime_uses_runtime_local_parsers() -> None:
    assert pymc_runtime.extract_python_code is pymc_completion.extract_python_code
    assert pymc_runtime.replace_python_code is pymc_completion.replace_python_code
    assert pymc_runtime._pymc_validate_format is pymc_completion.validate_completion_format


class TestPyMCRuntime:
    rt = PyMCRuntime()

    def test_extract_code_submit_tag(self) -> None:
        text = (
            "<think>ok</think>\n<submit_program>\n"
            "import pymc as pm\ndef model(data): pass\n</submit_program>"
        )
        code = self.rt.extract_code(text)
        assert code is not None
        assert "import pymc" in code

    def test_extract_code_python_fence(self) -> None:
        text = "<think>ok</think>\n```python\nimport pymc as pm\ndef model(data): pass\n```"
        code = self.rt.extract_code(text)
        assert code is not None
        assert "import pymc" in code

    def test_extract_code_missing(self) -> None:
        assert self.rt.extract_code("just plain text") is None

    def test_replace_code(self) -> None:
        text = "<think>ok</think>\n<submit_program>\nold_code\n</submit_program>"
        replaced = self.rt.replace_code(text, "new_code")
        assert "new_code" in replaced
        assert "old_code" not in replaced

    def test_validate_format_ok(self) -> None:
        text = "<think>ok</think>\n<submit_program>\ndef model(data): pass\n</submit_program>"
        result = self.rt.validate_completion_format(text)
        assert result.ok
        assert result.issues == []

    def test_validate_format_missing_think(self) -> None:
        text = "<submit_program>\ndef model(data): pass\n</submit_program>"
        result = self.rt.validate_completion_format(text)
        assert not result.ok
        assert any("think" in i for i in result.issues)

    def test_metadata(self) -> None:
        assert self.rt.metadata_version_key() == "pymc_version"
        assert self.rt.runtime_version_tag() == "v5"
        assert self.rt.harbor_task_id_prefix() == "pymc"
        assert self.rt.completion_system_prompt_name() == "completion_gen.md"


_VALID_LAZYPPL = (
    "<think>reasoning</think>\n"
    '<submit_program language="haskell">\n'
    "module Submission where\n"
    "import LazyPPL\n"
    "import LazyPPL.Distributions\n"
    "import TaskData\n"
    "model :: [Row] -> Meas State\n"
    "model = undefined\n"
    "logLikRow :: State -> Row -> Double\n"
    "logLikRow = undefined\n"
    "</submit_program>"
)

_VALID_LAZYPPL_FENCE = (
    "<think>reasoning</think>\n"
    "```haskell\n"
    "module Submission where\n"
    "model :: [Row] -> Meas State\n"
    "model = undefined\n"
    "logLikRow :: State -> Row -> Double\n"
    "logLikRow = undefined\n"
    "```"
)


class TestLazyPPLRuntime:
    rt = LazyPPLRuntime()

    def test_extract_code_submit_tag(self) -> None:
        code = self.rt.extract_code(_VALID_LAZYPPL)
        assert code is not None
        assert "module Submission" in code
        assert "model ::" in code

    def test_extract_code_haskell_fence(self) -> None:
        code = self.rt.extract_code(_VALID_LAZYPPL_FENCE)
        assert code is not None
        assert "model ::" in code

    def test_extract_code_bare_submit_tag(self) -> None:
        text = (
            "<think>ok</think>\n<submit_program>\n"
            "module Submission where\nmodel :: [Row] -> Meas State\n"
            "model = undefined\nlogLikRow :: State -> Row -> Double\n"
            "logLikRow = undefined\n</submit_program>"
        )
        code = self.rt.extract_code(text)
        assert code is not None
        assert "model ::" in code

    def test_extract_code_missing(self) -> None:
        assert self.rt.extract_code("just plain text") is None

    def test_replace_code_submit_tag(self) -> None:
        replaced = self.rt.replace_code(_VALID_LAZYPPL, "module New where\nfoo = 1")
        assert "module New" in replaced
        assert "module Submission" not in replaced

    def test_replace_code_fence(self) -> None:
        replaced = self.rt.replace_code(_VALID_LAZYPPL_FENCE, "module New where\nfoo = 1")
        assert "module New" in replaced

    def test_replace_code_no_existing_block(self) -> None:
        replaced = self.rt.replace_code("plain text", "module X where")
        assert "submit_program" in replaced
        assert 'language="haskell"' in replaced
        assert "module X" in replaced

    def test_validate_format_ok(self) -> None:
        result = self.rt.validate_completion_format(_VALID_LAZYPPL)
        assert result.ok
        assert result.issues == []

    def test_validate_format_missing_think(self) -> None:
        text = (
            '<submit_program language="haskell">\n'
            "model :: [Row] -> Meas State\nmodel = undefined\n"
            "logLikRow :: State -> Row -> Double\nlogLikRow = undefined\n"
            "</submit_program>"
        )
        result = self.rt.validate_completion_format(text)
        assert not result.ok
        assert any("think" in i for i in result.issues)

    def test_validate_format_missing_model_sig(self) -> None:
        text = (
            "<think>ok</think>\n<submit_program>\n"
            "logLikRow :: State -> Row -> Double\nlogLikRow = undefined\n</submit_program>"
        )
        result = self.rt.validate_completion_format(text)
        assert not result.ok
        assert any("model" in i for i in result.issues)

    def test_validate_format_missing_loglikrow_sig(self) -> None:
        text = (
            "<think>ok</think>\n<submit_program>\n"
            "model :: [Row] -> Meas State\nmodel = undefined\n</submit_program>"
        )
        result = self.rt.validate_completion_format(text)
        assert not result.ok
        assert any("logLikRow" in i for i in result.issues)

    def test_validate_format_missing_both_sigs(self) -> None:
        text = "<think>ok</think>\n<submit_program>\nfoo = 42\n</submit_program>"
        result = self.rt.validate_completion_format(text)
        assert not result.ok
        assert len(result.issues) == 2

    def test_validate_syntax_blocked_import(self) -> None:
        code = (
            "module Bad where\nimport System.IO.Unsafe\n"
            "model :: [Row] -> Meas State\nmodel = undefined\n"
            "logLikRow :: State -> Row -> Double\nlogLikRow = undefined"
        )
        result = self.rt.validate_syntax(code)
        assert not result.ok
        assert "blocked import" in result.message

    def test_validate_syntax_blocked_qualified_import(self) -> None:
        code = (
            "module Bad where\nimport qualified System.IO.Unsafe as U\n"
            "model :: [Row] -> Meas State\nmodel = undefined\n"
            "logLikRow :: State -> Row -> Double\nlogLikRow = undefined"
        )
        result = self.rt.validate_syntax(code)
        assert not result.ok
        assert "blocked import" in result.message

    def test_validate_syntax_safe_imports(self) -> None:
        code = (
            "module Submission where\nimport LazyPPL\n"
            "import LazyPPL.Distributions\nimport TaskData\n"
            "model :: [Row] -> Meas State\nmodel = undefined\n"
            "logLikRow :: State -> Row -> Double\nlogLikRow = undefined"
        )
        result = self.rt.validate_syntax(code)
        # GHC may fail on missing TaskData/State; that's fine, it means
        # the blocked-import check passed (no blocked modules).
        # Only fail if the message mentions a blocked import.
        assert "blocked import" not in result.message

    def test_check_dangerous_pragmas_blocks_case_variants(self) -> None:
        assert (
            _check_dangerous_pragmas("{-# LANGUAGE TemplateHaskell #-}")
            == "blocked pragma: LANGUAGE TemplateHaskell"
        )
        assert (
            _check_dangerous_pragmas("{-# language templatehaskell #-}")
            == "blocked pragma: LANGUAGE TemplateHaskell"
        )
        assert _check_dangerous_pragmas("{-# language cpp #-}") == "blocked pragma: LANGUAGE CPP"

    def test_metadata(self) -> None:
        assert self.rt.metadata_version_key() == "lazyppl_version"
        assert self.rt.runtime_version_tag() == "lazyppl-1.0"
        assert self.rt.harbor_task_id_prefix() == "lazyppl"
        assert self.rt.completion_system_prompt_name() == "completion_gen_lazyppl.md"

    def test_repair_prompt_returns_two_strings(self) -> None:
        system, user = self.rt.repair_system_prompt("prompt", "broken", "error")
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert "broken" in user
        assert "error" in user

    @pytest.mark.narrow_unit_not_acceptance
    @patch("subprocess.run", side_effect=FileNotFoundError("ghc"))
    def test_validate_syntax_ghc_absent_returns_failure(self, _mock_run) -> None:
        code = (
            "module Submission where\n"
            "model :: [Row] -> Meas State\nmodel = undefined\n"
            "logLikRow :: State -> Row -> Double\nlogLikRow = undefined"
        )
        result = self.rt.validate_syntax(code)
        assert not result.ok
        assert "ghc not found" in result.message


def test_all_runtimes_implement_full_interface() -> None:
    for name in ("pymc", "lazyppl"):
        rt = get_runtime(name)
        assert isinstance(rt, TargetRuntime)
        assert rt.name == name
        assert isinstance(rt.completion_system_prompt_name(), str)
        assert isinstance(rt.metadata_version_key(), str)
        assert isinstance(rt.runtime_version_tag(), str)
        assert isinstance(rt.harbor_task_id_prefix(), str)
