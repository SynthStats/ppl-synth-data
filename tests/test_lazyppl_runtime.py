from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ppl_synth_data.runtimes.lazyppl_runtime import (
    LazyPPLRuntime,
    _build_typecheck_taskdata,
    _run_docker_typecheck,
)

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "lazyppl_synthesis" / "seeds"

rt = LazyPPLRuntime()


# --- _build_typecheck_taskdata ---


def test_typecheck_taskdata_infers_double_fields() -> None:
    code = "model rows = do\n  let v = value r\n  let a = age r\n  return ()"
    taskdata = _build_typecheck_taskdata(code)
    assert "value :: Double" in taskdata
    assert "age :: Double" in taskdata
    assert "module TaskData" in taskdata
    assert "Row(..)" in taskdata


def test_typecheck_taskdata_infers_int_fields() -> None:
    code = "foo = count r + nTrials r"
    taskdata = _build_typecheck_taskdata(code)
    assert "count :: Integer" in taskdata
    assert "nTrials :: Int" in taskdata


def test_typecheck_taskdata_uses_stage_c_integer_values_over_name_guess() -> None:
    code = (
        "logLikRow st row =\n"
        "  let yRaw = diet_category row\n"
        "      y = if yRaw < 0 then 0 else if yRaw > 2 then 2 else yRaw\n"
        "  in categoricalLogPmf [0.2, 0.3, 0.5] y\n"
    )
    taskdata = _build_typecheck_taskdata(
        code,
        sample_data={"n": 3, "diet_category": [0, 1, 2]},
    )

    assert "diet_category :: Int" in taskdata
    assert "Row { diet_category = 0 }" in taskdata
    assert "Row { diet_category = 1 }" in taskdata
    assert "loadData = return []" not in taskdata


def test_typecheck_taskdata_infers_bool_fields() -> None:
    code = "if heads r then 1 else 0"
    taskdata = _build_typecheck_taskdata(code)
    assert "heads :: Bool" in taskdata


def test_typecheck_taskdata_uses_fieldless_row_when_no_fields_match() -> None:
    code = "module Submission where\nfoo = 42"
    taskdata = _build_typecheck_taskdata(code)
    assert "_stub" not in taskdata
    assert "data Row = Row deriving (Show)" in taskdata


def test_typecheck_taskdata_mixed_types() -> None:
    code = "bar = value r + fromIntegral (subjectId r)"
    taskdata = _build_typecheck_taskdata(code)
    assert "value :: Double" in taskdata
    assert "subjectId :: Int" in taskdata


def test_typecheck_taskdata_merges_parser_fields_with_known_guesses() -> None:
    code = (
        "logLikRow st row =\n"
        "  let y = followup_systolic_bp row\n"
        "      ageC = age row - 50.0\n"
        "      baseC = baseline_systolic_bp row - 130.0\n"
        "      tr = if treated row then 1.0 else 0.0\n"
        "  in y + ageC + baseC + tr\n"
    )
    taskdata = _build_typecheck_taskdata(code)
    assert "followup_systolic_bp :: Double" in taskdata
    assert "age :: Double" in taskdata
    assert "baseline_systolic_bp :: Double" in taskdata
    assert "treated :: Bool" in taskdata


def test_typecheck_taskdata_has_loaddata() -> None:
    taskdata = _build_typecheck_taskdata("foo = value r")
    assert "loadData :: IO [Row]" in taskdata
    assert "TaskData sample rows were not provided" in taskdata
    assert "loadData = return []" not in taskdata


# --- extract_code edge cases ---


def test_extract_prefers_submit_over_fence() -> None:
    text = (
        "<think>ok</think>\n"
        "```haskell\nfence_code\n```\n"
        '<submit_program language="haskell">\nsubmit_code\n</submit_program>'
    )
    code = rt.extract_code(text)
    assert code is not None
    assert "submit_code" in code


def test_extract_falls_back_to_fence_when_no_submit() -> None:
    text = "<think>ok</think>\n```haskell\nfence_only\n```"
    code = rt.extract_code(text)
    assert code is not None
    assert "fence_only" in code


def test_extract_handles_bare_submit_no_language_attr() -> None:
    text = "<think>ok</think>\n<submit_program>\nbare_code\n</submit_program>"
    code = rt.extract_code(text)
    assert code is not None
    assert "bare_code" in code


# --- replace_code edge cases ---


def test_replace_preserves_think_block() -> None:
    text = (
        "<think>my reasoning</think>\n"
        '<submit_program language="haskell">\nold_code\n</submit_program>'
    )
    replaced = rt.replace_code(text, "new_code")
    assert "<think>my reasoning</think>" in replaced
    assert "new_code" in replaced
    assert "old_code" not in replaced


def test_replace_creates_wrapper_when_no_existing_block() -> None:
    replaced = rt.replace_code("no blocks here", "fresh_code")
    assert "submit_program" in replaced
    assert 'language="haskell"' in replaced
    assert "fresh_code" in replaced
    assert "<think>" in replaced


# Syntax validation runs through the Docker/GHC tests below.


# --- validate_runtime ---


def test_validate_runtime_requires_stage_c_sample_data() -> None:
    code = (
        "module Submission where\n"
        "model :: [Row] -> Meas State\nmodel = undefined\n"
        "logLikRow :: State -> Row -> Double\nlogLikRow = undefined"
    )
    result = rt.validate_runtime(code)
    assert result.ok is False
    assert "requires Stage C sample_data" in result.message


@pytest.mark.narrow_unit_not_acceptance
def test_validate_runtime_requires_docker_for_harbor_scorer() -> None:
    with patch("shutil.which", return_value=None):
        code = (
            "module Submission where\n"
            "model :: [Row] -> Meas State\nmodel = undefined\n"
            "logLikRow :: State -> Row -> Double\nlogLikRow = undefined"
        )
        result = rt.validate_runtime(code, sample_data={"n": 1, "value": [0.0]})
        assert result.ok is False
        assert "does not fall back to syntax-only" in result.message


def test_validate_runtime_routes_through_lazyppl_semantic_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")

    calls: list[dict[str, object]] = []

    def fake_score_lazyppl_semantics(code_arg, **kwargs):
        calls.append({"code": code_arg, **kwargs})
        return SimpleNamespace(ok=True, message="LazyPPL semantic validation passed")

    monkeypatch.setattr(
        "ppl_synth_data.validation.lazyppl_semantics.score_lazyppl_semantics",
        fake_score_lazyppl_semantics,
    )
    code = (
        "module Submission where\n"
        "import LazyPPL\nimport LazyPPL.Distributions\nimport TaskData (Row(..))\n"
        "data State = State { mu :: Double } deriving (Show)\n"
        "model :: [Row] -> Meas State\n"
        "model rows = do\n  m <- sample (normal 0 10)\n  return (State m)\n"
        "logLikRow :: State -> Row -> Double\n"
        "logLikRow (State m) r = normalLogPdf m 1.0 (value r)\n"
    )
    result = rt.validate_runtime(
        code,
        sample_data={"n": 1, "value": [0.0]},
        timeout_sec=13,
    )
    assert result.ok is True
    assert result.message == "LazyPPL semantic validation passed"
    assert calls == [
        {
            "code": code,
            "prompt": "",
            "sample_data": {"n": 1, "value": [0.0]},
            "hidden_data": None,
            "timeout_sec": 13,
        }
    ]


def test_docker_typecheck_workspace_is_traversable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAZYPPL_HARBOR_IMAGE", "lazyppl-harbor:test")

    def fake_run_typecheck_command(cmd: list[str], **_kwargs):
        mount_arg = next(arg for arg in cmd if ":/app/workspace:ro" in arg)
        workspace = Path(mount_arg.split(":", 1)[0])
        assert workspace.stat().st_mode & 0o755 == 0o755
        assert (workspace / "Submission.hs").exists()
        assert (workspace / "TaskData.hs").exists()

        from ppl_synth_data.runtimes.base import ValidationResult

        return ValidationResult(ok=True, message="ok")

    monkeypatch.setattr(
        "ppl_synth_data.runtimes.lazyppl_runtime._run_typecheck_command",
        fake_run_typecheck_command,
    )

    result = _run_docker_typecheck(
        "module Submission where\n"
        "model :: [Row] -> Meas State\nmodel = undefined\n"
        "logLikRow :: State -> Row -> Double\nlogLikRow = undefined\n",
        timeout_sec=1,
    )

    assert result.ok is True


def test_docker_typecheck_uses_stage_c_taskdata_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYPPL_HARBOR_IMAGE", "lazyppl-harbor:test")

    def fake_run_typecheck_command(cmd: list[str], **_kwargs):
        mount_arg = next(arg for arg in cmd if ":/app/workspace:ro" in arg)
        workspace = Path(mount_arg.split(":", 1)[0])
        taskdata = (workspace / "TaskData.hs").read_text(encoding="utf-8")
        assert "diet_category :: Int" in taskdata

        from ppl_synth_data.runtimes.base import ValidationResult

        return ValidationResult(ok=True, message="ok")

    monkeypatch.setattr(
        "ppl_synth_data.runtimes.lazyppl_runtime._run_typecheck_command",
        fake_run_typecheck_command,
    )

    result = _run_docker_typecheck(
        "module Submission where\n"
        "import LazyPPL.Distributions\n"
        "logLikRow st row = categoricalLogPmf [0.2, 0.3, 0.5] (diet_category row)\n",
        timeout_sec=1,
        sample_data={"n": 3, "diet_category": [0, 1, 2]},
    )

    assert result.ok is True


def test_validate_static_uses_stage_c_taskdata_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAZYPPL_HARBOR_IMAGE", "lazyppl-harbor:test")

    def fake_run_typecheck_command(cmd: list[str], **_kwargs):
        mount_arg = next(arg for arg in cmd if ":/app/workspace:ro" in arg)
        workspace = Path(mount_arg.split(":", 1)[0])
        taskdata = (workspace / "TaskData.hs").read_text(encoding="utf-8")
        assert "diet_category :: Int" in taskdata

        from ppl_synth_data.runtimes.base import ValidationResult

        return ValidationResult(ok=True, message="ok")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        "ppl_synth_data.runtimes.lazyppl_runtime._run_typecheck_command",
        fake_run_typecheck_command,
    )

    result = rt.validate_static(
        "module Submission where\n"
        "import LazyPPL.Distributions\n"
        "logLikRow st row = categoricalLogPmf [0.2, 0.3, 0.5] (diet_category row)\n",
        sample_data={"n": 3, "diet_category": [0, 1, 2]},
    )

    assert result.ok is True


# --- dangerous pragma / FFI blocking ---


def test_blocks_template_haskell_pragma() -> None:
    code = (
        "{-# LANGUAGE TemplateHaskell #-}\n"
        "module Submission where\n"
        "model :: a\nmodel = undefined\n"
        "logLikRow :: a\nlogLikRow = undefined\n"
    )
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "TemplateHaskell" in result.message


def test_blocks_quasiquotes_pragma() -> None:
    code = "{-# LANGUAGE QuasiQuotes #-}\nmodule Submission where\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "QuasiQuotes" in result.message


def test_blocks_cpp_pragma() -> None:
    code = "{-# LANGUAGE CPP #-}\nmodule Submission where\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "CPP" in result.message


def test_blocks_unsafe_pragma() -> None:
    code = "{-# LANGUAGE Unsafe #-}\nmodule Submission where\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "Unsafe" in result.message


def test_blocks_template_haskell_in_comma_list() -> None:
    code = "{-# LANGUAGE OverloadedStrings, TemplateHaskell, GADTs #-}\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "TemplateHaskell" in result.message


def test_blocks_options_ghc_pragma() -> None:
    code = "{-# OPTIONS_GHC -pgmF some-preprocessor #-}\nmodule Submission where\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "OPTIONS_GHC" in result.message


def test_blocks_ann_pragma() -> None:
    code = '{-# ANN module "HLint: ignore" #-}\nmodule Submission where\n'
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "ANN" in result.message


def test_blocks_foreign_import() -> None:
    code = 'module Submission where\nforeign import ccall "exp" c_exp :: Double -> Double\n'
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "foreign import" in result.message


def test_blocks_template_haskell_import() -> None:
    code = "module Submission where\nimport Language.Haskell.TH.Syntax\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "Language.Haskell.TH.Syntax" in result.message


def test_blocks_template_haskell_th_import() -> None:
    code = "module Submission where\nimport Language.Haskell.TH\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "Language.Haskell.TH" in result.message


def test_blocks_template_haskell_lib_import() -> None:
    code = "module Submission where\nimport Language.Haskell.TH.Lib\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "Language.Haskell.TH.Lib" in result.message


def test_blocks_template_haskell_quote_import() -> None:
    code = "module Submission where\nimport Language.Haskell.TH.Quote\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "Language.Haskell.TH.Quote" in result.message


def test_blocks_template_haskell_ppr_import() -> None:
    code = "module Submission where\nimport Language.Haskell.TH.Ppr\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "Language.Haskell.TH.Ppr" in result.message


def test_blocks_qualified_template_haskell_import() -> None:
    code = "module Submission where\nimport qualified Language.Haskell.TH as TH\n"
    result = rt.validate_syntax(code)
    assert result.ok is False
    assert "Language.Haskell.TH" in result.message


def test_allows_safe_language_pragmas() -> None:
    code = "{-# LANGUAGE OverloadedStrings #-}\nmodule Submission where\n"
    result = rt.validate_syntax(code)
    if not result.ok:
        assert "blocked pragma" not in result.message
        assert "blocked declaration" not in result.message


@pytest.mark.narrow_unit_not_acceptance
def test_validate_syntax_timeout_not_infra_prefix() -> None:
    import subprocess as _sp

    with patch("shutil.which", return_value=None):
        with patch(
            "subprocess.run",
            side_effect=_sp.TimeoutExpired(cmd=["ghc"], timeout=15),
        ):
            result = rt.validate_syntax("module Submission where")
    assert result.ok is False
    assert not result.message.startswith("INFRA:")
    assert "timeout" in result.message


def test_validate_syntax_uses_docker_after_local_missing_lazyppl(monkeypatch) -> None:
    monkeypatch.setenv("LAZYPPL_HARBOR_IMAGE", "lazyppl-harbor:test")
    code = (
        "module Submission where\n"
        "import LazyPPL\n"
        "import LazyPPL.Distributions\n"
        "import TaskData (Row(..))\n"
        "data State = State { mu :: Double }\n"
        "model :: [Row] -> Meas State\nmodel = undefined\n"
        "logLikRow :: State -> Row -> Double\nlogLikRow = undefined\n"
    )
    local_missing = subprocess.CompletedProcess(
        args=["ghc"],
        returncode=1,
        stdout="",
        stderr="Could not find module 'LazyPPL'",
    )
    docker_ok = subprocess.CompletedProcess(
        args=["docker"],
        returncode=0,
        stdout="",
        stderr="",
    )

    with patch("shutil.which", return_value="/usr/bin/docker"):
        with patch("subprocess.run", side_effect=[local_missing, docker_ok]) as run:
            result = rt.validate_syntax(code)

    assert result.ok is True
    assert run.call_count == 2
    assert run.call_args_list[1].args[0][0] == "docker"


@pytest.mark.narrow_unit_not_acceptance
def test_validate_syntax_reports_infra_when_lazyppl_missing_and_no_docker() -> None:
    code = (
        "module Submission where\n"
        "import LazyPPL\n"
        "model :: [Row] -> Meas State\nmodel = undefined\n"
        "logLikRow :: State -> Row -> Double\nlogLikRow = undefined\n"
    )
    local_missing = subprocess.CompletedProcess(
        args=["ghc"],
        returncode=1,
        stdout="",
        stderr="Could not find module 'LazyPPL'",
    )

    with patch("shutil.which", return_value=None):
        with patch("subprocess.run", return_value=local_missing):
            result = rt.validate_syntax(code)

    assert result.ok is False
    assert result.message.startswith("INFRA:")
    assert "LAZYPPL_HARBOR_IMAGE" in result.message


# --- validate_completion_format on edge cases ---


def test_format_no_code_block_reports_all_missing() -> None:
    result = rt.validate_completion_format("<think>ok</think>\njust text, no code")
    assert not result.ok
    assert len(result.issues) >= 3  # no code block + no model + no logLikRow


# Docker/GHC integration tests execute the real compiler and require Docker.


def _require_docker() -> None:
    if not shutil.which("docker"):
        raise RuntimeError(
            "heavy_lane test requires a real docker binary on PATH; run this suite on "
            "a Linux host with Docker available"
        )


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_docker_typecheck_uses_stage_c_taskdata_types_real_docker() -> None:
    _require_docker()
    result = _run_docker_typecheck(
        "module Submission where\n"
        "import TaskData (Row(..))\n"
        "import LazyPPL.Distributions\n"
        "logLikRow st row = categoricalLogPmf [0.2, 0.3, 0.5] (diet_category row)\n",
        timeout_sec=180,
        sample_data={"n": 3, "diet_category": [0, 1, 2]},
    )
    assert result.ok is True, result.message


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_docker_typecheck_workspace_is_traversable_real_docker() -> None:
    _require_docker()
    result = _run_docker_typecheck(
        "module Submission where\n"
        "import LazyPPL\n"
        "import TaskData (Row(..))\n"
        "data State = State Double\n"
        "model :: [Row] -> Meas State\nmodel = undefined\n"
        "logLikRow :: State -> Row -> Double\nlogLikRow = undefined\n",
        timeout_sec=180,
    )
    assert result.ok is True, result.message


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_validate_static_uses_stage_c_taskdata_types_real_docker() -> None:
    _require_docker()
    result = rt.validate_static(
        "module Submission where\n"
        "import TaskData (Row(..))\n"
        "import LazyPPL.Distributions\n"
        "logLikRow st row = categoricalLogPmf [0.2, 0.3, 0.5] (diet_category row)\n",
        sample_data={"n": 3, "diet_category": [0, 1, 2]},
    )
    assert result.ok is True, result.message


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_validate_syntax_catches_type_error_via_real_docker() -> None:
    """Real-Docker twin of ``test_validate_syntax_catches_type_error`` (row 130).

    Routes through the real lazyppl-harbor GHC typecheck instead of requiring a
    standalone local GHC toolchain, per the manifest's stated alternative
    substrate ("route through the real lazyppl-harbor Docker image").

    ``_run_docker_typecheck`` compiles ``Submission.hs`` standalone (no
    ``Main.hs``), so a signature merely declaring the "wrong" return type
    (``[Row] -> Int`` instead of ``[Row] -> Meas State``) is not by itself a
    type error GHC can see: ``model = undefined`` unifies with any signature.
    The body below is a genuine, self-contained type mismatch (a `String`
    literal where `Int` is required) so the real GHC boundary has something
    concrete to reject, regardless of how the code is later wired into
    Main.hs.
    """
    _require_docker()
    code = (
        "module Submission where\n"
        "import TaskData (Row(..))\n"
        "model :: [Row] -> Int\n"
        'model rows = "not an int"\n'  # genuine self-contained type mismatch
        "logLikRow :: Int -> Row -> Double\n"
        "logLikRow = undefined\n"
    )
    result = _run_docker_typecheck(code, timeout_sec=180)
    assert result.ok is False
    assert result.message.strip() != ""


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_validate_syntax_valid_seed_file_via_real_docker() -> None:
    """Real-Docker twin of ``test_validate_syntax_valid_seed_file`` (row 134)."""
    _require_docker()
    seed_path = SEED_DIR / "simple_normal.hs"
    if not seed_path.exists():
        raise RuntimeError(f"required seed file missing: {seed_path}")
    code = seed_path.read_text(encoding="utf-8")
    result = _run_docker_typecheck(code, timeout_sec=180)
    assert result.ok is True, result.message


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_validate_runtime_routes_through_lazyppl_semantic_scorer_real_docker() -> None:
    _require_docker()
    code = (
        "module Submission where\n"
        "import LazyPPL\nimport LazyPPL.Distributions\nimport TaskData (Row(..))\n"
        "data State = State { mu :: Double } deriving (Show)\n"
        "model :: [Row] -> Meas State\n"
        "model rows = do\n"
        "  m <- sample (normal 0 10)\n"
        "  let state = State m\n"
        "  mapM_ (\\row -> score (exp (logLikRow state row))) rows\n"
        "  return state\n"
        "logLikRow :: State -> Row -> Double\n"
        "logLikRow (State m) r = normalLogPdf m 1.0 (value r)\n"
    )
    result = rt.validate_runtime(
        code,
        sample_data={"n": 3, "value": [0.1, -0.2, 0.05]},
        timeout_sec=600,
    )
    assert result.ok is True, result.message
    assert result.message == "LazyPPL semantic validation passed"


@pytest.mark.heavy_lane
@pytest.mark.real_boundary
def test_validate_syntax_uses_docker_after_local_missing_lazyppl_real_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-Docker twin of ``test_validate_syntax_uses_docker_after_local_missing_lazyppl``
    (row 133).

    Design choice (documented, not a fake): only the *local* GHC precondition
    (a package DB missing LazyPPL) is simulated; that precondition is
    incidental to the routing behaviour under test. The Docker fallback call
    that follows is the real boundary and is NOT mocked: ``subprocess.run`` is
    wrapped so the first invocation (the local ``ghc`` probe) returns a canned
    "Could not find module 'LazyPPL'" result, while every subsequent
    invocation (the ``docker run`` fallback) is forwarded to the real
    ``subprocess.run``.
    """
    _require_docker()
    code = (
        "module Submission where\n"
        "import LazyPPL\n"
        "import LazyPPL.Distributions\n"
        "import TaskData (Row(..))\n"
        "data State = State { mu :: Double }\n"
        "model :: [Row] -> Meas State\nmodel = undefined\n"
        "logLikRow :: State -> Row -> Double\nlogLikRow = undefined\n"
    )
    real_run = subprocess.run
    calls: list[list[str]] = []

    def _wrapped_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "ghc":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="Could not find module 'LazyPPL'",
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _wrapped_run)

    result = rt.validate_syntax(code)

    assert result.ok is True, result.message
    assert len(calls) == 2
    assert calls[0][0] == "ghc"
    assert calls[1][0] == "docker"
