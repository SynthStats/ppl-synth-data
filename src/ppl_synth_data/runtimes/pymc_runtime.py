from __future__ import annotations

from typing import Any

from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.runtime_registry import get_runtime_spec
from ppl_synth_data.runtimes.base import CompletionFormatResult, TargetRuntime, ValidationResult
from ppl_synth_data.runtimes.pymc_completion import (
    extract_python_code,
    replace_python_code,
)
from ppl_synth_data.runtimes.pymc_completion import (
    validate_completion_format as _pymc_validate_format,
)
from ppl_synth_data.validation.pymc_runtime import runtime_compile_check
from ppl_synth_data.validation.static_code import StaticCodeError, validate_python_safety

_RUNTIME_SPEC = get_runtime_spec("pymc")


class PyMCRuntime(TargetRuntime):
    @property
    def name(self) -> str:
        return str(_RUNTIME_SPEC.name)

    def extract_code(self, completion_text: str) -> str | None:
        return extract_python_code(completion_text)

    def replace_code(self, completion_text: str, new_code: str) -> str:
        return replace_python_code(completion_text, new_code)

    def validate_completion_format(self, completion_text: str) -> CompletionFormatResult:
        ok, issues = _pymc_validate_format(completion_text)
        return CompletionFormatResult(ok=ok, issues=issues)

    def validate_syntax(self, code: str) -> ValidationResult:
        try:
            validate_python_safety(code)
        except (StaticCodeError, SyntaxError) as exc:
            return ValidationResult(ok=False, message=f"{type(exc).__name__}: {exc}")
        return ValidationResult(ok=True, message="ok")

    def validate_runtime(
        self,
        code: str,
        sample_data: dict[str, Any] | None = None,
        timeout_sec: int = 120,
    ) -> ValidationResult:
        return runtime_compile_check(code, timeout_sec=timeout_sec, sample_data=sample_data)

    def completion_system_prompt_name(self) -> str:
        return "completion_gen.md"

    def repair_system_prompt(
        self, prompt_text: str, broken_code: str, error_trace: str
    ) -> tuple[str, str]:
        system = load_prompt_text("repair_completion.md")
        user = (
            f"Original prompt:\n{prompt_text}\n\n"
            f"Broken code:\n{broken_code}\n\n"
            f"Error:\n{error_trace}"
        )
        return system, user

    def runtime_version_tag(self) -> str:
        return str(_RUNTIME_SPEC.default_version)

    def metadata_version_key(self) -> str:
        return str(_RUNTIME_SPEC.metadata_version_key)

    def harbor_task_id_prefix(self) -> str:
        return str(_RUNTIME_SPEC.harbor_task_id_prefix)
