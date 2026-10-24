from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class CompletionFormatResult:
    ok: bool
    issues: list[str]


class TargetRuntime(ABC):
    """Protocol for a target PPL language supported by the pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical identifier, e.g. 'pymc' or 'lazyppl'."""
        ...

    @abstractmethod
    def extract_code(self, completion_text: str) -> str | None:
        """Extract program source from a raw LLM completion; None if absent."""
        ...

    @abstractmethod
    def replace_code(self, completion_text: str, new_code: str) -> str: ...

    @abstractmethod
    def validate_completion_format(self, completion_text: str) -> CompletionFormatResult:
        """Returns a result with ok flag and issues list (empty on success)."""
        ...

    @abstractmethod
    def validate_syntax(self, code: str) -> ValidationResult:
        """Check syntax and safety without executing the code."""
        ...

    def validate_static(
        self,
        code: str,
        sample_data: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Check syntax/safety with optional task data but without execution."""
        return self.validate_syntax(code)

    @abstractmethod
    def validate_runtime(
        self,
        code: str,
        sample_data: dict[str, Any] | None = None,
        # Wall-clock hang backstop; real compute is bounded by RLIMIT_CPU in
        # the subprocess sandbox. Tight wall budgets misclassify good code as
        # "timeout" on loaded hosts, where interpreter spawn + framework
        # import dominate at low CPU.
        timeout_sec: int = 120,
    ) -> ValidationResult:
        """Execute the code against sample data in a subprocess."""
        ...

    @abstractmethod
    def completion_system_prompt_name(self) -> str:
        """Filename in llm/prompts/ for the completion system prompt."""
        ...

    @abstractmethod
    def repair_system_prompt(
        self, prompt_text: str, broken_code: str, error_trace: str
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for a repair call."""
        ...

    @abstractmethod
    def runtime_version_tag(self) -> str:
        """Version string stored in dataset metadata, e.g. 'v5'."""
        ...

    @abstractmethod
    def metadata_version_key(self) -> str:
        """Key name in metadata, e.g. 'pymc_version'."""
        ...

    @abstractmethod
    def harbor_task_id_prefix(self) -> str:
        """Prefix for generated Harbor task IDs."""
        ...
