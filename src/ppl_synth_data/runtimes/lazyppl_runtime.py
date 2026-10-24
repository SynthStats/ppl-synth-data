from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.completion_markup import THINK_RE
from ppl_synth_data.llm.prompt_loader import load_prompt_text
from ppl_synth_data.runtime_registry import get_runtime_spec
from ppl_synth_data.runtimes.base import CompletionFormatResult, TargetRuntime, ValidationResult

log = logging.getLogger(__name__)
_SUBMIT_HASKELL_RE = re.compile(
    r'<submit_program(?:\s+language="haskell")?>\s*\n?(.*?)</submit_program>',
    re.DOTALL,
)
_HASKELL_FENCE_RE = re.compile(r"```haskell\s*\n(.*?)```", re.DOTALL)

_REQUIRED_SIGS = [
    re.compile(r"^\s*model\s*::", re.MULTILINE),
    re.compile(r"^\s*logLikRow\s*::", re.MULTILINE),
]

_GHC_SYNTAX_TIMEOUT = 15  # shorter than full runtime; GHC type-check only
_RUNTIME_SPEC = get_runtime_spec("lazyppl")

_BLOCKED_MODULES = frozenset(
    {
        "System.IO.Unsafe",
        "Unsafe.Coerce",
        "Foreign",
        "Foreign.Ptr",
        "Foreign.C",
        "Network",
        "System.Process",
        "GHC.Exts",
        "Language.Haskell.TH",
        "Language.Haskell.TH.Syntax",
        "Language.Haskell.TH.Lib",
        "Language.Haskell.TH.Quote",
        "Language.Haskell.TH.Ppr",
    }
)

# LANGUAGE pragmas that allow compile-time code execution or bypass safety.
# TemplateHaskell/QuasiQuotes: arbitrary splice execution even under -fno-code.
# CPP: C preprocessor can include arbitrary files.
# Unsafe: opts out of Safe Haskell restrictions.
_BLOCKED_LANGUAGE_PRAGMAS = frozenset(
    {
        "TemplateHaskell",
        "QuasiQuotes",
        "CPP",
        "Unsafe",
    }
)
_BLOCKED_LANGUAGE_PRAGMA_LOOKUP = {
    pragma.casefold(): pragma for pragma in _BLOCKED_LANGUAGE_PRAGMAS
}

# Regexes for dangerous pragma/declaration forms.
_LANGUAGE_PRAGMA_RE = re.compile(r"\{-#\s*LANGUAGE\s+([^#]+)#-\}", re.IGNORECASE)
_OPTIONS_GHC_RE = re.compile(r"\{-#\s*OPTIONS_GHC\b", re.IGNORECASE)
_ANN_PRAGMA_RE = re.compile(r"\{-#\s*ANN\b", re.IGNORECASE)
_FOREIGN_IMPORT_RE = re.compile(r"^\s*foreign\s+import\b", re.MULTILINE)

# The held-out rows live in this module inside the scorer container. A
# submission has no honest reason to name it, so any reference is rejected
# outright (import, qualified import, or a bare qualified name in an
# expression). The scorer additionally keeps the module off the submission's
# module path until Submission.hs has type-checked; this token ban is the
# independent first layer.
_HIDDEN_DATA_MODULE = "HiddenTaskData"


def _check_dangerous_pragmas(code: str) -> str | None:
    for m in _LANGUAGE_PRAGMA_RE.finditer(code):
        extensions = [ext.strip() for ext in m.group(1).split(",")]
        for ext in extensions:
            blocked = _BLOCKED_LANGUAGE_PRAGMA_LOOKUP.get(ext.casefold())
            if blocked is not None:
                return f"blocked pragma: LANGUAGE {blocked}"
    if _OPTIONS_GHC_RE.search(code):
        return "blocked pragma: OPTIONS_GHC"
    if _ANN_PRAGMA_RE.search(code):
        return "blocked pragma: ANN"
    if _FOREIGN_IMPORT_RE.search(code):
        return "blocked declaration: foreign import"
    return None


_FIELD_TYPE_GUESSES: dict[str, str] = {
    "value": "Double",
    "xVal": "Double",
    "yVal": "Double",
    "x1Val": "Double",
    "x2Val": "Double",
    "yLength": "Double",
    "age": "Double",
    "duration": "Double",
    "time": "Double",
    "concentration": "Double",
    "rating": "Double",
    "rt": "Double",
    "measurement": "Double",
    "count": "Integer",
    "nTrials": "Int",
    "nSuccesses": "Int",
    "attemptCount": "Int",
    "subjectId": "Int",
    "timeIdx": "Int",
    "groupId": "Int",
    "heads": "Bool",
    "label": "Bool",
    "isWeekday": "Bool",
}


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _sample_values(sample_data: dict[str, Any] | None, field_name: str) -> list[Any]:
    if sample_data is None or field_name not in sample_data:
        return []
    value = sample_data[field_name]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        return []
    return list(value)


def _rows_from_sample_data(
    columns: list[dict[str, str]],
    sample_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if sample_data is None or not columns:
        return []

    values_by_field: dict[str, list[Any]] = {}
    for column in columns:
        values = _sample_values(sample_data, column["name"])
        if not values:
            return []
        values_by_field[column["name"]] = values

    row_count = min(len(values) for values in values_by_field.values())
    declared_n = _python_scalar(sample_data.get("n"))
    if isinstance(declared_n, Integral) and not isinstance(declared_n, bool):
        row_count = min(row_count, int(declared_n))
    if row_count <= 0:
        return []

    return [
        {
            column["name"]: _python_scalar(values_by_field[column["name"]][index])
            for column in columns
        }
        for index in range(row_count)
    ]


def _build_fail_loud_typecheck_taskdata(fields: dict[str, str]) -> str:
    lines = [
        "module TaskData",
        "  ( Row(..)",
        "  , loadData",
        "  ) where",
        "",
        "import System.IO.Error (ioError, userError)",
        "",
    ]
    if not fields:
        lines.append("data Row = Row deriving (Show)")
    else:
        lines.append("data Row = Row")
        for index, (name, hs_type) in enumerate(fields.items()):
            prefix = "  { " if index == 0 else "  , "
            lines.append(f"{prefix}{name} :: {hs_type}")
        lines.append("  } deriving (Show)")
    lines.extend(
        [
            "",
            "loadData :: IO [Row]",
            "loadData =",
            "  ioError",
            '    (userError "TaskData sample rows were not provided for LazyPPL typechecking.")',
            "",
        ]
    )
    return "\n".join(lines)


def _hs_type_from_sample_values(
    *,
    field_name: str,
    values: list[Any],
    fallback: str,
) -> str:
    observed = [_python_scalar(value) for value in values if _python_scalar(value) is not None]
    if not observed:
        return fallback
    if all(isinstance(value, bool) for value in observed):
        return "Bool"
    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in observed):
        if field_name == "count" or field_name.endswith("_count"):
            return "Integer"
        return "Int"
    if all(isinstance(value, Real) and not isinstance(value, bool) for value in observed):
        return "Double"
    return fallback


def _build_typecheck_taskdata(
    code: str,
    *,
    sample_data: dict[str, Any] | None = None,
) -> str:
    fields: dict[str, str] = {}
    try:
        from ppl_synth_data.harbor.taskdata_gen import parse_seed_row_fields

        parsed = parse_seed_row_fields(code)
        for col in parsed:
            field_name = col["name"]
            fields[field_name] = _hs_type_from_sample_values(
                field_name=field_name,
                values=_sample_values(sample_data, field_name),
                fallback=col["hs_type"],
            )
    except (ImportError, ValueError, KeyError, TypeError, re.error):
        log.debug(
            "taskdata_gen.parse_seed_row_fields failed, using field guesses",
            exc_info=True,
        )

    for word, typ in _FIELD_TYPE_GUESSES.items():
        if re.search(rf"\b{word}\b", code):
            fields[word] = _hs_type_from_sample_values(
                field_name=word,
                values=_sample_values(sample_data, word),
                fallback=typ,
            )

    if not fields:
        return _build_fail_loud_typecheck_taskdata(fields)

    columns = [{"name": field_name, "hs_type": hs_type} for field_name, hs_type in fields.items()]
    rows = _rows_from_sample_data(columns, sample_data)
    if not rows:
        return _build_fail_loud_typecheck_taskdata(fields)

    from ppl_synth_data.harbor.taskdata_gen import generate_taskdata_hs

    return generate_taskdata_hs(columns, rows)


def _workspace_parent_for_docker() -> Path | None:
    # colima (macOS Docker) only shares /Users/; default tmpdir is /var/folders
    home = Path.home()
    return home if str(home).startswith("/Users") else None


def _write_validation_workspace(
    workspace: Path,
    code: str,
    *,
    sample_data: dict[str, Any] | None = None,
) -> Path:
    submission_path = workspace / "Submission.hs"
    submission_path.write_text(code, encoding="utf-8")
    (workspace / "TaskData.hs").write_text(
        _build_typecheck_taskdata(code, sample_data=sample_data),
        encoding="utf-8",
    )
    return submission_path


def _run_typecheck_command(
    cmd: list[str],
    *,
    timeout_sec: int,
    missing_binary_message: str,
    timeout_message: str,
) -> ValidationResult:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        return ValidationResult(ok=False, message=missing_binary_message)
    except subprocess.TimeoutExpired:
        return ValidationResult(ok=False, message=timeout_message)

    if proc.returncode != 0:
        return ValidationResult(ok=False, message=proc.stderr.strip())
    return ValidationResult(ok=True, message="ok")


def _is_missing_lazyppl_module(message: str) -> bool:
    lowered = message.lower()
    return (
        "could not find module" in lowered or "could not load module" in lowered
    ) and "lazyppl" in lowered


def _run_docker_typecheck(
    code: str,
    *,
    timeout_sec: int,
    sample_data: dict[str, Any] | None = None,
) -> ValidationResult:
    image = Settings().lazyppl_harbor_image
    if not image:
        return ValidationResult(
            ok=False,
            message=(
                "INFRA: LAZYPPL_HARBOR_IMAGE is not set; set it to an image built from "
                "docker/lazyppl-harbor"
            ),
        )
    with tempfile.TemporaryDirectory(dir=_workspace_parent_for_docker()) as td:
        workspace = Path(td)
        _write_validation_workspace(workspace, code, sample_data=sample_data)
        # tempfile creates 0700 directories. The Harbor image runs GHC as a
        # non-root user, so the bind-mounted workspace must be traversable.
        workspace.chmod(0o755)
        return _run_typecheck_command(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--memory=512m",
                "--cpus=2",
                "-v",
                f"{workspace}:/app/workspace:ro",
                image,
                "ghc",
                "-fno-code",
                "-i/app/workspace",
                "/app/workspace/Submission.hs",
            ],
            timeout_sec=timeout_sec,
            missing_binary_message="docker not found",
            timeout_message="timeout: ghc type-check exceeded time limit (Docker)",
        )


def static_safety_error(code: str) -> str | None:
    """Return a reason to reject ``code`` before it reaches GHC, or ``None``.

    Shared by the runtime's validate_syntax/validate_static path and by the
    Harbor ELPD scorer (``validation.lazyppl_semantics``), so every road to a
    LazyPPL reward passes the same static gate.
    """
    if _HIDDEN_DATA_MODULE in code:
        return f"blocked reference: {_HIDDEN_DATA_MODULE} (held-out data module)"
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "import\t")):
            parts = stripped.split()
            # handles: import M, import qualified M
            idx = 2 if len(parts) > 2 and parts[1] == "qualified" else 1
            mod = parts[idx].split("(")[0] if idx < len(parts) else ""
            if mod in _BLOCKED_MODULES:
                return f"blocked import: {mod}"

    return _check_dangerous_pragmas(code)


def static_safety_grep_pattern() -> str:
    """One POSIX ERE (for ``grep -E``) equivalent to :func:`static_safety_error`.

    Rendered into the Harbor ``test.sh`` so the in-container scorer path, which
    has no Python package available, applies the same gate as the host. Uses
    ``[[:space:]]`` rather than ``\\s`` so BSD and GNU grep agree, and contains
    no single quote so it can sit inside a single-quoted shell word.
    """
    modules = "|".join(re.escape(m) for m in sorted(_BLOCKED_MODULES, key=len, reverse=True))
    pragmas = "|".join(sorted(_BLOCKED_LANGUAGE_PRAGMAS))
    alternatives = [
        rf"^[[:space:]]*import[[:space:]]+(qualified[[:space:]]+)?({modules})([[:space:]]|\(|$)",
        r"\{-#[[:space:]]*(OPTIONS_GHC|ANN)([[:space:]]|#|-)",
        rf"\{{-#[[:space:]]*LANGUAGE[^#]*({pragmas})",
        r"^[[:space:]]*foreign[[:space:]]+import([[:space:]]|$)",
        _HIDDEN_DATA_MODULE,
    ]
    pattern = "|".join(alternatives)
    assert "'" not in pattern
    return pattern


class LazyPPLRuntime(TargetRuntime):
    @property
    def name(self) -> str:
        return str(_RUNTIME_SPEC.name)

    def extract_code(self, completion_text: str) -> str | None:
        match = _SUBMIT_HASKELL_RE.search(completion_text)
        if match:
            return match.group(1).strip()
        match = _HASKELL_FENCE_RE.search(completion_text)
        if match:
            return match.group(1).strip()
        return None

    def replace_code(self, completion_text: str, new_code: str) -> str:
        normalized = new_code.rstrip() + "\n"

        match = _SUBMIT_HASKELL_RE.search(completion_text)
        if match:
            start, end = match.span(1)
            return completion_text[:start] + "\n" + normalized + completion_text[end:]

        match = _HASKELL_FENCE_RE.search(completion_text)
        if match:
            start, end = match.span(1)
            return completion_text[:start] + normalized + completion_text[end:]

        think = (
            "<think>\n"
            "Repaired program chosen to satisfy compilation and type constraints.\n"
            "</think>\n\n"
        )
        return f'{think}<submit_program language="haskell">\n{normalized}</submit_program>'

    def validate_completion_format(self, completion_text: str) -> CompletionFormatResult:
        issues: list[str] = []
        if not THINK_RE.search(completion_text):
            issues.append("missing <think> block")
        code = self.extract_code(completion_text)
        if code is None:
            issues.append("missing code block (no <submit_program> or haskell fence)")
            issues.append("missing model :: [Row] -> Meas State signature")
            issues.append("missing logLikRow :: State -> Row -> Double signature")
        else:
            for sig_re in _REQUIRED_SIGS:
                if not sig_re.search(code):
                    name = "model" if "model" in sig_re.pattern else "logLikRow"
                    issues.append(f"missing {name} type signature")
        return CompletionFormatResult(ok=len(issues) == 0, issues=issues)

    def validate_syntax(self, code: str) -> ValidationResult:
        pragma_err = static_safety_error(code)
        if pragma_err:
            return ValidationResult(ok=False, message=pragma_err)

        with tempfile.TemporaryDirectory() as td:
            submission_path = _write_validation_workspace(Path(td), code)
            local_result = _run_typecheck_command(
                ["ghc", "-fno-code", "-i" + td, str(submission_path)],
                timeout_sec=_GHC_SYNTAX_TIMEOUT,
                missing_binary_message=(
                    "INFRA: ghc not found; install GHC 9.6+ or use Docker-based validation"
                ),
                timeout_message="timeout: ghc type-check exceeded time limit",
            )
        if local_result.ok or not _is_missing_lazyppl_module(local_result.message):
            return local_result
        if not shutil.which("docker"):
            return ValidationResult(
                ok=False,
                message=(
                    "INFRA: local GHC cannot resolve LazyPPL modules and Docker "
                    "validation is unavailable; install LazyPPL for local GHC or set "
                    "LAZYPPL_HARBOR_IMAGE to a usable LazyPPL Harbor image"
                ),
            )
        return _run_docker_typecheck(code, timeout_sec=_GHC_SYNTAX_TIMEOUT)

    def validate_static(
        self,
        code: str,
        sample_data: dict[str, Any] | None = None,
    ) -> ValidationResult:
        pragma_err = static_safety_error(code)
        if pragma_err:
            return ValidationResult(ok=False, message=pragma_err)
        if not shutil.which("docker"):
            return self.validate_syntax(code)
        return _run_docker_typecheck(
            code,
            timeout_sec=_GHC_SYNTAX_TIMEOUT,
            sample_data=sample_data,
        )

    def validate_runtime(
        self,
        code: str,
        sample_data: dict[str, Any] | None = None,
        timeout_sec: int = 120,
    ) -> ValidationResult:
        """Execute the LazyPPL Harbor scorer against real Stage C data."""
        if sample_data is None:
            return ValidationResult(
                ok=False,
                message=(
                    "LazyPPL runtime validation requires Stage C sample_data; "
                    "use validate_static only for compile-only checks"
                ),
            )
        if not shutil.which("docker"):
            return ValidationResult(
                ok=False,
                message=(
                    "INFRA: docker not found; LazyPPL runtime validation requires "
                    "the Harbor semantic scorer and does not fall back to syntax-only"
                ),
            )

        from ppl_synth_data.validation.lazyppl_semantics import score_lazyppl_semantics

        try:
            result = score_lazyppl_semantics(
                code,
                prompt="",
                sample_data=sample_data,
                hidden_data=None,
                timeout_sec=timeout_sec,
            )
        except FileNotFoundError as exc:
            return ValidationResult(
                ok=False,
                message=f"INFRA: LazyPPL Harbor runtime validation failed: {exc}",
            )
        return ValidationResult(ok=result.ok, message=result.message)

    def completion_system_prompt_name(self) -> str:
        return "completion_gen_lazyppl.md"

    def repair_system_prompt(
        self, prompt_text: str, broken_code: str, error_trace: str
    ) -> tuple[str, str]:
        system = load_prompt_text("repair_completion_lazyppl.md")
        user = (
            f"Original prompt:\n{prompt_text}\n\n"
            f"Broken code:\n{broken_code}\n\n"
            f"GHC error:\n{error_trace}"
        )
        return system, user

    def runtime_version_tag(self) -> str:
        return str(_RUNTIME_SPEC.default_version)

    def metadata_version_key(self) -> str:
        return str(_RUNTIME_SPEC.metadata_version_key)

    def harbor_task_id_prefix(self) -> str:
        return str(_RUNTIME_SPEC.harbor_task_id_prefix)
