from __future__ import annotations

import pytest

from ppl_synth_data.validation.static_code import (
    BLOCKED_CALLS,
    BLOCKED_DUNDERS,
    BLOCKED_MODULES,
    BLOCKED_NAMES,
    StaticCodeError,
    validate_python_safety,
)


def test_blocks_os_import() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("import os\n")


def test_allows_simple_pymc_code() -> None:
    code = "def model(data):\n    return data\n"
    validate_python_safety(code)


def test_blocks_builtins_eval_attr_call() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("import builtins\nbuiltins.eval('1+1')\n")


def test_blocks_reflective_eval_alias() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("f = getattr(__builtins__, 'eval')\nf('1+1')\n")


def test_blocks_dunder_attribute_escape() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("x = (1).__class__.__mro__\n")


def test_blocks_eval_in_list_indirection() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("func_list = [eval]; func_list[0]('1')\n")


def test_blocks_bare_eval_assignment() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("x = eval\n")


def test_blocks_bare_exec_assignment() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("x = exec\n")


def test_blocks_bare_open_reference() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("f = open\n")


def test_blocks_module_alias_attribute_call() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("import os as O\nO.system('ls')\n")


def test_blocks_from_import_aliased_call() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("from os import system as s\ns('ls')\n")


def test_blocks_signal_import() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("import signal\n")


def test_blocks_getattr_call() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("x = getattr(obj, 'method')\n")


def test_blocks_breakpoint_call() -> None:
    with pytest.raises(StaticCodeError):
        validate_python_safety("breakpoint()\n")


def test_bare_except_rejected() -> None:
    with pytest.raises(StaticCodeError, match="bare except"):
        validate_python_safety("try:\n    pass\nexcept:\n    pass\n")


def test_except_base_exception_rejected() -> None:
    with pytest.raises(StaticCodeError, match="except BaseException"):
        validate_python_safety("try:\n    pass\nexcept BaseException:\n    pass\n")


def test_specific_exception_allowed() -> None:
    validate_python_safety("try:\n    pass\nexcept ValueError:\n    pass\n")


def test_blocklists_match_harbor_validator() -> None:
    """Ensure static_code.py blocklists stay in sync with validator_template.py."""
    from ppl_synth_data.harbor.validator_template import (
        _BLOCKED_CALLS,
        _BLOCKED_DUNDERS,
        _BLOCKED_MODULES,
        _BLOCKED_NAMES,
    )

    assert BLOCKED_MODULES == _BLOCKED_MODULES, (
        f"BLOCKED_MODULES diverged.\n"
        f"  only in static_code: {BLOCKED_MODULES - _BLOCKED_MODULES}\n"
        f"  only in validator:   {_BLOCKED_MODULES - BLOCKED_MODULES}"
    )
    assert BLOCKED_CALLS == _BLOCKED_CALLS, (
        f"BLOCKED_CALLS diverged.\n"
        f"  only in static_code: {BLOCKED_CALLS - _BLOCKED_CALLS}\n"
        f"  only in validator:   {_BLOCKED_CALLS - BLOCKED_CALLS}"
    )
    assert BLOCKED_NAMES == _BLOCKED_NAMES, (
        f"BLOCKED_NAMES diverged.\n"
        f"  only in static_code: {BLOCKED_NAMES - _BLOCKED_NAMES}\n"
        f"  only in validator:   {_BLOCKED_NAMES - BLOCKED_NAMES}"
    )
    assert BLOCKED_DUNDERS == _BLOCKED_DUNDERS, (
        f"BLOCKED_DUNDERS diverged.\n"
        f"  only in static_code: {BLOCKED_DUNDERS - _BLOCKED_DUNDERS}\n"
        f"  only in validator:   {_BLOCKED_DUNDERS - BLOCKED_DUNDERS}"
    )


def test_except_handler_sync_with_harbor_validator() -> None:
    """Both static_code and validator_template must reject the same except patterns."""
    from ppl_synth_data.harbor.validator_template import check_l1_ast_safety

    bare = "try:\n    pass\nexcept:\n    pass\n"
    base_exc = "try:\n    pass\nexcept BaseException:\n    pass\n"
    specific = "try:\n    pass\nexcept ValueError:\n    pass\n"

    ok_bare, _ = check_l1_ast_safety(bare)
    ok_base, _ = check_l1_ast_safety(base_exc)
    ok_specific, _ = check_l1_ast_safety(specific)

    assert not ok_bare, "validator_template should reject bare except"
    assert not ok_base, "validator_template should reject except BaseException"
    assert ok_specific, "validator_template should allow except ValueError"

    with pytest.raises(StaticCodeError):
        validate_python_safety(bare)
    with pytest.raises(StaticCodeError):
        validate_python_safety(base_exc)
    validate_python_safety(specific)
