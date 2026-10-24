"""AST-based safety filter for LLM-generated Python code.

Not a security sandbox: Python's dynamic introspection can bypass static
analysis. Production deployments should run generated code in an isolated
environment (nsjail, bwrap, or Docker).
"""

from __future__ import annotations

import ast

BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "http",
    "urllib",
    "requests",
    "pathlib",
    "importlib",
    "builtins",
    "io",
    "ctypes",
    "types",
    "signal",
    "code",
    "codeop",
    "compileall",
    "runpy",
    "rlcompleter",
    "pickle",
    "shelve",
    "marshal",
    "multiprocessing",
    "threading",
    "_thread",
    "gc",
    "inspect",
    "dis",
    "asyncio",
    "tempfile",
    "glob",
    "sqlite3",
    "webbrowser",
    "pdb",
    "pty",
}

BLOCKED_CALLS = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "__import__",
    "system",
    "popen",
    "import_module",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "memoryview",
    "type",
    "dir",
}

BLOCKED_NAMES = {
    "__builtins__",
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "memoryview",
    "type",
    "dir",
}

BLOCKED_DUNDERS = {
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__module__",
    "__reduce__",
    "__reduce_ex__",
    "__getstate__",
    "__setstate__",
    "__dict__",
    "__setattr__",
    "__delattr__",
}


class StaticCodeError(ValueError):
    pass


def _root_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _root_name(node.value)
    return None


def _collect_import_aliases(
    node: ast.Import,
    blocked_aliases: set[str],
    module_aliases: dict[str, str],
) -> None:
    for alias in node.names:
        mod = alias.name.split(".")[0]
        if mod in BLOCKED_MODULES and alias.asname:
            blocked_aliases.add(alias.asname)
            module_aliases[alias.asname] = mod


def _collect_import_from_aliases(
    node: ast.ImportFrom,
    blocked_aliases: set[str],
) -> None:
    mod = (node.module or "").split(".")[0]
    if mod not in BLOCKED_MODULES:
        return
    for alias in node.names:
        if alias.name in BLOCKED_CALLS:
            blocked_aliases.add(alias.asname or alias.name)


def _collect_assign_aliases(node: ast.Assign, blocked_aliases: set[str]) -> None:
    rhs_name: str | None = None
    if isinstance(node.value, ast.Name) and node.value.id in BLOCKED_CALLS:
        rhs_name = node.value.id
    elif isinstance(node.value, ast.Attribute) and node.value.attr in BLOCKED_CALLS:
        rhs_name = node.value.attr
    if rhs_name is None:
        return
    for target in node.targets:
        if isinstance(target, ast.Name):
            blocked_aliases.add(target.id)


def _collect_blocked_aliases(tree: ast.AST) -> tuple[set[str], dict[str, str]]:
    blocked_aliases: set[str] = set()
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _collect_import_aliases(node, blocked_aliases, module_aliases)
        elif isinstance(node, ast.ImportFrom):
            _collect_import_from_aliases(node, blocked_aliases)
        elif isinstance(node, ast.Assign):
            _collect_assign_aliases(node, blocked_aliases)
    return blocked_aliases, module_aliases


def _collect_imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                roots.add(alias.asname or alias.name)
    return roots


def _check_name_node(node: ast.AST) -> None:
    if not isinstance(node, ast.Name):
        return
    if node.id in BLOCKED_NAMES:
        raise StaticCodeError(f"blocked name: {node.id}")
    if node.id in BLOCKED_DUNDERS:
        raise StaticCodeError(f"blocked dunder name: {node.id}")


def _check_import_node(node: ast.AST) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            mod = alias.name.split(".")[0]
            if mod in BLOCKED_MODULES:
                raise StaticCodeError(f"blocked import: {mod}")
        return

    if isinstance(node, ast.ImportFrom):
        mod = (node.module or "").split(".")[0]
        if mod in BLOCKED_MODULES:
            raise StaticCodeError(f"blocked import: {mod}")


def _check_direct_call(func: ast.Name, blocked_aliases: set[str]) -> None:
    if func.id in BLOCKED_CALLS:
        raise StaticCodeError(f"blocked call: {func.id}")
    if func.id in blocked_aliases:
        raise StaticCodeError(f"blocked aliased call: {func.id}")


def _check_attribute_call(
    func: ast.Attribute,
    module_aliases: dict[str, str] | None,
) -> None:
    if func.attr in BLOCKED_DUNDERS:
        raise StaticCodeError(f"blocked dunder attribute call: {func.attr}")
    root = _root_name(func)
    if root in BLOCKED_MODULES:
        raise StaticCodeError(f"blocked module attribute call: {root}.{func.attr}")
    if module_aliases and root in module_aliases:
        raise StaticCodeError(
            f"blocked module attribute call: {root}.{func.attr} (alias for {module_aliases[root]})"
        )
    if func.attr in BLOCKED_CALLS:
        raise StaticCodeError(f"blocked call: {func.attr}")


def _check_call_node(
    node: ast.AST,
    blocked_aliases: set[str],
    module_aliases: dict[str, str] | None = None,
) -> None:
    if not isinstance(node, ast.Call):
        return
    if isinstance(node.func, ast.Name):
        _check_direct_call(node.func, blocked_aliases)
    elif isinstance(node.func, ast.Attribute):
        _check_attribute_call(node.func, module_aliases)


def _check_attribute_node(node: ast.AST) -> None:
    if isinstance(node, ast.Attribute) and node.attr in BLOCKED_DUNDERS:
        raise StaticCodeError(f"blocked dunder attribute: {node.attr}")


def _check_module_attr_store(node: ast.AST, blocked_roots: set[str]) -> None:
    def _blocked_attr_target(target: ast.AST, ctx_type: type[ast.expr_context]) -> str | None:
        if not (isinstance(target, ast.Attribute) and isinstance(target.ctx, ctx_type)):
            return None
        root = _root_name(target)
        if root in blocked_roots:
            return f"blocked module-attribute reassignment: {root}.{target.attr}"
        return None

    if isinstance(node, ast.Assign):
        for t in node.targets:
            for subnode in ast.walk(t):
                reason = _blocked_attr_target(subnode, ast.Store)
                if reason:
                    raise StaticCodeError(reason)
    elif isinstance(node, ast.AugAssign):
        reason = _blocked_attr_target(node.target, ast.Store)
        if reason:
            raise StaticCodeError(reason)
    elif isinstance(node, ast.AnnAssign):
        reason = _blocked_attr_target(node.target, ast.Store)
        if reason:
            raise StaticCodeError(reason)
    elif isinstance(node, ast.Delete):
        for t in node.targets:
            reason = _blocked_attr_target(t, ast.Del)
            if reason:
                raise StaticCodeError(reason)


def _check_except_handler(node: ast.AST) -> None:
    if not isinstance(node, ast.ExceptHandler):
        return
    if node.type is None:
        raise StaticCodeError("bare except: clause not allowed (catches timeout signals)")
    if isinstance(node.type, ast.Name) and node.type.id == "BaseException":
        raise StaticCodeError("except BaseException not allowed (catches timeout signals)")


def validate_python_safety(code: str) -> None:
    tree = ast.parse(code)
    blocked_aliases, module_aliases = _collect_blocked_aliases(tree)
    blocked_roots = _collect_imported_roots(tree) | {"np", "numpy", "pm", "pt", "pymc", "pytensor"}

    for node in ast.walk(tree):
        _check_name_node(node)
        _check_import_node(node)
        _check_call_node(node, blocked_aliases, module_aliases)
        _check_attribute_node(node)
        _check_module_attr_store(node, blocked_roots)
        _check_except_handler(node)
