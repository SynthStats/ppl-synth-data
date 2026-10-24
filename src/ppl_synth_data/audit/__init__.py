from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AuditJob",
    "BatchSpec",
    "EntryResult",
    "build_batch_text",
    "build_entry_list",
    "collect",
    "load_outputs",
    "make_batches",
    "outputs_complete",
    "parse_log",
    "render_report",
    "submit",
]

_EXPORTS = {
    "AuditJob": ("ppl_synth_data.audit.runner", "AuditJob"),
    "BatchSpec": ("ppl_synth_data.audit.batch", "BatchSpec"),
    "EntryResult": ("ppl_synth_data.audit.parser", "EntryResult"),
    "build_batch_text": ("ppl_synth_data.audit.batch", "build_batch_text"),
    "build_entry_list": ("ppl_synth_data.audit.batch", "build_entry_list"),
    "collect": ("ppl_synth_data.audit.runner", "collect"),
    "load_outputs": ("ppl_synth_data.audit.runner", "load_outputs"),
    "make_batches": ("ppl_synth_data.audit.batch", "make_batches"),
    "outputs_complete": ("ppl_synth_data.audit.runner", "outputs_complete"),
    "parse_log": ("ppl_synth_data.audit.parser", "parse_log"),
    "render_report": ("ppl_synth_data.audit.report", "render_report"),
    "submit": ("ppl_synth_data.audit.runner", "submit"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
