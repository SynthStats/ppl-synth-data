#!/usr/bin/env python3
"""Subprocess harness for gold code validation.

Accepts JSON on stdin with keys "code" and "data" (and optionally
"holdout_data"), runs the L0-L5 validation cascade from validator_template,
and writes the result JSON to stdout. Designed to be called via
subprocess.run() so that exec() of untrusted (gold-shaped) code never happens
in the host process.

When "holdout_data" is present and the L0-L5 cascade passes, also runs L6
(check_l6_holdout_elpd) and merges its raw result under the "l6" key. This is
the single source of truth export_pymc.py uses to compute both a gold's own
held-out ELPD and the intercept-only baseline's held-out ELPD (by calling this
subprocess twice: once with the gold's real data, once with covariate-blanked
data): the exact same check_l6_holdout_elpd function that ships (vendored)
inside the container, so export-time and verify-time ELPD math never drift.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, cast

import numpy as np


def _deserialize_data(raw: dict | None) -> dict | None:
    if raw is None:
        return None
    out = {}
    for k, v in raw.items():
        if isinstance(v, list):
            out[k] = np.array(v)
        else:
            out[k] = v
    return out


def _observe_fit_provenance() -> dict[str, Any]:
    """What this scoring process actually observed, for the fit record.

    The validator's identity is authenticated HERE, from the module this process actually loaded,
    not from the path a caller passed in: a file that was replaced between the call and the import
    would otherwise be reported as though it were the one that scored.

    The numerical profile is read from the running configuration, not from the environment a caller
    meant to set, because an env var is what was intended and the config is what ran.
    """
    import hashlib
    import pathlib

    # Import it here if it is not already loaded, so this function always names the module it
    # actually loaded rather than assuming a caller imported it first.
    loaded = sys.modules.get("ppl_synth_data.harbor.validator_template")
    if loaded is None:
        import ppl_synth_data.harbor.validator_template as _validator_template

        loaded = _validator_template
    loaded_path = getattr(loaded, "__file__", None)
    if loaded_path is None:
        raise RuntimeError("The validator module is not loaded in this process")
    module_path = pathlib.Path(loaded_path)
    profile: dict[str, Any] = {}
    try:
        import pytensor

        config = pytensor.config
        profile = {
            "cxx": getattr(config, "cxx", None),
            "mode": getattr(config, "mode", None),
            "floatX": getattr(config, "floatX", None),
            "observed_from": "pytensor.config",
        }
    except Exception as exc:  # noqa: BLE001 - report the absence, never invent a profile
        profile = {"observed_from": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    helper_path = pathlib.Path(__file__)
    import datetime
    import os
    import socket

    from ppl_synth_data.harbor.export_pymc import container_identity

    started = datetime.datetime.now(datetime.UTC).isoformat()
    return {
        "validator_module": {
            "module": getattr(loaded, "__name__", "ppl_synth_data.harbor.validator_template"),
            "file": str(module_path),
            "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        },
        # The helper that runs the fit is hashed too, not only the validator.
        "helper_module": {
            "module": __name__,
            "file": str(helper_path),
            "sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        },
        "numerical_profile": profile,
        # Record the process identity and start time before the fit begins.
        "process": {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at_utc": started,
        },
        "container": container_identity(),
    }


def _stamp_fit_finished(provenance: dict[str, Any]) -> dict[str, Any]:
    """Record the UTC moment the fit stopped, on a record that already names its start."""
    import datetime

    process = provenance.setdefault("process", {})
    process["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    return provenance


def main() -> None:
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    data = cast(dict[str, Any], _deserialize_data(payload["data"]))
    holdout_data = _deserialize_data(payload.get("holdout_data"))
    # Optional draws/tune override, test-only: production callers omit these
    # and get validator_template's real L6_DRAWS/L6_TUNE (200/200) defaults.
    l6_kwargs: dict[str, int] = {}
    if payload.get("l6_draws") is not None:
        l6_kwargs["draws"] = int(payload["l6_draws"])
    if payload.get("l6_tune") is not None:
        l6_kwargs["tune"] = int(payload["l6_tune"])

    from ppl_synth_data.harbor.validator_template import (
        check_l6_holdout_elpd,
        run_cascade,
    )

    # Record the loaded validator and process identity before running the fit.
    result_provenance = _observe_fit_provenance()

    target_name = payload.get("target_name")
    # Return the fit identity alongside any cascade exception.
    try:
        result = run_cascade(code, data)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "fit_provenance": _stamp_fit_finished(result_provenance),
                    "cascade_exception": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
                default=str,
            )
        )
        raise
    if result["reward"] >= 1.0:
        if holdout_data is None or target_name is None:
            # Never leave "l6" absent (export_pymc.py's caller then only sees
            # gold_result.get("l6") is None, i.e. the uninformative "failed:
            # None": surface exactly which precondition was unmet instead,
            # since the L0-L5 structural cascade itself already passed here.
            result["l6"] = {
                "outcome": "l6_not_attempted",
                "error": (
                    "L6 skipped: holdout_data missing"
                    if holdout_data is None
                    else "L6 skipped: target_name missing"
                ),
            }
        else:
            try:
                result["l6"] = check_l6_holdout_elpd(
                    code, data, holdout_data, target_name=target_name, **l6_kwargs
                )
            except Exception as exc:  # noqa: BLE001 - report every real crash, never swallow
                # A real exception from inside check_l6_holdout_elpd (e.g. an
                # OOM/near-OOM MemoryError under host contention, a PyTensor
                # compile failure) must reach export_pymc.py's caller with a
                # real type/message/traceback, not vanish into a missing "l6"
                # key that renders as "failed: None".
                result["l6"] = {
                    "outcome": "l6_subprocess_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
    result["fit_provenance"] = _stamp_fit_finished(result_provenance)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
