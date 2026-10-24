"""Three soundness screens for LazyPPL programs, applied to the numbers the in-container
scorer reports (docker/lazyppl-harbor/app/score.py, likelihood_screen_metrics). They are required
before a LazyPPL program becomes an SFT target or RL reference; passing the compile-and-finite
gate does not establish a sound likelihood.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

ROUNDING = 1e-9


def flags(metrics: Mapping[str, float | None], *, outcome: Sequence[float]) -> list[str]:
    """Screen names that fire, in a fixed order. A missing screen number raises (a KeyError), never
    passes.
    """
    fired = []
    sd = metrics["screen_lpd_sd"]
    if sd is not None and sd <= ROUNDING:
        fired.append("row_ignoring")
    # Any entry at log(1e-300): a sound likelihood never gives a real observation
    # probability 1e-300, so one such entry
    # marks a defect for review.
    floor_share = metrics["screen_floor_entry_share"]
    if floor_share is None:
        raise TypeError("'>' not supported between instances of 'NoneType' and 'int'")
    if floor_share > 0:
        fired.append("floor_reached")
    discrete = all(float(v).is_integer() for v in outcome)
    if discrete:
        maximum = metrics["screen_max_log_lik"]
        if maximum is None:
            raise TypeError("'>' not supported between instances of 'NoneType' and 'float'")
        if maximum > ROUNDING:
            fired.append("positive_log_prob_discrete")
    return fired
