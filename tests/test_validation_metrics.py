from __future__ import annotations

from ppl_synth_data.validation.metrics import AcceptanceMetrics


def test_pass_rate_all_accepted() -> None:
    m = AcceptanceMetrics(total=10, accepted=10)
    assert m.pass_rate == 1.0


def test_pass_rate_none_accepted() -> None:
    m = AcceptanceMetrics(total=10, accepted=0)
    assert m.pass_rate == 0.0


def test_pass_rate_partial() -> None:
    m = AcceptanceMetrics(total=4, accepted=3)
    assert m.pass_rate == 0.75


def test_pass_rate_empty() -> None:
    m = AcceptanceMetrics(total=0, accepted=0)
    assert m.pass_rate == 0.0
