"""Seed-ledger determinism: same root + same subsystem -> same seed, always.

This is the bookkeeping invariant that makes replay possible. Anything stochastic in the eval
runtime MUST pull its seed through the ledger; the tests here pin the derivation and the
idempotence of repeated calls.
"""

from __future__ import annotations

import pytest

from ppl_synth_data.eval.seed_ledger import SeedLedger, derive_seed


class TestDeriveSeed:
    def test_same_inputs_same_output(self) -> None:
        a = derive_seed(42, "env")
        b = derive_seed(42, "env")
        assert a == b

    def test_different_root_different_seed(self) -> None:
        a = derive_seed(42, "env")
        b = derive_seed(43, "env")
        assert a != b

    def test_different_subsystem_different_seed(self) -> None:
        a = derive_seed(42, "env")
        b = derive_seed(42, "agent")
        assert a != b

    def test_output_is_in_positive_int32_range(self) -> None:
        for seed in (0, 1, 42, 123456, 2**31 - 2):
            for subsystem in ("env", "agent", "pymc.turn_0"):
                out = derive_seed(seed, subsystem)
                assert 0 <= out < 2**31

    def test_root_seed_must_be_nonnegative_int(self) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            derive_seed(-1, "env")

    def test_root_seed_must_be_int_not_float(self) -> None:
        with pytest.raises(ValueError, match="non-negative int"):
            derive_seed(42.0, "env")  # type: ignore[arg-type]

    def test_subsystem_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            derive_seed(42, "")


class TestSeedLedger:
    def test_for_subsystem_is_deterministic(self) -> None:
        ledger_a = SeedLedger(root_seed=42)
        ledger_b = SeedLedger(root_seed=42)
        assert ledger_a.for_subsystem("env") == ledger_b.for_subsystem("env")
        assert ledger_a.for_subsystem("agent.turn_0") == ledger_b.for_subsystem("agent.turn_0")

    def test_for_subsystem_is_idempotent(self) -> None:
        ledger = SeedLedger(root_seed=42)
        first = ledger.for_subsystem("pymc.sample")
        second = ledger.for_subsystem("pymc.sample")
        assert first == second

    def test_snapshot_captures_consulted_subsystems(self) -> None:
        ledger = SeedLedger(root_seed=42)
        ledger.for_subsystem("env")
        ledger.for_subsystem("agent.turn_0")
        ledger.for_subsystem("env")  # idempotent re-read
        snap = ledger.snapshot()
        assert set(snap.keys()) == {"env", "agent.turn_0"}

    def test_order_reflects_first_call(self) -> None:
        ledger = SeedLedger(root_seed=42)
        ledger.for_subsystem("env")
        ledger.for_subsystem("agent.turn_0")
        ledger.for_subsystem("agent.turn_1")
        ledger.for_subsystem("env")  # already seen; order unchanged
        assert ledger.order() == ["env", "agent.turn_0", "agent.turn_1"]

    def test_different_roots_give_different_seeds_same_subsystem(self) -> None:
        ledger_a = SeedLedger(root_seed=42)
        ledger_b = SeedLedger(root_seed=99)
        assert ledger_a.for_subsystem("env") != ledger_b.for_subsystem("env")

    def test_snapshot_isolated_from_mutation(self) -> None:
        ledger = SeedLedger(root_seed=42)
        ledger.for_subsystem("env")
        snap = ledger.snapshot()
        snap["env"] = 0  # mutating the snapshot
        assert ledger.snapshot()["env"] != 0  # ledger unchanged
