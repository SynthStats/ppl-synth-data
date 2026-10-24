"""Superset contract: our Trajectory schema projects into synthstats shape.

Pinned at synthstats commit a9fb24c1400b847adfaa257b45b16818c93c9cc6.

The test pins the EXACT top-level field set of synthstats Trajectory at the
pinned SHA. If synthstats adds a field in HEAD, this test does not break; if
synthstats removes or renames a field, the next re-pin must update the
expected set AND `Trajectory.as_synthstats_dict()`.

If the synthstats package is importable in the test environment, we also
construct the synthstats dataclass from our dict as a stronger consumer check.
Otherwise we check the shape only.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401 (used as forward-ref string literal below)

import pytest

from ppl_synth_data.schemas.trajectory import Trajectory

# Copied-in, not imported, so this test doesn't require synthstats to run.
# Source of truth: synthstats/src/synthstats/core/foundation/types.py @ a9fb24c
SYNTHSTATS_TRAJECTORY_FIELDS_AT_a9fb24c: frozenset[str] = frozenset(
    {
        "messages",
        "token_ids",
        "token_logprobs",
        "loss_mask",
        "reward",
        "eos_logprobs",
    }
)
SYNTHSTATS_REWARD_FIELDS_AT_a9fb24c: frozenset[str] = frozenset(
    {"total", "components", "info", "factors", "scalarization"}
)
SYNTHSTATS_MESSAGE_FIELDS_AT_a9fb24c: frozenset[str] = frozenset(
    {"role", "content", "tool_call_id", "reasoning_content"}
)


@pytest.fixture
def trajectory() -> Trajectory:
    from tests.eval.test_trajectory_schema import _make_trajectory

    return _make_trajectory()


class TestSupersetShape:
    def test_projection_contains_every_synthstats_field(self, trajectory: Trajectory) -> None:
        projected = trajectory.as_synthstats_dict()
        missing = SYNTHSTATS_TRAJECTORY_FIELDS_AT_a9fb24c - projected.keys()
        assert not missing, f"missing synthstats fields in projection: {sorted(missing)}"

    def test_reward_projection_contains_every_synthstats_reward_field(
        self, trajectory: Trajectory
    ) -> None:
        projected_reward = trajectory.as_synthstats_dict()["reward"]
        missing = SYNTHSTATS_REWARD_FIELDS_AT_a9fb24c - projected_reward.keys()
        assert not missing, f"missing synthstats reward fields: {sorted(missing)}"

    def test_reward_info_carries_provenance(self, trajectory: Trajectory) -> None:
        """Until synthstats types Reward.info, we round-trip provenance through info."""
        projected = trajectory.as_synthstats_dict()
        info = projected["reward"]["info"]
        assert "reward_source" in info
        assert "evidence_kind" in info
        assert "truth_owner" in info
        assert "producer" in info

    def test_token_shapes_match_synthstats_conventions(self, trajectory: Trajectory) -> None:
        """synthstats: list[list[int]] per assistant generation.

        Our projection: one inner list per turn.
        """
        projected = trajectory.as_synthstats_dict()
        assert isinstance(projected["token_ids"], list)
        assert all(isinstance(inner, list) for inner in projected["token_ids"])
        assert all(isinstance(i, int) for inner in projected["token_ids"] for i in inner)

    def test_loss_mask_bool_typed(self, trajectory: Trajectory) -> None:
        """synthstats loss_mask is list[list[bool]] at Trajectory level.

        (Separately, extract_generation_data_from_agent_metadata returns list[int] 0/1 at the
        ATIF extractor level; that is the extractor's contract.)
        """
        projected = trajectory.as_synthstats_dict()
        for inner in projected["loss_mask"]:
            assert all(isinstance(b, bool) for b in inner)

    def test_messages_roles_are_valid(self, trajectory: Trajectory) -> None:
        projected = trajectory.as_synthstats_dict()
        valid_roles = {"system", "user", "assistant", "tool"}
        for msg in projected["messages"]:
            assert msg["role"] in valid_roles
            assert set(msg.keys()) <= SYNTHSTATS_MESSAGE_FIELDS_AT_a9fb24c, (
                f"message has unknown keys vs synthstats@a9fb24c: {msg.keys()}"
            )

    def test_messages_are_not_duplicated(self) -> None:
        """A two-turn trajectory projection emits each prior message once."""
        from ppl_synth_data.schemas.trajectory import (
            Episode,
            Message,
            Step,
            StopReason,
            Trajectory,
            Trial,
            Turn,
        )
        from tests.eval.test_trajectory_schema import (
            _make_observation,
            _make_reward,
            _make_task_spec,
        )

        sys_msg = Message(role="system", content="sys")
        assistant_0 = Message(role="assistant", content="a0")
        tool_0 = Message(role="tool", content="t0")
        assistant_1 = Message(role="assistant", content="a1")

        turn_0 = Turn(
            turn_index=0,
            messages_input=[sys_msg],
            assistant_message=assistant_0,
            token_ids=[1, 2, 3],
            token_logprobs=[-0.1] * 3,
            loss_mask=[True] * 3,
        )
        turn_1 = Turn(
            turn_index=1,
            messages_input=[sys_msg, assistant_0, tool_0],
            assistant_message=assistant_1,
            token_ids=[4, 5, 6],
            token_logprobs=[-0.1] * 3,
            loss_mask=[True] * 3,
        )
        obs = _make_observation(0, 0)
        step = Step(
            step_index=0,
            triggering_turn_index=0,
            observation_id=obs.observation_id,
            env_state_digest_before="sha256:a",
            env_state_digest_after="sha256:b",
            wall_seconds=0.0,
        )
        episode = Episode(
            episode_id="ep-multi",
            turns=[turn_0, turn_1],
            steps=[step],
            observations=[obs],
            belief_snapshots=[],
            belief_updates=[],
            tool_calls=[],
            metric_breakdowns=[],
            transition_rewards=[0.0, 0.0],
            turn_discounts=[1.0, 1.0],
            terminal_stop_reason=StopReason.completed,
        )
        trial = Trial(
            trial_id="trial-multi",
            submitted_program="pass",
            language="pymc",
            reward=_make_reward(),
        )
        trajectory = Trajectory(
            trajectory_id="traj-multi",
            task_spec=_make_task_spec(),
            episode=episode,
            trial=trial,
            seed=0,
            created_at_iso="2026-04-22T00:00:00Z",
        )

        projected = trajectory.as_synthstats_dict()
        roles = [m["role"] for m in projected["messages"]]
        # Correct: [sys, a0, tool_0, a1]; 4 messages.
        # Naive bug: [sys, a0, sys, a0, tool_0, a1]; 6 messages, sys/a0 duplicated.
        assert roles == ["system", "assistant", "tool", "assistant"], (
            f"messages duplicated; got {len(roles)} messages with roles {roles}"
        )
        assert projected["messages"][0]["content"] == "sys"
        assert projected["messages"][1]["content"] == "a0"
        assert projected["messages"][2]["content"] == "t0"
        assert projected["messages"][3]["content"] == "a1"

    def test_messages_full_conversation_via_orchestrator(
        self,
        tmp_path: Path,  # type: ignore[name-defined]
    ) -> None:
        """End-to-end: after a 3-turn run (query, query, submit), messages must be [sys, a0,
        tool_0, a1, tool_1, a2]; six messages, no dupes.
        """
        from tests.eval.test_orchestrator_toy_env import _run

        orch, traj_dict = _run(tmp_path, seed=42)
        traj = Trajectory.model_validate(traj_dict)
        projected = traj.as_synthstats_dict()
        roles = [m["role"] for m in projected["messages"]]
        assert roles == ["system", "assistant", "tool", "assistant", "tool", "assistant"], (
            f"expected 6 messages with interleaved assistant/tool, got {roles}"
        )

    def test_messages_count_equals_expected(
        self,
        tmp_path: Path,  # type: ignore[name-defined]
    ) -> None:
        """Quantitative check: 3-turn scripted run -> 6 messages."""
        from tests.eval.test_orchestrator_toy_env import _run

        orch, traj_dict = _run(tmp_path, seed=42)
        traj = Trajectory.model_validate(traj_dict)
        assert len(traj.as_synthstats_dict()["messages"]) == 6


class TestConsumerCheck:
    """Strong check: synthstats actually consumes our dict."""

    def test_synthstats_trajectory_from_dict_succeeds(self, trajectory: Trajectory) -> None:
        synthstats_types = pytest.importorskip(
            "synthstats.core.foundation.types",
            reason=(
                "synthstats not installed in this env; shape check in "
                "TestSupersetShape is sufficient."
            ),
        )
        projected = trajectory.as_synthstats_dict()
        synth_trajectory = synthstats_types.Trajectory.from_dict(projected)
        # Round-trip: their to_dict back to our schema shape
        re_dumped = synth_trajectory.to_dict()
        assert re_dumped.keys() == projected.keys(), "synthstats round-trip dropped or added keys"
