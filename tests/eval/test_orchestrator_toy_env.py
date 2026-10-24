"""Internal ToyOrchestrator kernel tests.

Checkpoint 1 from the parent plan: one toy env, one internal metric, full
trajectory, local JSONL, byte-identical trajectories across two runs at the
same seed.

Byte-identical means every content byte matches; timestamps are the one
field we do NOT assert on (the clock is allowed to tick between runs). The
seed-ledger snapshot, trial reward, per-turn token data, and state digests
must all match.
"""

from __future__ import annotations

import json
from pathlib import Path

from synthstats_schemas import EvidenceKind, TruthOwner

from ppl_synth_data.eval.agents.scripted import ScriptedAgent, ScriptedTurn
from ppl_synth_data.eval.orchestrator import OrchestratorConfig, ToyOrchestrator
from ppl_synth_data.eval.persistence import EpisodeJournal, FsyncMode
from ppl_synth_data.eval.sandbox.trivial import TrivialSandbox
from ppl_synth_data.eval.seed_ledger import SeedLedger
from ppl_synth_data.eval.toy_env import ToyScalarEnv
from ppl_synth_data.schemas.trajectory import (
    ActionType,
    BeliefSpec,
    EvalTaskSpec,
    GoalSpec,
    MetricSchedule,
    MetricSpec,
)


def _default_task_spec() -> EvalTaskSpec:
    return EvalTaskSpec(
        task_id="task-toy-scalar-001",
        domain="toy_scalar",
        env_spec_id="envspec-toy-scalar-001",
        env_spec_digest="sha256:toy-envspec",
        compiled_env_code_digest="sha256:toy-compiled",
        goal=GoalSpec(
            goal_id="identify_linear_params",
            goal_type="parameter_identification",
            target={"params": ["alpha", "beta"]},
        ),
        belief_spec=BeliefSpec(
            model_family="analytic",
            sampler_config={},
            sampler_config_digest="sha256:toy-sampler",
        ),
        metric_specs=[
            MetricSpec(
                metric_id="trivial",
                metric_name="trivial_terminal",
                schedule=MetricSchedule.terminal,
                reward_source="validation.compilation",
                evidence_kind=EvidenceKind.verifier,
                truth_owner=TruthOwner.verifier,
            ),
        ],
        visible_data_ref="inline://toy_scalar",
        hidden_data_ref="inline://toy_scalar_hidden",
        prompt_template_hash="sha256:toy-prompt-v1",
    )


def _scripted_turns_submit_at_end() -> list[ScriptedTurn]:
    return [
        ScriptedTurn(content="Query 1", action_type=ActionType.query, action={"x": 1.0}),
        ScriptedTurn(content="Query 2", action_type=ActionType.query, action={"x": -0.5}),
        ScriptedTurn(
            content="Submit",
            action_type=ActionType.submit_program,
            action={"language": "pymc", "program": "import pymc as pm\n"},
        ),
    ]


def _scripted_turns_no_submission() -> list[ScriptedTurn]:
    # All query, no submission; forces the stop-without-submission branch.
    return [
        ScriptedTurn(content=f"Query {i}", action_type=ActionType.query, action={"x": float(i)})
        for i in range(3)
    ]


def _run(
    run_dir: Path,
    *,
    seed: int = 42,
    turns: list[ScriptedTurn] | None = None,
    episode_id: str = "ep-test",
    max_turns: int = 5,
    fixed_clock: str = "2026-04-22T00:00:00Z",
) -> tuple[ToyOrchestrator, dict]:
    turns = turns or _scripted_turns_submit_at_end()
    env = ToyScalarEnv()
    agent = ScriptedAgent(turns)
    task_spec = _default_task_spec()
    journal = EpisodeJournal.for_episode(run_dir, episode_id, fsync_mode=FsyncMode.never)
    orchestrator = ToyOrchestrator(
        env=env,
        agent=agent,
        task_spec=task_spec,
        journal=journal,
        seed_ledger=SeedLedger(root_seed=seed),
        sandbox=TrivialSandbox(),
        config=OrchestratorConfig(
            episode_id=episode_id,
            trajectory_id=f"traj-{episode_id}",
            max_turns=max_turns,
            clock=lambda: fixed_clock,
        ),
    )
    orchestrator.run_episode()
    return orchestrator, json.loads(journal.paths.trajectory_path.read_text(encoding="utf-8"))


class TestToyEpisodeCompletes:
    def test_submit_branch_reaches_trial(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42)
        assert traj["trial"]["reward"]["total"] == 1.0
        assert traj["trial"]["reward"]["reward_source"] == "validation.compilation"
        assert traj["trial"]["sandbox_backend"] == "trivial"
        assert traj["episode"]["terminal_stop_reason"] == "completed"

    def test_turn_count_matches_script_up_to_submit(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42)
        assert len(traj["episode"]["turns"]) == 3

    def test_each_query_produces_observation_and_step(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42)
        assert len(traj["episode"]["observations"]) == 2
        assert len(traj["episode"]["steps"]) == 2

    def test_observation_evidence_kind_is_verifier(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42)
        for obs in traj["episode"]["observations"]:
            assert obs["evidence_kind"] == "verifier"
            assert obs["truth_owner"] == "verifier"
            assert obs["source"] == "toy_scalar_env"

    def test_seed_ledger_captured_in_run_metadata(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42)
        ledger = traj["run_metadata"]["seed_ledger"]
        assert "env" in ledger
        assert "agent.turn_0" in ledger
        assert "agent.turn_1" in ledger


class TestDeterminism:
    def test_same_seed_produces_byte_identical_trajectory(self, tmp_path: Path) -> None:
        """Two runs with the same seed + deterministic clock -> identical JSON."""
        run_a = tmp_path / "a"
        run_b = tmp_path / "b"
        _, traj_a = _run(run_a, seed=42, episode_id="ep")
        _, traj_b = _run(run_b, seed=42, episode_id="ep")
        assert json.dumps(traj_a, sort_keys=True) == json.dumps(traj_b, sort_keys=True), (
            "byte-identical trajectories expected for same-seed runs"
        )

    def test_different_seeds_produce_different_observations(self, tmp_path: Path) -> None:
        _, traj_a = _run(tmp_path / "a", seed=42)
        _, traj_b = _run(tmp_path / "b", seed=99)
        obs_a = [o["payload"]["y"] for o in traj_a["episode"]["observations"]]
        obs_b = [o["payload"]["y"] for o in traj_b["episode"]["observations"]]
        assert obs_a != obs_b

    def test_seed_ledger_order_is_stable(self, tmp_path: Path) -> None:
        _, traj_a = _run(tmp_path / "a", seed=42)
        _, traj_b = _run(tmp_path / "b", seed=42)
        assert (
            traj_a["run_metadata"]["seed_ledger_order"]
            == traj_b["run_metadata"]["seed_ledger_order"]
        )


class TestStopWithoutSubmission:
    def test_no_submission_emits_unknown_trial(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42, turns=_scripted_turns_no_submission(), max_turns=3)
        assert traj["trial"]["reward"]["reward_source"] == "heuristic.no_submission"
        assert traj["trial"]["reward"]["evidence_kind"] == "heuristic"
        assert traj["trial"]["submitted_program"] is None
        assert traj["episode"]["terminal_stop_reason"] == "length"


class TestWALInvariant:
    def test_trajectory_json_matches_event_projection(self, tmp_path: Path) -> None:
        from ppl_synth_data.eval.persistence import events_to_trajectory

        orch, traj = _run(tmp_path, seed=42)
        events = orch._journal.read_events()  # intentional: re-project from events
        projected = events_to_trajectory(events)
        assert projected.model_dump(mode="json") == traj

    def test_finalised_marker_exists(self, tmp_path: Path) -> None:
        orch, _ = _run(tmp_path, seed=42)
        assert orch._journal.finalised
        assert orch._journal.paths.finalised_marker.exists()


class TestMetricEmission:
    """Milestone: one metric fires per episode."""

    def test_submit_path_emits_terminal_metric(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42)
        metrics = traj["episode"]["metric_breakdowns"]
        assert len(metrics) == 1
        mb = metrics[0]
        assert mb["metric_id"] == "trivial.trial_reward"
        assert mb["schedule"] == "terminal"
        assert mb["turn_index"] is None
        assert mb["value"] == traj["trial"]["reward"]["total"]
        assert mb["reward_source"] == traj["trial"]["reward"]["reward_source"]

    def test_unknown_path_also_emits_terminal_metric(self, tmp_path: Path) -> None:
        _, traj = _run(tmp_path, seed=42, turns=_scripted_turns_no_submission(), max_turns=3)
        metrics = traj["episode"]["metric_breakdowns"]
        assert len(metrics) == 1
        assert metrics[0]["reward_source"] == "heuristic.no_submission"
        assert metrics[0]["value"] == 0.0


class TestOtelEmitterHook:
    """The otel_emitter callback fires at episode start, turns, finalise."""

    def test_emitter_receives_episode_lifecycle_spans(self, tmp_path: Path) -> None:
        captured: list[tuple[str, dict]] = []

        def capture(span: str, attrs: dict) -> None:
            captured.append((span, attrs))

        env = ToyScalarEnv()
        agent = ScriptedAgent(_scripted_turns_submit_at_end())
        journal = EpisodeJournal.for_episode(tmp_path, "ep-otel", fsync_mode=FsyncMode.never)
        orchestrator = ToyOrchestrator(
            env=env,
            agent=agent,
            task_spec=_default_task_spec(),
            journal=journal,
            seed_ledger=SeedLedger(root_seed=42),
            sandbox=TrivialSandbox(),
            config=OrchestratorConfig(
                episode_id="ep-otel",
                trajectory_id="traj-otel",
                max_turns=5,
                clock=lambda: "2026-04-22T00:00:00Z",
                otel_emitter=capture,
            ),
        )
        orchestrator.run_episode()

        span_names = [s for s, _ in captured]
        assert "episode.started" in span_names
        assert "episode.finalised" in span_names

    def test_missing_emitter_does_not_crash(self, tmp_path: Path) -> None:
        """Default config leaves emitter=None; orchestrator must no-op."""
        _, traj = _run(tmp_path, seed=42)  # default emitter is None
        assert traj["trial"]["reward"]["total"] == 1.0


class TestCliIntegration:
    """Public CLI must not expose the internal toy kernel."""

    def test_eval_subapp_is_not_registered(self) -> None:
        from ppl_synth_data.cli import app

        group_names = {group.name for group in app.registered_groups}
        assert "eval" not in group_names


class TestResilience:
    """Regression guards: the orchestrator must always emit a finalisable episode.

    Exceptions in agent.generate_turn or sandbox.execute leave the WAL with
    a half-emitted trial if the orchestrator does not catch them. The
    finalise path below compensates by emitting an unknown/error trial
    scoring before episode_finalised.
    """

    def test_agent_exception_mid_turn_still_finalises(self, tmp_path: Path) -> None:
        from ppl_synth_data.eval.agents.base import EpisodeAgent
        from ppl_synth_data.schemas.trajectory import Message

        class ExplodingAgent(EpisodeAgent):
            def initial_messages(self, task_spec):
                return [Message(role="system", content="sys")]

            def generate_turn(self, *, task_spec, messages, turn_index, seed):
                raise RuntimeError("agent exploded")

        journal = EpisodeJournal.for_episode(tmp_path, "ep-boom", fsync_mode=FsyncMode.never)
        orchestrator = ToyOrchestrator(
            env=ToyScalarEnv(),
            agent=ExplodingAgent(),
            task_spec=_default_task_spec(),
            journal=journal,
            seed_ledger=SeedLedger(root_seed=42),
            sandbox=TrivialSandbox(),
            config=OrchestratorConfig(
                episode_id="ep-boom",
                trajectory_id="traj-boom",
                clock=lambda: "2026-04-22T00:00:00Z",
            ),
        )
        result = orchestrator.run_episode()
        assert result.trajectory.episode.terminal_stop_reason.value == "error"
        assert result.trajectory.trial.reward.reward_source == "heuristic.no_submission"
        err = result.trajectory.run_metadata["error_info"]
        assert err["kind"] == "RuntimeError"
        assert "exploded" in err["detail"]

    def test_scripted_agent_exhaustion_stops_cleanly(self, tmp_path: Path) -> None:
        """Agent running out of scripted turns signals stop_token, not crash."""
        single_query = [ScriptedTurn(content="Q", action_type=ActionType.query, action={"x": 0.5})]
        _, traj = _run(tmp_path, seed=42, turns=single_query, max_turns=5, episode_id="ep-exhaust")
        assert traj["episode"]["terminal_stop_reason"] == "stop_token"
        assert traj["trial"]["reward"]["reward_source"] == "heuristic.no_submission"
        err = traj["run_metadata"]["error_info"]
        assert err["kind"] == "agent_exhausted"

    def test_metric_failure_after_trial_scored_does_not_double_emit(self, tmp_path: Path) -> None:
        """A metric error after a durable trial_scored event must not append a second trial_scored.

        """
        journal = EpisodeJournal.for_episode(tmp_path, "ep-metric-fail", fsync_mode=FsyncMode.never)
        orchestrator = ToyOrchestrator(
            env=ToyScalarEnv(),
            agent=ScriptedAgent(_scripted_turns_submit_at_end()),
            task_spec=_default_task_spec(),
            journal=journal,
            seed_ledger=SeedLedger(root_seed=42),
            sandbox=TrivialSandbox(),
            config=OrchestratorConfig(
                episode_id="ep-metric-fail",
                trajectory_id="traj-metric-fail",
                clock=lambda: "2026-04-22T00:00:00Z",
            ),
        )

        # Make the terminal-metric emission raise on first call. The event-
        # order guarantee: trial_scored was already appended successfully
        # before this raises, so trial_phase must be "scored" to prevent a
        # double-emit in the post-loop fallback.
        original_metric = orchestrator._emit_terminal_metric
        calls = [0]

        def raising_metric(reward):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("metric journal IO error")
            return original_metric(reward)

        orchestrator._emit_terminal_metric = raising_metric  # type: ignore[method-assign]

        orchestrator.run_episode()

        # Post-condition: exactly ONE trial_scored event in the journal.
        events = journal.read_events()
        trial_scored_count = sum(1 for e in events if e.event_type == "trial_scored")
        assert trial_scored_count == 1, f"expected 1 trial_scored event, got {trial_scored_count}"
        # Post-condition: the journal still projects cleanly (no
        # "multiple trial_scored events" ProjectionError).
        from ppl_synth_data.eval.persistence import events_to_trajectory

        trajectory = events_to_trajectory(events)
        assert trajectory.trial.trial_id == "trial-ep-metric-fail"
        assert trajectory.episode.terminal_stop_reason.value == "error"

    def test_sandbox_exception_emits_error_trial_scored(self, tmp_path: Path) -> None:
        class ExplodingSandbox(TrivialSandbox):
            backend_name = "exploding"

            def execute(self, **kwargs):
                raise RuntimeError("sandbox fuse blown")

        submit_turns = [
            ScriptedTurn(
                content="Submit",
                action_type=ActionType.submit_program,
                action={"language": "pymc", "program": "pass"},
            )
        ]
        journal = EpisodeJournal.for_episode(tmp_path, "ep-sbxfail", fsync_mode=FsyncMode.never)
        orchestrator = ToyOrchestrator(
            env=ToyScalarEnv(),
            agent=ScriptedAgent(submit_turns),
            task_spec=_default_task_spec(),
            journal=journal,
            seed_ledger=SeedLedger(root_seed=42),
            sandbox=ExplodingSandbox(),
            config=OrchestratorConfig(
                episode_id="ep-sbxfail",
                trajectory_id="traj-sbxfail",
                clock=lambda: "2026-04-22T00:00:00Z",
            ),
        )
        result = orchestrator.run_episode()
        assert result.trajectory.trial.reward.reward_source == "heuristic.judge_exception"
        assert result.trajectory.episode.terminal_stop_reason.value == "error"
        err = result.trajectory.run_metadata["error_info"]
        assert err["kind"] == "sandbox_error"
        assert "fuse blown" in err["detail"]
        # Critical invariant: episode_finalised was still emitted; trajectory.json on disk.
        assert journal.finalised
        assert journal.paths.trajectory_path.exists()
