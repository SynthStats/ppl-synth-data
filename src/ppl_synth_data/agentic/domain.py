"""Multi-turn state machine over one compiled infection_process environment.

This is the real environment side of the cold-start agentic rollout: a policy
queries the environment that provides observations with ``observe_data`` (Bayesian
experimental-design style: pick a time t, observe an infection count) and then
terminates with ``submit_program`` (a PyMC program explaining the visible data).
The submitted program is scored by the native held-out log-likelihood verifier.

Real components only. If PyMC, the env compiler, or the verifier cannot run, the
relevant call raises ``RuntimeError`` rather than substituting a stub.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthstats_schemas.reward import (
    EvidenceKind,
    RewardArtifact,
    TruthOwner,
    get_reward_source,
)

from ppl_synth_data.envs.base import BaseEnv
from ppl_synth_data.envs.compiler import TemplateCompiler
from ppl_synth_data.pipeline.env_loader import load_env_instance
from ppl_synth_data.runtimes.pymc_runtime import PyMCRuntime
from ppl_synth_data.specs.models import (
    Complexity,
    DataPlan,
    EnvSpec,
    InputType,
    IOSchema,
)
from ppl_synth_data.validation.pymc_semantics import score_pymc_semantics

# The registered held-out log-probability verifier source and its only allowed
# producer (synthstats-schemas catalog). Submitting a program is the terminal
# action; its reward provenance is this verifier source.
REWARD_SOURCE = "validation.pymc_semantic_holdout_logp"
REWARD_PRODUCER = "ppl_synth_data.validation.pymc_semantics"
TRAINING_LOG_REWARD_FLOOR = -20.0


def build_infection_process_spec(seed: int) -> EnvSpec:
    """A concrete infection_process EnvSpec the template compiler can render."""
    return EnvSpec(
        env_id=f"infection-process-{seed:06d}",
        env_name=f"infection_process_{seed:06d}",
        domain="epidemiology",
        complexity=Complexity.medium,
        model_type="latent-rate count process",
        key_concepts=["infection rate", "binomial counts", "saturating probability"],
        io_schema=IOSchema(
            input_type=InputType.float,
            input_description="a positive time value t at which to observe the infected count",
            output_type="int",
            target_name="infected",
            feature_names=["t"],
            feature_domains={"t": (0.01, 100.0)},
        ),
        data_plan=DataPlan(n_observations=12, grouping=[], covariates=[]),
        generative_family="infection_process",
        generative_params={
            "population_n": 100,
            "rate_mean": 0.05,
            "rate_sigma": 0.01,
        },
    )


def _column_dict(env: BaseEnv, rows: list[tuple[Any, ...]]) -> dict[str, list[float]]:
    """Column-keyed dict (the `data` argument to the submitted `model(data)`)."""
    cols = env.get_ordered_column_names()
    out: dict[str, list[float]] = {name: [] for name in cols}
    for row in rows:
        for name, value in zip(cols, row, strict=True):
            out[name].append(float(value))
    return out


@dataclass(frozen=True)
class ObserveResult:
    ok: bool
    t: float
    infected: int | None
    message: str
    n_visible: int


@dataclass(frozen=True)
class SubmitResult:
    reward_artifact: RewardArtifact
    metric_name: str | None
    metric_value: float | None
    semantic_status: str
    semantic_message: str
    parsed_program: str | None
    n_visible: int
    n_hidden: int


@dataclass
class InfectionProcessDomain:
    """One episode over a single compiled infection_process environment."""

    seed: int = 0
    turn_budget: int = 6
    hidden_grid: tuple[float, ...] = (1.5, 6.5, 11.5, 16.5, 21.5, 31.5)
    verifier_draws: int = 50
    verifier_tune: int = 50

    _env: BaseEnv = field(init=False)
    _spec: EnvSpec = field(init=False)
    _runtime: PyMCRuntime = field(init=False)
    _tmpdir: Any = field(init=False, default=None)
    _visible_rows: list[tuple[Any, ...]] = field(init=False, default_factory=list)
    turns_used: int = field(init=False, default=0)
    done: bool = field(init=False, default=False)

    def reset(self) -> dict[str, Any]:
        # Validate the verifier reward source is registered before we promise a
        # reward; fail loud if the contract catalog is missing it.
        spec = get_reward_source(REWARD_SOURCE)
        if spec.evidence_kind is not EvidenceKind.verifier:
            raise RuntimeError(f"reward source {REWARD_SOURCE!r} is not a verifier source")

        self._spec = build_infection_process_spec(self.seed)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="agentic_env_")
        out_dir = Path(self._tmpdir.name)
        module_path = TemplateCompiler().compile_envspec(self._spec, out_dir)
        self._env = load_env_instance(
            module_path,
            env_name=self._spec.env_name,
            seed=self.seed,
            trusted_root=out_dir,
        )
        self._runtime = PyMCRuntime()
        self._visible_rows = []
        self.turns_used = 0
        self.done = False

        # Seed the visible buffer with a couple of grounding observations so the
        # initial summary is non-empty (real env queries, not fabricated rows).
        for t in (2.0, 12.0):
            self._observe_raw(t)

        cols = self._env.get_ordered_column_names()
        return {
            "task": self._task_text(),
            "system_message": self._env.generate_system_message(include_prior=False),
            "data_columns": cols,
            "target_name": self._spec.io_schema.target_name,
            "visible_data": _column_dict(self._env, self._visible_rows),
            "turns_remaining": self.turn_budget,
        }

    def _task_text(self) -> str:
        return (
            "You are a Bayesian experimental-design agent investigating an "
            "infection-process environment. You may query the environment at "
            "different time values t (>0) to observe how many of "
            f"{self._spec.generative_params['population_n']} individuals are "
            "infected by time t. After gathering evidence, submit a PyMC "
            "probabilistic program `def model(data):` that explains the observed "
            f"counts. The data dictionary keys are exactly "
            f"{self._env.get_ordered_column_names()}; each maps to a list of "
            "observations. Your program is scored by held-out predictive "
            "log-likelihood on unseen time points."
        )

    def _observe_raw(self, t: float) -> ObserveResult:
        result, ok = self._env.run_experiment(f"{float(t):.6f}")
        if not ok:
            return ObserveResult(
                ok=False,
                t=float(t),
                infected=None,
                message=str(result),
                n_visible=len(self._visible_rows),
            )
        # run_experiment appended (t, y) to env.observed_data; mirror it.
        self._visible_rows = list(self._env.get_data())
        return ObserveResult(
            ok=True,
            t=float(t),
            infected=int(result),
            message="observed",
            n_visible=len(self._visible_rows),
        )

    def observe_data(self, payload: dict[str, Any]) -> ObserveResult:
        if self.done:
            raise RuntimeError("episode is already terminal; cannot observe_data")
        if self.turns_used >= self.turn_budget:
            raise RuntimeError("turn budget exhausted")
        if "t" not in payload:
            raise ValueError("observe_data payload must include a numeric 't'")
        self.turns_used += 1
        return self._observe_raw(float(payload["t"]))

    def submit_program(self, code: str) -> SubmitResult:
        if self.done:
            raise RuntimeError("episode is already terminal; cannot submit again")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("submit_program requires non-empty program source")
        self.done = True

        visible_data = _column_dict(self._env, self._visible_rows)
        hidden_data = self._build_hidden_data()

        result = score_pymc_semantics(
            code,
            visible_data=visible_data,
            hidden_data=hidden_data,
            draws=self.verifier_draws,
            tune=self.verifier_tune,
        )

        log_reward = self._to_log_reward(result.metric_name, result.metric_value)
        reward_artifact = RewardArtifact(
            reward_source=REWARD_SOURCE,
            evidence_kind=EvidenceKind.verifier,
            truth_owner=TruthOwner.verifier,
            producer=REWARD_PRODUCER,
            training_log_reward=log_reward,
            reward_transform={
                "kind": "holdout_logp_floored",
                "floor": TRAINING_LOG_REWARD_FLOOR,
                "metric_name": result.metric_name,
            },
            metrics={
                k: v
                for k, v in {
                    "holdout_logp": result.metric_value
                    if result.metric_name == "avg_holdout_logp"
                    else None,
                    "train_elpd_loo": result.details.get("train_elpd_loo"),
                    "pareto_k_max": result.details.get("pareto_k_max"),
                }.items()
                if isinstance(v, (int, float))
            },
            diagnostics={
                "semantic_status": str(result.status),
                "semantic_message": result.message,
                "semantic_details": result.details,
                "verifier_ok": result.ok,
                "n_visible": len(self._visible_rows),
                "n_hidden": len(self.hidden_grid),
            },
            provenance={
                "env_id": self._spec.env_id,
                "env_name": self._spec.env_name,
                "generative_family": self._spec.generative_family,
                "seed": self.seed,
            },
        )

        return SubmitResult(
            reward_artifact=reward_artifact,
            metric_name=result.metric_name,
            metric_value=result.metric_value,
            semantic_status=str(result.status),
            semantic_message=result.message,
            parsed_program=code,
            n_visible=len(self._visible_rows),
            n_hidden=len(self.hidden_grid),
        )

    def _build_hidden_data(self) -> dict[str, list[float]]:
        """A held-out grid of (t, infected) rows from the SAME env (unseen t)."""
        rows: list[tuple[Any, ...]] = []
        for t in self.hidden_grid:
            y = self._env.step((float(t),))
            rows.append((float(t), int(y)))
        return _column_dict(self._env, rows)

    def _to_log_reward(self, metric_name: str | None, metric_value: float | None) -> float:
        if metric_value is None or metric_name != "avg_holdout_logp":
            return TRAINING_LOG_REWARD_FLOOR
        floored = max(TRAINING_LOG_REWARD_FLOOR, float(metric_value))
        return floored

    def extract_program(self, completion_text: str) -> str | None:
        return self._runtime.extract_code(completion_text)

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
