"""Pinned OTel attribute constants for SynthStats trajectories.

We do NOT inherit OpenTelemetry GenAI semantic-convention names directly because every
`gen_ai.*` attribute is currently in Development status in the OpenTelemetry semantic-
conventions repository (https://opentelemetry.io/docs/specs/semconv/gen-ai/). No `gen_ai.*`
attribute carries the Stable badge as of 2026; the GenAI conventions note an opt-in
`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` for newer experimental versions,
which confirms ongoing churn. Pinning our own names under the `synthstats.*` namespace gives us
backend-stable identifiers independent of that churn.

Naming rule: `synthstats.<domain>.<attribute>`. Domains: episode, turn, trial, observation,
belief, metric, reward, task, sandbox. Gradual adoption of any stabilised OTel GenAI names
happens via exporter-side translation, not by renaming these constants.

Do not delete or rename constants here without bumping the observability schema version.
Backends are indexed on these names.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[str] = "1.0.0"

# --- Episode ------------------------------------------------------------------
EPISODE_ID: Final[str] = "synthstats.episode.id"
EPISODE_TRAJECTORY_ID: Final[str] = "synthstats.episode.trajectory_id"
EPISODE_TASK_ID: Final[str] = "synthstats.episode.task_id"
EPISODE_SEED: Final[str] = "synthstats.episode.seed"
EPISODE_TURN_COUNT: Final[str] = "synthstats.episode.turn_count"
EPISODE_RELEASE_ID: Final[str] = "synthstats.episode.benchmark_release_id"

# --- Turn ---------------------------------------------------------------------
TURN_INDEX: Final[str] = "synthstats.turn.index"
TURN_ACTION_TYPE: Final[str] = "synthstats.turn.action_type"
TURN_ASSISTANT_TOKEN_COUNT: Final[str] = "synthstats.turn.assistant_token_count"
TURN_TRANSITION_REWARD: Final[str] = "synthstats.turn.transition_reward"

# --- Trial --------------------------------------------------------------------
TRIAL_ID: Final[str] = "synthstats.trial.id"
TRIAL_LANGUAGE: Final[str] = "synthstats.trial.language"
TRIAL_SANDBOX_BACKEND: Final[str] = "synthstats.trial.sandbox_backend"
TRIAL_WALL_SECONDS: Final[str] = "synthstats.trial.wall_seconds"

# --- Observation --------------------------------------------------------------
OBSERVATION_ID: Final[str] = "synthstats.observation.id"
OBSERVATION_EVIDENCE_KIND: Final[str] = "synthstats.observation.evidence_kind"
OBSERVATION_TRUTH_OWNER: Final[str] = "synthstats.observation.truth_owner"
OBSERVATION_SOURCE: Final[str] = "synthstats.observation.source"

# --- Belief -------------------------------------------------------------------
BELIEF_SNAPSHOT_ID: Final[str] = "synthstats.belief.snapshot_id"
BELIEF_OWNER: Final[str] = "synthstats.belief.owner"
BELIEF_MODEL_FAMILY: Final[str] = "synthstats.belief.model_family"
BELIEF_CONTENT_DIGEST: Final[str] = "synthstats.belief.content_digest"
BELIEF_SAMPLER_CONFIG_DIGEST: Final[str] = "synthstats.belief.sampler_config_digest"
BELIEF_SEED: Final[str] = "synthstats.belief.seed"

# --- Metric -------------------------------------------------------------------
METRIC_ID: Final[str] = "synthstats.metric.id"
METRIC_NAME: Final[str] = "synthstats.metric.name"
METRIC_VALUE: Final[str] = "synthstats.metric.value"
METRIC_SCHEDULE: Final[str] = "synthstats.metric.schedule"
METRIC_REWARD_SOURCE: Final[str] = "synthstats.metric.reward_source"

# --- Reward -------------------------------------------------------------------
REWARD_TOTAL: Final[str] = "synthstats.reward.total"
REWARD_SOURCE: Final[str] = "synthstats.reward.source"
REWARD_EVIDENCE_KIND: Final[str] = "synthstats.reward.evidence_kind"
REWARD_TRUTH_OWNER: Final[str] = "synthstats.reward.truth_owner"
REWARD_PRODUCER: Final[str] = "synthstats.reward.producer"
REWARD_SCALARIZATION: Final[str] = "synthstats.reward.scalarization"

# --- Task spec ----------------------------------------------------------------
TASK_SPEC_ID: Final[str] = "synthstats.task.spec_id"
TASK_DOMAIN: Final[str] = "synthstats.task.domain"
TASK_GOAL: Final[str] = "synthstats.task.goal"
TASK_ENV_SPEC_DIGEST: Final[str] = "synthstats.task.env_spec_digest"
TASK_PROMPT_TEMPLATE_HASH: Final[str] = "synthstats.task.prompt_template_hash"

# --- Sandbox ------------------------------------------------------------------
SANDBOX_BACKEND: Final[str] = "synthstats.sandbox.backend"
SANDBOX_IMAGE_DIGEST: Final[str] = "synthstats.sandbox.image_digest"
SANDBOX_EXIT_CODE: Final[str] = "synthstats.sandbox.exit_code"
SANDBOX_TIMEOUT_SECONDS: Final[str] = "synthstats.sandbox.timeout_seconds"

# --- OTel GenAI mirrors (stable subset we echo for backend compatibility) -----
# These are OTel-standard names we emit in addition to our synthstats.* keys,
# so dashboards that key on gen_ai.* still render correctly. Adding here does
# not break the pinning guarantee: we never rename, only extend.
GEN_AI_SYSTEM: Final[str] = "gen_ai.system"
GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL: Final[str] = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"

ALL_CONSTANTS: Final[tuple[str, ...]] = (
    EPISODE_ID,
    EPISODE_TRAJECTORY_ID,
    EPISODE_TASK_ID,
    EPISODE_SEED,
    EPISODE_TURN_COUNT,
    EPISODE_RELEASE_ID,
    TURN_INDEX,
    TURN_ACTION_TYPE,
    TURN_ASSISTANT_TOKEN_COUNT,
    TURN_TRANSITION_REWARD,
    TRIAL_ID,
    TRIAL_LANGUAGE,
    TRIAL_SANDBOX_BACKEND,
    TRIAL_WALL_SECONDS,
    OBSERVATION_ID,
    OBSERVATION_EVIDENCE_KIND,
    OBSERVATION_TRUTH_OWNER,
    OBSERVATION_SOURCE,
    BELIEF_SNAPSHOT_ID,
    BELIEF_OWNER,
    BELIEF_MODEL_FAMILY,
    BELIEF_CONTENT_DIGEST,
    BELIEF_SAMPLER_CONFIG_DIGEST,
    BELIEF_SEED,
    METRIC_ID,
    METRIC_NAME,
    METRIC_VALUE,
    METRIC_SCHEDULE,
    METRIC_REWARD_SOURCE,
    REWARD_TOTAL,
    REWARD_SOURCE,
    REWARD_EVIDENCE_KIND,
    REWARD_TRUTH_OWNER,
    REWARD_PRODUCER,
    REWARD_SCALARIZATION,
    TASK_SPEC_ID,
    TASK_DOMAIN,
    TASK_GOAL,
    TASK_ENV_SPEC_DIGEST,
    TASK_PROMPT_TEMPLATE_HASH,
    SANDBOX_BACKEND,
    SANDBOX_IMAGE_DIGEST,
    SANDBOX_EXIT_CODE,
    SANDBOX_TIMEOUT_SECONDS,
    GEN_AI_SYSTEM,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
