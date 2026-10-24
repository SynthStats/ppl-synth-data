"""Cold-start (teacher) agentic rollout runner.

The policy is the repo's real OpenAI backend. It is shown the env
system message and a JSON tool schema, then drives a multi-turn loop:

    observe_data({"t": ...})  -> query the environment that provides observations
    submit_program({"code": ...}) -> terminate, score with the held-out verifier

Each turn appends a real ``TraceStep`` (assistant message + tool_calls +
observation). On submission a real ``RewardArtifact`` (registered verifier
source, evidence_kind=verifier) and a real ``ATIFRecord`` are written to
``artifacts/agentic_episodes/<episode>/`` as ``trajectory.json`` + ``reward.json``.

Fail-loud: missing OpenAI key/SDK, missing PyMC/env/verifier all raise. No stub
policy, no fake verifier, no silent fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthstats_schemas.trace import ATIFRecord, TraceStep

from ppl_synth_data.agentic.domain import InfectionProcessDomain
from ppl_synth_data.config import Settings
from ppl_synth_data.llm.backend import LLMBackend, LLMRequest
from ppl_synth_data.pipeline.backends import build_backend, resolve_stage_model
from ppl_synth_data.pipeline.io import write_json

log = logging.getLogger(__name__)

ARTIFACT_ROOT = Path("artifacts/agentic_episodes")

_TOOL_SCHEMA = {
    "observe_data": {
        "description": (
            "Query the environment at a positive time t. Returns the number of "
            "infected individuals observed at that t and appends it to the "
            "visible data buffer."
        ),
        "arguments": {"t": "float > 0"},
    },
    "submit_program": {
        "description": (
            "Terminal action. Submit a PyMC program. The 'code' string must "
            "define `def model(data):` using only the data dictionary keys "
            "provided, and return a pm.Model. Scored by held-out predictive "
            "log-likelihood."
        ),
        "arguments": {"code": "string (python source defining def model(data):)"},
    },
}


def _build_system_prompt(reset_payload: dict[str, Any]) -> str:
    return (
        f"{reset_payload['system_message']}\n\n"
        f"{reset_payload['task']}\n\n"
        "You act by emitting EXACTLY ONE JSON object per turn and nothing else "
        "(no prose, no markdown fences). The JSON object must have this shape:\n"
        '  {"tool": "observe_data", "arguments": {"t": <float>}}\n'
        "or\n"
        '  {"tool": "submit_program", "arguments": {"code": "<python source>"}}\n\n'
        "Tool catalog:\n"
        f"{json.dumps(_TOOL_SCHEMA, indent=2)}\n\n"
        f"Data dictionary keys for `def model(data):` are exactly "
        f"{reset_payload['data_columns']}; each key maps to a python list of "
        "observations. Gather a few observations with observe_data across a "
        "range of t, then submit_program. Do not exceed the turn budget."
    )


def _render_state(reset_payload: dict[str, Any], domain: InfectionProcessDomain) -> str:
    visible = json.dumps(reset_payload["visible_data"])
    return (
        f"Current visible data (dict of column -> list): {visible}\n"
        f"Turns used: {domain.turns_used}/{domain.turn_budget}. "
        "Emit your next JSON action."
    )


def _parse_action(text: str) -> dict[str, Any]:
    """Structurally parse the policy's JSON tool-call.

    Tries strict JSON first, then the first balanced ``{...}`` object substring.
    No keyword/regex content banks: we parse structure, then validate the
    'tool' and 'arguments' shape.
    """
    candidate = text.strip()
    obj = _try_json_object(candidate)
    if obj is None:
        obj = _try_json_object(_first_json_object(candidate))
    if obj is None:
        raise ValueError(f"policy output is not a JSON tool-call: {candidate[:200]!r}")
    tool = obj.get("tool")
    arguments = obj.get("arguments")
    if tool not in _TOOL_SCHEMA:
        raise ValueError(f"unknown tool {tool!r}; expected one of {list(_TOOL_SCHEMA)}")
    if not isinstance(arguments, dict):
        raise ValueError("tool-call 'arguments' must be a JSON object")
    return {"tool": tool, "arguments": arguments}


def _try_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _first_json_object(text: str) -> str | None:
    """Return the first balanced brace-delimited substring (handles strings)."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def run_one_episode(
    *,
    backend: LLMBackend,
    model: str,
    seed: int,
    turn_budget: int,
    out_dir: Path,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    domain = InfectionProcessDomain(seed=seed, turn_budget=turn_budget)
    reset_payload = domain.reset()
    system_prompt = _build_system_prompt(reset_payload)

    steps: list[TraceStep] = []
    steps.append(
        TraceStep(
            step_id="env_reset",
            source="environment",
            role="system",
            content=system_prompt,
            observation={
                "visible_data": reset_payload["visible_data"],
                "data_columns": reset_payload["data_columns"],
            },
            metrics={"turns_remaining": float(turn_budget)},
        )
    )

    transcript: list[str] = [system_prompt, _render_state(reset_payload, domain)]
    submit_result = None

    try:
        for turn in range(turn_budget + 1):
            user_text = "\n\n".join(transcript[1:])
            request = LLMRequest(
                model=model,
                system=transcript[0],
                user=user_text,
                reasoning_effort=reasoning_effort,
                max_output_tokens=4000,
            )
            output = backend.complete(request)
            raw = output.content
            action = _parse_action(raw)
            tool = action["tool"]
            arguments = action["arguments"]

            if tool == "observe_data":
                obs = domain.observe_data(arguments)
                observation = {
                    "ok": obs.ok,
                    "t": obs.t,
                    "infected": obs.infected,
                    "message": obs.message,
                    "n_visible": obs.n_visible,
                }
                steps.append(
                    TraceStep(
                        step_id=f"turn_{turn:02d}",
                        source="policy",
                        role="assistant",
                        content=raw,
                        tool_calls=[{"tool": tool, "arguments": arguments}],
                        observation=observation,
                        metrics={"turns_used": float(domain.turns_used)},
                    )
                )
                reset_payload["visible_data"] = _refresh_visible(domain)
                transcript.append(
                    f"observe_data result: {json.dumps(observation)}\n\n"
                    + _render_state(reset_payload, domain)
                )
                continue

            # submit_program (terminal)
            code = _extract_submission_code(arguments, raw, domain)
            submit_result = domain.submit_program(code)
            steps.append(
                TraceStep(
                    step_id=f"turn_{turn:02d}_submit",
                    source="policy",
                    role="assistant",
                    content=raw,
                    tool_calls=[{"tool": tool, "arguments": {"code": code}}],
                    observation={
                        "semantic_status": submit_result.semantic_status,
                        "semantic_message": submit_result.semantic_message,
                        "metric_name": submit_result.metric_name,
                        "metric_value": submit_result.metric_value,
                        "n_visible": submit_result.n_visible,
                        "n_hidden": submit_result.n_hidden,
                    },
                    metrics={
                        "training_log_reward": float(
                            submit_result.reward_artifact.training_log_reward
                        )
                    },
                )
            )
            break
    finally:
        if submit_result is None:
            domain.close()

    if submit_result is None:
        raise RuntimeError(
            f"episode (seed={seed}) ended without a submitted program within "
            f"the turn budget of {turn_budget}; the teacher policy must submit"
        )

    atif = ATIFRecord(
        session_id=f"agentic-infection-{seed:06d}-{uuid.uuid4().hex[:8]}",
        steps=steps,
        metadata={
            "env_family": "infection_process",
            "policy_model": model,
            "seed": seed,
            "turn_budget": turn_budget,
            "turns_used": domain.turns_used,
            "created_utc": datetime.now(UTC).isoformat(),
            "rollout_kind": "cold_start_teacher",
        },
        reward_artifact=submit_result.reward_artifact,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_dir / "trajectory.json"
    reward_path = out_dir / "reward.json"
    write_json(trajectory_path, atif.model_dump(mode="json"))
    write_json(reward_path, submit_result.reward_artifact.model_dump(mode="json"))
    domain.close()

    return {
        "seed": seed,
        "training_log_reward": float(submit_result.reward_artifact.training_log_reward),
        "metric_name": submit_result.metric_name,
        "metric_value": submit_result.metric_value,
        "semantic_status": submit_result.semantic_status,
        "turns_used": domain.turns_used,
        "trajectory_path": str(trajectory_path),
        "reward_path": str(reward_path),
    }


def _refresh_visible(domain: InfectionProcessDomain) -> dict[str, Any]:
    from ppl_synth_data.agentic.domain import _column_dict

    return _column_dict(domain._env, domain._visible_rows)


def _extract_submission_code(
    arguments: dict[str, Any],
    raw: str,
    domain: InfectionProcessDomain,
) -> str:
    code = arguments.get("code")
    if isinstance(code, str) and code.strip():
        return code
    # The policy may have emitted the program inside a <submit_program> block or
    # python fence instead of the 'code' argument; use the real extractor.
    extracted = domain.extract_program(raw)
    if extracted:
        return extracted
    raise ValueError("submit_program had no usable 'code' argument or program block")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--turn-budget", type=int, default=6)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ARTIFACT_ROOT,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    settings = Settings()
    # Fail loud now if the API key is missing, before any episode runs.
    settings.require_openai()

    backend = build_backend(args.backend or "openai", settings=settings)
    model = resolve_stage_model("completions", args.model)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.out_root / run_id

    results: list[dict[str, Any]] = []
    for i in range(args.episodes):
        seed = args.seed + i
        episode_dir = run_root / f"episode_{i:03d}_seed_{seed:06d}"
        log.info("running agentic episode %d (seed=%d, model=%s)", i, seed, model)
        result = run_one_episode(
            backend=backend,
            model=model,
            seed=seed,
            turn_budget=args.turn_budget,
            out_dir=episode_dir,
            reasoning_effort=args.reasoning_effort,
        )
        results.append(result)
        log.info(
            "episode %d done: training_log_reward=%.4f status=%s turns=%d -> %s",
            i,
            result["training_log_reward"],
            result["semantic_status"],
            result["turns_used"],
            result["trajectory_path"],
        )

    summary_path = run_root / "summary.json"
    write_json(
        summary_path,
        {
            "run_id": run_id,
            "model": model,
            "episodes": len(results),
            "rewards": [r["training_log_reward"] for r in results],
            "results": results,
        },
    )
    log.info("wrote run summary -> %s", summary_path)
    print(
        json.dumps(
            {"run_id": run_id, "summary": str(summary_path), "results": results},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
