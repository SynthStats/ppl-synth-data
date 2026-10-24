from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ppl_synth_data.harbor import agent as agent_module
from ppl_synth_data.harbor.agent import (
    build_atif_record,
    build_messages,
    complete_subtb_generation_metadata,
    extract_generation_metadata,
    extract_submitted_program,
    parse_json_env,
    render_openai_message,
    require_llm_api_base,
    require_openai_api_key,
    runtime_from_task_config,
    submission_path_for_runtime,
    write_submission_program,
    write_text_to_environment,
)


def test_require_llm_api_base_requires_explicit_endpoint() -> None:
    with pytest.raises(ValueError, match="LLM_API_BASE"):
        require_llm_api_base({})

    assert require_llm_api_base({"LLM_API_BASE": "http://127.0.0.1:8000/v1"}) == (
        "http://127.0.0.1:8000/v1"
    )


def test_require_openai_api_key_requires_explicit_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        require_openai_api_key({})

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        require_openai_api_key({"LLM_API_BASE": "https://api.openai.com/v1"})

    assert (
        require_openai_api_key({"LLM_API_BASE": "http://127.0.0.1:8000/v1"})
        == "synthstats-local-openai-compatible-endpoint"
    )

    assert require_openai_api_key({"OPENAI_API_KEY": "sk-real-provider-key"}) == (
        "sk-real-provider-key"
    )


def test_parse_json_env_requires_object() -> None:
    assert parse_json_env("TASK_CONFIG", env={}) == {}
    assert parse_json_env("TASK_CONFIG", env={"TASK_CONFIG": '{"runtime":"pymc"}'}) == {
        "runtime": "pymc"
    }

    with pytest.raises(ValueError, match="JSON object"):
        parse_json_env("TASK_CONFIG", env={"TASK_CONFIG": '["not", "an", "object"]'})


def test_runtime_and_submission_paths_are_domain_owned() -> None:
    assert runtime_from_task_config({"runtime": "LazyPPL"}) == "lazyppl"
    assert submission_path_for_runtime("pymc").as_posix() == "/workspace/submission.py"
    assert submission_path_for_runtime("lazyppl").as_posix() == "/app/Submission.hs"

    with pytest.raises(ValueError, match="Unsupported"):
        submission_path_for_runtime("stan")


def test_extract_submitted_program_prefers_tool_call() -> None:
    text = (
        "reasoning\n"
        "<tool_call>"
        + json.dumps(
            {
                "name": "submit_program",
                "arguments": {"code": "module Submission where\nanswer = 1"},
            }
        )
        + "</tool_call>"
    )

    assert extract_submitted_program(text, runtime="lazyppl").startswith("module Submission")


def test_extract_submitted_program_uses_matching_fence() -> None:
    text = "```text\nignore me\n```\n```python\nimport pymc as pm\n```"

    assert extract_submitted_program(text, runtime="pymc") == "import pymc as pm"


def test_extract_submitted_program_uses_submit_program_tag() -> None:
    text = (
        "<think>reason privately</think>\n"
        '<submit_program language="haskell">\n'
        "module Submission where\nanswer = 1\n"
        "</submit_program>"
    )

    assert extract_submitted_program(text, runtime="lazyppl") == (
        "module Submission where\nanswer = 1"
    )


def test_build_messages_requests_non_thinking_program_only() -> None:
    messages = build_messages("write a model", {"runtime": "lazyppl"})

    assert "Do not output hidden reasoning" in messages[0]["content"]
    assert "<submit_program>" in messages[0]["content"]
    assert messages[1]["content"].startswith("/no_think\n\n")


def test_render_openai_message_includes_tool_call() -> None:
    choice = {
        "message": {
            "content": "draft",
            "tool_calls": [
                {
                    "function": {
                        "name": "submit_program",
                        "arguments": '{"code":"x = 1"}',
                    }
                }
            ],
        }
    }

    rendered = render_openai_message(choice)

    assert "draft" in rendered
    assert "<tool_call>" in rendered
    assert "submit_program" in rendered


def test_extract_generation_metadata_requires_aligned_token_ids_and_logprobs() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [10, 11],
                "completion_logprobs": [-0.1, -0.2],
                "eos_logprobs": [-2.0, -1.5],
                "token_entropies": [0.4, 0.5],
                "message": {"content": "x"},
            }
        ]
    }

    assert extract_generation_metadata(
        response,
        model_name="Qwen/Qwen3.5-0.8B",
        response_text="x",
        prompt_messages=build_messages("write a model", {"runtime": "pymc"}),
    ) == {
        "token_ids": [10, 11],
        "prompt_ids": [1, 2, 3],
        "mask_ids": [1, 1],
        "logprobs": [-0.1, -0.2],
        "stop_reason": "stop",
        "eos_logprobs": [-2.0, -1.5],
        "token_entropies": [0.4, 0.5],
        "entropy_source": "llm_response.token_entropies",
    }

    with pytest.raises(ValueError, match="aligned"):
        extract_generation_metadata(
            {"choices": [{"completion_token_ids": [10], "message": {"content": "x"}}]},
            model_name="Qwen/Qwen3.5-0.8B",
            response_text="x",
            prompt_messages=build_messages("write a model", {"runtime": "pymc"}),
        )


@pytest.mark.narrow_unit_not_acceptance
def test_extract_generation_metadata_tokenizes_when_response_omits_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del add_special_tokens
            return [ord(char) for char in text]

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> list[int]:
            del tokenize, add_generation_prompt, kwargs
            return [len(message["content"]) for message in messages]

    monkeypatch.setitem(agent_module._TOKENIZER_CACHE, "fake/model", FakeTokenizer())
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "completion_logprobs": [-0.1, -0.2],
                "message": {"content": "xy"},
            }
        ]
    }
    messages = build_messages("write a model", {"runtime": "pymc"})

    metadata = extract_generation_metadata(
        response,
        model_name="fake/model",
        response_text="xy",
        prompt_messages=messages,
    )

    assert metadata["token_ids"] == [120, 121]
    assert metadata["prompt_ids"] == [len(message["content"]) for message in messages]


@pytest.mark.narrow_unit_not_acceptance
def test_prompt_tokenization_accepts_batch_encoding_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBatchEncoding(dict):
        @property
        def input_ids(self) -> list[int]:
            return self["input_ids"]

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del add_special_tokens
            return [ord(char) for char in text]

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> FakeBatchEncoding:
            del messages, tokenize, add_generation_prompt, kwargs
            return FakeBatchEncoding(input_ids=[101, 102, 103])

    monkeypatch.setitem(agent_module._TOKENIZER_CACHE, "fake/batch", FakeTokenizer())
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "completion_logprobs": [-0.1, -0.2],
                "message": {"content": "xy"},
            }
        ]
    }

    metadata = extract_generation_metadata(
        response,
        model_name="fake/batch",
        response_text="xy",
        prompt_messages=build_messages("write a model", {"runtime": "pymc"}),
    )

    assert metadata["prompt_ids"] == [101, 102, 103]


def test_build_atif_record_preserves_prompt_and_eos_logprobs() -> None:
    messages = build_messages("write a model", {"runtime": "pymc"})
    atif = build_atif_record(
        messages=messages,
        response_text="```python\nx = 1\n```",
        generation_metadata={
            "prompt_ids": [9, 8, 7],
            "token_ids": [1, 2],
            "logprobs": [-0.1, -0.2],
            "eos_logprobs": [-3.0, -2.0],
            "stop_reason": "stop",
        },
        task_config={"runtime": "pymc"},
        task_provenance={"task_name": "dugongs"},
    )

    assert atif["schema_version"] == "ATIF-v1.6"
    assert [step["source"] for step in atif["steps"]] == ["system", "user", "agent"]
    assert atif["steps"][2]["metrics"]["prompt_token_ids"] == [9, 8, 7]
    assert atif["steps"][2]["metrics"]["completion_token_ids"] == [1, 2]
    assert atif["steps"][2]["metrics"]["eos_logprobs"] == [-3.0, -2.0]


@pytest.mark.narrow_unit_not_acceptance
def test_complete_subtb_generation_metadata_appends_terminal_eos_and_scores_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        eos_token_id = 99

        def decode(self, token_ids: list[int], **_: Any) -> str:
            return json.dumps(token_ids)

    class FakeCompletions:
        def __init__(self) -> None:
            self.prompt_batches: list[list[str]] = []

        async def create(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["max_tokens"] == 0
            assert kwargs["echo"] is True
            assert kwargs["extra_body"]["prompt_logprobs"] == 1
            assert kwargs["extra_body"]["return_token_ids"] is True
            prompt_batch = kwargs["prompt"]
            self.prompt_batches.append(prompt_batch)
            choices = []
            for index, prompt_text in enumerate(prompt_batch):
                prompt_ids = json.loads(prompt_text)
                assert prompt_ids[-1] == 99
                choices.append(
                    {
                        "prompt_token_ids": prompt_ids,
                        "logprobs": {
                            "token_logprobs": [None] * (len(prompt_ids) - 1) + [-3.0 - index],
                        },
                    }
                )
            return {"choices": choices}

    class FakeClient:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    monkeypatch.setitem(agent_module._TOKENIZER_CACHE, "fake/model", FakeTokenizer())
    client = FakeClient()

    completed = asyncio.run(
        complete_subtb_generation_metadata(
            client,
            model_name="fake/model",
            generation_metadata={
                "prompt_ids": [1, 2],
                "token_ids": [10, 11],
                "logprobs": [-0.1, -0.2],
                "mask_ids": [1, 1],
                "stop_reason": "stop",
            },
        )
    )

    assert completed["token_ids"] == [10, 11, 99]
    assert completed["logprobs"] == [-0.1, -0.2, -5.0]
    assert completed["mask_ids"] == [1, 1, 1]
    assert completed["eos_logprobs"] == [-3.0, -4.0, -5.0]
    assert completed["terminal_eos_token_count"] == 1
    assert completed["terminal_eos_token_coverage"] == pytest.approx(1.0)
    assert client.completions.prompt_batches == [
        ["[1, 2, 99]", "[1, 2, 10, 99]", "[1, 2, 10, 11, 99]"]
    ]


@pytest.mark.narrow_unit_not_acceptance
def test_complete_subtb_generation_metadata_honors_eos_score_chunk_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        eos_token_id = 99

        def decode(self, token_ids: list[int], **_: Any) -> str:
            return json.dumps(token_ids)

    class FakeCompletions:
        def __init__(self) -> None:
            self.prompt_batches: list[list[str]] = []

        async def create(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["max_tokens"] == 0
            assert kwargs["echo"] is True
            assert kwargs["extra_body"]["prompt_logprobs"] == 1
            assert kwargs["extra_body"]["return_token_ids"] is True
            prompt_batch = kwargs["prompt"]
            self.prompt_batches.append(prompt_batch)
            choices = []
            for prompt_text in prompt_batch:
                prompt_ids = json.loads(prompt_text)
                choices.append(
                    {
                        "prompt_token_ids": prompt_ids,
                        "logprobs": {
                            "token_logprobs": [None] * (len(prompt_ids) - 1) + [-4.0],
                        },
                    }
                )
            return {"choices": choices}

    class FakeClient:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    monkeypatch.setitem(agent_module._TOKENIZER_CACHE, "fake/model", FakeTokenizer())
    monkeypatch.setenv("SUBTB_EOS_SCORE_CHUNK_SIZE", "1")
    client = FakeClient()

    completed = asyncio.run(
        complete_subtb_generation_metadata(
            client,
            model_name="fake/model",
            generation_metadata={
                "prompt_ids": [1, 2],
                "token_ids": [10, 11],
                "logprobs": [-0.1, -0.2],
                "mask_ids": [1, 1],
            },
        )
    )

    assert completed["eos_logprobs"] == [-4.0, -4.0, -4.0]
    assert client.completions.prompt_batches == [
        ["[1, 2, 99]"],
        ["[1, 2, 10, 99]"],
        ["[1, 2, 10, 11, 99]"],
    ]


def test_write_submission_program_uses_runtime_specific_path(tmp_path) -> None:
    output_path = write_submission_program("x = 1", runtime="pymc", root=tmp_path)

    assert output_path == tmp_path / "workspace" / "submission.py"
    assert output_path.read_text() == "x = 1"


@pytest.mark.narrow_unit_not_acceptance
def test_write_text_to_environment_writes_via_harbor_exec(tmp_path: Path) -> None:
    calls: list[tuple[object, str]] = []

    class FakeAgent:
        async def exec_as_root(self, environment: object, *, command: str) -> None:
            calls.append((environment, command))

    environment = object()
    target = tmp_path / "logs" / "agent" / "trajectory.json"

    asyncio.run(
        write_text_to_environment(
            FakeAgent(),
            environment,
            path=target,
            text='{"ok": true}',
        )
    )

    assert len(calls) == 1
    assert calls[0][0] is environment
    assert str(target) in calls[0][1]
    assert '{"ok": true}' not in calls[0][1]


@pytest.mark.narrow_unit_not_acceptance
def test_write_text_to_environment_chunks_large_payload_and_executes_shell(
    tmp_path: Path,
) -> None:
    commands: list[str] = []

    class ShellAgent:
        async def exec_as_root(self, environment: object, *, command: str) -> None:
            del environment
            commands.append(command)
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)

    target = tmp_path / "logs" / "agent" / "trajectory.json"
    payload = json.dumps({"text": "abc123\n" * 40_000})

    asyncio.run(
        write_text_to_environment(
            ShellAgent(),
            object(),
            path=target,
            text=payload,
        )
    )

    assert target.read_text(encoding="utf-8") == payload
    assert len(commands) > 1
    assert max(len(command) for command in commands) < 20_000
    assert "abc123" not in "\n".join(commands)


@pytest.mark.narrow_unit_not_acceptance
def test_synthstats_harbor_agent_satisfies_base_agent_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harbor_module = types.ModuleType("harbor")
    agents_module = types.ModuleType("harbor.agents")
    base_module = types.ModuleType("harbor.agents.base")

    class FakeBaseAgent(ABC):
        SUPPORTS_ATIF = False

        def __init__(
            self,
            *,
            logs_dir: Path,
            model_name: str | None = None,
            **kwargs: Any,
        ) -> None:
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"unexpected kwargs: {unexpected}")
            self.logs_dir = logs_dir
            self.model_name = model_name

        @staticmethod
        @abstractmethod
        def name() -> str:
            raise NotImplementedError

        @abstractmethod
        def version(self) -> str | None:
            raise NotImplementedError

        @abstractmethod
        async def setup(self, environment: Any) -> None:
            raise NotImplementedError

        @abstractmethod
        async def run(self, instruction: str, environment: Any, context: Any) -> None:
            raise NotImplementedError

    base_module.BaseAgent = FakeBaseAgent
    monkeypatch.setitem(sys.modules, "harbor", harbor_module)
    monkeypatch.setitem(sys.modules, "harbor.agents", agents_module)
    monkeypatch.setitem(sys.modules, "harbor.agents.base", base_module)

    spec = importlib.util.spec_from_file_location(
        "ppl_synth_data_harbor_agent_contract_probe",
        Path(agent_module.__file__),
    )
    assert spec is not None
    assert spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    Agent = probe.SynthStatsInstalledAgent
    assert probe.HARBOR_AGENT_AVAILABLE is True
    assert Agent is not None
    assert not Agent.__abstractmethods__
    assert Agent.SUPPORTS_ATIF is True
    assert Agent.name() == "synthstats"

    agent = Agent(
        logs_dir=tmp_path,
        model_name="Qwen/Qwen3.5-0.8B",
        extra_env={
            "LLM_API_BASE": "http://127.0.0.1:8000/v1",
            "MODEL_NAME": "Qwen/Qwen3.5-0.8B",
            "TASK_CONFIG": '{"runtime":"pymc"}',
            "TASK_PROVENANCE": "{}",
        },
        collect_rollout_details=True,
    )
    assert agent.version() == "ppl-synth-data"

    @dataclass
    class ExecResult:
        return_code: int = 0
        stderr: str = ""

    class FakeEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def exec(self, *, command: str, **kwargs: Any) -> ExecResult:
            del kwargs
            self.commands.append(command)
            return ExecResult()

    environment = FakeEnvironment()
    asyncio.run(agent.setup(environment))
    asyncio.run(agent.exec_as_root(environment, command="true"))

    assert environment.commands == [
        "mkdir -p /logs/agent /logs/artifacts /workspace && chmod 777 /logs/agent /logs/artifacts",
        "true",
    ]

    lazyppl_agent = Agent(
        logs_dir=tmp_path,
        model_name="Qwen/Qwen3.5-0.8B",
        extra_env={
            "LLM_API_BASE": "http://127.0.0.1:8000/v1",
            "MODEL_NAME": "Qwen/Qwen3.5-0.8B",
            "TASK_CONFIG": '{"runtime":"lazyppl"}',
            "TASK_PROVENANCE": "{}",
        },
        collect_rollout_details=True,
    )
    lazyppl_environment = FakeEnvironment()
    asyncio.run(lazyppl_agent.setup(lazyppl_environment))
    assert lazyppl_environment.commands == [
        "mkdir -p /logs/agent /logs/artifacts /app && chmod 777 /logs/agent /logs/artifacts",
    ]

    captured_chat_kwargs: dict[str, Any] = {}

    class FakeChatCompletions:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            captured_chat_kwargs.update(kwargs)
            return {"choices": [{"message": {"content": "```haskell\nx = 1\n```"}}]}

    lazyppl_agent._openai_client = lambda: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeChatCompletions())
    )
    asyncio.run(lazyppl_agent._create_chat_completion(build_messages("x", {"runtime": "lazyppl"})))
    assert captured_chat_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    async def fake_create_chat_completion(messages: list[dict[str, str]]) -> dict[str, Any]:
        assert messages[0]["role"] == "system"
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [10, 11, 12],
                    "completion_logprobs": [-0.1, -0.2, -0.3],
                    "eos_logprobs": [-2.0, -1.8, -1.6],
                    "message": {
                        "content": (
                            "```python\nimport pymc as pm\n\ndef model(data):\n    return None\n```"
                        )
                    },
                }
            ]
        }

    class StrictContext:
        __slots__ = ("metadata",)

        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {"existing": "kept"}

    context = StrictContext()
    agent._create_chat_completion = fake_create_chat_completion
    probe._TOKENIZER_CACHE["Qwen/Qwen3.5-0.8B"] = types.SimpleNamespace(eos_token_id=99)

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("TB agent runs must not trigger prompt-logprob EOS scoring")

    agent._openai_client = lambda: types.SimpleNamespace(completions=FakeCompletions())
    asyncio.run(agent.run("write a PyMC model", environment, context))

    assert context.metadata["existing"] == "kept"
    assert context.metadata["language"] == "pymc"
    assert context.metadata["trajectory"]["schema_version"] == "ATIF-v1.6"
    assert "terminal_eos_token_count" not in context.metadata
    assert not hasattr(context, "trajectory")
    assert any("/workspace/submission.py" in command for command in environment.commands)
    assert any("/logs/agent/synthstats_result.json" in command for command in environment.commands)
    assert any("/logs/agent/trajectory.json" in command for command in environment.commands)
    assert any(
        "/logs/artifacts/synthstats_result.json" in command for command in environment.commands
    )
    assert any("/logs/artifacts/trajectory.json" in command for command in environment.commands)


@pytest.mark.narrow_unit_not_acceptance
def test_agent_subtb_mode_completes_eos_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harbor_module = types.ModuleType("harbor")
    agents_module = types.ModuleType("harbor.agents")
    base_module = types.ModuleType("harbor.agents.base")

    class FakeBaseAgent(ABC):
        SUPPORTS_ATIF = False

        def __init__(
            self,
            *,
            logs_dir: Path,
            model_name: str | None = None,
            **kwargs: Any,
        ) -> None:
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"unexpected kwargs: {unexpected}")
            self.logs_dir = logs_dir
            self.model_name = model_name

        @staticmethod
        @abstractmethod
        def name() -> str:
            raise NotImplementedError

        @abstractmethod
        def version(self) -> str | None:
            raise NotImplementedError

        @abstractmethod
        async def setup(self, environment: Any) -> None:
            raise NotImplementedError

        @abstractmethod
        async def run(self, instruction: str, environment: Any, context: Any) -> None:
            raise NotImplementedError

    base_module.BaseAgent = FakeBaseAgent
    monkeypatch.setitem(sys.modules, "harbor", harbor_module)
    monkeypatch.setitem(sys.modules, "harbor.agents", agents_module)
    monkeypatch.setitem(sys.modules, "harbor.agents.base", base_module)

    spec = importlib.util.spec_from_file_location(
        "ppl_synth_data_harbor_agent_subtb_probe",
        Path(agent_module.__file__),
    )
    assert spec is not None
    assert spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    Agent = probe.SynthStatsInstalledAgent
    assert Agent is not None

    agent = Agent(
        logs_dir=tmp_path,
        model_name="Qwen/Qwen3.5-0.8B",
        extra_env={
            "LLM_API_BASE": "http://127.0.0.1:8000/v1",
            "MODEL_NAME": "Qwen/Qwen3.5-0.8B",
            "REQUIRE_EOS_LOGPROBS": "1",
            "TASK_CONFIG": '{"runtime":"pymc"}',
            "TASK_PROVENANCE": "{}",
        },
        collect_rollout_details=True,
    )

    @dataclass
    class ExecResult:
        return_code: int = 0
        stderr: str = ""

    class FakeEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def exec(self, *, command: str, **kwargs: Any) -> ExecResult:
            del kwargs
            self.commands.append(command)
            return ExecResult()

    environment = FakeEnvironment()
    asyncio.run(agent.setup(environment))

    async def fake_create_chat_completion(messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [10, 11],
                    "completion_logprobs": [-0.1, -0.2],
                    "message": {"content": "```python\nimport pymc as pm\n```"},
                }
            ]
        }

    class StrictContext:
        __slots__ = ("metadata",)

        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}

    context = StrictContext()
    agent._create_chat_completion = fake_create_chat_completion

    class FakeTokenizer:
        eos_token_id = 99

        def decode(self, token_ids: list[int], **_: Any) -> str:
            return json.dumps(token_ids)

    probe._TOKENIZER_CACHE["Qwen/Qwen3.5-0.8B"] = FakeTokenizer()

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            choices = []
            for prompt_text in kwargs["prompt"]:
                prompt_ids = json.loads(prompt_text)
                choices.append(
                    {
                        "prompt_token_ids": prompt_ids,
                        "logprobs": {
                            "token_logprobs": [None] * (len(prompt_ids) - 1) + [-4.0],
                        },
                    }
                )
            return {"choices": choices}

    agent._openai_client = lambda: types.SimpleNamespace(completions=FakeCompletions())
    asyncio.run(agent.run("write a PyMC model", environment, context))

    assert context.metadata["terminal_eos_token_count"] == 1
    assert context.metadata["terminal_eos_token_coverage"] == pytest.approx(1.0)
    assert context.metadata["eos_logprob_source"] == "vllm.prompt_logprobs"
    assert context.metadata["token_ids"] == [10, 11, 99]
    assert context.metadata["eos_logprobs"] == [-4.0, -4.0, -4.0]
