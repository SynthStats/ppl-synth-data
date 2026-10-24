from __future__ import annotations

import pytest

from ppl_synth_data.llm.response_utils import extract_chat_text


def test_string_content() -> None:
    raw = {"choices": [{"message": {"content": "hello world"}}]}
    assert extract_chat_text(raw, backend_name="test") == "hello world"


def test_list_content_with_text_dicts() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "part1"},
                        {"type": "text", "text": "part2"},
                    ]
                }
            }
        ]
    }
    assert extract_chat_text(raw, backend_name="test") == "part1part2"


def test_list_content_with_strings() -> None:
    raw = {"choices": [{"message": {"content": ["ab", "cd"]}}]}
    assert extract_chat_text(raw, backend_name="test") == "abcd"


def test_missing_choices_raises() -> None:
    with pytest.raises(RuntimeError, match="missing choices"):
        extract_chat_text({}, backend_name="mybackend")


def test_empty_choices_raises() -> None:
    with pytest.raises(RuntimeError, match="missing choices"):
        extract_chat_text({"choices": []}, backend_name="test")


def test_non_dict_choice_raises() -> None:
    with pytest.raises(RuntimeError, match="unsupported choices format"):
        extract_chat_text({"choices": ["bad"]}, backend_name="test")


def test_missing_message_raises() -> None:
    with pytest.raises(RuntimeError, match="unsupported message format"):
        extract_chat_text({"choices": [{}]}, backend_name="test")


def test_unsupported_content_type_raises() -> None:
    raw = {"choices": [{"message": {"content": 12345}}]}
    with pytest.raises(RuntimeError, match="unsupported content format"):
        extract_chat_text(raw, backend_name="test")
