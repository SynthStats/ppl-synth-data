from __future__ import annotations

import pytest
from pydantic import ValidationError

from ppl_synth_data.specs.models import CompletionRecord, PromptRecord


def test_prompt_record_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        PromptRecord.model_validate(
            {
                "id": "prompt-001",
                "split": "train",
                "env_id": "env-001",
                "prompt_text": "prompt",
                "metadata": {},
                "unexpected": "boom",
            }
        )


def test_completion_record_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        CompletionRecord.model_validate(
            {
                "id": "completion-001",
                "prompt_id": "prompt-001",
                "prompt_text": "prompt",
                "raw_completion": "completion",
                "metadata": {},
                "unexpected": "boom",
            }
        )
