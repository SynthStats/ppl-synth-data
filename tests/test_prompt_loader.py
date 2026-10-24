from __future__ import annotations

import importlib.util
from importlib import resources

import pytest

from ppl_synth_data.llm import prompt_loader
from ppl_synth_data.llm.prompt_loader import load_prompt_text, strip_html_comments

PROMPT_NAMES = sorted(
    entry.name
    for entry in resources.files("ppl_synth_data").joinpath("llm/prompts").iterdir()
    if entry.name.endswith(".md")
)


def test_every_prompt_template_is_enumerated() -> None:
    # Known positive: this template carries HTML comments with local paths.
    assert "envspec_gen.md" in PROMPT_NAMES


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_loaded_prompt_has_no_html_comment_or_local_path(name: str) -> None:
    text = load_prompt_text(name)

    assert "<!--" not in text
    assert "/Users/" not in text


def test_envspec_prompt_keeps_the_text_around_its_comments() -> None:
    text = load_prompt_text("envspec_gen.md")

    assert "- regression_continuous: coefficients, intercept, noise_sigma." in text
    assert "Infection uses p=1-exp(-theta*z_time)" in text
    assert "delayed utility is delayed_reward/(1+k*delay_days)." in text


def test_strip_html_comments_removes_each_comment_and_keeps_the_rest() -> None:
    text = "keep A\n<!-- note one\n/Users/someone/path -->\nkeep B <!-- note two --> keep C\n"

    stripped = strip_html_comments(text, source="example.md")

    for kept in ("keep A", "keep B", "keep C"):
        assert kept in stripped
    for removed in ("note one", "note two", "/Users/", "<!--", "-->"):
        assert removed not in stripped


def test_strip_html_comments_refuses_an_unterminated_comment() -> None:
    with pytest.raises(ValueError, match="example.md"):
        strip_html_comments("keep\n<!-- never closed\nprivate text", source="example.md")


def test_load_existing_prompt() -> None:
    text = load_prompt_text("completion_gen_lazyppl.md")
    assert len(text) > 0
    assert isinstance(text, str)


def test_load_prompt_contains_expected_content() -> None:
    text = load_prompt_text("completion_gen_lazyppl.md")
    assert "LazyPPL" in text or "Haskell" in text or "module" in text


def test_load_nonexistent_prompt() -> None:
    with pytest.raises((FileNotFoundError, TypeError)):
        load_prompt_text("nonexistent_prompt_file_xyz.md")


def test_prompt_loader_call_sites_use_llm_module() -> None:
    from ppl_synth_data.pipeline.stages import (
        stage_a_envspec,
        stage_d_prompts,
        stage_e_completions,
        stage_f_validate,
    )
    from ppl_synth_data.runtimes import lazyppl_runtime, pymc_runtime

    modules = (
        stage_a_envspec,
        stage_d_prompts,
        stage_e_completions,
        pymc_runtime,
        lazyppl_runtime,
    )
    for module in modules:
        assert module.load_prompt_text is prompt_loader.load_prompt_text
    assert not hasattr(stage_f_validate, "load_prompt_text")


def test_pipeline_runtime_prompt_loader_module_removed() -> None:
    assert importlib.util.find_spec("ppl_synth_data.pipeline.runtime") is None
