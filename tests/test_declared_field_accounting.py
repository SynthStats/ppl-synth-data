"""Compile-time read accounting, not proof of stochastic semantic fidelity."""

import shutil
from pathlib import Path

import pytest

from ppl_synth_data.envs.compiler import TemplateCompiler
from tests.test_response_law_declarations import PARAMETERS, _spec


@pytest.mark.parametrize("family", list(PARAMETERS))
def test_unknown_declaration_reports_json_pointer_before_write(tmp_path: Path, family: str):
    spec = _spec(family)
    spec.generative_params["unimplemented_effect"] = 0.3
    with pytest.raises(ValueError, match="/generative_params/unimplemented_effect"):
        TemplateCompiler().compile_envspec(spec, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ["sigma", "negative_binomial_dispersion"])
def test_legacy_declaration_is_refused_with_pointer(tmp_path: Path, field: str):
    family = "regression_continuous" if field == "sigma" else "regression_count"
    spec = _spec(family)
    spec.generative_params[field] = 5.5
    with pytest.raises(ValueError, match=f"/generative_params/{field}"):
        TemplateCompiler().compile_envspec(spec, tmp_path)
    assert not (tmp_path / f"{spec.env_name}.py").exists()


def test_real_template_with_parameter_reads_removed_is_refused(tmp_path: Path):
    import ppl_synth_data.envs.compiler as module

    templates = tmp_path / "templates"
    shutil.copytree(Path(module.__file__).parent / "templates", templates)
    declaration = templates / "_response_law.j2"
    original = declaration.read_text()
    assert "{{ gp[name] | tojson }}" in original
    declaration.write_text(original.replace("{{ gp[name] | tojson }}", "0"))
    spec = _spec("regression_continuous")
    with pytest.raises(ValueError, match="unread.*?/generative_params/intercept"):
        TemplateCompiler(templates).compile_envspec(spec, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("family", list(PARAMETERS))
def test_existing_supported_parameters_compile(tmp_path: Path, family: str):
    assert TemplateCompiler().compile_envspec(_spec(family), tmp_path).is_file()
