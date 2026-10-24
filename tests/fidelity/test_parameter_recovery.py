"""independent raw-scale recovery using the real declared sampler."""

import hashlib
import json
import os
from pathlib import Path

import pytest

SOURCE = Path(
    os.environ.get(
        "FIDELITY_SPECS",
        Path(__file__).resolve().parents[1] / "fixtures/declared_specs/selected-120.jsonl",
    )
)
SOURCE_BYTES = SOURCE.read_bytes()
SPECS = [json.loads(line) for line in SOURCE_BYTES.splitlines() if line.strip()]
PILOT_IDS = {f"4f-v01-fwd-{number:03d}" for number in (0, 4, 5, 6, 7, 8)}
SELECTED = (
    SPECS
    if os.environ.get("FIDELITY_ALL") == "1"
    else [s for s in SPECS if s["env_id"] in PILOT_IDS]
)


def test_corpus_and_pilot_are_complete():
    assert len(SPECS) == len({s["env_id"] for s in SPECS}) == 120
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "sha256": hashlib.sha256(SOURCE_BYTES).hexdigest(),
                "count": len(SPECS),
            }
        )
    )
    if os.environ.get("FIDELITY_ALL") != "1":
        assert {s["env_id"] for s in SELECTED} == PILOT_IDS
        assert len({s["generative_family"] for s in SELECTED}) == 4


@pytest.mark.parametrize("spec", SELECTED, ids=lambda s: s["env_id"])
def test_parameter_recovery(spec):
    from recovery_run import check_spec, parameter_budget

    result = check_spec(spec, parameter_budget(SPECS))
    print(json.dumps(result, sort_keys=True))
    assert result["status"] == "PASS", result
    assert result["inputs"]["passed"], result


@pytest.mark.parametrize(
    "mutation", ["minmax", "drop", "reverse", "poisson_for_nb2", "ignore_zero_inflation"]
)
def test_mutation_controls(mutation):
    from recovery_run import check_spec, parameter_budget

    number = 5 if mutation == "poisson_for_nb2" else 6 if mutation == "ignore_zero_inflation" else 4
    spec = next(s for s in SPECS if s["env_id"] == f"4f-v01-fwd-{number:03d}")
    result = check_spec(spec, parameter_budget(SPECS), mutation=mutation)
    print(json.dumps(result, sort_keys=True))
    assert result["status"] == "REJECT", result
    assert result["violations"], result


def test_fixed_string_coding_uses_reference_level():
    from recovery_mle import design_columns, design_matrix

    spec = {
        "features": [
            {"name": "arm", "law": {"dist": "fixed_counts", "levels": ["base", "treated"]}}
        ],
        "response": {"coefficients": {"arm=treated": 0.3}},
    }
    assert design_columns(spec) == [("arm=treated", "arm", "treated")]
    assert design_matrix(spec, [{"arm": "base"}, {"arm": "treated"}]).tolist() == [
        [1.0, 0.0],
        [1.0, 1.0],
    ]


def test_mixed_fixed_coding_is_refused():
    from recovery_mle import UndefinedCoding, design_columns

    spec = {
        "features": [{"name": "arm", "law": {"dist": "fixed_counts", "levels": [0, "treated"]}}],
        "response": {"coefficients": {"arm=treated": 0.3}},
    }
    with pytest.raises(UndefinedCoding, match="/features/0/law/levels"):
        design_columns(spec)
