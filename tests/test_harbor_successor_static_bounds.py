"""Static coverage probes must not crash on valid larger index constants."""

from pathlib import Path

from ppl_synth_data.harbor import validator_template as v

ROOT = Path(__file__).resolve().parent / "fixtures/successor_laws"


def test_partition_index_beyond_static_probe_length_is_unresolved_not_a_crash():
    code = """def model(data):
    y = np.asarray(data["y"])
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        pm.Normal("a", mu, 1, observed=y[:50])
        pm.Normal("b", mu, 1, observed=y[50])
        pm.Normal("c", mu, 1, observed=y[51:])
    return m
"""
    result = v.analyze_program(code)
    assert result["status"] == "MATCH"
    assert "TARGET_OBSERVED_MORE_THAN_ONCE" not in result["flags"]
