"""Baseline constants from the sealed role manifest.

Each package's gold program is refitted with its ordinary predictors replaced by constants
computed from the visible split alone, and its structural inputs preserved. An ordinary
quantitative predictor uses its visible-split arithmetic mean in float64, whatever its number
of distinct values. An ordinary categorical predictor uses its visible-split mode, ties to the
first value in sorted order, with valid dtype and level labels. The cardinality heuristic
``_is_categorical_like`` takes no part in the baseline.

Each expected number below is computed independently from this construction, rather than
copied from the code's output. No expected value is a snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ppl_synth_data.harbor.export_pymc import (
    BaselineBlank,
    RoleManifestError,
    blank_predictors,
    load_role_manifest,
    require_package_roles,
    visible_split_blanks,
)

MANIFEST_SHA = "1f2aa2b1f3674b2b4cfc41506ebbb73791c6039a5eae26a29fdd30b84fc4ed02"


# --- Fixtures built independently of the implementation ------------------------


def package_entry(columns, *, package_id="pkg-a", target="y", status="complete"):
    """A manifest package entry in the sealed manifest's own shape."""
    return {
        "package_id": package_id,
        "target": target,
        "columns": columns,
        "status": status,
    }


def replace_col(name, reducer_type):
    return {
        "column": name,
        "role": "PREDICTOR",
        "treatment": "replace",
        "reducer_type": reducer_type,
    }


def preserve_col(name, role="PRESERVED"):
    return {"column": name, "role": role, "treatment": "preserve", "reducer_type": None}


def unresolved_col(name):
    return {"column": name, "role": "UNRESOLVED", "treatment": "unresolved", "reducer_type": None}


# --- Reducer types and deterministic ties -------------------------------


def test_quantitative_is_the_visible_mean_for_an_all_distinct_column_of_32_rows():
    """32 rows with 32 distinct values use the arithmetic mean.

    The mean of 0..31 in float64 is 15.5. A cardinality heuristic would call this categorical
    (32 <= 50 distinct) and use its mode, the smallest value, 0.
    """
    visible = {"y": np.zeros(32), "x": np.arange(32, dtype=np.float64)}
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    blanks = visible_split_blanks(entry, visible, target_name="y")
    # 0+1+...+31 = 496; 496/32 = 15.5 exactly, and float64 holds it exactly.
    assert blanks["x"].constant == pytest.approx(15.5, rel=0, abs=0)
    assert isinstance(blanks["x"].constant, float)
    assert blanks["x"].branch == "mean"


def test_quantitative_is_the_visible_mean_for_an_integer_column_with_repeats():
    """The quantitative values [1, 1, 2, 3] have arithmetic mean 7/4 = 1.75.

    The mode would be 1. An integer dtype and repeated values must not change the reducer.
    """
    visible = {"y": np.zeros(4), "x": np.array([1, 1, 2, 3])}
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    blanks = visible_split_blanks(entry, visible, target_name="y")
    assert blanks["x"].constant == pytest.approx(1.75, rel=0, abs=0)
    assert blanks["x"].branch == "mean"


def test_quantitative_mean_is_float64_even_from_an_integer_dtype():
    visible = {"y": np.zeros(3), "x": np.array([1, 2, 4])}
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    constant = visible_split_blanks(entry, visible, target_name="y")["x"].constant
    # The arithmetic mean of 1, 2 and 4 is 7/3. Stated here as 7/3, not recomputed with
    # the same call the code makes, so the assertion can disagree with the code.
    assert isinstance(constant, float)
    assert constant == pytest.approx(7 / 3, rel=0, abs=1e-12)


def test_categorical_tie_resolves_to_the_first_value_in_sorted_order():
    """"Ties go to the first value in sorted order".

    b and a both occur twice. Sorted order is a, b, so the constant is "a", not "b" and not
    whichever happened to appear first in the data.
    """
    visible = {"y": np.zeros(4), "x": np.array(["b", "a", "a", "b"], dtype=object)}
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "categorical")])
    blanks = visible_split_blanks(entry, visible, target_name="y")
    assert blanks["x"].constant == "a"
    assert blanks["x"].branch == "mode"


def test_categorical_tie_ignores_the_order_the_values_appear_in():
    """The same multiset in a different row order must give the same constant."""
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "categorical")])
    first = visible_split_blanks(
        entry,
        {"y": np.zeros(4), "x": np.array(["b", "a", "a", "b"], dtype=object)},
        target_name="y",
    )
    second = visible_split_blanks(
        entry,
        {"y": np.zeros(4), "x": np.array(["a", "b", "b", "a"], dtype=object)},
        target_name="y",
    )
    assert first["x"].constant == second["x"].constant == "a"


def test_categorical_mode_keeps_a_non_tied_majority():
    visible = {"y": np.zeros(5), "x": np.array(["z", "z", "z", "a", "b"], dtype=object)}
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "categorical")])
    assert visible_split_blanks(entry, visible, target_name="y")["x"].constant == "z"


# The manifest determines which columns are replaced and how they are reduced.


def test_a_package_with_no_manifest_entry_raises():
    entry = package_entry([replace_col("x", "quantitative")], package_id="pkg-a")
    manifest = {"packages": [entry], "package_count": 1, "complete": 1, "unresolved": 0}
    with pytest.raises(ValueError, match="ROLE_MANIFEST"):
        require_package_roles(manifest, "pkg-absent")


def test_an_unresolved_column_raises():
    entry = package_entry([replace_col("x", "quantitative"), unresolved_col("w")])
    manifest = {"packages": [entry], "package_count": 1, "complete": 0, "unresolved": 1}
    with pytest.raises(ValueError, match="UNRESOLVED"):
        require_package_roles(manifest, "pkg-a")


def test_an_unresolved_column_raises_when_blanking_too():
    entry = package_entry(
        [preserve_col("y", role="TARGET"), replace_col("x", "quantitative"), unresolved_col("w")]
    )
    manifest = {"packages": [entry]}
    with pytest.raises(ValueError, match="UNRESOLVED"):
        visible_split_blanks(
            require_package_roles(manifest, "pkg-a"),
            {"y": np.zeros(3), "x": np.array([1.0, 2.0, 3.0]), "w": np.array([7.0, 8.0, 9.0])},
            target_name="y",
        )


def test_a_replaced_column_missing_from_the_split_raises():
    entry = package_entry([replace_col("absent", "quantitative")])
    with pytest.raises(ValueError, match="missing from the visible split"):
        visible_split_blanks(entry, {"y": np.zeros(3)}, target_name="y")


def test_an_unrecognised_reducer_type_raises():
    entry = package_entry([replace_col("x", "quantish")])
    with pytest.raises(ValueError, match="reducer_type"):
        visible_split_blanks(entry, {"x": np.array([1.0, 2.0, 3.0])}, target_name="y")


def test_manifest_sha256_is_verified_at_load(tmp_path):
    """The manifest is verified by sha256 at load."""
    good = tmp_path / "ROLE-MANIFEST-220.json"
    good.write_text(json.dumps({"packages": []}))
    actual = hashlib.sha256(good.read_bytes()).hexdigest()
    load_role_manifest(good, expected_sha256=actual)  # matching hash loads
    with pytest.raises(ValueError, match="ROLE_MANIFEST_SHA256"):
        load_role_manifest(good, expected_sha256="0" * 64)


# Preserve declared columns and apply the declared reducers to replaced columns.


def test_replaced_columns_are_constant_across_the_rows():
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    visible = {"y": np.zeros(5), "x": np.array([10.0, -3.0, 7.5, 0.0, 2.0])}
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    # mean = 16.5/5 = 3.3, by the arithmetic-mean rule
    assert out["x"] == pytest.approx(3.3, rel=0, abs=1e-12)
    assert len(np.unique(np.asarray(out["x"]))) == 1


def test_replaced_columns_are_constant_and_equal_across_the_two_splits():
    """The same constants apply to both splits."""
    entry = package_entry(
        [
            preserve_col("y", role="TARGET"),
            replace_col("x", "quantitative"),
            replace_col("c", "categorical"),
        ]
    )
    visible = {
        "y": np.zeros(4),
        "x": np.array([1.0, 2.0, 3.0, 6.0]),
        "c": np.array(["p", "q", "p", "q"], dtype=object),
    }
    hidden = {
        "y": np.zeros(3),
        "x": np.array([-100.0, 50.0, 7.0]),
        "c": np.array(["z", "z", "z"], dtype=object),
    }
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out_v = blank_predictors(visible, target_name="y", blanks=blanks)
    out_h = blank_predictors(hidden, target_name="y", blanks=blanks)
    # visible mean = 12/4 = 3.0; the hidden values must not move it
    xv, xh = np.asarray(out_v["x"]).reshape(-1), np.asarray(out_h["x"]).reshape(-1)
    assert float(xv[0]) == pytest.approx(3.0, rel=0, abs=1e-12)
    assert float(xh[0]) == pytest.approx(3.0, rel=0, abs=1e-12)
    assert len(np.unique(xv)) == 1
    assert len(np.unique(xh)) == 1
    # p and q tie at two each; sorted order puts p first
    cv, ch = np.asarray(out_v["c"]).reshape(-1), np.asarray(out_h["c"]).reshape(-1)
    assert str(cv[0]) == "p" and str(ch[0]) == "p"
    assert len(np.unique(cv)) == 1
    assert len(np.unique(ch)) == 1


def test_the_target_is_never_blanked():
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    visible = {"y": np.array([1.0, 2.0, 3.0]), "x": np.array([4.0, 5.0, 6.0])}
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    assert out["y"] is visible["y"]


def test_a_scalar_row_count_is_preserved_when_the_manifest_preserves_it():
    """A scalar row count marked preserve comes back unchanged."""
    entry = package_entry(
        [
            preserve_col("y", role="TARGET"),
            replace_col("x", "quantitative"),
            preserve_col("n", role="UNUSED"),
        ]
    )
    visible = {"y": np.zeros(3), "x": np.array([2.0, 4.0, 6.0]), "n": 3}
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    assert out["n"] == 3
    assert isinstance(out["n"], int)


# --- Properties ---------------------------------------------------------------


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=1, max_size=40)
)
@settings(max_examples=60, deadline=None)
def test_property_a_quantitative_constant_equals_the_visible_mean_for_any_cardinality(values):
    """The rule does not depend on how many distinct values there are.

    The expected value is math.fsum over the generated values divided by their count: exact
    summation of the floats, independent of numpy's reduction. Never np.mean on the same data,
    which would restate the implementation's own answer.
    """
    arr = np.asarray(values, dtype=np.float64)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    visible = {"y": np.zeros(arr.shape[0]), "x": arr}
    constant = visible_split_blanks(entry, visible, target_name="y")["x"].constant
    expected = math.fsum(float(v) for v in arr) / arr.size
    assert constant == pytest.approx(expected, rel=1e-12, abs=1e-12)


@given(st.lists(st.integers(min_value=-50, max_value=50), min_size=1, max_size=40))
@settings(max_examples=60, deadline=None)
def test_property_an_integer_quantitative_column_still_takes_the_mean(values):
    """Repeats or not, integers follow the mean rule, never the mode.

    The expected value is the exact rational sum divided by the count, from fractions.Fraction.
    The generated integers are small enough that both that quotient's nearest float64 and the
    implementation's float64 mean are the same number, so this asserts equality, not closeness.
    """
    arr = np.asarray(values)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    blanks = visible_split_blanks(entry, {"y": np.zeros(arr.shape[0]), "x": arr}, target_name=(
                'y'
            ))
    expected = float(Fraction(sum(values), len(values)))
    assert blanks["x"].constant == expected


@given(st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=40))
@settings(max_examples=60, deadline=None)
def test_property_a_categorical_tie_resolves_to_the_first_sorted_value(values):
    """Among the most frequent values, the one first in sorted order wins."""
    arr = np.asarray(values, dtype=object)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "categorical")])
    constant = visible_split_blanks(
        entry, {"y": np.zeros(arr.shape[0]), "x": arr}, target_name="y"
    )["x"].constant
    counts = {v: int(np.sum(arr == v)) for v in sorted(set(values))}
    top = max(counts.values())
    expected = min(v for v, c in counts.items() if c == top)  # first in sorted order
    assert constant == expected


@given(st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=30))
@settings(max_examples=40, deadline=None)
def test_property_preserved_columns_are_byte_identical_for_any_input(values):
    """Preserved columns of different dtypes reach the fit as the original array objects."""
    arr = np.asarray(values, dtype=np.int64)
    offset = np.asarray(values, dtype=np.float64) / 3.0
    entry = package_entry(
        [
            preserve_col("y", role="TARGET"),
            replace_col("x", "quantitative"),
            preserve_col("keep", role="GROUP"),
            preserve_col("exposure", role="OFFSET"),
        ]
    )
    visible = {
        "y": np.zeros(arr.shape[0]),
        "x": np.arange(1.0, arr.shape[0] + 1.0),
        "keep": arr,
        "exposure": offset,
    }
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    for name in ("y", "keep", "exposure"):
        assert out[name] is visible[name], f"{name} must be the very same array"
        assert out[name].tobytes() == visible[name].tobytes(), f"{name} must be byte-identical"


# --- the defect, pinned through the path it travelled ---------------------------


def test_the_old_cardinality_heuristic_would_have_wrongly_selected_the_mode():
    """32 all-distinct rows distinguish the mean from a cardinality-based mode.

    The mean is 15.5. A heuristic that treats at most 50 distinct values as categorical would
    give the mode instead, which for all-distinct values is the smallest. The assertion catches
    a future edit that reintroduces that cardinality test.
    """
    arr = np.arange(32, dtype=np.float64)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    constant = visible_split_blanks(entry, {"y": np.zeros(32), "x": arr}, target_name="y")[
        "x"
    ].constant
    assert constant == pytest.approx(15.5, rel=0, abs=0)


# Manifest and reducer expectations use literal values or independently computed arithmetic.


def _mean_fraction(values):
    """The arithmetic mean as an exact fraction: (sum / n), independently of numpy."""
    total = sum(Fraction(int(v)) for v in values)
    return total / len(values)


# The fit must receive the fractional mean as float64.


def test_an_int64_quantitative_column_is_filled_as_float64_with_the_mean():
    """Quantitative [1, 1, 2, 3] is filled as float64 with its visible mean, 7/4 = 1.75.

    Integer storage must not truncate the constant to 1.
    """
    x = np.array([1, 1, 2, 3], dtype=np.int64)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    visible = {"y": np.zeros(4), "x": x}
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    filled = np.asarray(out["x"])
    assert filled.dtype == np.float64
    assert float(_mean_fraction([1, 1, 2, 3])) == pytest.approx(1.75, rel=0, abs=0)
    assert np.all(filled == 1.75), f"the fit would see {filled.tolist()!r}, not 1.75 in every row"


def test_a_zero_one_int64_column_keeps_its_fractional_mean():
    """A quantitative [1, 0, 1, 0] indicator retains its mean 1/2 rather than truncating to zero."""
    x = np.array([1, 0, 1, 0], dtype=np.int64)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    blanks = visible_split_blanks(entry, {"y": np.zeros(4), "x": x}, target_name="y")
    out = blank_predictors({"y": np.zeros(4), "x": x}, target_name="y", blanks=blanks)
    filled = np.asarray(out["x"])
    assert float(_mean_fraction([1, 0, 1, 0])) == pytest.approx(0.5, rel=0, abs=0)
    assert np.all(filled == 0.5), f"the fit would see {filled.tolist()!r}"


def test_a_fractional_mean_survives_the_fill():
    """The quantitative 0/1 predictor has mean 13/24, which must survive the fill as float64."""
    ones, zeros = 13, 11  # 13/24 = 0.541666...
    x = np.array([1] * ones + [0] * zeros, dtype=np.int64)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    blanks = visible_split_blanks(entry, {"y": np.zeros(ones + zeros), "x": x}, target_name="y")
    out = blank_predictors({"y": np.zeros(ones + zeros), "x": x}, target_name="y", blanks=blanks)
    filled = np.asarray(out["x"])
    want = float(_mean_fraction(list(x)))
    assert want == pytest.approx(13 / 24, rel=0, abs=1e-12)
    assert np.all(np.abs(filled - want) < 1e-12), f"the fit would see {filled[0]!r}, not {want!r}"


def test_a_categorical_column_keeps_its_own_dtype_when_filled():
    """Categorical columns fill with the visible column's dtype."""
    x = np.array(["p", "q", "p", "q"], dtype=object)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "categorical")])
    blanks = visible_split_blanks(entry, {"y": np.zeros(4), "x": x}, target_name="y")
    out = blank_predictors({"y": np.zeros(4), "x": x}, target_name="y", blanks=blanks)
    assert np.asarray(out["x"]).dtype == x.dtype


# The sealed manifest can be supplied explicitly on hosts that have it.


def test_the_real_sealed_manifest_resolves_every_package(tmp_path):
    """The sealed manifest gives n the treatment
    "preserve (scalar row count, never replaced)". The loader must accept this treatment
    without restricting it to the four bare role names.
    """
    import hashlib as _h

    manifest_path = Path(
        os.environ.get(
            "PYMC_ROLE_MANIFEST",
            "/tmp/exampleROLE-MANIFEST-220.json",
        )
    )
    if not manifest_path.is_file():
        pytest.skip("the sealed manifest is not on this host")
    raw = manifest_path.read_bytes()
    assert (
        _h.sha256(raw).hexdigest()
        == "1f2aa2b1f3674b2b4cfc41506ebbb73791c6039a5eae26a29fdd30b84fc4ed02"
    )
    manifest = load_role_manifest(manifest_path, expected_sha256=_h.sha256(raw).hexdigest())
    ids = [p["package_id"] for p in manifest["packages"]]
    for package_id in ids:
        entry = require_package_roles(manifest, package_id)
        assert entry["package_id"] == package_id
    assert len(ids) == 220


def test_the_sealed_preserve_wording_maps_to_preserve():
    """The exact declared treatment maps to preserve; unregistered treatments raise."""
    entry = package_entry(
        [
            {
                "column": "n",
                "role": "UNUSED",
                "treatment": "preserve (scalar row count, never replaced)",
                "reducer_type": None,
            },
        ]
    )
    manifest = {"packages": [entry]}
    resolved = require_package_roles(manifest, "pkg-a")
    assert resolved["columns"][0]["treatment"] == "preserve (scalar row count, never replaced)"


def test_an_unmapped_treatment_string_raises():
    entry = package_entry(
        [{"column": "x", "role": "PREDICTOR", "treatment": "preserve-ish", "reducer_type": None}]
    )
    with pytest.raises(ValueError, match="ROLE_MANIFEST"):
        require_package_roles({"packages": [entry]}, "pkg-a")


# Missing manifest entries are rejected.


def test_a_data_key_the_manifest_does_not_list_raises():
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    visible = {"y": np.zeros(3), "x": np.array([1.0, 2.0, 3.0]), "censored": np.array([0, 1, 0])}
    with pytest.raises(ValueError, match="ROLE_MANIFEST"):
        visible_split_blanks(entry, visible, target_name="y")


def test_a_target_that_differs_from_the_manifest_raises():
    entry = package_entry(
        [preserve_col("z", role="TARGET"), replace_col("x", "quantitative")], target="z"
    )
    with pytest.raises(ValueError, match="ROLE_MANIFEST"):
        visible_split_blanks(
            entry, {"z": np.zeros(3), "x": np.array([1.0, 2.0, 3.0])}, target_name="y"
        )


def test_a_scalar_marked_replace_raises():
    """A scalar marked replace raises. It has no rows to fill."""
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("n", "quantitative")])
    with pytest.raises(ValueError, match="ROLE_MANIFEST"):
        visible_split_blanks(entry, {"y": np.zeros(3), "n": 3}, target_name="y")


def test_only_one_manifest_argument_raises(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"packages": []}))
    with pytest.raises(ValueError, match="ROLE_MANIFEST"):
        load_role_manifest(path, expected_sha256=None)


# Unregistered column types are rejected.


def test_unregistered_blanks_does_not_exist():
    import ppl_synth_data.harbor.export_pymc as m

    assert not hasattr(m, "_unregistered_blanks"), (
        "the dtype fallback must not select a second reducer"
    )


def test_an_envspec_export_keeps_an_undeclared_numeric_column_identical():
    """Without a manifest, only EnvSpec-declared predictors are blanked.

    An undeclared numeric censoring indicator stays byte-identical.
    """
    from ppl_synth_data.harbor.export_pymc import _blanks_from_spec

    censored = np.array([0, 1, 0, 1, 0, 0], dtype=np.int64)
    visible = {
        "y": np.zeros(6),
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "censored": censored,
    }
    blanks = _blanks_from_spec(
        feature_names=["x"], feature_types={"x": "quantitative"}, visible=visible
    )
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    assert out["censored"] is censored
    assert out["censored"].tobytes() == censored.tobytes()


# The density-gate record uses the validator's emitted fields.


def test_the_density_gate_reads_the_validators_own_fields():
    """Map the validator's emitted density-gate fields to the export status.

    Every dict below is a literal of the fields the validator puts on a result. Status "applied"
    with density_gate_passed False and reason_code "min_predictive_scale" maps to failed;
    "applied" with density_gate_passed True maps to passed; "not_applicable*" maps to
    not_applicable; anything else maps to unavailable. The old failed_density_gate string
    is not a second signal.
    """
    from ppl_synth_data.harbor.export_pymc import _gate_from_l6
    from ppl_synth_data.harbor.validator_template import MIN_PRED_SCALE_FRACTION

    data_std = 1.0
    threshold = MIN_PRED_SCALE_FRACTION * data_std

    # A minimum below the threshold yields a failed gate with matching reason and refusal codes.
    failing = {
        "outcome": "density_gate_failed",
        "disposition": "refused",
        "reason_code": "min_predictive_scale",
        "refusal_code": "min_predictive_scale",
        "phase": "predictive_spread",
        "error": "predictive spread below registered threshold",
        "density_gate_applicable": True,
        "density_gate_assessed": True,
        "density_gate_passed": False,
        "density_gate_status": "applied",
        "density_gate_threshold": threshold,
        "data_std": data_std,
        "min_pred_std": threshold / 2.0,
        "failed_density_gate": None,
    }
    failed = _gate_from_l6(failing)
    assert failed["state"] == "failed"
    assert failed["threshold"] == threshold
    assert failed["measurements"]["density_gate_passed"] is False
    assert failed["measurements"]["reason_code"] == "min_predictive_scale"

    # A minimum at or above the threshold passes the gate.
    passing = {
        "outcome": "scored",
        "reason_code": None,
        "density_gate_applicable": True,
        "density_gate_assessed": True,
        "density_gate_passed": True,
        "density_gate_status": "applied",
        "density_gate_threshold": threshold,
        "data_std": data_std,
        "min_pred_std": threshold * 2.0,
        "failed_density_gate": None,
    }
    passed = _gate_from_l6(passing)
    assert passed["state"] == "passed"
    assert passed["threshold"] == threshold

    # A verified discrete target is exempt from the predictive-spread gate.
    discrete = {
        "outcome": "scored",
        "density_gate_applicable": False,
        "density_gate_assessed": False,
        "density_gate_passed": None,
        "density_gate_status": "not_applicable_discrete",
        "verified_discrete_exemption": True,
        "failed_density_gate": None,
    }
    assert _gate_from_l6(discrete)["state"] == "not_applicable"

    # An applicable gate that was not assessed is unavailable.
    unassessed = {
        "outcome": "unjudged",
        "density_gate_applicable": True,
        "density_gate_assessed": False,
        "density_gate_passed": None,
        "density_gate_status": "unassessed",
        "data_std": data_std,
        "min_pred_std": None,
        "density_gate_threshold": None,
        "failed_density_gate": None,
    }
    assert _gate_from_l6(unassessed)["state"] == "unavailable"

    # A result with no gate status is unavailable.
    assert _gate_from_l6({"outcome": "scored"})["state"] == "unavailable"
    assert (
        _gate_from_l6({"outcome": "scored", "density_gate_status": "unknown"})["state"]
        == "unavailable"
    )

    # The recorded threshold must equal the validator's fraction times data_std.
    disagreeing = dict(passing, density_gate_threshold=threshold * 10.0)
    with pytest.raises(ValueError, match="DENSITY_GATE"):
        _gate_from_l6(disagreeing)

    # A record that says "scored" while the gate failed is a contradiction the export refuses.
    contradictory = dict(failing, outcome="scored")
    with pytest.raises(ValueError, match="DENSITY_GATE"):
        _gate_from_l6(contradictory)


def test_the_record_carries_no_varying_fields_in_the_package_artifact():
    """The package artifact carries the three keys the verifier reads and nothing else, so two runs
    that differ only in host, timing or path produce byte-identical packages. The full fit
    record goes to export_summary.json.
    """
    import ppl_synth_data.harbor.export_pymc as m

    assert not hasattr(m, "VARYING_RECORD_FIELDS"), (
        "the package payload contains only its three declared keys"
    )
    out = m.package_reference_payload(
        gold_elpd=1.0,
        baseline_elpd=2.0,
        target_name="y",
        fit_record={
            "host": "example-host",
            "pid": 7,
            "worker": "w",
            "timing_s": 1.5,
            "validator_module": {"sha256": "a" * 64},
            "profile": {"floatX": "float64"},
        },
    )
    assert set(out) == {"gold_elpd", "baseline_elpd", "target_name"}
    assert out["gold_elpd"] == 1.0 and out["baseline_elpd"] == 2.0 and out["target_name"] == (
                'y'
            )


def test_the_readback_raise_is_reached():
    """Reject a fill that changes the constant, naming the column and both values."""
    from ppl_synth_data.harbor.export_pymc import blank_predictors

    # A quantitative constant 0.5 filled into an int64 column truncates to 0 under
    # casting="unsafe", so the readback does not see the recorded constant.
    blanks = {
        "x": BaselineBlank(
            column="x",
            reducer_type="quantitative",
            branch="mean",
            constant=0.5,
            fill_dtype=np.int64,
        )
    }
    visible = {"y": np.zeros(3), "x": np.array([1, 2, 3], dtype=np.int64)}
    with pytest.raises(RoleManifestError, match="BASELINE_FILL_TRUNCATED") as excinfo:
        blank_predictors(visible, target_name="y", blanks=blanks)
    message = str(excinfo.value)
    assert "'x'" in message
    assert "0.5" in message


def test_the_export_refuses_a_manifest_whose_sha_is_wrong(tmp_path):
    """Refuse a manifest with the wrong hash before any fit runs."""
    import hashlib

    from ppl_synth_data.harbor.export_pymc import reexport_harbor_tasks

    manifest = tmp_path / "ROLE-MANIFEST.json"
    manifest.write_text(json.dumps({"packages": []}))
    validator = tmp_path / "validator.py"
    validator.write_text("x = 1\n")
    with pytest.raises(ValueError, match="ROLE_MANIFEST_SHA256"):
        reexport_harbor_tasks(
            [tmp_path / "source"],
            tmp_path / "out",
            validator_path=validator,
            validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest(),
            role_manifest_path=manifest,
            role_manifest_sha256="0" * 64,
            excluded=(),
            attempt_id="attempt-1",
        )


def test_the_held_out_split_narrower_than_the_visible_one_is_not_blanked_by_width():
    """String arrays of different storage widths retain the same blanking constant."""
    from ppl_synth_data.harbor.export_pymc import blank_predictors

    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "categorical")])
    visible_x = np.array(["ab", "cd", "efg"], dtype="<U3")
    hidden_x = np.array(["ab", "cd"], dtype="<U2")
    visible = {"y": np.zeros(3), "x": visible_x}
    hidden = {"y": np.zeros(2), "x": hidden_x}
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out_v = blank_predictors(visible, target_name="y", blanks=blanks)
    out_h = blank_predictors(hidden, target_name="y", blanks=blanks)
    # The constant comes from the visible split alone. Both fills collapse to that one value, and
    # the value does not depend on the storage width of the column it filled.
    filled_v = list(np.asarray(out_v["x"]).reshape(-1))
    filled_h = list(np.asarray(out_h["x"]).reshape(-1))
    assert filled_v == [blanks["x"].constant] * 3
    assert filled_h == [blanks["x"].constant] * 2
    assert blanks["x"].constant in ("ab", "cd", "efg")


def test_a_run_with_no_manifest_refuses_rather_than_falling_back():
    """The rows export rejects a missing role manifest."""
    from ppl_synth_data.harbor.export_pymc import export_harbor_tasks_from_rows

    rows = [
        {
            "row_id": "r1",
            "task": {"prompt": "p"},
            "visible_data": {"y": [1.0], "x": [2.0]},
            "heldout_data": {"y": [3.0], "x": [4.0]},
        }
    ]
    validator = Path("tests/fixtures/harbor_reference/ROLE-MANIFEST-fixture.json")
    with pytest.raises((ValueError, TypeError)):
        export_harbor_tasks_from_rows(
            rows,
            {"r1": "code"},
            Path("/tmp/never-written"),
            validator_path=validator,
            validator_sha256="0" * 64,
            excluded_rows=(),
            attempt_id="attempt-1",
            role_manifest_path=None,
            role_manifest_sha256=None,
        )


def test_a_preserved_predictor_is_not_blanked():
    """A preserved predictor reaches the fit unchanged."""
    entry = package_entry(
        [
            preserve_col("y", role="TARGET"),
            replace_col("x", "quantitative"),
            preserve_col("group", role="GROUP"),
        ]
    )
    group = np.array([0, 1, 0, 1], dtype=np.int64)
    visible = {"y": np.zeros(4), "x": np.array([1.0, 2.0, 3.0, 4.0]), "group": group}
    blanks = visible_split_blanks(entry, visible, target_name="y")
    assert set(blanks) == {"x"}, "only the replaced column has a constant"
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    assert out["group"] is group
    assert out["group"].tobytes() == group.tobytes()


# --- Re-export regressions ----------------------------------------------------


def test_two_reexports_with_varying_records_write_byte_identical_packages(tmp_path):
    """Two re-exports whose fit records differ only in worker, timings and paths
    must write byte-identical tests/elpd_reference.json, each with exactly the three keys
    the verifier reads. Returning the full record from package_reference_payload would
    make this test fail on the extra keys.
    """
    import hashlib

    from ppl_synth_data.harbor.export_pymc import package_reference_payload

    a = package_reference_payload(
        gold_elpd=-31.5,
        baseline_elpd=-52.25,
        target_name="y",
        fit_record={
            "worker": {"host": "host-a", "pid": 11},
            "timing_s": {"gold": 1.5, "baseline": 2.5},
            "paths": ["/tmp/a"],
            "attempt_id": "attempt-a",
        },
    )
    b = package_reference_payload(
        gold_elpd=-31.5,
        baseline_elpd=-52.25,
        target_name="y",
        fit_record={
            "worker": {"host": "host-b", "pid": 99},
            "timing_s": {"gold": 90.0, "baseline": 91.0},
            "paths": ["/elsewhere/b"],
            "attempt_id": "attempt-b",
        },
    )
    assert set(a) == {"gold_elpd", "baseline_elpd", "target_name"}
    left = json.dumps(a, sort_keys=True).encode("utf-8")
    right = json.dumps(b, sort_keys=True).encode("utf-8")
    assert hashlib.sha256(left).hexdigest() == hashlib.sha256(right).hexdigest()
    assert left == right


def test_a_failed_fit_leaves_its_partial_record_in_the_summary(tmp_path, monkeypatch):
    """Keep both attempts and their partial records in export_summary.json.

    This test permits an unlocked reference runtime to exercise the fit-failure handler
    directly.
    """
    import ppl_synth_data.harbor.export_pymc as m

    def failing(
        code,
        visible,
        hidden,
        *,
        env_id,
        target_name,
        blanks,
        expected_validator_sha256,
        attempt_id,
        l6_draws=None,
        l6_tune=None,
    ):
        exc = RuntimeError("gold L6 held-out ELPD scoring failed: boom")
        exc.fit_record = {
            "env_id": env_id,
            "attempt_id": attempt_id,
            "failure": {"message": "RuntimeError: boom"},
            "gold_fit_provenance": {"validator_module": {"sha256": "a" * 64}},
        }
        raise exc

    monkeypatch.setattr(m, "_compute_elpd_reference", failing)
    monkeypatch.setattr(
        m,
        "_run_gold_subprocess",
        lambda payload, timeout: (_ for _ in ()).throw(AssertionError("no fit")),
    )

    # Drive the rows export through its reference-failure handler.
    rows = [
        {
            "row_id": "r-fail",
            "runtime": "pymc",
            "prompt": "Scenario for r-fail",
            "structural_class": "mixture",
            "visible_data": {"y": [1.0, 2.0], "x": [3.0, 4.0]},
            "heldout_data": {"y": [5.0], "x": [6.0]},
        }
    ]
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    manifest = tmp_path / "ROLE-MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "package_id": "r-fail",
                        "target": "y",
                        "columns": entry["columns"],
                        "status": "complete",
                    }
                ]
            }
        )
    )
    import hashlib

    validator = tmp_path / "validator.py"
    validator.write_text("x = 1\n")

    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="ELPD reference scoring failed"):
        m.export_harbor_tasks_from_rows(
            rows,
            {"r-fail": "import pymc\ndef model(data):\n    return None\n"},
            out,
            validator_path=validator,
            validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest(),
            excluded_rows=(),
            attempt_id="attempt-9",
            allow_unlocked_reference_runtime=True,
            role_manifest_path=manifest,
            role_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )
    summary = json.loads((out / "export_summary.json").read_text())
    failures = summary["reference_failures"]
    assert failures, "the failure must reach the summary"
    record = failures[0].get("fit_record") or failures[0].get("partial_record")
    assert record is not None, "a failed fit must keep its partial record in the summary"
    assert record["attempt_id"] == "attempt-9"
    assert record["env_id"] == "r-fail"
    assert record["failure"]["message"] == "RuntimeError: boom"


def test_the_fill_receipt_digests_a_literal_canonical_json_string():
    """The fill digest agrees with the independently specified canonical JSON encoding."""
    import hashlib

    from ppl_synth_data.harbor.export_pymc import applied_fill_receipt

    # Two preserved columns, both filled unchanged. The canonical JSON is written out by hand.
    x = np.array([1.5, 2.5], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    literal = '{"x":[1.5,2.5],"y":[0.0,1.0]}'
    expected = hashlib.sha256(literal.encode("utf-8")).hexdigest()

    receipt = applied_fill_receipt({"x": x, "y": y}, blanks={})
    assert receipt["filled_split_sha256"] == expected


def test_the_fill_receipt_has_a_defined_encoding_for_non_finite_values():
    """NaN and infinities have explicit tokens in the fill digest encoding."""
    import hashlib

    from ppl_synth_data.harbor.export_pymc import applied_fill_receipt

    data = {"x": np.array([float("nan"), float("inf"), float("-inf"), 1.0])}
    receipt = applied_fill_receipt(data, blanks={})
    literal = '{"x":["NaN","Infinity","-Infinity",1.0]}'
    assert receipt["filled_split_sha256"] == hashlib.sha256(literal.encode("utf-8")).hexdigest()


def test_the_fill_receipt_refuses_an_unsupported_dtype_up_front():
    """The export rejects unsupported dtypes before fitting."""
    from ppl_synth_data.harbor.export_pymc import applied_fill_receipt

    data = {"t": np.array(["2026-10-13", "2026-10-14"], dtype="datetime64[D]")}
    with pytest.raises(RoleManifestError, match="BASELINE_FILL_DTYPE"):
        applied_fill_receipt(data, blanks={})


def test_the_receipt_covers_both_filled_splits():
    """The record reports one digest for each split, visible and held-out,
    rather than one for whichever split happened to be passed last.
    """
    from ppl_synth_data.harbor.export_pymc import applied_fill_receipt

    visible = {"y": np.zeros(3), "x": np.array([1.0, 2.0, 3.0])}
    hidden = {"y": np.zeros(2), "x": np.array([4.0, 5.0])}
    blanks = {
        "x": BaselineBlank(
            column="x",
            reducer_type="quantitative",
            branch="mean",
            constant=2.0,
            fill_dtype=np.float64,
        )
    }
    filled_visible = blank_predictors(visible, target_name="y", blanks=blanks)
    filled_hidden = blank_predictors(hidden, target_name="y", blanks=blanks)
    receipt = applied_fill_receipt(
        {"visible": filled_visible, "hidden": filled_hidden}, blanks=blanks, both_splits=True
    )
    assert set(receipt["filled_split_sha256"]) == {"visible", "hidden"}
    assert receipt["filled_split_sha256"]["visible"] != receipt["filled_split_sha256"]["hidden"]


def test_fixed_counts_is_categorical_only_with_reference_coding():
    """A fixed_counts feature is categorical only when it uses reference coding. Without it the
    program uses the column's numeric value in arithmetic, so it is quantitative.

    The 13/24 case: an allocation of 13 and 11 rows over two numeric levels has visible mean
    13/24 exactly, computed here with fractions.Fraction. Categorical would give the mode
    instead.
    """
    from fractions import Fraction

    from ppl_synth_data.harbor.export_pymc import reducer_for_law

    assert reducer_for_law("fixed_counts", uses_reference_coding=True) == "categorical"
    assert reducer_for_law("fixed_counts", uses_reference_coding=False) == "quantitative"

    # 13 rows at 1 and 11 rows at 0: the quantitative mean is exactly 13/24. The counts are
    # integers here so the exact rational is built from integers, not from float arithmetic.
    counts = [1] * 13 + [0] * 11
    expected = Fraction(13, 24)
    assert Fraction(sum(counts), len(counts)) == expected
    values = [float(v) for v in counts]
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    arr = np.asarray(values, dtype=np.float64)
    blanks = visible_split_blanks(entry, {"y": np.zeros(24), "x": arr}, target_name="y")
    assert blanks["x"].constant == float(expected)


# --- Additional manifest regressions ------------------------------------------


def test_the_leak_check_covers_preserved_columns_too():
    """Reject any non-target column identical to the target, including preserved columns."""
    from ppl_synth_data.harbor.export_pymc import _check_target_not_duplicated

    target = np.array([0.0, 1.0, 1.0, 0.0])
    visible = {"y": target, "x": np.array([1.0, 2.0, 3.0, 4.0]), "leak": target.copy()}
    # "leak" is preserved, so it is not in blanks; the check must still see it.
    with pytest.raises(RuntimeError, match="bit-identical to target"):
        _check_target_not_duplicated("e", "y", sorted(set(visible) - {"y"}), visible)


def test_a_string_scalar_marked_replace_is_refused():
    """A scalar string cannot be replaced as a column of rows."""
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    visible = {"y": np.zeros(3), "x": "abc"}
    with pytest.raises(RoleManifestError, match="scalar"):
        visible_split_blanks(entry, visible, target_name="y")


def test_a_nan_constant_names_the_real_reason():
    """A non-finite fill constant raises an error that identifies the non-finite value."""
    from ppl_synth_data.harbor.export_pymc import blank_predictors

    visible = {"y": np.zeros(3), "x": np.array([1.0, float("nan"), 3.0])}
    blanks = {
        "x": BaselineBlank(
            column="x",
            reducer_type="quantitative",
            branch="mean",
            constant=float("nan"),
            fill_dtype=np.float64,
        )
    }
    with pytest.raises(RoleManifestError) as excinfo:
        blank_predictors(visible, target_name="y", blanks=blanks)
    assert "TRUNCATED" not in str(excinfo.value)
    assert "not finite" in str(excinfo.value) or "nan" in str(excinfo.value).lower()


def test_a_crash_in_the_cascade_still_reports_its_provenance():
    """A cascade exception preserves the identity of the loaded validator."""
    import ppl_synth_data.harbor._validate_gold_subprocess as helper

    def boom(code, data):
        raise RuntimeError("cascade exploded")

    # Drive the helper's main on a payload whose cascade raises, and require the record.
    import json as _json

    payload = _json.dumps({"code": "x = 1", "data": {"y": [0.0]}})

    class _FakeStdin:
        def read(self):
            return payload

    import sys as _sys

    monkey_stdin = _sys.stdin
    out: list[str] = []

    class _FakeStdout:
        def write(self, s):
            out.append(s)

        def flush(self):
            pass

    monkey_stdout = _sys.stdout
    _sys.stdin = _FakeStdin()
    _sys.stdout = _FakeStdout()
    import ppl_synth_data.harbor.validator_template as vt

    original = vt.run_cascade
    try:
        vt.run_cascade = boom
        try:
            helper.main()
        except Exception:
            # the crash is the point; the record must have been printed first
            pass
    finally:
        vt.run_cascade = original
        _sys.stdin = monkey_stdin
        _sys.stdout = monkey_stdout

    printed = "".join(out)
    assert printed, "the helper printed nothing: a crash swallowed its provenance record"
    record = _json.loads(printed)
    assert "fit_provenance" in record
    assert (record["fit_provenance"].get("validator_module") or {}).get("sha256")


def test_the_export_refuses_splits_with_different_columns():
    """The export rejects visible and held-out splits with different columns."""
    from ppl_synth_data.harbor.export_pymc import check_split_columns

    check_split_columns(
        "e", {"y": np.zeros(2), "x": np.zeros(2)}, {"y": np.zeros(1), "x": np.zeros(1)}
    )

    with pytest.raises(RuntimeError, match="columns differ"):
        check_split_columns(
            "e",
            {"y": np.zeros(2), "x": np.zeros(2), "z": np.zeros(2)},
            {"y": np.zeros(1), "x": np.zeros(1)},
        )
    with pytest.raises(RuntimeError, match="columns differ"):
        check_split_columns(
            "e",
            {"y": np.zeros(2), "x": np.zeros(2)},
            {"y": np.zeros(1), "x": np.zeros(1), "z": np.zeros(1)},
        )


# Baseline construction expectations use independently computed arithmetic.


def test_the_cli_requires_and_forwards_both_manifest_arguments(tmp_path, monkeypatch):
    """The reexport-harbor CLI requires and forwards --role-manifest and --role-manifest-sha256."""
    from ppl_synth_data.cli import commands as cli

    seen = {}

    def fake_reexport(sources, out, **kwargs):
        seen.update(kwargs)
        return {"tasks_created": 0}

    # The command imports the function from the module at call time, so the module is what to patch.
    import ppl_synth_data.harbor.export_pymc as m

    monkeypatch.setattr(m, "reexport_harbor_tasks", fake_reexport)

    manifest = tmp_path / "ROLE-MANIFEST.json"
    manifest.write_text(json.dumps({"packages": []}))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    from typer.testing import CliRunner

    runner = CliRunner()
    validator = tmp_path / "validator.py"
    validator.write_text("x = 1\n")
    result = runner.invoke(
        cli.app,
        [
            "reexport-harbor",
            str(tmp_path / "src"),
            "--out",
            str(tmp_path / "out"),
            "--validator-path",
            str(validator),
            "--validator-sha256",
            hashlib.sha256(validator.read_bytes()).hexdigest(),
            "--attempt-id",
            "attempt-1",
            "--role-manifest",
            str(manifest),
            "--role-manifest-sha256",
            sha,
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["role_manifest_path"] == manifest
    assert (
        seen["role_sha256"] == sha if "role_sha256" in seen else seen["role_manifest_sha256"] == sha
    )


def test_the_validator_hash_must_match_the_callers_pinned_hash():
    """The pin is the caller's validator_sha256, never a constant in the module. A validator
    replaced between the call and the import cannot be reported as the one that scored.
    """
    import ppl_synth_data.harbor.export_pymc as m

    assert not hasattr(m, "EXPECTED_VALIDATOR_SHA256"), (
                'the stale validator pin constant is deleted'
            )

    pin = "b" * 64
    good = {
        "validator_module": {"sha256": pin},
        "helper_module": {"sha256": "c" * 64},
        "numerical_profile": {"floatX": "float64", "observed_from": "pytensor.config"},
    }
    m.require_fit_provenance(good, expected_validator_sha256=pin)

    swapped = dict(good, validator_module={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="FIT_PROVENANCE"):
        m.require_fit_provenance(swapped, expected_validator_sha256=pin)

    unhashed = dict(good, helper_module={"sha256": ""})
    with pytest.raises(ValueError, match="FIT_PROVENANCE"):
        m.require_fit_provenance(unhashed, expected_validator_sha256=pin)

    no_profile = dict(good, numerical_profile={"observed_from": "unavailable"})
    with pytest.raises(ValueError, match="FIT_PROVENANCE"):
        m.require_fit_provenance(no_profile, expected_validator_sha256=pin)


def test_the_helper_is_hashed_alongside_the_validator():
    """The helper that runs the fit is hashed too, not only the validator."""
    from ppl_synth_data.harbor._validate_gold_subprocess import _observe_fit_provenance

    provenance = _observe_fit_provenance()
    assert "helper_module" in provenance
    assert len(provenance["helper_module"]["sha256"]) == 64


def test_the_record_carries_container_identity():
    """Record PID, host, container identity and UTC start/end for each fit."""
    from ppl_synth_data.harbor._validate_gold_subprocess import _observe_fit_provenance
    from ppl_synth_data.harbor.export_pymc import container_identity

    identity = container_identity()
    assert isinstance(identity, dict) and identity
    assert "observed_from" in identity

    from ppl_synth_data.harbor._validate_gold_subprocess import _stamp_fit_finished

    provenance = _observe_fit_provenance()
    process = provenance.get("process") or {}
    for field in ("pid", "hostname", "started_at_utc"):
        assert process.get(field) is not None, f"the fit record must name its {field}"
    assert "container" in provenance
    assert provenance["container"].get("observed_from") in ("HOSTNAME", "unavailable")

    finished = _stamp_fit_finished(provenance)["process"]["finished_at_utc"]
    assert finished.endswith("+00:00") or finished.endswith("Z"), f"{finished!r} is not UTC"
    assert finished >= process["started_at_utc"]


def test_both_fits_are_checked_against_the_callers_pin():
    """The pin is the caller's validator_sha256, and BOTH the gold fit and the baseline fit are
    checked against it. A validator that was swapped between the call and the import cannot be
    reported as the one that scored.
    """
    import ppl_synth_data.harbor.export_pymc as m

    pin = "d" * 64
    provenance = {
        "validator_module": {"sha256": pin},
        "helper_module": {"sha256": "e" * 64},
        "numerical_profile": {"floatX": "float64", "observed_from": "pytensor.config"},
        "process": {
            "pid": 1,
            "hostname": "h",
            "started_at_utc": "2026-10-13T00:00:00+00:00",
            "finished_at_utc": "2026-10-13T00:00:01+00:00",
        },
    }

    def ok_result(**_):
        return {
            "reward": 1.0,
            "l6": {
                "outcome": "scored",
                "holdout_elpd": -1.0,
                "density_gate_status": "applied",
                "density_gate_passed": True,
                "density_gate_threshold": 0.01,
                "reason_code": None,
                "failed_density_gate": None,
                "data_std": 1.0,
                "min_pred_std": 2.0,
            },
            "fit_provenance": provenance,
        }

    monkey_calls = []

    def fake_run(payload, *, timeout):
        monkey_calls.append(payload)
        return ok_result()

    blanks = {
        "x": m.BaselineBlank(
            column="x",
            reducer_type="quantitative",
            branch="mean",
            constant=0.0,
            fill_dtype=np.float64,
        )
    }
    visible = {"y": np.zeros(4), "x": np.array([1.0, 2.0, 3.0, 4.0])}
    hidden = {"y": np.zeros(2), "x": np.array([5.0, 6.0])}

    import ppl_synth_data.harbor.export_pymc as mod

    orig = mod._run_gold_subprocess
    mod._run_gold_subprocess = fake_run
    try:
        out = m._compute_elpd_reference(
            "code",
            visible,
            hidden,
            env_id="e",
            target_name="y",
            blanks=blanks,
            expected_validator_sha256=pin,
            attempt_id="a1",
        )
    finally:
        mod._run_gold_subprocess = orig
    assert out["attempt_id"] == "a1"
    assert len(monkey_calls) == 2, "both the gold and the baseline fit must run"

    # The baseline fit must be checked too, not only the gold.
    def swapped(payload, *, timeout):
        monkey_calls.append(payload)
        bad = dict(provenance, validator_module={"sha256": "0" * 64})
        if len(monkey_calls) == 2:
            return {
                "reward": 1.0,
                "l6": {
                    "outcome": "scored",
                    "holdout_elpd": -1.0,
                    "density_gate_status": "applied",
                    "density_gate_passed": True,
                    "density_gate_threshold": 0.01,
                    "reason_code": None,
                    "failed_density_gate": None,
                    "data_std": 1.0,
                    "min_pred_std": 2.0,
                },
                "fit_provenance": bad,
            }
        return ok_result()

    monkey_calls.clear()
    mod._run_gold_subprocess = swapped
    try:
        with pytest.raises(ValueError, match="FIT_PROVENANCE"):
            m._compute_elpd_reference(
                "code",
                visible,
                hidden,
                env_id="e",
                target_name="y",
                blanks=blanks,
                expected_validator_sha256=pin,
                attempt_id="a1",
            )
    finally:
        mod._run_gold_subprocess = orig

    # The GOLD fit must be checked too, not only the baseline. The first call is the gold one,
    # so a mismatch there must raise before the baseline is ever run.
    def swapped_first(payload, *, timeout):
        monkey_calls.append(payload)
        bad = dict(provenance, validator_module={"sha256": "0" * 64})
        if len(monkey_calls) == 1:
            return {
                "reward": 1.0,
                "l6": {
                    "outcome": "scored",
                    "holdout_elpd": -1.0,
                    "density_gate_status": "applied",
                    "density_gate_passed": True,
                    "density_gate_threshold": 0.01,
                    "reason_code": None,
                    "failed_density_gate": None,
                    "data_std": 1.0,
                    "min_pred_std": 2.0,
                },
                "fit_provenance": bad,
            }
        return ok_result()

    monkey_calls.clear()
    mod._run_gold_subprocess = swapped_first
    try:
        with pytest.raises(ValueError, match="FIT_PROVENANCE"):
            m._compute_elpd_reference(
                "code",
                visible,
                hidden,
                env_id="e",
                target_name="y",
                blanks=blanks,
                expected_validator_sha256=pin,
                attempt_id="a1",
            )
    finally:
        mod._run_gold_subprocess = orig


def test_a_failed_fit_keeps_its_partial_record():
    """A fit that fails still leaves its record. The attempt id, what ran and the failure are
    evidence about the attempt even when no ELPD came back.
    """
    import ppl_synth_data.harbor.export_pymc as m

    pin = "f" * 64

    def failing(payload, *, timeout):
        return {
            "reward": 1.0,
            "l6": {"outcome": "l6_subprocess_exception", "error": "MemoryError: boom"},
            "fit_provenance": {
                "validator_module": {"sha256": pin},
                "helper_module": {"sha256": "a" * 64},
                "numerical_profile": {"floatX": "float64", "observed_from": "pytensor.config"},
            },
        }

    blanks = {
        "x": m.BaselineBlank(
            column="x",
            reducer_type="quantitative",
            branch="mean",
            constant=0.0,
            fill_dtype=np.float64,
        )
    }
    visible = {"y": np.zeros(4), "x": np.array([1.0, 2.0, 3.0, 4.0])}
    hidden = {"y": np.zeros(2), "x": np.array([5.0, 6.0])}

    import ppl_synth_data.harbor.export_pymc as mod

    orig = mod._run_gold_subprocess
    mod._run_gold_subprocess = failing
    try:
        with pytest.raises(RuntimeError) as excinfo:
            m._compute_elpd_reference(
                "code",
                visible,
                hidden,
                env_id="env-7",
                target_name="y",
                blanks=blanks,
                expected_validator_sha256=pin,
                attempt_id="attempt-9",
            )
    finally:
        mod._run_gold_subprocess = orig

    partial = getattr(excinfo.value, "fit_record", None)
    assert partial is not None, "a failed fit must keep its partial record"
    assert partial["attempt_id"] == "attempt-9"
    assert partial["env_id"] == "env-7"
    assert "failure" in partial


def test_the_attempt_id_comes_from_the_caller():
    """The attempt is identified by a caller-supplied id, not a literal count of 1."""
    import inspect

    from ppl_synth_data.harbor.export_pymc import _compute_elpd_reference

    parameters = inspect.signature(_compute_elpd_reference).parameters
    assert "attempt_id" in parameters, (
                'the fit record must take the attempt id from '
                'its caller'
            )
    assert parameters["attempt_id"].default is inspect.Parameter.empty


def test_the_fill_digest_agrees_for_equal_string_contents():
    """Equal string contents have equal digests regardless of storage width.

    An array of width U2 and a slice of width U3 contain the same two strings. Canonical JSON of
    the values agrees even though raw storage bytes differ.
    """
    from ppl_synth_data.harbor.export_pymc import applied_fill_receipt

    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "categorical")])
    narrow = np.array(["ab", "cd"], dtype="<U2")
    wide = np.array(["ab", "cd", "efg"], dtype="<U3")[:2]
    assert list(narrow) == list(wide), "the two columns hold the same strings"

    def receipt(values):
        blanks = visible_split_blanks(entry, {"y": np.zeros(2), "x": values}, target_name="y")
        return applied_fill_receipt(
            blank_predictors({"y": np.zeros(2), "x": values}, target_name="y", blanks=blanks),
            blanks=blanks,
        )

    left, right = receipt(narrow), receipt(wide)
    assert left["filled_split_sha256"] == right["filled_split_sha256"], (
        "equal string contents must digest equal"
    )


def test_the_record_carries_each_applied_constant_and_the_filled_split_sha256():
    """Each applied constant is read back from the filled arrays, with the sha256 of each filled
    split, so the construction comparison describes the construction that actually ran.
    """
    from ppl_synth_data.harbor.export_pymc import applied_fill_receipt

    x = np.array([1, 1, 2, 3], dtype=np.int64)
    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    visible = {"y": np.zeros(4), "x": x}
    blanks = visible_split_blanks(entry, visible, target_name="y")
    out = blank_predictors(visible, target_name="y", blanks=blanks)
    receipt = applied_fill_receipt(out, blanks=blanks)
    assert receipt["applied"]["x"] == 1.75
    assert len(receipt["filled_split_sha256"]) == 64


# Density-gate expectations follow the validator's emitted fields and exact arithmetic.


# Gate states use the validator's emitted field shapes.


def test_the_threshold_is_the_validator_fraction_of_data_std():
    """MIN_PRED_SCALE_FRACTION is imported from the validator, not restated here."""
    from ppl_synth_data.harbor.export_pymc import MIN_PRED_SCALE_FRACTION
    from ppl_synth_data.harbor.validator_template import (
        MIN_PRED_SCALE_FRACTION as VALIDATOR_FRACTION,
    )

    assert MIN_PRED_SCALE_FRACTION == VALIDATOR_FRACTION
    assert MIN_PRED_SCALE_FRACTION == 0.01


def test_an_outcome_that_disagrees_with_the_state_raises():
    """Raise when outcome and state disagree."""
    from ppl_synth_data.harbor.export_pymc import _gate_from_l6

    # Reject an outcome that claims scoring succeeded despite a failed density gate.
    with pytest.raises(ValueError, match="DENSITY_GATE"):
        _gate_from_l6(
            {
                "outcome": "scored",
                "density_gate_status": "applied",
                "density_gate_passed": False,
                "reason_code": "min_predictive_scale",
                "data_std": 10.0,
                "min_pred_std": 0.05,
                "density_gate_threshold": 0.1,
                "failed_density_gate": None,
            }
        )


# The package carries three keys; the summary retains the full fit record.


def test_a_package_reference_carries_only_three_keys_and_the_record_goes_to_the_summary():
    """Two fits that differ only in worker, timings and paths must give byte-identical packages."""
    from ppl_synth_data.harbor.export_pymc import package_reference_payload

    a = package_reference_payload(
        gold_elpd=-1.0,
        baseline_elpd=-2.0,
        target_name="y",
        fit_record={"worker": {"host": "h1"}, "timing_s": {"gold": 1.0}},
    )
    b = package_reference_payload(
        gold_elpd=-1.0,
        baseline_elpd=-2.0,
        target_name="y",
        fit_record={"worker": {"host": "h2"}, "timing_s": {"gold": 9.0}, "paths": ["elsewhere"]},
    )
    assert set(a) == {"gold_elpd", "baseline_elpd", "target_name"}
    assert a == b, "the varying fields must not reach the package"


# The caller supplies the validator identity.


def test_the_stale_validator_constant_is_gone_and_the_caller_pin_is_required():
    """The validator hash is supplied by the caller."""
    import inspect

    import ppl_synth_data.harbor.export_pymc as m
    from ppl_synth_data.harbor.export_pymc import _compute_elpd_reference

    assert not hasattr(m, "EXPECTED_VALIDATOR_SHA256"), (
        "the stale validator pin constant must be deleted"
    )
    parameters = inspect.signature(_compute_elpd_reference).parameters
    assert "expected_validator_sha256" in parameters
    assert parameters["expected_validator_sha256"].default is inspect.Parameter.empty
    assert parameters["attempt_id"].default is inspect.Parameter.empty


def test_an_unavailable_numerical_profile_refuses():
    """A fit whose provenance cannot name its profile is refused, not recorded."""
    from ppl_synth_data.harbor.export_pymc import require_fit_provenance

    with pytest.raises(ValueError, match="FIT_PROVENANCE"):
        require_fit_provenance(
            {
                "numerical_profile": {"observed_from": "unavailable"},
                "validator_module": {"sha256": "a" * 64},
                "helper_module": {"sha256": "b" * 64},
            },
            expected_validator_sha256="a" * 64,
        )


# The fill digest covers the canonical JSON of both splits.


def test_the_fill_receipt_digests_canonical_json_not_raw_bytes():
    """Equal contents from distinct objects must give the same digest."""
    from ppl_synth_data.harbor.export_pymc import applied_fill_receipt

    blanks = {"x": BaselineBlank("x", "quantitative", "mean", 1.75, np.dtype(np.float64))}
    one = {"x": np.array([1.75, 1.75], dtype=np.float64)}
    two = {"x": np.array([1.75, 1.75], dtype=np.float64)}
    assert applied_fill_receipt(one, blanks=blanks) == applied_fill_receipt(two, blanks=blanks)


# The feature-type mapping accepts only registered names.


def test_the_no_manifest_mapping_refuses_a_type_it_does_not_know():
    """Stage C's "continuous" maps to quantitative through an exact table; anything else raises."""
    from ppl_synth_data.harbor.export_pymc import _blanks_from_spec

    # Map the declared word continuous to quantitative; unknown types raise.
    blanks = _blanks_from_spec(
        feature_names=["x"],
        feature_types={"x": "continuous"},
        visible={"x": np.array([1, 0, 1, 0])},
    )
    assert blanks["x"].branch == "mean"
    with pytest.raises(ValueError, match="ROLE_MANIFEST"):
        _blanks_from_spec(
            feature_names=["x"], feature_types={"x": "text"}, visible={"x": np.ones(3)}
        )


# --- Package output: the SkyRL bundle manifest --------------------------------------


def test_the_bundle_manifest_carries_the_skyrl_v1_shape(tmp_path):
    """The package tree must be directly consumable as a package receipt.

    src/synthstats/scripts/verification/materialize_bundle_task_rows.py requires schema_version
    "ppl_synth_data.harbor.skyrl_bundle.v1", tasks[] with relative_path starting in tasks/ and a
    digest_sha256 over the package tree, and a task_rows.jsonl whose rows agree with those
    digests. The tree digest is path NUL sha256 NUL per file, sorted by relative path
    (sha256_tree there).
    """

    from ppl_synth_data.harbor.export_pymc import sha256_tree, write_bundle_manifest

    def make_task(root, name, *, extra=None):
        task = root / "tasks" / name
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text('version = "1.0"\n')
        (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
        (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
        for rel, body in (extra or {}).items():
            target = task / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
        return task

    make_task(tmp_path, "pkg-a", extra={"tests/validator.py": "x = 1\n"})
    make_task(tmp_path, "pkg-b", extra={"tests/validator.py": "x = 1\n"})

    manifest = write_bundle_manifest(
        tmp_path,
        validator_sha256="a" * 64,
        split="train",
        group="diag",
    )

    assert manifest["schema_version"] == "ppl_synth_data.harbor.skyrl_bundle.v1"
    assert set(manifest) >= {"schema_version", "tasks", "task_rows_path"}
    names = {entry["task_name"]: entry for entry in manifest["tasks"]}
    assert set(names) == {"pkg-a", "pkg-b"}
    for name, entry in names.items():
        assert entry["relative_path"] == f"tasks/{name}"
        assert entry["split"] == "train"
        assert entry["group"] == "diag"
        assert entry["validator_sha256"] == "a" * 64
        assert len(entry["digest_sha256"]) == 64
        # the digest is the tree digest the consumer recomputes
        assert entry["digest_sha256"] == sha256_tree(tmp_path / "tasks" / name)

    rows = [
        json.loads(line)
        for line in (tmp_path / manifest["task_rows_path"]).read_text().splitlines()
        if line.strip()
    ]
    assert {row["task_path"] for row in rows} == {"tasks/pkg-a", "tasks/pkg-b"}
    for row in rows:
        assert (
            row["task_package_digest"] == names[row["task_path"].split("/", 1)[1]][(
                'digest_sha256'
            )]
        )


def test_the_tree_digest_is_path_nul_sha_nul_per_file():
    """The digest the consumer recomputes: relative path, NUL, that file's sha256 hex, NUL, in
    sorted path order. Asserted against a literal built here with hashlib, not read back.
    """
    import hashlib

    from ppl_synth_data.harbor.export_pymc import sha256_tree

    root = _tmp = None

    def build(base):
        (base / "b.txt").write_text("bee\n")
        (base / "a.txt").write_text("ay\n")
        return base

    import tempfile

    with tempfile.TemporaryDirectory() as work:
        root = build(Path(work))
        one = hashlib.sha256(b"ay\n").hexdigest()
        two = hashlib.sha256(b"bee\n").hexdigest()
        literal = hashlib.sha256(bytes(f"a.txt\0{one}\0b.txt\0{two}\0", "utf-8")).hexdigest()
        assert sha256_tree(root) == literal


# --- Package preflight ------------------------------------------------------------


def test_a_start_gate_is_consulted_before_each_reference_fit():
    """The gate blocks starting the next fit, and a running fit always finishes. So the gate is
    called before each fit, not once for the run.
    """
    import ppl_synth_data.harbor.export_pymc as m

    calls = []

    def gate():
        calls.append(1)

    def fake_reference(
        code,
        visible,
        hidden,
        *,
        env_id,
        target_name,
        blanks,
        expected_validator_sha256,
        attempt_id,
        l6_draws=None,
        l6_tune=None,
    ):
        return {
            "gold_elpd": -1.0,
            "baseline_elpd": -2.0,
            "attempt_id": attempt_id,
            "gold_elpd_done": True,
        }

    rows = [
        {
            "row_id": f"r{i}",
            "runtime": "pymc",
            "prompt": f"p{i}",
            "structural_class": "mixture",
            "visible_data": {"y": [1.0, 2.0], "x": [3.0, 4.0]},
            "heldout_data": {"y": [5.0], "x": [6.0]},
        }
        for i in range(3)
    ]
    golds = {f"r{i}": "import pymc\ndef model(data):\n    return None\n" for i in range(3)}

    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    manifest = Path("/tmp/export-gate-manifest.json")
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "package_id": f"r{i}",
                        "target": "y",
                        "columns": entry["columns"],
                        "status": "complete",
                    }
                    for i in range(3)
                ]
            }
        )
    )
    validator = Path("/tmp/export-gate-validator.py")
    validator.write_text("x = 1\n")
    import hashlib
    import tempfile

    orig = m._compute_elpd_reference
    m._compute_elpd_reference = fake_reference
    try:
        with tempfile.TemporaryDirectory() as work:
            m.export_harbor_tasks_from_rows(
                rows,
                golds,
                Path(work) / "out",
                validator_path=validator,
                validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest(),
                excluded_rows=(),
                attempt_id="attempt-1",
                allow_unlocked_reference_runtime=True,
                role_manifest_path=manifest,
                role_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                start_gate=gate,
            )
    finally:
        m._compute_elpd_reference = orig

    assert len(calls) == 3, "the gate must be consulted before every fit, not once for the run"


def test_a_stop_signal_stops_the_run_at_a_package_boundary():
    """A STOP file means drain: finish the fit that is running, start no more. The run ends with a
    partial result rather than refusing the packages it already wrote.
    """
    import ppl_synth_data.harbor.export_pymc as m

    started = []

    def fake_reference(
        code,
        visible,
        hidden,
        *,
        env_id,
        target_name,
        blanks,
        expected_validator_sha256,
        attempt_id,
        l6_draws=None,
        l6_tune=None,
    ):
        started.append(env_id)
        return {"gold_elpd": -1.0, "baseline_elpd": -2.0, "attempt_id": attempt_id}

    def gate():
        if len(started) >= 2:
            raise m.ExportStopped("STOP file appeared")

    rows = [
        {
            "row_id": f"r{i}",
            "runtime": "pymc",
            "prompt": f"p{i}",
            "structural_class": "mixture",
            "visible_data": {"y": [1.0, 2.0], "x": [3.0, 4.0]},
            "heldout_data": {"y": [5.0], "x": [6.0]},
        }
        for i in range(4)
    ]
    golds = {f"r{i}": "import pymc\ndef model(data):\n    return None\n" for i in range(4)}

    entry = package_entry([preserve_col("y", role="TARGET"), replace_col("x", "quantitative")])
    manifest = Path("/tmp/export-stop-manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "package_id": f"r{i}",
                        "target": "y",
                        "columns": entry["columns"],
                        "status": "complete",
                    }
                    for i in range(4)
                ]
            }
        )
    )
    validator = Path("/tmp/export-stop-validator.py")
    validator.write_text("x = 1\n")
    import hashlib
    import tempfile

    orig = m._compute_elpd_reference
    m._compute_elpd_reference = fake_reference
    try:
        with tempfile.TemporaryDirectory() as work:
            summary = m.export_harbor_tasks_from_rows(
                rows,
                golds,
                Path(work) / "out",
                validator_path=validator,
                validator_sha256=hashlib.sha256(validator.read_bytes()).hexdigest(),
                excluded_rows=(),
                attempt_id="attempt-1",
                allow_unlocked_reference_runtime=True,
                role_manifest_path=manifest,
                role_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                start_gate=gate,
            )
    finally:
        m._compute_elpd_reference = orig

    assert len(started) == 2, "the run must stop starting fits once the gate refuses"
    assert summary.get("stopped") is True
    assert summary["tasks_created"] == 2


def test_the_record_says_whether_the_parent_decoder_rejected_a_worker_result():
    """The record identifies packages rejected by the parent's result decoder.

    The validator signals this as outcome "unjudged" with phase "worker_deserialization".
    """
    from ppl_synth_data.harbor.export_pymc import worker_result_flags

    rejected = {
        "outcome": "unjudged",
        "phase": "worker_deserialization",
        "reason_code": "verifier_contract_failure",
        "error": "worker result is not a strict JSON object",
    }
    assert worker_result_flags(rejected)["worker_deserialization_rejected"] is True

    clean = {"outcome": "scored", "phase": None, "holdout_elpd": -1.0}
    assert worker_result_flags(clean)["worker_deserialization_rejected"] is False

    other = {
        "outcome": "unjudged",
        "phase": "posterior_fit",
        "reason_code": "verifier_contract_failure",
    }
    assert worker_result_flags(other)["worker_deserialization_rejected"] is False
