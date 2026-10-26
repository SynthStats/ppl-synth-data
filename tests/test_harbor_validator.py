from __future__ import annotations

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import pytest
from synthstats_schemas import RewardArtifact

import ppl_synth_data.harbor.validator_template as validator_template
from ppl_synth_data.harbor.validator_template import (
    check_l0_parse,
    check_l1_ast_safety,
    check_l2_model_build,
    check_l3_harbor_pymc_strict_profile,
    check_l6_holdout_elpd,
    run_cascade,
)
from ppl_synth_data.harbor_timeouts import (
    cascade_timeout_budget_s,
    pymc_harbor_verifier_timeout_sec,
)
from ppl_synth_data.runtime_registry import get_runtime_spec

# ---------------------------------------------------------------------------
# L0: Parse
# ---------------------------------------------------------------------------


class TestL0Parse:
    def test_valid_model(self):
        code = "def model(data):\n    return None\n"
        ok, msg = check_l0_parse(code)
        assert ok

    def test_syntax_error(self):
        ok, msg = check_l0_parse("def model(data\n")
        assert not ok
        assert "SyntaxError" in msg

    def test_missing_model_function(self):
        ok, msg = check_l0_parse("def foo(data):\n    pass\n")
        assert not ok
        assert "no 'def model(data)'" in msg

    def test_model_no_params(self):
        ok, msg = check_l0_parse("def model():\n    pass\n")
        assert not ok
        assert "missing" in msg


# ---------------------------------------------------------------------------
# L1: AST Safety
# ---------------------------------------------------------------------------


class TestL1ASTSafety:
    def test_clean_code(self):
        code = "import pymc as pm\nimport numpy as np\ndef model(data):\n    pass\n"
        ok, msg = check_l1_ast_safety(code)
        assert ok

    def test_blocked_import_os(self):
        ok, msg = check_l1_ast_safety("import os\ndef model(data):\n    pass\n")
        assert not ok
        assert "blocked import" in msg

    def test_blocked_import_subprocess(self):
        ok, msg = check_l1_ast_safety("import subprocess\ndef model(data):\n    pass\n")
        assert not ok
        assert "blocked import" in msg

    def test_blocked_eval_call(self):
        ok, msg = check_l1_ast_safety("def model(data):\n    eval('1+1')\n")
        assert not ok
        assert "blocked call" in msg

    def test_blocked_exec_call(self):
        ok, msg = check_l1_ast_safety("def model(data):\n    exec('pass')\n")
        assert not ok
        assert "blocked" in msg

    def test_blocked_open_call(self):
        ok, msg = check_l1_ast_safety("def model(data):\n    open('/etc/passwd')\n")
        assert not ok

    def test_blocked_dunder_access(self):
        ok, msg = check_l1_ast_safety("def model(data):\n    x = data.__globals__\n")
        assert not ok
        assert "blocked dunder" in msg

    def test_aliased_import_blocked(self):
        code = "import os as myos\ndef model(data):\n    myos.system('ls')\n"
        ok, msg = check_l1_ast_safety(code)
        assert not ok

    def test_from_import_blocked(self):
        ok, msg = check_l1_ast_safety("from os import system\ndef model(data):\n    system('ls')\n")
        assert not ok

    def test_l1_blocks_module_attribute_reassignment(self):
        code = "def model(data):\n    np.asarray = lambda *a, **k: None\n    return None\n"
        ok, msg = check_l1_ast_safety(code)
        assert not ok
        assert msg == "blocked module-attribute reassignment: np.asarray"

        dunder_code = 'def model(data):\n np.__dict__["asarray"]=1\n return None'
        ok_dunder, msg_dunder = check_l1_ast_safety(dunder_code)
        assert not ok_dunder
        assert msg_dunder == "blocked dunder attribute: __dict__"


# ---------------------------------------------------------------------------
# L2/L3/L4/L5: require PyMC - tested via cascade
# ---------------------------------------------------------------------------

GOOD_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        y = data.get("y", np.array([0.0, 1.0]))
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""

POTENTIAL_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        pm.Potential("hack", mu * 10)
    return m
"""

IMPORT_OS_MODEL = """\
import os
import pymc as pm

def model(data):
    os.system("echo hacked")
    with pm.Model() as m:
        pm.Normal("x", mu=0, sigma=1)
    return m
"""

DATA_IGNORING_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=np.array([1.0, 2.0]))
    return m
"""

BAD_BUILD_MODEL = """\
import pymc as pm

def model(data):
    return "not a model"
"""

# observes 1000 points when data only has 4 (should fail L4)
FABRICATED_DATA_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.ones(1000))
    return m
"""

DATA_CONTAINER_MODEL = """\
import pymc as pm

def model(data):
    with pm.Model() as m:
        y = pm.Data("y_data", data["y"])
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""

MUTABLE_DATA_CONTAINER_MODEL = """\
import pymc as pm

def model(data):
    with pm.Model() as m:
        y = pm.MutableData("y_data", data["y"])
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return m
"""


class TestCascade:
    @pytest.fixture()
    def sample_data(self):
        import numpy as np

        return {"y": np.array([1.0, 2.0, 3.0, 4.0]), "n": 4}

    def test_good_model_passes_l0_l1(self):
        ok_l0, _ = check_l0_parse(GOOD_MODEL)
        ok_l1, _ = check_l1_ast_safety(GOOD_MODEL)
        assert ok_l0
        assert ok_l1

    def test_import_os_fails_l1(self):
        ok, msg = check_l1_ast_safety(IMPORT_OS_MODEL)
        assert not ok
        assert "os" in msg

    def test_cascade_import_os_fails_at_l1(self):
        import numpy as np

        data = {"y": np.array([1.0, 2.0]), "n": 2}
        result = run_cascade(IMPORT_OS_MODEL, data)
        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L1_ast_safety"
        assert result["reward_source"] == "validation.compilation"
        assert result["diagnostics"]["semantic_status"] == "critical"
        assert result["evidence_kind"] == "verifier"
        assert result["truth_owner"] == "verifier"
        assert result["training_log_reward"] == -20.0
        assert result["reward_components"][0]["name"] == "compilation_cascade"
        assert result["reward_components"][0]["log_contribution"] == -20.0
        assert result["scalarization"]["component_names"] == ["compilation_cascade"]

    def test_cascade_good_model(self, sample_data):
        result = run_cascade(GOOD_MODEL, sample_data)
        assert result["reward"] == 1.0
        assert result["diagnostics"]["failed_layer"] is None
        assert result["reward_source"] == "validation.compilation"
        assert result["diagnostics"]["semantic_status"] == "pass"
        assert result["evidence_kind"] == "verifier"
        assert result["truth_owner"] == "verifier"
        assert result["training_log_reward"] == 0.0
        assert result["reward_components"][0]["name"] == "compilation_cascade"
        assert result["reward_components"][0]["log_contribution"] == 0.0
        assert result["scalarization"]["method"] == "weighted_sum"
        assert "L0_parse" in result["diagnostics"]["details"]
        assert "L5_data_perturbation" in result["diagnostics"]["details"]

    def test_cascade_emits_contract_reward_artifact(self, sample_data):
        result = run_cascade(GOOD_MODEL, sample_data)

        assert set(result) >= {
            "schema_version",
            "reward_source",
            "evidence_kind",
            "truth_owner",
            "producer",
            "metrics",
            "diagnostics",
            "provenance",
            "reward_transform",
            "training_log_reward",
            "reward_components",
            "scalarization",
        }
        assert result["reward_source"] == "validation.compilation"
        assert result["producer"] == "ppl_synth_data.harbor.validator_template"
        assert result["metrics"]["cascade_reward"] == 1.0
        assert result["metrics"]["passed_layers"] == 6
        assert result["diagnostics"]["failed_layer"] is None
        assert result["provenance"]["verifier"] == "harbor_pymc_l0_l5"
        assert result["reward_transform"]["kind"] == "binary_log_reward"
        assert result["reward_transform"]["lower_log_reward_bound"] == -20.0
        assert result["reward_components"][0]["source"] == "validation.compilation"
        assert result["reward_components"][0]["producer"] == (
            "ppl_synth_data.harbor.validator_template"
        )
        assert result["scalarization"]["component_names"] == ["compilation_cascade"]
        RewardArtifact.model_validate(result)

    @pytest.mark.parametrize(
        "code",
        [DATA_CONTAINER_MODEL, MUTABLE_DATA_CONTAINER_MODEL],
    )
    def test_cascade_accepts_pymc_data_containers(self, sample_data, code):
        result = run_cascade(code, sample_data)
        assert result["reward"] == 1.0
        assert result["diagnostics"]["failed_layer"] is None

    def test_cascade_potential_fails_l3(self, sample_data):
        result = run_cascade(POTENTIAL_MODEL, sample_data)
        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L3_harbor_pymc_strict_profile"
        assert "Potential" in (result["diagnostics"]["failure_reason"] or "")

    def test_l2_potential_model_does_not_emit_prior_predictive_warning(self, sample_data):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ok, msg = check_l2_model_build(POTENTIAL_MODEL, sample_data)

        assert ok
        assert msg == "ok"
        assert not any(
            "Potentials on other parameters is ignored during prior predictive sampling"
            in str(w.message)
            for w in caught
        )

    def test_cascade_syntax_error_fails_l0(self, sample_data):
        result = run_cascade("def model(data\n", sample_data)
        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L0_parse"

    @pytest.mark.narrow_unit_not_acceptance
    def test_cascade_l0_failure_does_not_preload_runtime(self, monkeypatch, sample_data):
        called = False

        def fake_load_runtime_modules():
            nonlocal called
            called = True
            raise AssertionError("runtime preload should not run before L0/L1 succeed")

        monkeypatch.setattr(
            "ppl_synth_data.harbor.validator_template._load_runtime_modules",
            fake_load_runtime_modules,
        )

        result = run_cascade("def model(data\n", sample_data)

        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L0_parse"
        assert called is False

    def test_cascade_bad_build_fails_l2(self, sample_data):
        result = run_cascade(BAD_BUILD_MODEL, sample_data)
        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L2_model_build"
        assert "pm.Model" in (result["diagnostics"]["failure_reason"] or "")

    @pytest.mark.narrow_unit_not_acceptance
    def test_cascade_runtime_preload_failure_is_reported_as_l2(self, monkeypatch, sample_data):
        def fake_load_runtime_modules():
            raise RuntimeError("bootstrap failed")

        monkeypatch.setattr(
            "ppl_synth_data.harbor.validator_template._load_runtime_modules",
            fake_load_runtime_modules,
        )

        result = run_cascade(GOOD_MODEL, sample_data)

        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L2_model_build"
        assert "bootstrap failed" in (result["diagnostics"]["failure_reason"] or "")

    def test_pymc_harbor_timeout_covers_cascade_budget(self):
        runtime_spec = get_runtime_spec("pymc")

        assert not hasattr(validator_template, "cascade_timeout_budget_s")
        assert not hasattr(validator_template, "pymc_harbor_verifier_timeout_sec")
        assert cascade_timeout_budget_s() == 176
        assert runtime_spec.harbor_verifier_timeout_sec == pymc_harbor_verifier_timeout_sec()
        assert runtime_spec.harbor_verifier_timeout_sec > cascade_timeout_budget_s()

    def test_l6_holdout_timeout_stays_synced_and_grows_the_wall_budget(self):
        """The standalone validator and host mirror have equal L6 timeout constants.

        The container wall-clock budget covers the L6 timeout. The validator is copied into
        tasks and cannot import the host timeout module.
        """
        from ppl_synth_data import harbor_timeouts

        assert (
            harbor_timeouts.L6_HOLDOUT_ELPD_TIMEOUT_S
            == validator_template.L6_HOLDOUT_ELPD_TIMEOUT_S
        )
        runtime_spec = get_runtime_spec("pymc")
        assert (
            runtime_spec.harbor_verifier_timeout_sec
            > harbor_timeouts.full_cascade_timeout_budget_s()
        )
        assert harbor_timeouts.full_cascade_timeout_budget_s() == (
            cascade_timeout_budget_s() + validator_template.L6_HOLDOUT_ELPD_TIMEOUT_S
        )

    @pytest.mark.narrow_unit_not_acceptance
    def test_validator_template_vendors_timeout_constants(self, monkeypatch):
        validator_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ppl_synth_data"
            / "harbor"
            / "validator_template.py"
        )
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ppl_synth_data.harbor_timeouts":
                raise ModuleNotFoundError("template must be self-contained")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", fake_import)

        spec = importlib.util.spec_from_file_location(
            "validator_template_timeout_policy_test",
            validator_path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.L2_RUNTIME_PRELOAD_TIMEOUT_S == 90

    def test_cascade_data_ignoring_model_fails_l5(self, sample_data):
        result = run_cascade(DATA_IGNORING_MODEL, sample_data)
        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L5_data_perturbation"
        assert "ignores data" in (result["diagnostics"]["failure_reason"] or "")

    def test_cascade_fabricated_data_fails_l4(self, sample_data):
        result = run_cascade(FABRICATED_DATA_MODEL, sample_data)
        assert result["reward"] == 0.0
        assert result["diagnostics"]["failed_layer"] == "L4_exploit_heuristics"
        assert "fabricated" in (result["diagnostics"]["failure_reason"] or "")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "linux",
    reason=(
        "main() --role reducer spawns a masked worker via "
        "`unshare --mount --pid --fork` which requires Linux"
    ),
)
class TestMain:
    def test_main_writes_reward_json(self, tmp_path, monkeypatch):
        """A structurally-good model with real holdout + elpd_reference scores
        through L6 to a real ELPD reward; never a silent binary pass=1.0.
        """
        import json

        import numpy as np
        import pandas as pd

        import ppl_synth_data.harbor.validator_template as validator_template
        from ppl_synth_data.harbor.validator_template import main

        monkeypatch.setattr(validator_template, "L6_DRAWS", 25)
        monkeypatch.setattr(validator_template, "L6_TUNE", 25)

        sub = tmp_path / "submission.py"
        sub.write_text(GOOD_MODEL)

        df = pd.DataFrame({"y": np.array([1.0, 2.0, 3.0, 4.0])})
        parquet = tmp_path / "data.parquet"
        df.to_parquet(parquet, index=False)

        holdout_df = pd.DataFrame({"y": np.array([2.5, 3.5])})
        holdout_parquet = tmp_path / "holdout.parquet"
        holdout_df.to_parquet(holdout_parquet, index=False)

        reference = tmp_path / "elpd_reference.json"
        reference.write_text(
            json.dumps({"gold_elpd": -3.0, "baseline_elpd": -10.0, "target_name": "y"})
        )

        artifact_out = tmp_path / "results" / "reward.json"
        harbor_rewards_out = tmp_path / "verifier" / "reward.json"

        monkeypatch.setattr(
            "sys.argv",
            [
                "validator.py",
                "--role",
                "reducer",
                "--submission",
                str(sub),
                "--data",
                str(parquet),
                "--holdout-data",
                str(holdout_parquet),
                "--elpd-reference",
                str(reference),
                "--reward-artifact-out",
                str(artifact_out),
                "--harbor-rewards-out",
                str(harbor_rewards_out),
            ],
        )

        rc = main()
        assert rc == 0
        assert artifact_out.exists()
        assert harbor_rewards_out.exists()
        payload = json.loads(artifact_out.read_text())
        assert payload["reward_source"] == "validation.pymc_elpd"
        assert payload["producer"] == "ppl_synth_data.harbor.validator_template"
        assert payload["diagnostics"]["elpd_status"] == "scored"
        assert payload["diagnostics"]["semantic_status"] == "pass"
        assert math.isfinite(payload["training_log_reward"])
        harbor_rewards = json.loads(harbor_rewards_out.read_text())
        assert harbor_rewards["passed_layers"] == 7
        assert harbor_rewards["total_layers"] == 7
        assert all(isinstance(v, (float, int)) for v in harbor_rewards.values())

    def test_main_returns_nonzero_on_failure(self, tmp_path, monkeypatch):
        import numpy as np
        import pandas as pd

        from ppl_synth_data.harbor.validator_template import main

        sub = tmp_path / "submission.py"
        sub.write_text("def model(data\n")  # syntax error

        df = pd.DataFrame({"y": np.array([1.0])})
        parquet = tmp_path / "data.parquet"
        df.to_parquet(parquet, index=False)

        artifact_out = tmp_path / "reward.json"
        harbor_rewards_out = tmp_path / "harbor-rewards.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "validator.py",
                "--role",
                "reducer",
                "--submission",
                str(sub),
                "--data",
                str(parquet),
                "--holdout-data",
                str(tmp_path / "missing-holdout.parquet"),
                "--elpd-reference",
                str(tmp_path / "missing-elpd-reference.json"),
                "--reward-artifact-out",
                str(artifact_out),
                "--harbor-rewards-out",
                str(harbor_rewards_out),
            ],
        )

        rc = main()
        assert rc == 1
        payload = json.loads(artifact_out.read_text())
        assert payload["diagnostics"]["semantic_status"] == "critical"
        harbor_rewards = json.loads(harbor_rewards_out.read_text())
        assert harbor_rewards["reward"] == 0.0
        assert harbor_rewards["training_log_reward"] == -20.0

    def test_main_missing_holdout_is_fail_loud_not_silent_pass(self, tmp_path, monkeypatch):
        """A structurally valid model without holdout data receives the reward floor."""
        import json

        import numpy as np
        import pandas as pd

        from ppl_synth_data.harbor.validator_template import main

        sub = tmp_path / "submission.py"
        sub.write_text(GOOD_MODEL)

        df = pd.DataFrame({"y": np.array([1.0, 2.0, 3.0, 4.0])})
        parquet = tmp_path / "data.parquet"
        df.to_parquet(parquet, index=False)

        artifact_out = tmp_path / "reward.json"
        harbor_rewards_out = tmp_path / "harbor-rewards.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "validator.py",
                "--role",
                "reducer",
                "--submission",
                str(sub),
                "--data",
                str(parquet),
                "--holdout-data",
                str(tmp_path / "missing-holdout.parquet"),
                "--elpd-reference",
                str(tmp_path / "missing-elpd-reference.json"),
                "--reward-artifact-out",
                str(artifact_out),
                "--harbor-rewards-out",
                str(harbor_rewards_out),
            ],
        )

        rc = main()
        assert rc == 1
        payload = json.loads(artifact_out.read_text())
        assert payload["reward_source"] == "heuristic.pymc_elpd_error"
        assert payload["diagnostics"]["elpd_status"] == "missing_holdout"
        assert payload["training_log_reward"] == -20.0
        harbor_rewards = json.loads(harbor_rewards_out.read_text())
        assert harbor_rewards["reward"] == 0.0


# L3 checks observed-RV parameter graphs and forward-graph branching.
# pm.Data predictors and unobserved predictive nodes are accepted; forbidden
# parameter dependencies and data-dependent branches are rejected.

# A pm.Data predictor feeds mu while observed receives a NumPy response array.
PMDATA_PREDICTOR_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    xm = float(np.mean(x)); xs = float(np.std(x)) or 1.0
    with pm.Model() as m:
        xd = pm.Data("x", x)
        xz = (xd - xm) / xs
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        s = pm.HalfNormal("s", sigma=1.0)
        pm.Normal("y", mu=a + b * xz, sigma=s, observed=y)
    return m
"""

# An unobserved posterior-predictive node conditions on new pm.Data.
PPC_NEW_NODE_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    xm = float(np.mean(x)); xs = float(np.std(x)) or 1.0
    with pm.Model() as m:
        xd = pm.Data("x", x)
        xz = (xd - xm) / xs
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        s = pm.HalfNormal("s", sigma=1.0)
        pm.Normal("y", mu=a + b * xz, sigma=s, observed=y)
        x_new = pm.Data("x_new", np.array([xm], dtype=float))
        xz_new = (x_new - xm) / xs
        pm.Normal("y_new", mu=a + b * xz_new, sigma=s)
    return m
"""

# Softmax/Categorical regression with pm.Data covariates passes validation.
SOFTMAX_CATEGORICAL_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype="int64")
    X = np.asarray(data["X"], dtype=float)
    xm = X.mean(axis=0); xs = X.std(axis=0); xs = np.where(xs > 0, xs, 1.0)
    Xz = (X - xm) / xs
    n = y.shape[0]
    coords = {
        "obs": np.arange(n),
        "pred": ["p0", "p1"],
        "cat_nb": ["c1", "c2"],
        "cat": ["c0", "c1", "c2"],
    }
    with pm.Model(coords=coords) as m:
        Xd = pm.Data("X", Xz, dims=("obs", "pred"))
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0, dims="cat_nb")
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims=("pred", "cat_nb"))
        eta_nb = intercept + pm.math.dot(Xd, beta)
        eta0 = pm.math.zeros_like(eta_nb[:, :1])
        eta = pm.math.concatenate([eta0, eta_nb], axis=1)
        p = pm.Deterministic("p", pm.math.softmax(eta, axis=1), dims=("obs", "cat"))
        pm.Categorical("y", p=p, observed=y, dims="obs")
    return m
"""

# A user-written data-dependent branch in the forward model graph is refused.
DATA_BRANCH_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        s = pm.HalfNormal("s", sigma=1.0)
        mu = pm.math.switch(xd > 0.0, a, -a)
        pm.Normal("y", mu=mu, sigma=s, observed=y)
    return m
"""

# GENUINE violation: empirical-Bayes leak: data flows into the prior LOCATION
# of a model parameter that is upstream of the observed likelihood. MUST reject.
DATA_IN_PRIOR_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        b = pm.Normal("b", mu=pm.math.sum(xd), sigma=1.0)
        s = pm.HalfNormal("s", sigma=1.0)
        pm.Normal("y", mu=b, sigma=s, observed=y)
    return m
"""

# GENUINE violation: pm.Data declared but wired to nothing (decorative data).
# MUST reject.
DEAD_DATA_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        pm.Data("x", x)
        a = pm.Normal("a", mu=0.0, sigma=1.0)
        s = pm.HalfNormal("s", sigma=1.0)
        pm.Normal("y", mu=a, sigma=s, observed=y)
    return m
"""


class TestL3StrictProfileGraph:
    @pytest.fixture()
    def reg_data(self):
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.normal(size=32)
        y = 0.5 + 1.3 * x + rng.normal(size=32) * 0.1
        return {"y": y, "x": x}

    @pytest.fixture()
    def cat_data(self):
        import numpy as np

        rng = np.random.default_rng(1)
        X = rng.normal(size=(48, 2))
        y = rng.integers(0, 3, size=48)
        return {"y": y, "X": X}

    def test_pmdata_predictor_passes_l3(self, reg_data):
        ok, reason = check_l3_harbor_pymc_strict_profile(PMDATA_PREDICTOR_MODEL, reg_data)
        assert ok, reason

    def test_posterior_predictive_new_node_passes_l3(self, reg_data):
        ok, reason = check_l3_harbor_pymc_strict_profile(PPC_NEW_NODE_MODEL, reg_data)
        assert ok, reason

    def test_softmax_categorical_passes_l3(self, cat_data):
        ok, reason = check_l3_harbor_pymc_strict_profile(SOFTMAX_CATEGORICAL_MODEL, cat_data)
        assert ok, reason

    def test_data_dependent_branch_fails_l3(self, reg_data):
        ok, reason = check_l3_harbor_pymc_strict_profile(DATA_BRANCH_MODEL, reg_data)
        assert not ok
        assert "data-dependent branching" in reason

    def test_data_in_prior_fails_l3(self, reg_data):
        ok, reason = check_l3_harbor_pymc_strict_profile(DATA_IN_PRIOR_MODEL, reg_data)
        assert not ok
        assert "data used in prior" in reason

    def test_unread_data_passes_l3(self, reg_data):
        ok, reason = check_l3_harbor_pymc_strict_profile(DEAD_DATA_MODEL, reg_data)
        assert ok, reason


# ---------------------------------------------------------------------------
# L3: observe each response once; do not use it in its own parameters.
# ---------------------------------------------------------------------------

DUPLICATE_LENGTH_OBSERVATION_MODEL = """\
import numpy as np
import pymc as pm

def model(data):
    with pm.Model() as m:
        {response}
        mu = pm.Normal("mu", mu=2.0, sigma=1.0)
        pm.Normal("length", mu=mu, sigma=0.25, observed=y)
        pm.Normal("length_again", mu=mu, sigma=0.25, observed=y)
    return m
"""

OWN_RESPONSE_PARAMETER_MODEL = """\
import numpy as np
import pymc as pm

def model(data):
    with pm.Model() as m:
        {response}
        pm.Normal("length", mu={mu}, sigma={sigma}, observed=y)
    return m
"""

_L3_RESPONSE_BINDINGS = [
    pytest.param('y = pm.Data("length_data", data["length"])', id="pmdata"),
    pytest.param('y = np.asarray(data["length"], dtype=float)', id="raw_numpy"),
]


class TestL3ObservationDiscipline:
    @pytest.fixture()
    def length_data(self):
        import numpy as np

        return {"length": np.array([1.8, 2.0, 2.3, 2.5])}

    @pytest.fixture()
    def run_length_cascade(self, tmp_path, monkeypatch, length_data):
        import pandas as pd

        sub = tmp_path / "submission.py"
        parquet = tmp_path / "data.parquet"
        pd.DataFrame(length_data).to_parquet(parquet, index=False)
        holdout_parquet = tmp_path / "holdout.parquet"
        pd.DataFrame({"length": [2.5, 3.5]}).to_parquet(holdout_parquet, index=False)
        reference = tmp_path / "elpd_reference.json"
        reference.write_text(
            json.dumps({"gold_elpd": -3.0, "baseline_elpd": -10.0, "target_name": "length"})
        )
        artifact_out = tmp_path / "reward.json"
        harbor_rewards_out = tmp_path / "harbor-rewards.json"

        def run(code):
            sub.write_text(code)
            monkeypatch.setattr(
                "sys.argv",
                [
                    "validator.py",
                    "--role",
                    "reducer",
                    "--submission",
                    str(sub),
                    "--data",
                    str(parquet),
                    "--holdout-data",
                    str(holdout_parquet),
                    "--elpd-reference",
                    str(reference),
                    "--reward-artifact-out",
                    str(artifact_out),
                    "--harbor-rewards-out",
                    str(harbor_rewards_out),
                ],
            )

            rc = validator_template.main()
            payload = json.loads(artifact_out.read_text())
            RewardArtifact.model_validate(payload)
            diagnostics = payload["diagnostics"]
            assert rc == 1, payload
            assert diagnostics["semantic_status"] == "critical", diagnostics
            assert diagnostics.get("elpd_status") != "scored", diagnostics
            assert payload["reward"] == 0.0
            harbor_rewards = json.loads(harbor_rewards_out.read_text())
            assert harbor_rewards["reward"] == 0.0
            return diagnostics

        return run

    @pytest.mark.heavy_lane
    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="main() --role reducer requires Linux `unshare --mount --pid --fork`",
    )
    @pytest.mark.parametrize("response", _L3_RESPONSE_BINDINGS)
    def test_repeated_length_observation_is_refused_as_fabricated_data(
        self, run_length_cascade, response
    ):
        code = DUPLICATE_LENGTH_OBSERVATION_MODEL.format(response=response)

        diagnostics = run_length_cascade(code)

        assert diagnostics["failed_layer"] == "L4_exploit_heuristics", diagnostics
        assert "refusal_code" not in diagnostics
        assert diagnostics["failure_reason"].startswith("fabricated data"), diagnostics

    @pytest.mark.heavy_lane
    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="main() --role reducer requires Linux `unshare --mount --pid --fork`",
    )
    @pytest.mark.parametrize("response", _L3_RESPONSE_BINDINGS)
    @pytest.mark.parametrize(
        ("mu", "sigma", "refusal_code"),
        [
            pytest.param("y", "0.01", "y_self_conditioned", id="direct_mean"),
            pytest.param(
                "pm.math.sum(y) / y.shape[0]",
                "0.01",
                "y_statistic_in_likelihood",
                id="transformed_mean",
            ),
            pytest.param(
                "0.0", "pm.math.abs(y) + 0.01", "y_self_conditioned", id="transformed_scale"
            ),
        ],
    )
    def test_own_response_parameter_is_refused_at_l6(
        self, run_length_cascade, response, mu, sigma, refusal_code
    ):
        code = OWN_RESPONSE_PARAMETER_MODEL.format(response=response, mu=mu, sigma=sigma)

        diagnostics = run_length_cascade(code)

        assert diagnostics["failed_layer"] == "L6_holdout_elpd", diagnostics
        assert diagnostics["elpd_status"] == "binding_refused", diagnostics
        assert diagnostics["refusal_code"] == refusal_code, diagnostics

    @pytest.mark.parametrize("container", [False, True], ids=["raw_numpy", "pmdata"])
    def test_distinct_equal_responses_pass_l3(self, length_data, container):
        data = {**length_data, "width": length_data["length"].copy()}
        length = 'pm.Data("length_data", data["length"])' if container else 'data["length"]'
        width = 'pm.Data("width_data", data["width"])' if container else 'data["width"]'
        code = f"""\
import pymc as pm

def model(data):
    with pm.Model() as m:
        length = {length}
        width = {width}
        mu = pm.Normal("mu", mu=2.0, sigma=1.0)
        pm.Normal("length", mu=mu, sigma=0.25, observed=length)
        pm.Normal("width", mu=mu, sigma=0.25, observed=width)
    return m
"""

        ok, reason = check_l3_harbor_pymc_strict_profile(code, data)

        assert ok, reason

    def test_age_covariate_with_observed_length_passes_l3(self, length_data):
        import numpy as np

        data = {**length_data, "age": np.array([1.0, 3.0, 8.0, 15.0])}
        code = """\
import pymc as pm

def model(data):
    with pm.Model() as m:
        age = pm.Data("age_data", data["age"])
        y = pm.Data("length_data", data["length"])
        alpha = pm.Normal("alpha", mu=2.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0)
        pm.Normal("length", mu=alpha + beta * age, sigma=0.25, observed=y)
    return m
"""

        ok, reason = check_l3_harbor_pymc_strict_profile(code, data)

        assert ok, reason

    @pytest.mark.parametrize("reference", ["dugongs", "disease_spread"])
    def test_literal_reference_program_passes_l3(self, reference):
        import numpy as np

        # These fixtures preserve the two demonstration reference programs.
        code = (
            Path(__file__).parent / "fixtures" / "l3_observation" / f"{reference}_reference.py"
        ).read_text()
        datasets = {
            "dugongs": {
                "age": np.array([1.0, 3.0, 8.0, 15.0]),
                "length": np.array([1.8, 2.0, 2.3, 2.5]),
            },
            "disease_spread": {
                "Time": np.array([0.1, 0.3, 0.6, 1.0]),
                "Infected_Count": np.array([4, 12, 21, 31]),
            },
        }

        ok, reason = check_l3_harbor_pymc_strict_profile(code, datasets[reference])

        assert ok, reason

    def test_new_prediction_data_passes_l3(self):
        import numpy as np

        data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([2.0, 3.5, 4.0])}

        ok, reason = check_l3_harbor_pymc_strict_profile(PPC_NEW_NODE_MODEL, data)

        assert ok, reason


# L6 target-isolation tests compare direct target reuse and a valid model.
# Small draw/tune overrides reduce test time without changing production defaults.

HONEST_GOLD_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        a = pm.Normal("a", mu=0.0, sigma=2.0)
        b = pm.Normal("b", mu=0.0, sigma=2.0)
        s = pm.HalfNormal("s", sigma=1.0)
        pm.Normal("y", mu=a + b * xd, sigma=s, observed=y)
    return m
"""

# A mean wired directly to the observed target must reach the reward floor.
GREEDY_TARGET_LEAK_MODEL = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        leak = pm.Data("leak", y)
        s = pm.HalfNormal("s", 1.0)
        b = pm.Normal("b", 0, 1)
        mu = leak + 0.0 * xd * b
        pm.Normal("y", mu=mu, sigma=s, observed=y)
    return m
"""

# Direct target reuse with a small fixed sigma must reach the reward floor.
TUNED_TARGET_LEAK_MODEL_TEMPLATE = """\
import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    with pm.Model() as m:
        xd = pm.Data("x", x)
        leak = pm.Data("leak", y)
        b = pm.Normal("b", 0, 1)
        mu = leak + 0.0 * xd * b
        pm.Normal("y", mu=mu, sigma={sigma:.6f}, observed=y)
    return m
"""


@pytest.mark.slow
class TestL6TargetIsolation:
    """both probe cheats must floor (outcome !=

    "scored", or avg_holdout_logp <= -20) while the honest gold still scores.
    Real NUTS fits, real check_l6_holdout_elpd; no mocks.
    """

    _DRAWS = 30
    _TUNE = 30

    @pytest.fixture()
    def data(self):
        import numpy as np

        rng = np.random.default_rng(7)
        x_visible = rng.normal(size=24)
        y_visible = 0.4 + 1.1 * x_visible + rng.normal(size=24) * 0.15
        x_holdout = rng.normal(size=8)
        y_holdout = 0.4 + 1.1 * x_holdout + rng.normal(size=8) * 0.15
        visible = {"y": y_visible, "x": x_visible}
        holdout = {"y": y_holdout, "x": x_holdout}
        return visible, holdout

    def test_honest_gold_is_scored(self, data):
        visible, holdout = data
        result = check_l6_holdout_elpd(
            HONEST_GOLD_MODEL, visible, holdout, target_name="y", draws=self._DRAWS, tune=self._TUNE
        )
        assert result["outcome"] == "scored", result
        assert math.isfinite(result["avg_holdout_logp"])

    def test_greedy_target_leak_is_floored(self, data):
        visible, holdout = data
        result = check_l6_holdout_elpd(
            GREEDY_TARGET_LEAK_MODEL,
            visible,
            holdout,
            target_name="y",
            draws=self._DRAWS,
            tune=self._TUNE,
        )
        floored = result["outcome"] != "scored" or result["avg_holdout_logp"] <= -20
        assert floored, result

    def test_tuned_target_leak_is_floored(self, data):
        import numpy as np

        visible, holdout = data
        sigma = 0.05 * float(np.std(visible["y"]))
        code = TUNED_TARGET_LEAK_MODEL_TEMPLATE.format(sigma=sigma)
        result = check_l6_holdout_elpd(
            code,
            visible,
            holdout,
            target_name="y",
            draws=self._DRAWS,
            tune=self._TUNE,
        )
        floored = result["outcome"] != "scored" or result["avg_holdout_logp"] <= -20
        assert floored, result


# Check worker namespace flags, PR_SET_DUMPABLE and temporal-barrier ordering.
# These Linux checks exercise process isolation independently of L1's import restrictions.


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="_spawn_worker_and_score spawns `unshare --mount --pid --fork` (Linux only)",
)
class TestNewAttackClassMechanisms:
    def test_worker_spawn_uses_pid_and_mount_namespaces_and_denies_dumpable(
        self, tmp_path, monkeypatch
    ):
        """parent-memory (/proc parent-memory scrape): the worker must run inside a
        real PID namespace (so /proc shows only its own subtree; the
        reducer's true-target-holding process is not addressable at all,
        not merely permission-denied) and the reducer must call
        prctl(PR_SET_DUMPABLE, 0) on itself before spawning (denies a
        same-uid /proc/<reducer_pid>/mem read as defense in depth, even
        though PID-namespacing already makes the reducer's real pid
        unaddressable from inside the worker's namespace).
        """
        import ctypes
        import subprocess as subprocess_mod

        import numpy as np

        import ppl_synth_data.harbor.validator_template as validator_template

        prctl_calls: list[tuple[int, int]] = []
        real_cdll = ctypes.CDLL

        class _SpyLibc:
            def __init__(self, real):
                self._real = real

            def prctl(self, option, arg2, arg3, arg4, arg5):
                prctl_calls.append((option, arg2))
                return self._real.prctl(option, arg2, arg3, arg4, arg5)

        def _spy_cdll(name, **kwargs):
            real = real_cdll(name, **kwargs)
            return _SpyLibc(real) if name is None else real

        monkeypatch.setattr(ctypes, "CDLL", _spy_cdll)

        captured_cmd: list[str] = []
        real_popen = subprocess_mod.Popen

        class _SpyPopen(real_popen):
            def __init__(self, cmd, *args, **kwargs):
                captured_cmd.extend(cmd)
                super().__init__(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess_mod, "Popen", _SpyPopen)

        sub = tmp_path / "submission.py"
        sub.write_text(GOOD_MODEL)
        result = validator_template._spawn_worker_and_score(
            submission_path=str(sub),
            visible_data={"y": np.array([1.0, 2.0, 3.0, 4.0])},
            censored_holdout={"y": np.array([2.5, 3.5])},
            true_target=np.array([9.0, 10.0]),
            target_name="y",
            timeout=60,
        )

        assert "--mount" in captured_cmd
        assert "--pid" in captured_cmd
        assert "--fork" in captured_cmd
        assert ("unshare" in captured_cmd[0]) or captured_cmd[0].endswith("unshare")
        # PR_SET_DUMPABLE=0 (option 4, arg2 0) called on the reducer itself,
        # before the worker (holding no true-target knowledge at all) runs.
        assert (4, 0) in prctl_calls
        assert result.get("phase") in ("structural", "l6_early", "l6_scored") or "outcome" in result

    def test_worker_env_is_scrubbed_of_reducer_secrets(self, tmp_path, monkeypatch):
        """worker-environment (env-var leak): the worker subprocess's environment is a
        fixed, hardcoded dict (PATH + PYTENSOR_FLAGS only); gold_elpd,
        baseline_elpd, and the true target NEVER pass through env at any
        point in the reducer/worker protocol (they cross the temporal-
        barrier pipe, never an env var), so they structurally cannot leak
        into the worker's environment regardless of what a submission tries.
        """
        import subprocess as subprocess_mod

        import numpy as np

        import ppl_synth_data.harbor.validator_template as validator_template

        captured_envs: list[dict] = []
        real_popen = subprocess_mod.Popen

        class _SpyPopen(real_popen):
            def __init__(self, cmd, *args, **kwargs):
                captured_envs.append(dict(kwargs.get("env") or {}))
                super().__init__(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess_mod, "Popen", _SpyPopen)

        sub = tmp_path / "submission.py"
        sub.write_text(GOOD_MODEL)
        secret_gold_elpd = -3.14159265
        secret_baseline_elpd = -9.99999999
        validator_template._spawn_worker_and_score(
            submission_path=str(sub),
            visible_data={"y": np.array([1.0, 2.0, 3.0, 4.0])},
            censored_holdout={"y": np.array([2.5, 3.5])},
            true_target=np.array([9.0, 10.0]),
            target_name="y",
            timeout=60,
        )

        assert len(captured_envs) == 1
        env = captured_envs[0]
        env_repr = repr(env)
        assert str(secret_gold_elpd) not in env_repr
        assert str(secret_baseline_elpd) not in env_repr
        assert "y" not in env  # the target_name is not a passed env var
        assert set(env.keys()) == {"PATH", "PYTENSOR_FLAGS"}

    def test_true_target_write_only_happens_after_worker_handshake(self):
        """target-pipe (anonymous-pipe / fd race): source-level proof of the
        temporal-barrier ordering _spawn_worker_and_score implements: the
        reducer writes the true target to target_wfd ONLY inside the
        `if handshake == b"\\x01":` branch, i.e. strictly after reading the
        worker's own handshake byte (which the worker only sends after every
        submission exec() (L0-L5 and the L6 censored-data build) has
        already completed; see _main_worker). A submission cannot race this:
        even if it somehow held a reference to the pipe's read end (it does
        not; pass_fds only grants the WORKER process the fd; nothing in
        the submission's exec namespace exposes fds at all), the write does
        not happen until after the submission's last exec() call has
        already returned control to the worker's own (non-exec'd) code.
        """
        import inspect

        import ppl_synth_data.harbor.validator_template as validator_template

        source = inspect.getsource(validator_template._spawn_worker_and_score)
        handshake_write_idx = source.index('if handshake == b"\\x01":')
        target_write_idx = source.index("_write_framed(target_wfd, buf.getvalue())")
        handshake_read_idx = source.index("handshake = os.read(handshake_rfd, 1)")
        assert handshake_read_idx < handshake_write_idx < target_write_idx, (
            "the true-target write must be lexically inside (and after) the "
            "handshake-read branch: if this ordering breaks, the temporal "
            "barrier the whole isolation design depends on is gone"
        )
