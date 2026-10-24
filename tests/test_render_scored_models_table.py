"""Tests for scripts/render_scored_models_table.py.

Tests read real receipts copied byte for byte into fixtures/scored_models_table;
nothing is mocked and no receipt is invented. Paths are relative to this test
file. Missing fixtures raise with their exact path. Edge tests copy these real
receipts into temporary trees and alter only the condition under test.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import render_scored_models_table as rsmt  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scored_models_table"
MANUAL_DIR = _FIXTURES / "elpd_manual_docker_check"
LIVE_TRIAL_DIR = _FIXTURES / "2026-07-22_harbor-0.20-live-trial"
MIGRATION_DIR = _FIXTURES / "2026-07-22_current-pymc-task-migration"
REAL_COMPLETIONS = _FIXTURES / "completions.jsonl"

ARMS = {
    "2026-07-22_harbor-0.20-live-trial": "harbor-0.20",
    "2026-07-22_current-pymc-task-migration": "current-task",
}

GATE_RUN = _FIXTURES / "gate-run"
GATE_ARMS = ["known_answer", "gold", "intercept", "semantic_wrong", "generated", "iterative"]


class TestGateReceipts:
    def test_scored_diagnostics_are_values_or_not_retained(self):
        row = _rows([GATE_RUN / "results/intercept"])[0]
        assert "reliability_flags" not in row
        assert row["pareto_k_warning"] == "false"
        assert row["pareto_k_max"] == "0.3468"
        assert row["density_gate_status"] == "applied"
        for key in ["ess_bulk", "ess_tail", "bfmi", "divergences", "r_hat"]:
            assert row[key] == "not retained", key

    def test_build_failure_diagnostics_are_not_evaluated(self):
        row = _rows([GATE_RUN / "results/generated"])[0]
        for key in [
            "pareto_k_warning",
            "pareto_k_max",
            "density_gate_status",
            "ess_bulk",
            "ess_tail",
            "bfmi",
            "divergences",
            "r_hat",
        ]:
            assert row[key] == "not evaluated", key

    @pytest.mark.parametrize(
        ("arm", "seconds"),
        [
            ("known_answer", "10.96"),
            ("gold", "604.84"),
            ("intercept", "11.18"),
            ("semantic_wrong", "12.01"),
            ("generated", "9.82"),
            ("iterative", "12.12"),
            ("gold_diag", "1142.262"),
        ],
    )
    def test_container_wall_time_uses_real_receipt(self, arm, seconds):
        row = _rows([GATE_RUN / "results" / arm])[0]
        assert row["container_wall_seconds"] == seconds
        assert row["wall_seconds"] == "absent"

    def test_missing_container_time_is_not_trial_time(self):
        row = _rows([LIVE_TRIAL_DIR])[0]
        assert row["container_wall_seconds"] == "absent"
        assert 40 < float(row["wall_seconds"]) < 50

    def test_malformed_container_time_raises(self, tmp_path):
        arm = tmp_path / "results/intercept"
        shutil.copytree(GATE_RUN / "results/intercept", arm)
        (arm / "wall-time.txt").write_text("real unavailable\n")
        with pytest.raises(ValueError, match="wall-time.txt"):
            _rows([arm])

    def test_identity_and_candidate_join_real_gate_receipts(self):
        row = _rows([GATE_RUN / "results/intercept"])[0]
        assert row["env_id"] == "psychology_latent_rt_mixture_12536972429857146790"
        assert row["env_name"] == "latent_strategy_reaction_time_mixture"
        assert row["target_name"] == "log_reaction_time"
        assert Path(row["candidate"]) == GATE_RUN / "results/intercept/candidate.py"
        task = Path(row["task_directory"])
        assert task.is_dir()
        assert row["task_id"] == task.name
        assert json.loads((task / "tests/elpd_reference.json").read_text()) == json.loads(
            (GATE_RUN / "results/intercept/elpd_reference.json").read_text()
        )

    def test_missing_receipts_do_not_borrow_identity_from_directory_name(self, tmp_path):
        source = GATE_RUN / "results/intercept/verifier/reward_artifact.json"
        trial = tmp_path / "plausible-task-name"
        trial.mkdir()
        shutil.copyfile(source, trial / "reward_artifact.json")
        row = _rows([trial])[0]
        for key in ["task_id", "task_directory", "env_id", "env_name", "target_name", "candidate"]:
            assert row[key] == "absent", key

    def test_ambiguous_reference_does_not_choose_a_task(self, tmp_path):
        arm = tmp_path / "results/intercept"
        shutil.copytree(GATE_RUN / "results/intercept", arm)
        for name in ["task-a", "task-b"]:
            tests = tmp_path / "tasks" / name / "tests"
            tests.mkdir(parents=True)
            shutil.copyfile(arm / "elpd_reference.json", tests / "elpd_reference.json")
        row = _rows([arm])[0]
        assert row["task_directory"] == "absent"
        assert row["task_id"] == "absent"
        assert row["env_id"] == "psychology_latent_rt_mixture_12536972429857146790"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required real artifact tree not on this machine: {path}")


def _rows(dirs, arms=None):
    return rsmt.build_rows(dirs, arms=arms or {})


class TestManualDockerCheck:
    """7-of-7 layers pass artifact under elpd_manual_docker_check."""

    def setup_method(self):
        _require(MANUAL_DIR)
        self.rows = _rows([MANUAL_DIR])

    def test_one_candidate_found(self):
        assert len(self.rows) == 1

    def test_verdict_pass_from_l3(self):
        assert self.rows[0]["safepymc_verdict"] == "pass"

    def test_layers_passed(self):
        assert self.rows[0]["layers_passed"] == "7/7"

    def test_rewards(self):
        assert self.rows[0]["reward"] == "2.719488"
        assert self.rows[0]["training_log_reward"] == "1.000444"

    def test_metrics(self):
        row = self.rows[0]
        assert row["avg_holdout_logp"] == "1.246894"
        assert row["holdout_elpd"] == "2.4938"
        assert row["train_elpd_loo"] == "8.551"
        assert row["pareto_k_max"] == "0.8366"

    def test_absent_keys_render_absent(self):
        row = self.rows[0]
        # This artifact predates density_gate_status, holdout_pred_percentiles,
        # and has no sibling result.json (harbor_out layout, not a trial dir).
        assert row["density_gate_status"] == "not retained"
        assert "pred_p50" not in row
        assert row["wall_seconds"] == "absent"
        assert row["task_id"] == "absent"

    def test_pareto_warning_from_diagnostics(self):
        assert self.rows[0]["pareto_k_warning"] == "true"


class TestLiveTrial:
    """Scored trial records retain passes, negative ELPD and result.json metrics."""

    def setup_method(self):
        _require(LIVE_TRIAL_DIR)
        self.rows = _rows([LIVE_TRIAL_DIR], arms=ARMS)

    def test_task_id_from_result_json(self):
        assert self.rows[0]["task_id"] == "pymc-dugongs-00000"

    def test_arm_label(self):
        assert self.rows[0]["arm"] == "harbor-0.20"

    def test_verdict_and_layers(self):
        assert self.rows[0]["safepymc_verdict"] == "pass"
        assert self.rows[0]["layers_passed"] == "7/7"

    def test_rewards(self):
        assert self.rows[0]["reward"] == "0.288084"
        assert self.rows[0]["training_log_reward"] == "-1.244505"

    def test_density_gate_status(self):
        assert self.rows[0]["density_gate_status"] == "applied"

    def test_negative_elpd(self):
        assert self.rows[0]["holdout_elpd"] == "-18.7676"

    def test_wall_seconds_from_result_json(self):
        wall = float(self.rows[0]["wall_seconds"])
        assert 40.0 < wall < 50.0

    def test_tokens_absent_without_completions(self):
        assert "prompt_tokens" not in self.rows[0]
        assert "completion_tokens" not in self.rows[0]


class TestMigrationErrorArtifact:
    """Error artifacts preserve heuristic rewards and missing metrics."""

    def setup_method(self):
        _require(MIGRATION_DIR)
        self.rows = _rows([MIGRATION_DIR], arms=ARMS)

    def test_error_reward_values(self):
        assert self.rows[0]["reward"] == "0.0"
        assert self.rows[0]["training_log_reward"] == "-20.0"

    def test_zero_layers(self):
        assert self.rows[0]["layers_passed"] == "0/7"

    def test_verdict_absent_when_no_l3_record(self):
        assert self.rows[0]["safepymc_verdict"] == "absent"

    def test_missing_metrics_render_absent_never_zero(self):
        row = self.rows[0]
        for key in ("avg_holdout_logp", "holdout_elpd", "train_elpd_loo"):
            assert row[key] == "absent", key
        assert row["pareto_k_max"] == "not evaluated"

    def test_no_diagnostic_receipt_does_not_pass(self):
        assert self.rows[0]["pareto_k_warning"] == "not evaluated"


class TestSortAndArms:
    def test_cli_order_accepts_current_directory(self, tmp_path):
        result = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(_SCRIPTS.parent),
                "--no-sync",
                "python",
                str(_SCRIPTS / "render_scored_models_table.py"),
                ".",
                "--order",
                "intercept",
                "--out-csv",
                str(tmp_path / "out.csv"),
                "--out-html",
                str(tmp_path / "out.html"),
            ],
            cwd=GATE_RUN / "results/intercept",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        with (tmp_path / "out.csv").open(newline="") as stream:
            assert next(csv.DictReader(stream))["Arm"] == "intercept"

    def test_order_and_mapping_use_symlink_input_name(self, tmp_path):
        alias = tmp_path / "transferred"
        alias.symlink_to(GATE_RUN / "results/iterative", target_is_directory=True)
        rows = rsmt.build_rows([alias], arms={"transferred": "iterative"}, order=["iterative"])
        assert rows[0]["arm"] == "iterative"
        assert Path(rows[0]["candidate"]) == GATE_RUN / "results/iterative/candidate.py"

    def setup_method(self):
        _require(LIVE_TRIAL_DIR)
        _require(MIGRATION_DIR)
        self.rows = _rows([LIVE_TRIAL_DIR, MIGRATION_DIR], arms=ARMS)

    def test_two_candidates_total(self):
        assert len(self.rows) == 2

    def test_default_order_preserves_input_directories(self):
        assert [r["arm"] for r in self.rows] == ["harbor-0.20", "current-task"]

    def test_explicit_order_uses_arm_labels(self):
        rows = rsmt.build_rows(
            [LIVE_TRIAL_DIR, MIGRATION_DIR],
            arms=ARMS,
            order=["current-task", "harbor-0.20"],
        )
        assert [r["arm"] for r in rows] == ["current-task", "harbor-0.20"]

    @pytest.mark.parametrize(
        "order",
        [
            ["current-task"],
            ["current-task", "unknown"],
            ["current-task", "current-task", "harbor-0.20"],
        ],
    )
    def test_order_rejects_incomplete_unknown_or_duplicate_arms(self, order):
        with pytest.raises(ValueError, match="order"):
            rsmt.build_rows([LIVE_TRIAL_DIR, MIGRATION_DIR], arms=ARMS, order=order)

    def test_candidate_index_assigned_within_dir(self):
        assert {r["candidate_index"] for r in self.rows} == {"0"}


class TestRealCompletionsFile:
    """The real stage-E records have no usage counts; token columns are omitted."""

    def test_tokens_absent_with_real_completions(self):
        _require(LIVE_TRIAL_DIR)
        _require(REAL_COMPLETIONS)
        records = [json.loads(line) for line in REAL_COMPLETIONS.read_text().splitlines() if line]
        assert all(row.get("prompt_tokens") is None for row in records)
        rows = _rows([LIVE_TRIAL_DIR], arms=ARMS)
        assert "prompt_tokens" not in rows[0]
        assert "completion_tokens" not in rows[0]


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tr_count = 0
        self.text_parts: list[str] = []
        self.in_score_table = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_score_table = dict(attrs).get("id") == "scores"
        if tag == "tr" and self.in_score_table:
            self.tr_count += 1

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_score_table = False

    def handle_data(self, data):
        self.text_parts.append(data)


class TestOutputs:
    def setup_method(self):
        _require(LIVE_TRIAL_DIR)
        _require(MIGRATION_DIR)
        self.rows = _rows([LIVE_TRIAL_DIR, MIGRATION_DIR], arms=ARMS)

    def test_csv_roundtrip(self, tmp_path):
        out = tmp_path / "table.csv"
        rsmt.write_csv(self.rows, out)
        with out.open(newline="") as fh:
            parsed = list(csv.DictReader(fh))
        assert len(parsed) == 2
        header = set(parsed[0].keys())
        assert "Holdout ELPD" in header
        assert "Task ID" in header

    def test_html_parseable_and_complete(self, tmp_path):
        out = tmp_path / "table.html"
        rsmt.write_html(
            self.rows,
            out,
            input_dirs=[LIVE_TRIAL_DIR, MIGRATION_DIR],
            csv_path=tmp_path / "table.csv",
        )
        text = out.read_text()
        parser = _TableParser()
        parser.feed(text)
        # 1 header row + 2 candidate rows.
        assert parser.tr_count == 3
        body = "".join(parser.text_parts)
        assert "absent" in body
        assert "pymc-dugongs-00000" in body
        # No em-dashes, no emoji.
        assert "—" not in text
        assert all(ord(ch) < 0x1F300 or ord(ch) > 0x1FAFF for ch in text)
        # Footer carries absolute paths.
        assert str(LIVE_TRIAL_DIR) in text
        assert str(out.resolve()) in text

    def test_html_self_contained(self, tmp_path):
        out = tmp_path / "table.html"
        rsmt.write_html(
            self.rows,
            out,
            input_dirs=[LIVE_TRIAL_DIR],
            csv_path=tmp_path / "table.csv",
        )
        text = out.read_text()
        assert "<style" in text
        assert "http://" not in text and "https://" not in text

    def test_gate_columns_show_receipts_and_omit_unsupported_fields(self, tmp_path):
        rows = _rows([GATE_RUN / "results" / arm for arm in GATE_ARMS])
        out = tmp_path / "gate.csv"
        rsmt.write_csv(rows, out)
        with out.open(newline="") as stream:
            reader = csv.DictReader(stream)
            parsed = list(reader)
            headers = reader.fieldnames
        assert "Container wall time (s)" in headers
        assert "Candidate path" in headers
        assert "Task directory" in headers
        assert "Environment ID" in headers
        assert "Target" in headers
        assert not any(
            "token" in label or "pred_p50" in label or "reliability" in label for label in headers
        )
        assert parsed[2]["Container wall time (s)"] == "11.18"
        assert parsed[2]["Candidate path"] == str(GATE_RUN / "results/intercept/candidate.py")

    def test_html_separates_scores_diagnostics_and_receipt_links(self, tmp_path):
        rows = _rows([GATE_RUN / "results/intercept"])
        out = tmp_path / "gate.html"
        rsmt.write_html(
            rows, out, input_dirs=[GATE_RUN / "results/intercept"], csv_path=tmp_path / "gate.csv"
        )
        text = out.read_text()
        assert "<caption>Scores</caption>" in text
        assert "<caption>Diagnostics</caption>" in text
        assert "Container wall time (s)" in text
        assert (GATE_RUN / "results/intercept/candidate.py").as_uri() in text
        assert "not retained" in text
        assert "reliability" not in text.lower()
        assert "prefers-color-scheme: dark" in text
