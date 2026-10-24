"""Un-mocked integration tests for the LazyPPL Harbor export pipeline.

Uses real seed files and the real ``taskdata_gen`` module; no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppl_synth_data.harbor.export_lazyppl import export_lazyppl_harbor_tasks

SEEDS_DIR = Path(__file__).resolve().parent.parent / "data" / "lazyppl_synthesis" / "seeds"
DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker" / "lazyppl-harbor"
DOCKER_APP_DIR = DOCKER_DIR / "app"
DOCKER_FILE = DOCKER_DIR / "Dockerfile"

# seeds that exercise different field patterns
SEED_CASES: dict[str, dict] = {
    "simple_normal": {"expected_fields": ["value"]},
    "linear_regression": {"expected_fields": ["xVal", "yVal"]},
    "dp_clustering": {"expected_fields": ["value"]},
}


def _data_rows(seed_names: list[str]) -> dict[str, list[dict[str, float]]]:
    rows_by_seed: dict[str, list[dict[str, float]]] = {}
    for seed_name in seed_names:
        fields = SEED_CASES[seed_name]["expected_fields"]
        rows_by_seed[seed_name] = [
            {field: float(row_index + field_index + 1) for field_index, field in enumerate(fields)}
            for row_index in range(3)
        ]
    return rows_by_seed


def _hidden_rows(seed_names: list[str]) -> dict[str, list[dict[str, float]]]:
    rows_by_seed: dict[str, list[dict[str, float]]] = {}
    for seed_name in seed_names:
        fields = SEED_CASES[seed_name]["expected_fields"]
        rows_by_seed[seed_name] = [
            {field: float(10 + field_index) for field_index, field in enumerate(fields)}
        ]
    return rows_by_seed


@pytest.fixture()
def single_seed_dir(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Copy a single seed into an isolated directory."""
    seed_name: str = request.param
    src = SEEDS_DIR / f"{seed_name}.hs"
    assert src.exists(), f"seed file missing: {src}"
    d = tmp_path / "seeds"
    d.mkdir()
    (d / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return d


# per-seed parametrized tests.


@pytest.mark.parametrize(
    "single_seed_dir, seed_name",
    [(name, name) for name in SEED_CASES],
    indirect=["single_seed_dir"],
)
class TestPerSeed:
    """Run assertions against each seed individually."""

    def _export(self, single_seed_dir: Path, tmp_path: Path) -> tuple[dict, Path]:
        out = tmp_path / "harbor_out"
        seed_name = next(single_seed_dir.glob("*.hs")).stem
        result = export_lazyppl_harbor_tasks(
            single_seed_dir,
            out,
            docker_app_dir=DOCKER_APP_DIR,
            docker_file=DOCKER_FILE,
            data_rows=_data_rows([seed_name]),
            hidden_data_rows=_hidden_rows([seed_name]),
        )
        return result, out

    def test_creates_one_task(self, single_seed_dir: Path, seed_name: str, tmp_path: Path) -> None:
        result, out = self._export(single_seed_dir, tmp_path)
        assert result["tasks_created"] == 1
        assert result["tasks_skipped"] == 0
        task_dirs = list(out.iterdir())
        assert len(task_dirs) == 1

    def test_all_files_present(self, single_seed_dir: Path, seed_name: str, tmp_path: Path) -> None:
        _, out = self._export(single_seed_dir, tmp_path)
        td = next(out.iterdir())

        assert (td / "task.toml").is_file()
        assert (td / "instruction.md").is_file()
        assert (td / "environment" / "Dockerfile").is_file()
        assert (td / "environment" / "workspace" / "Main.hs").is_file()
        assert (td / "environment" / "workspace" / "LazyPPLCompat.hs").is_file()
        assert (td / "environment" / "workspace" / "TaskData.hs").is_file()
        # Held-out rows must never reach the agent-visible image.
        assert not (td / "environment" / "workspace" / "HiddenTaskData.hs").exists()
        assert (td / "environment" / "workspace" / "score.py").is_file()
        assert (td / "environment" / "workspace" / "check.sh").is_file()
        assert (td / "environment" / "workspace" / "REFERENCE.md").is_file()
        assert (td / "tests" / "test.sh").is_file()
        assert (td / "solution" / "Submission.hs").is_file()
        assert (td / "solution" / "solve.sh").is_file()

    def test_taskdata_module_header(
        self, single_seed_dir: Path, seed_name: str, tmp_path: Path
    ) -> None:
        _, out = self._export(single_seed_dir, tmp_path)
        td = next(out.iterdir())
        taskdata = (td / "environment" / "workspace" / "TaskData.hs").read_text()

        assert taskdata.startswith("module TaskData")
        assert "loadData" in taskdata

    def test_taskdata_contains_expected_fields(
        self, single_seed_dir: Path, seed_name: str, tmp_path: Path
    ) -> None:
        _, out = self._export(single_seed_dir, tmp_path)
        td = next(out.iterdir())
        taskdata = (td / "environment" / "workspace" / "TaskData.hs").read_text()

        for field in SEED_CASES[seed_name]["expected_fields"]:
            assert field in taskdata, f"expected field {field!r} not in TaskData.hs"

    def test_solve_sh_hands_submission_to_separate_verifier(
        self, single_seed_dir: Path, seed_name: str, tmp_path: Path
    ) -> None:
        _, out = self._export(single_seed_dir, tmp_path)
        td = next(out.iterdir())
        solve_sh = (td / "solution" / "solve.sh").read_text()

        assert "cp /solution/Submission.hs /app/Submission.hs" in solve_sh
        assert "/tests/" not in solve_sh

    def test_test_sh_splits_canonical_artifact_from_harbor_scalar_rewards(
        self, single_seed_dir: Path, seed_name: str, tmp_path: Path
    ) -> None:
        _, out = self._export(single_seed_dir, tmp_path)
        td = next(out.iterdir())
        test_sh = (td / "tests" / "test.sh").read_text()

        assert "ARTIFACT_REWARD=/logs/verifier/reward_artifact.json" in test_sh
        assert "HARBOR_REWARD=/logs/verifier/reward.json" in test_sh
        # The held-out module is copied into /app/hidden only after the type-check.
        assert "cp /tests/HiddenTaskData.hs /app/HiddenTaskData.hs" not in test_sh
        assert "cp /tests/HiddenTaskData.hs /app/hidden/HiddenTaskData.hs" in test_sh
        assert "Path(sys.argv[2]).write_text(json.dumps(artifact, indent=2))" in test_sh
        assert "'training_log_reward': float(artifact['training_log_reward'])" in test_sh


# multi-seed test.


def test_all_three_seeds_together(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for name in SEED_CASES:
        src = SEEDS_DIR / f"{name}.hs"
        (seeds / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    out = tmp_path / "harbor_out"
    result = export_lazyppl_harbor_tasks(
        seeds,
        out,
        docker_app_dir=DOCKER_APP_DIR,
        docker_file=DOCKER_FILE,
        data_rows=_data_rows(list(SEED_CASES)),
        hidden_data_rows=_hidden_rows(list(SEED_CASES)),
    )

    assert result["tasks_created"] == 3
    assert result["tasks_skipped"] == 0

    task_names = sorted(d.name for d in out.iterdir())
    assert "lazyppl-dp-clustering-00001" in task_names
    assert "lazyppl-linear-regression-00001" in task_names
    assert "lazyppl-simple-normal-00001" in task_names
