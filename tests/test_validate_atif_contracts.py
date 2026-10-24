from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ppl_synth_data.cli import app
from ppl_synth_data.dataset.atif_export import export_atif_from_run
from ppl_synth_data.dataset.mixed_atif_assembly import (
    MixedATIFSource,
    assemble_mixed_atif_sources,
)
from scripts.verification.validate_atif_contracts import main, validate_atif_dir

runner = CliRunner()


def _write_real_stage_g_run(run_dir: Path, *, scores: tuple[float, float] = (-10.0, -12.0)) -> None:
    stage_dir = run_dir / "stage_G_dataset"
    stage_dir.mkdir(parents=True)
    data_dir = run_dir / "stage_C_data" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, 5.0]}).to_parquet(
        data_dir / "env-a.parquet",
        index=False,
    )
    pd.DataFrame({"temperature": [18.0, 19.5, 21.0], "output": [4.0, 4.4, 5.1]}).to_parquet(
        data_dir / "env-b.parquet",
        index=False,
    )
    rows = []
    for index, (env_id, score) in enumerate(zip(("env-a", "env-b"), scores, strict=True), start=1):
        rows.append(
            {
                "id": f"completion-{index:06d}",
                "prompt_id": f"prompt-{index:06d}",
                "env_id": env_id,
                "split": "train",
                "prompt": f"Fit a small model for {env_id}.",
                "completion": "<submit_program>def model(data): pass</submit_program>",
                "parsed_program": "def model(data): pass",
                "data_ref": f"stage_C_data/data/{env_id}.parquet",
                "trace_id": f"trace-{index:06d}",
                "env_spec_json": json.dumps(
                    {
                        "env_id": env_id,
                        "runtime_name": "pymc",
                        "task": f"fixture task for {env_id}",
                    }
                ),
                "provenance_json": json.dumps(
                    {
                        "provenance": {
                            "runtime_name": "pymc",
                            "source_category": "synthetic_invented",
                            "source_url_or_repo": None,
                            "license_signal": "not_applicable_invented",
                            "adaptation_mode": "invented",
                            "source_retrieval_date": "2026-05-27",
                        },
                        "metadata": {},
                    }
                ),
                "validation_json": json.dumps(
                    {
                        "ok": True,
                        "semantic_status": "pass",
                        "semantic_metric": "avg_holdout_logp",
                        "semantic_score": score,
                        "semantic_details": {
                            "avg_holdout_logp": score,
                            "hidden_n_obs": 3,
                        },
                    }
                ),
            }
        )
    (stage_dir / "pymc_with_data.train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (stage_dir / "pymc_with_data.validation.jsonl").write_text("", encoding="utf-8")
    (stage_dir / "pymc_with_data.test.jsonl").write_text("", encoding="utf-8")


def _args(atif_dir: Path, *, producer_root: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "atif_dir": atif_dir,
        "producer_root": producer_root,
        "output": producer_root / "unused-report.json",
        "min_trials": 2,
        "require_readable_data": True,
        "require_reward_variance": True,
        "require_shared_records": True,
        "require_explicit_reward": True,
        "require_release_digests": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_two_atif_sources(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    run_a = tmp_path / "source-run-a"
    run_b = tmp_path / "source-run-b"
    _write_real_stage_g_run(run_a, scores=(-10.0, -12.0))
    _write_real_stage_g_run(run_b, scores=(-8.0, -14.0))

    atif_a = tmp_path / "atif-a"
    atif_b = tmp_path / "atif-b"
    export_atif_from_run(run_a, atif_a, min_exported=2)
    export_atif_from_run(run_b, atif_b, min_exported=2)
    return run_a, atif_a, run_b, atif_b


def _write_mixed_atif_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    run_a, atif_a, run_b, atif_b = _write_two_atif_sources(tmp_path)

    mixed = tmp_path / "mixed-atif"
    manifest = assemble_mixed_atif_sources(
        [
            MixedATIFSource("source-a", atif_a, run_a),
            MixedATIFSource("source-b", atif_b, run_b),
        ],
        mixed,
        min_exported=4,
    )
    for trial_name, artifact in manifest["trial_artifacts"].items():
        assert artifact["trial_dir"] == str((mixed / trial_name).resolve())
        source_id = artifact["source_run_id"]
        assert artifact["source_run_dir"] == manifest["source_run_dirs"][source_id]
        assert artifact["source_atif_dir"] == manifest["source_atif_dirs"][source_id]
    return mixed, manifest["trial_source_run_ids"]


def test_validate_atif_contracts_passes_real_readable_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_real_stage_g_run(run_dir)
    out_dir = tmp_path / "atif"
    export_atif_from_run(run_dir, out_dir, min_exported=2)

    report_path = tmp_path / "atif-contract-report.json"
    exit_code = main(
        [
            "--atif-dir",
            str(out_dir),
            "--producer-root",
            str(tmp_path),
            "--output",
            str(report_path),
            "--min-trials",
            "2",
            "--require-readable-data",
            "--require-reward-variance",
            "--require-shared-records",
            "--require-explicit-reward",
            "--require-release-digests",
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["count_trials"] == 2
    assert report["reward_variance"] > 0.0
    assert report["failures"] == []
    assert sorted(trial["data_rows"] for trial in report["trials"]) == [3, 3]
    assert all(trial["manifest_artifact_present"] for trial in report["trials"])


def test_validate_atif_contracts_rejects_corrupt_release_digest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_real_stage_g_run(run_dir)
    out_dir = tmp_path / "atif"
    export_atif_from_run(run_dir, out_dir, min_exported=2)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_trial = next(iter(manifest["trial_artifacts"]))
    manifest["trial_artifacts"][first_trial]["data_ref_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_atif_dir(_args(out_dir, producer_root=tmp_path, require_release_digests=True))

    assert report["status"] == "failed"
    assert any("data_ref_sha256" in failure["failure"] for failure in report["failures"])


def test_validate_atif_contracts_rejects_unreadable_parquet(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_real_stage_g_run(run_dir)
    out_dir = tmp_path / "atif"
    export_atif_from_run(run_dir, out_dir, min_exported=2)
    (run_dir / "stage_C_data" / "data" / "env-a.parquet").write_bytes(b"PAR1")

    report = validate_atif_dir(_args(out_dir, producer_root=tmp_path))

    assert report["status"] == "failed"
    assert any("data_ref validation failed" in failure["failure"] for failure in report["failures"])


def test_validate_atif_contracts_rejects_sidecar_reward_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_real_stage_g_run(run_dir)
    out_dir = tmp_path / "atif"
    export_atif_from_run(run_dir, out_dir, min_exported=2)
    reward_path = next(out_dir.glob("*/reward.json"))
    reward_payload = json.loads(reward_path.read_text(encoding="utf-8"))
    reward_payload["training_log_reward"] = reward_payload["training_log_reward"] + 1.0
    reward_path.write_text(json.dumps(reward_payload), encoding="utf-8")

    report = validate_atif_dir(_args(out_dir, producer_root=tmp_path))

    assert report["status"] == "failed"
    assert any(
        "sidecar reward.json differs from embedded ATIF reward_artifact" == failure["failure"]
        for failure in report["failures"]
    )


def test_validate_atif_contracts_requires_reward_variance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_real_stage_g_run(run_dir, scores=(-10.0, -10.0))
    out_dir = tmp_path / "atif"
    export_atif_from_run(run_dir, out_dir, min_exported=2)

    report = validate_atif_dir(_args(out_dir, producer_root=tmp_path))

    assert report["status"] == "failed"
    assert any(
        "training_log_reward variance is zero or unavailable" == failure["failure"]
        for failure in report["failures"]
    )


def test_validate_atif_contracts_passes_explicit_mixed_source_map(tmp_path: Path) -> None:
    mixed, trial_source_run_ids = _write_mixed_atif_fixture(tmp_path)

    report = validate_atif_dir(
        _args(
            mixed,
            producer_root=tmp_path,
            min_trials=4,
            require_release_digests=True,
        ),
    )

    assert report["status"] == "passed"
    assert report["source_mode"] == "mixed"
    assert report["run_dir"] is None
    assert set(report["source_run_dirs"]) == {"source-a", "source-b"}
    assert report["trial_source_run_ids"] == trial_source_run_ids
    assert report["count_trials"] == 4
    assert report["failures"] == []
    assert sorted(trial["source_run_id"] for trial in report["trials"]) == [
        "source-a",
        "source-a",
        "source-b",
        "source-b",
    ]
    assert sorted(trial["data_rows"] for trial in report["trials"]) == [3, 3, 3, 3]
    assert all(trial["manifest_artifact_present"] for trial in report["trials"])


def test_validate_atif_contracts_rejects_mixed_manifest_without_source_dirs(
    tmp_path: Path,
) -> None:
    mixed, _ = _write_mixed_atif_fixture(tmp_path)
    manifest_path = mixed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("source_run_dirs")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source_run_dirs"):
        validate_atif_dir(_args(mixed, producer_root=tmp_path, min_trials=4))


def test_validate_atif_contracts_rejects_mixed_manifest_with_unmapped_trial(
    tmp_path: Path,
) -> None:
    mixed, _ = _write_mixed_atif_fixture(tmp_path)
    manifest_path = mixed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trial_source_run_ids"].pop(next(iter(manifest["trial_source_run_ids"])))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="trial_source_run_ids length"):
        validate_atif_dir(_args(mixed, producer_root=tmp_path, min_trials=4))


def test_validate_atif_contracts_rejects_mixed_manifest_with_unknown_source_id(
    tmp_path: Path,
) -> None:
    mixed, _ = _write_mixed_atif_fixture(tmp_path)
    manifest_path = mixed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_trial = next(iter(manifest["trial_source_run_ids"]))
    manifest["trial_source_run_ids"][first_trial] = "missing-source"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown source_run_id"):
        validate_atif_dir(_args(mixed, producer_root=tmp_path, min_trials=4))


def test_assemble_mixed_atif_cli_writes_validator_ready_manifest(tmp_path: Path) -> None:
    run_a, atif_a, run_b, atif_b = _write_two_atif_sources(tmp_path)
    sources_json = tmp_path / "sources.json"
    sources_json.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_run_id": "source-a",
                        "atif_dir": atif_a.name,
                        "run_dir": run_a.name,
                    },
                    {
                        "source_run_id": "source-b",
                        "atif_dir": atif_b.name,
                        "run_dir": run_b.name,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    mixed = tmp_path / "mixed-from-cli"

    result = runner.invoke(
        app,
        [
            "assemble-mixed-atif",
            "--sources-json",
            str(sources_json),
            "--out",
            str(mixed),
            "--min-exported",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((mixed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "ppl_synth_data.mixed_atif_assembly.v1"
    assert manifest["digest_algorithm"] == "sha256"
    assert set(manifest["source_run_dirs"]) == {"source-a", "source-b"}
    assert manifest["split_counts"] == {"train": 4}
    assert len(manifest["trial_artifacts"]) == 4
    assert all("trial_dir" in row for row in manifest["trial_artifacts"].values())
    assert set(manifest["trial_dirs"]) == {
        row["trial_dir"] for row in manifest["trial_artifacts"].values()
    }
    report = validate_atif_dir(_args(mixed, producer_root=tmp_path, min_trials=4))
    assert report["status"] == "passed"
