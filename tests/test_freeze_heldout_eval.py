"""Tests for the held-out eval-set freezing helper.

The synthetic tests cover the exclusion keying and the refusal paths. The tests marked
``real_boundary`` run the tool against the real candidate pool in the sealed corpus under
``artifacts/runs/``, the real committed split directory, and the real trajectory corpus next
door; they skip when those are not on the machine, because the corpus is not committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import freeze_heldout_eval as fhe  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_SEALED_RUN = _REPO / "artifacts" / "runs" / "evaluation-reference"
_COMMITTED_SPLIT_DIR = _REPO / "data" / "pymc_synthesis"
_TRAJECTORY_DIR = _REPO.parent / "synthstats" / "data" / "atif_trajectories"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sealed_candidates() -> list[dict[str, Any]]:
    """Every verifier-validated row of the sealed corpus as a candidate eval row.

    Joins the validated completions to their visible/hidden parquet pair and
    inlines both under ``eval_data``, which is the shape the freezing tool
    demands and the shape the holdout scorer later reads.
    """
    validated_path = _SEALED_RUN / "stage_F_validation" / "validated.jsonl"
    refs_path = _SEALED_RUN / "stage_C_data" / "data_refs.jsonl"
    if not validated_path.exists() or not refs_path.exists():
        pytest.skip(f"sealed corpus not on this machine: {_SEALED_RUN}")
    pd = pytest.importorskip("pandas")

    refs = {row["env_id"]: row for row in _read_jsonl(refs_path)}
    candidates: list[dict[str, Any]] = []
    for row in _read_jsonl(validated_path):
        program = row.get("parsed_program")
        if not isinstance(program, str) or not program.strip():
            continue
        ref = refs[row["env_id"]]
        candidates.append(
            {
                "task_id": row["id"],
                "prompt": row["prompt_text"],
                "completion": row["raw_completion"],
                "metadata": {"model_type": row["env_id"].rsplit("_", 1)[0]},
                "complexity": "unknown",
                "eval_data": {
                    "visible": {
                        k: v.tolist()
                        for k, v in pd.read_parquet(_SEALED_RUN / ref["data_ref"]).items()
                    },
                    "hidden": {
                        k: v.tolist()
                        for k, v in pd.read_parquet(_SEALED_RUN / ref["hidden_data_ref"]).items()
                    },
                },
            }
        )
    return candidates


def _require_real_inputs() -> None:
    for path in (_COMMITTED_SPLIT_DIR, _TRAJECTORY_DIR):
        if not path.exists():
            pytest.skip(f"real exclusion input not on this machine: {path}")


def _eval_row(task_id: str, prompt: str) -> dict:
    return {
        "task_id": task_id,
        "prompt": prompt,
        "completion": (
            "<think>x</think>\n<submit_program>\ndef model(data):\n    ...\n</submit_program>"
        ),
        "metadata": {"model_type": "inferential", "complexity": "simple"},
        "complexity": "simple",
        "eval_data": {"visible": {"y": [1.0]}, "hidden": {"y": [2.0]}},
    }


def _exclusion(prompts: list[str], source: str) -> fhe.ExclusionSet:
    exact, norm = fhe._prompt_keys(prompts)
    return fhe.ExclusionSet(frozenset(exact), frozenset(norm), source)


def test_filter_drops_exact_train_overlap() -> None:
    candidates = [_eval_row("a", "shared prompt"), _eval_row("b", "unique prompt")]
    exclusions = [_exclusion(["shared prompt"], "pymc_synthesis")]
    kept, report = fhe.filter_candidates(candidates=candidates, exclusions=exclusions)
    assert [r["task_id"] for r in kept] == ["b"]
    assert report["pymc_synthesis"] == 1


def test_filter_drops_whitespace_normalized_overlap() -> None:
    candidates = [_eval_row("a", "the   same\nprompt")]
    exclusions = [_exclusion(["the same prompt"], "atif_rescore")]
    kept, report = fhe.filter_candidates(candidates=candidates, exclusions=exclusions)
    assert kept == []
    assert report["atif_rescore"] == 1


def test_filter_drops_intra_candidate_duplicates() -> None:
    candidates = [_eval_row("a", "dup"), _eval_row("b", "dup")]
    kept, report = fhe.filter_candidates(candidates=candidates, exclusions=[])
    assert len(kept) == 1
    assert report["intra_candidate_duplicates"] == 1


def test_assert_zero_overlap_raises_on_contamination() -> None:
    eval_rows = [_eval_row("a", "leak")]
    exclusions = [_exclusion(["leak"], "pymc_synthesis")]
    with pytest.raises(RuntimeError, match="contamination"):
        fhe.assert_zero_overlap(eval_rows=eval_rows, exclusions=exclusions)


def test_validate_eval_row_requires_eval_data() -> None:
    row = {"task_id": "a", "prompt": "p", "completion": "c"}
    with pytest.raises(ValueError, match="eval_data"):
        fhe._validate_eval_row(row, index=0)


def test_atif_prompt_extracted_from_steps() -> None:
    doc = {
        "steps": [
            {"source": "system", "message": "sys"},
            {"source": "user", "message": "the task"},
        ]
    }
    assert fhe._atif_prompt_from_json(doc) == "the task"


def test_load_atif_prompts_empty_dir_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no ATIF prompts"):
        fhe.load_atif_prompts(tmp_path)


def test_run_dry_run_does_not_write(tmp_path: Path) -> None:
    candidates_path = tmp_path / "cands.jsonl"
    candidates_path.write_text(json.dumps(_eval_row("a", "p")) + "\n", encoding="utf-8")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    for name in ("train", "val", "test"):
        (split_dir / f"{name}.jsonl").write_text(
            json.dumps({"prompt": f"train-{name}"}) + "\n", encoding="utf-8"
        )
    atif_path = tmp_path / "atif.jsonl"
    atif_path.write_text(json.dumps({"prompt": "atif-p"}) + "\n", encoding="utf-8")
    out_path = tmp_path / "eval" / "frozen.jsonl"

    result = fhe.run(
        candidates_path=candidates_path,
        split_dir=split_dir,
        atif_path=atif_path,
        out_path=out_path,
        target_size=None,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["frozen"] is False
    assert result["kept_count"] == 1
    assert not out_path.exists()


def test_run_real_freeze_writes_clean_set_with_sha(tmp_path: Path) -> None:
    candidates_path = tmp_path / "cands.jsonl"
    rows = [_eval_row("a", "keep-1"), _eval_row("b", "train-train"), _eval_row("c", "keep-2")]
    candidates_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "train.jsonl").write_text(
        json.dumps({"prompt": "train-train"}) + "\n",
        encoding="utf-8",
    )
    (split_dir / "val.jsonl").write_text("", encoding="utf-8")
    (split_dir / "test.jsonl").write_text("", encoding="utf-8")
    atif_path = tmp_path / "atif.jsonl"
    atif_path.write_text(json.dumps({"prompt": "atif-only"}) + "\n", encoding="utf-8")
    out_path = tmp_path / "frozen.jsonl"

    result = fhe.run(
        candidates_path=candidates_path,
        split_dir=split_dir,
        atif_path=atif_path,
        out_path=out_path,
        target_size=None,
        dry_run=False,
    )
    assert result["frozen"] is True
    assert result["kept_count"] == 2
    assert result["dropped"]["pymc_synthesis"] == 1
    assert len(result["sha256"]) == 64
    frozen = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert {r["prompt"] for r in frozen} == {"keep-1", "keep-2"}


def test_run_refuses_to_write_an_empty_eval_set(tmp_path: Path) -> None:
    """Every candidate excluded is a refusal, not an empty file."""
    candidates_path = tmp_path / "cands.jsonl"
    candidates_path.write_text(json.dumps(_eval_row("a", "leak")) + "\n", encoding="utf-8")
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "train.jsonl").write_text(json.dumps({"prompt": "leak"}) + "\n", encoding="utf-8")
    for name in ("val", "test"):
        (split_dir / f"{name}.jsonl").write_text("", encoding="utf-8")
    atif_path = tmp_path / "atif.jsonl"
    atif_path.write_text(json.dumps({"prompt": "atif-only"}) + "\n", encoding="utf-8")
    out_path = tmp_path / "frozen.jsonl"

    with pytest.raises(RuntimeError, match="no clean eval rows"):
        fhe.run(
            candidates_path=candidates_path,
            split_dir=split_dir,
            atif_path=atif_path,
            out_path=out_path,
            target_size=None,
            dry_run=False,
        )
    assert not out_path.exists()


# --------------------------------------------------------------------------- #
# The real candidate pool
# --------------------------------------------------------------------------- #
@pytest.mark.real_boundary
def test_sealed_corpus_pool_carries_visible_and_hidden_data_for_every_row() -> None:
    """Checks the pool the upcoming evaluation would be frozen from.

    Counts are asserted against the corpus itself rather than a remembered
    number: as many candidates as validated rows, every one with both slices of
    real sample data, and no two sharing a prompt.
    """
    candidates = _sealed_candidates()
    validated = _read_jsonl(_SEALED_RUN / "stage_F_validation" / "validated.jsonl")
    prompts = _read_jsonl(_SEALED_RUN / "stage_D_prompts" / "prompts.jsonl")

    assert len(candidates) == len(validated)
    assert len(candidates) < len(prompts), "gold programs are a subset of the prompt pool"
    for index, row in enumerate(candidates):
        fhe._validate_eval_row(row, index=index)
        assert row["eval_data"]["visible"], f"{row['task_id']} has no visible data"
        assert row["eval_data"]["hidden"], f"{row['task_id']} has no hidden data"
    assert len({fhe.normalize_prompt(r["prompt"]) for r in candidates}) == len(candidates)


@pytest.mark.real_boundary
def test_sealing_accepts_the_real_pool_without_writing_anything(tmp_path: Path) -> None:
    """The read-only pass over the real pool, real split dir, real trajectories."""
    _require_real_inputs()
    candidates = _sealed_candidates()
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    out_path = tmp_path / "would_be_frozen.jsonl"

    result = fhe.run(
        candidates_path=candidates_path,
        split_dir=_COMMITTED_SPLIT_DIR,
        atif_path=_TRAJECTORY_DIR,
        out_path=out_path,
        target_size=None,
        dry_run=True,
    )

    assert result["kept_count"] == len(candidates)
    assert result["dropped"] == {"intra_candidate_duplicates": 0}
    assert result["exclusion_sizes"]["pymc_synthesis"] > 0
    assert result["exclusion_sizes"]["atif_rescore"] > 0
    assert result["frozen"] is False
    assert result["sha256"] is None
    assert not out_path.exists()


@pytest.mark.real_boundary
def test_sealing_rejects_a_real_pool_row_that_carries_a_real_train_prompt(
    tmp_path: Path,
) -> None:
    """A planted training prompt is excluded from the held-out rows."""
    _require_real_inputs()
    candidates = _sealed_candidates()
    train_rows = _read_jsonl(_COMMITTED_SPLIT_DIR / "train.jsonl")
    if not train_rows:
        pytest.skip(f"no train rows to plant from: {_COMMITTED_SPLIT_DIR / 'train.jsonl'}")

    planted = [dict(row) for row in candidates]
    planted[0]["prompt"] = train_rows[0]["prompt"]
    candidates_path = tmp_path / "planted.jsonl"
    candidates_path.write_text("".join(json.dumps(row) + "\n" for row in planted), encoding="utf-8")

    result = fhe.run(
        candidates_path=candidates_path,
        split_dir=_COMMITTED_SPLIT_DIR,
        atif_path=_TRAJECTORY_DIR,
        out_path=tmp_path / "unused.jsonl",
        target_size=None,
        dry_run=True,
    )
    assert result["dropped"]["pymc_synthesis"] == 1
    assert result["kept_count"] == len(candidates) - 1


@pytest.mark.real_boundary
def test_real_freeze_writes_every_clean_row_with_a_checksum(tmp_path: Path) -> None:
    """The write path, on the real pool, into a scratch directory."""
    _require_real_inputs()
    candidates = _sealed_candidates()
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    out_path = tmp_path / "eval" / "pymc_heldout.jsonl"

    result = fhe.run(
        candidates_path=candidates_path,
        split_dir=_COMMITTED_SPLIT_DIR,
        atif_path=_TRAJECTORY_DIR,
        out_path=out_path,
        target_size=None,
        dry_run=False,
    )

    assert result["frozen"] is True
    assert len(result["sha256"]) == 64
    frozen = _read_jsonl(out_path)
    assert len(frozen) == len(candidates)
    fhe.assert_zero_overlap(
        eval_rows=frozen,
        exclusions=fhe.build_exclusions(split_dir=_COMMITTED_SPLIT_DIR, atif_path=_TRAJECTORY_DIR),
    )
    assert all(row["eval_data"]["hidden"] for row in frozen)


@pytest.mark.real_boundary
def test_target_size_caps_the_frozen_pool(tmp_path: Path) -> None:
    _require_real_inputs()
    candidates = _sealed_candidates()
    if len(candidates) < 5:
        pytest.skip(f"real pool too small to cap: {len(candidates)} rows")
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )

    result = fhe.run(
        candidates_path=candidates_path,
        split_dir=_COMMITTED_SPLIT_DIR,
        atif_path=_TRAJECTORY_DIR,
        out_path=tmp_path / "capped.jsonl",
        target_size=5,
        dry_run=False,
    )
    assert result["kept_count"] == 5
    assert len(_read_jsonl(tmp_path / "capped.jsonl")) == 5
