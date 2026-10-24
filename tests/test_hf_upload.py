from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, load_dataset

from ppl_synth_data.dataset.features import pymc_sft_features, pymc_with_data_features
from ppl_synth_data.dataset.hf_upload import (
    build_datasetdict_from_jsonl_splits,
    jsonl_to_dataset_with_features,
    push_datasetdict_to_hub,
)


@pytest.mark.paid_lane
def test_real_hf_private_repo_roundtrip() -> None:
    """Real Hugging Face Hub boundary: push a tiny private DatasetDict, read it
    back, assert content equality, then delete the throwaway repo.
    """
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found (huggingface_hub.get_token() returned None). "
            "Run `hf auth login` or set HF_TOKEN before running the paid_lane."
        )
    api = HfApi(token=token)
    whoami = api.whoami()
    namespace = whoami["name"]
    repo_id = f"{namespace}/ppl-synth-data-roundtrip-{uuid.uuid4().hex[:12]}"

    ds = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"id": ["r1", "r2"], "text": ["hello world", "roundtrip check"]}
            )
        }
    )

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=False)
    try:
        push_datasetdict_to_hub(
            ds,
            repo_id=repo_id,
            token=token,
            config_name="default",
            private=True,
            commit_message="Verify dataset roundtrip",
        )

        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        assert repo_info.private is True

        roundtripped = load_dataset(repo_id, split="train", token=token)
        assert roundtripped["id"] == ["r1", "r2"]
        assert roundtripped["text"] == ["hello world", "roundtrip check"]
    finally:
        api.delete_repo(repo_id=repo_id, repo_type="dataset")
        assert not api.repo_exists(repo_id=repo_id, repo_type="dataset")

    print({"repo_id": repo_id, "namespace": namespace, "deleted": True})


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _sample_row(i: int, *, split: str) -> dict:
    return {
        "id": f"id-{i}",
        "prompt_id": f"p-{i}",
        "env_id": f"env-{i}",
        "split": split,
        "prompt": "prompt",
        "completion": "completion",
        "model_type": "predictive",
        "domain": "healthcare",
        "complexity": "medium",
        "key_concepts": ["hierarchical_model"],
        "has_observed_data": True,
        "pymc_version": "v5",
        "trace_id": f"id-{i}:validation",
        "provenance_json": "{}",
        "candidate_outputs_json": "[]",
    }


def test_build_datasetdict_from_jsonl_splits_applies_features(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_sample_row(1, split="train")])
    _write_jsonl(val, [_sample_row(2, split="validation")])
    _write_jsonl(test, [_sample_row(3, split="test")])

    features = pymc_sft_features()
    ds = build_datasetdict_from_jsonl_splits(train, val, test, features=features)

    assert set(ds.keys()) == {"train", "validation", "test"}
    assert ds["train"].features.type == features.type


@pytest.mark.narrow_unit_not_acceptance
def test_push_datasetdict_to_hub_forwards_schema_args(monkeypatch) -> None:
    ds = DatasetDict()
    captured: dict[str, object] = {}

    def _fake_push_to_hub(self, repo_id: str, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return {"repo_id": repo_id, "kwargs": kwargs}

    monkeypatch.setattr(DatasetDict, "push_to_hub", _fake_push_to_hub, raising=False)

    out = push_datasetdict_to_hub(
        ds,
        repo_id="org/repo",
        token="hf-token",
        config_name="pymc_sft",
        private=True,
        revision="main",
        commit_message="add split",
    )

    assert captured["repo_id"] == "org/repo"
    assert captured["token"] == "hf-token"
    assert captured["config_name"] == "pymc_sft"
    assert captured["private"] is True
    assert captured["revision"] == "main"
    assert captured["commit_message"] == "add split"
    assert out["repo_id"] == "org/repo"


def test_build_datasetdict_rejects_split_mismatch(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [_sample_row(1, split="train")])
    _write_jsonl(val, [_sample_row(2, split="train")])
    _write_jsonl(test, [_sample_row(3, split="test")])

    with pytest.raises(ValueError, match="Split mismatch"):
        build_datasetdict_from_jsonl_splits(
            train,
            val,
            test,
            features=pymc_sft_features(),
        )


def test_jsonl_to_dataset_with_features_reports_json_line_number(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"id":"ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="broken.jsonl:2"):
        jsonl_to_dataset_with_features(
            broken,
            pymc_sft_features(),
        )


def test_jsonl_to_dataset_with_features_reports_non_object_rows(tmp_path: Path) -> None:
    broken = tmp_path / "broken_non_object.jsonl"
    broken.write_text('{"id":"ok"}\n[]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Expected object JSON"):
        jsonl_to_dataset_with_features(
            broken,
            pymc_sft_features(),
        )


def test_jsonl_to_dataset_with_features_handles_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    ds = jsonl_to_dataset_with_features(
        empty,
        pymc_sft_features(),
        expected_split="train",
    )
    assert len(ds) == 0
    assert ds.features.type == pymc_sft_features().type


def test_jsonl_to_dataset_with_features_reports_feature_type_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad_seed.jsonl"
    row = {
        "id": "id-1",
        "prompt_id": "p-1",
        "env_id": "e-1",
        "split": "train",
        "prompt": "p",
        "completion": "c",
        "model_type": "predictive",
        "domain": "healthcare",
        "complexity": "medium",
        "key_concepts": ["k"],
        "has_observed_data": True,
        "pymc_version": "v5",
        "provenance_json": "{}",
        "candidate_outputs_json": "[]",
        "env_name": "e",
        "env_spec_json": "{}",
        "data_ref": "x",
        "seed": "not-an-int",
        "generator_version": "0.1.0",
        "trace_id": "id-1:validation",
        "trace_json": "{}",
        "validation_json": "{}",
    }
    _write_jsonl(bad, [row])

    with pytest.raises(ValueError, match="Feature validation failed"):
        jsonl_to_dataset_with_features(
            bad,
            pymc_with_data_features(),
            expected_split="train",
        )
