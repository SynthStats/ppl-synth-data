from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features


def jsonl_to_dataset(path: Path) -> Dataset:
    return Dataset.from_json(str(path))


def _empty_dataset(features: Features) -> Dataset:
    return Dataset.from_dict({name: [] for name in features.keys()}, features=features)


def _iter_jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected object JSON in {path}:{line_no}, got {type(payload).__name__}"
                )
            yield line_no, payload


def jsonl_to_dataset_with_features(
    path: Path,
    features: Features,
    *,
    expected_split: str | None = None,
) -> Dataset:
    n_rows = 0
    for line_no, row in _iter_jsonl_rows(path):
        n_rows += 1
        if expected_split is not None:
            split_val = row.get("split")
            if split_val != expected_split:
                raise ValueError(
                    f"Split mismatch in {path}:{line_no}: "
                    f"expected split={expected_split!r}, found {split_val!r}"
                )
        try:
            features.encode_example(row)
        except Exception as exc:
            raise ValueError(f"Feature validation failed in {path}:{line_no}: {exc}") from exc

    if n_rows == 0:
        return _empty_dataset(features)
    try:
        return Dataset.from_json(str(path), features=features)
    except Exception as exc:
        raise ValueError(f"Failed to build dataset from {path}: {exc}") from exc


def build_datasetdict_from_jsonl_splits(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    *,
    features: Features,
) -> DatasetDict:
    return DatasetDict(
        {
            "train": jsonl_to_dataset_with_features(
                train_path,
                features,
                expected_split="train",
            ),
            "validation": jsonl_to_dataset_with_features(
                validation_path,
                features,
                expected_split="validation",
            ),
            "test": jsonl_to_dataset_with_features(
                test_path,
                features,
                expected_split="test",
            ),
        }
    )


def push_datasetdict_to_hub(
    dataset_dict: DatasetDict,
    *,
    repo_id: str,
    token: str,
    config_name: str,
    private: bool = True,
    revision: str | None = None,
    commit_message: str | None = None,
) -> Any:
    # thin wrapper for package-level API discoverability: callers import
    # from dataset.hf_upload rather than reaching into DatasetDict methods
    return dataset_dict.push_to_hub(
        repo_id=repo_id,
        token=token,
        config_name=config_name,
        private=private,
        revision=revision,
        commit_message=commit_message,
    )
