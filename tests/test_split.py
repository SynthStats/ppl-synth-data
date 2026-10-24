from __future__ import annotations

import pytest

from ppl_synth_data.dataset.split import deterministic_split


def test_deterministic_split_is_stable() -> None:
    key = "prompt-123"
    a = deterministic_split(key)
    b = deterministic_split(key)
    assert a == b
    assert a in {"train", "validation", "test"}


def test_deterministic_split_validates_ratios() -> None:
    with pytest.raises(ValueError):
        deterministic_split("k", train_ratio=-0.1, val_ratio=0.1)
    with pytest.raises(ValueError):
        deterministic_split("k", train_ratio=0.8, val_ratio=0.4)


def test_deterministic_split_full_train_never_returns_test() -> None:
    for i in range(500):
        split = deterministic_split(f"k{i}", train_ratio=1.0, val_ratio=0.0)
        assert split == "train"
