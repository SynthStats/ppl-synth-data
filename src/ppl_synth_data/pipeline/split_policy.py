from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec

_TIME_FAMILIES = frozenset(
    {
        "time_series_level_seasonality",
        "infection_process",
    }
)
_SPATIAL_FAMILIES = frozenset({"spatial_signal_source"})
_COORDINATE_PAIRS = (
    ("x", "y"),
    ("lon", "lat"),
    ("longitude", "latitude"),
    ("coord_x", "coord_y"),
)
_TIME_TOKEN_RE = re.compile(r"(?:^|_)(time|step|day|days|week|month|year|timestamp|date)(?:$|_)")


class SplitPolicy(StrEnum):
    AUTO = "auto"
    GROUP_HOLDOUT = "group_holdout"
    SPATIAL_REGION = "spatial_region"
    ORDERED_TAIL = "ordered_tail"
    RANDOM_ROW = "random_row"


class SplitPolicyIneligibleError(ValueError):
    pass


def validate_visible_ratio(visible_ratio: float) -> None:
    if not math.isfinite(visible_ratio) or not 0.0 < visible_ratio < 1.0:
        raise SplitPolicyIneligibleError(
            "visible_ratio must be finite and satisfy "
            f"0.0 < visible_ratio < 1.0; received {visible_ratio!r}"
        )


@dataclass(frozen=True)
class DataSplitResult:
    policy_name: SplitPolicy
    visible: pd.DataFrame
    hidden: pd.DataFrame


def split_visible_hidden(
    df: pd.DataFrame,
    spec: EnvSpec | FourFamilySpec,
    *,
    split_policy: SplitPolicy,
    seed: int,
    visible_ratio: float,
) -> DataSplitResult:
    split_policy = SplitPolicy(split_policy)
    if isinstance(spec, FourFamilySpec):
        # This contract declares neither groups nor spatial or temporal topology.
        if split_policy not in (SplitPolicy.AUTO, SplitPolicy.RANDOM_ROW):
            raise SplitPolicyIneligibleError(
                f"{split_policy.value} split policy is ineligible: "
                "the four-family contract declares no grouping or ordering semantics"
            )
        split_policy = SplitPolicy.RANDOM_ROW
    n_total = len(df)
    n_visible = min(max(1, int(n_total * visible_ratio)), n_total)
    n_hidden = n_total - n_visible

    rng = np.random.default_rng(seed)

    if split_policy is SplitPolicy.GROUP_HOLDOUT:
        assert isinstance(spec, EnvSpec)
        result = _maybe_group_holdout(df, spec, rng=rng, n_hidden=n_hidden)
        if result is None:
            raise SplitPolicyIneligibleError(
                "group_holdout split policy is ineligible for this environment"
            )
        return result

    if split_policy is SplitPolicy.SPATIAL_REGION:
        assert isinstance(spec, EnvSpec)
        result = _maybe_spatial_region_holdout(df, spec, rng=rng, n_hidden=n_hidden)
        if result is None:
            raise SplitPolicyIneligibleError(
                "spatial_region split policy is ineligible for this environment"
            )
        return result

    if split_policy is SplitPolicy.ORDERED_TAIL:
        assert isinstance(spec, EnvSpec)
        result = _maybe_ordered_tail_holdout(df, spec, rng=rng, n_hidden=n_hidden)
        if result is None:
            raise SplitPolicyIneligibleError(
                "ordered_tail split policy is ineligible for this environment"
            )
        return result

    if split_policy is SplitPolicy.RANDOM_ROW:
        hidden_positions = sorted(rng.permutation(n_total)[:n_hidden].tolist())
        return _materialize_split(
            df,
            hidden_positions,
            policy_name=SplitPolicy.RANDOM_ROW,
        )

    assert isinstance(spec, EnvSpec)
    for resolver in (
        _maybe_group_holdout,
        _maybe_spatial_region_holdout,
        _maybe_ordered_tail_holdout,
    ):
        result = resolver(df, spec, rng=rng, n_hidden=n_hidden)
        if result is not None:
            return result

    hidden_positions = sorted(rng.permutation(n_total)[:n_hidden].tolist())
    return _materialize_split(
        df,
        hidden_positions,
        policy_name=SplitPolicy.RANDOM_ROW,
    )


def _maybe_group_holdout(
    df: pd.DataFrame,
    spec: EnvSpec,
    *,
    rng: np.random.Generator,
    n_hidden: int,
) -> DataSplitResult | None:
    group_columns = [column for column in spec.data_plan.grouping if column in df.columns]
    if not group_columns:
        return None

    groups: dict[tuple[object, ...], list[int]] = {}
    for position in range(len(df)):
        key = tuple(df.iloc[position][group_columns].tolist())
        groups.setdefault(key, []).append(position)
    if len(groups) < 2:
        return None

    group_keys = list(groups)
    shuffled_keys = [group_keys[int(i)] for i in rng.permutation(len(group_keys))]

    best_hidden_positions: list[int] | None = None
    best_gap: tuple[int, int] | None = None
    for prefix_size in range(1, len(shuffled_keys)):
        candidate = [position for key in shuffled_keys[:prefix_size] for position in groups[key]]
        hidden_count = len(candidate)
        visible_count = len(df) - hidden_count
        if hidden_count == 0 or visible_count == 0:
            continue
        gap = (abs(hidden_count - n_hidden), hidden_count)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_hidden_positions = candidate

    if best_hidden_positions is None:
        return None
    return _materialize_split(
        df,
        sorted(best_hidden_positions),
        policy_name=SplitPolicy.GROUP_HOLDOUT,
    )


def _maybe_spatial_region_holdout(
    df: pd.DataFrame,
    spec: EnvSpec,
    *,
    rng: np.random.Generator,
    n_hidden: int,
) -> DataSplitResult | None:
    coordinate_pair = _find_coordinate_pair(df, spec)
    if coordinate_pair is None:
        return None
    if spec.generative_family not in _SPATIAL_FAMILIES and coordinate_pair is None:
        return None

    coords = df.loc[:, list(coordinate_pair)].apply(pd.to_numeric, errors="coerce")
    if coords.isna().any().any():
        return None
    if len(coords.drop_duplicates()) < 2:
        return None

    anchor_position = int(rng.integers(len(df)))
    anchor = coords.iloc[anchor_position].to_numpy(dtype=float)
    deltas = coords.to_numpy(dtype=float) - anchor
    distances = np.sum(deltas * deltas, axis=1)
    hidden_positions = np.argsort(distances, kind="stable")[:n_hidden].tolist()
    return _materialize_split(
        df,
        sorted(hidden_positions),
        policy_name=SplitPolicy.SPATIAL_REGION,
    )


def _maybe_ordered_tail_holdout(
    df: pd.DataFrame,
    spec: EnvSpec,
    *,
    rng: np.random.Generator,
    n_hidden: int,
) -> DataSplitResult | None:
    del rng  # policy is deterministic once the order column is known
    order_columns = _find_temporal_order_columns(df, spec)
    if not order_columns:
        return None

    order_frame = pd.DataFrame({"__rowpos__": np.arange(len(df), dtype=int)})
    for column in order_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or values.nunique(dropna=True) < 2:
            return None
        order_frame[column] = values.to_numpy()

    sorted_positions = order_frame.sort_values(
        by=[*order_columns, "__rowpos__"],
        kind="mergesort",
    )["__rowpos__"].tolist()
    hidden_positions = sorted_positions[-n_hidden:]
    return _materialize_split(
        df,
        sorted(hidden_positions),
        policy_name=SplitPolicy.ORDERED_TAIL,
    )


def _materialize_split(
    df: pd.DataFrame,
    hidden_positions: list[int],
    *,
    policy_name: SplitPolicy,
) -> DataSplitResult:
    hidden_set = set(hidden_positions)
    visible_positions = [position for position in range(len(df)) if position not in hidden_set]
    hidden_count = len(df) - len(visible_positions)
    if not visible_positions or hidden_count == 0:
        raise SplitPolicyIneligibleError(
            "split requires non-empty visible and hidden sides; "
            f"received total={len(df)}, visible={len(visible_positions)}, hidden={hidden_count}"
        )
    return DataSplitResult(
        policy_name=policy_name,
        visible=df.iloc[visible_positions].copy(),
        hidden=df.iloc[sorted(hidden_positions)].copy(),
    )


def _find_coordinate_pair(
    df: pd.DataFrame,
    spec: EnvSpec,
) -> tuple[str, str] | None:
    available_columns = {str(column).lower(): str(column) for column in df.columns}

    prioritized_pairs: list[tuple[str, str]] = []
    if spec.generative_family in _SPATIAL_FAMILIES:
        prioritized_pairs.extend(_COORDINATE_PAIRS)
    for left, right in _COORDINATE_PAIRS:
        if left in available_columns and right in available_columns:
            prioritized_pairs.append((left, right))

    seen: set[tuple[str, str]] = set()
    for left, right in prioritized_pairs:
        if (left, right) in seen:
            continue
        seen.add((left, right))
        left_name = available_columns.get(left)
        right_name = available_columns.get(right)
        if left_name is not None and right_name is not None:
            return left_name, right_name
    return None


def _find_temporal_order_columns(
    df: pd.DataFrame,
    spec: EnvSpec,
) -> list[str]:
    ordered_candidates: list[str] = []
    seen: set[str] = set()
    for column in (
        *spec.data_plan.covariates,
        *spec.io_schema.feature_names,
        *df.columns,
    ):
        column_name = str(column)
        if column_name not in df.columns or column_name in seen:
            continue
        if _looks_temporal(column_name):
            seen.add(column_name)
            ordered_candidates.append(column_name)

    if ordered_candidates:
        return ordered_candidates
    if spec.generative_family in _TIME_FAMILIES:
        return [column for column in df.columns if _looks_temporal(str(column))]
    return []


def _looks_temporal(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return bool(_TIME_TOKEN_RE.search(normalized))
