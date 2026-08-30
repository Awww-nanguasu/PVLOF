"""PVLOF v1.7 penalized one-dimensional residual segmentation.

V1.7 preserves every v1.6 alert and adds a device-internal group branch. The
new branch partitions sorted residual ratios with a penalized SSE objective,
requires a sufficiently large higher-residual reference segment, and applies
the existing relative/absolute physical-effect gate. A homogeneous
whole-device reduction has no internal boundary and is therefore not alerted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["plant_id", "device_no", "string_no"]


@dataclass(frozen=True)
class PVLOFV17Config:
    """Configuration for segmentation, physical validation and memory."""

    version: str = "pvlof-v1.7-penalized-segmentation"
    segmentation_penalty: float = 0.01
    minimum_segment_strings: int = 2
    minimum_reference_strings: int = 5
    minimum_candidate_strings: int = 2
    minimum_relative_drop: float = 0.20
    minimum_absolute_drop: float = 0.50
    effect_gate_mode: str = "any"
    entry_consecutive: int = 3
    recovery_consecutive: int = 3
    expected_interval_minutes: int = 5

    def __post_init__(self) -> None:
        if self.segmentation_penalty <= 0:
            raise ValueError("segmentation_penalty must be positive")
        if self.minimum_segment_strings < 1:
            raise ValueError("minimum_segment_strings must be positive")
        if self.minimum_reference_strings < self.minimum_segment_strings:
            raise ValueError(
                "minimum_reference_strings must be >= minimum_segment_strings"
            )
        if self.minimum_candidate_strings < 1:
            raise ValueError("minimum_candidate_strings must be positive")
        if not 0 <= self.minimum_relative_drop < 1:
            raise ValueError("minimum_relative_drop must be in [0, 1)")
        if self.minimum_absolute_drop < 0:
            raise ValueError("minimum_absolute_drop must be non-negative")
        if self.effect_gate_mode not in {"all", "any"}:
            raise ValueError("effect_gate_mode must be 'all' or 'any'")
        if min(self.entry_consecutive, self.recovery_consecutive) < 1:
            raise ValueError("entry/recovery consecutive counts must be positive")
        if self.expected_interval_minutes < 1:
            raise ValueError("expected_interval_minutes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFV17Config":
        return cls(**dict(payload))


def save_config(config: PVLOFV17Config, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> PVLOFV17Config:
    return PVLOFV17Config.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _segment_sse(
    prefix: np.ndarray,
    prefix_sq: np.ndarray,
    start: int,
    end: int,
) -> float:
    count = end - start
    total = float(prefix[end] - prefix[start])
    total_sq = float(prefix_sq[end] - prefix_sq[start])
    return max(total_sq - total * total / count, 0.0)


def penalized_segments(
    values: np.ndarray,
    *,
    penalty: float,
    minimum_segment_size: int,
) -> list[tuple[int, int]]:
    """Return optimal half-open segments for finite values sorted ascending.

    The objective is within-segment SSE plus ``penalty`` for every change
    point. Dynamic programming is O(n^2), which is negligible for 13--24
    strings and supports more than one current staircase.
    """

    ordered = np.asarray(values, dtype=float)
    if ordered.ndim != 1 or not np.isfinite(ordered).all():
        raise ValueError("values must be a finite one-dimensional array")
    if len(ordered) < minimum_segment_size:
        return []
    if np.any(ordered[1:] < ordered[:-1]):
        raise ValueError("values must be sorted ascending")
    if penalty <= 0 or minimum_segment_size < 1:
        raise ValueError("penalty and minimum_segment_size must be positive")

    size = len(ordered)
    prefix = np.concatenate(([0.0], np.cumsum(ordered)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(ordered * ordered)))
    cost = np.full(size + 1, np.inf, dtype=float)
    previous = np.full(size + 1, -1, dtype=np.int32)
    cost[0] = -penalty  # The first segment has no change-point penalty.

    for end in range(minimum_segment_size, size + 1):
        for start in range(0, end - minimum_segment_size + 1):
            if start and not np.isfinite(cost[start]):
                continue
            candidate = (
                cost[start]
                + _segment_sse(prefix, prefix_sq, start, end)
                + penalty
            )
            if candidate < cost[end]:
                cost[end] = candidate
                previous[end] = start

    if previous[size] < 0:
        return []
    result: list[tuple[int, int]] = []
    end = size
    while end > 0:
        start = int(previous[end])
        result.append((start, end))
        end = start
    return list(reversed(result))


def _physical_effect(
    frame: pd.DataFrame,
    config: PVLOFV17Config,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    residual = pd.to_numeric(frame["residual_ratio"], errors="coerce")
    residual_median = pd.to_numeric(frame["residual_median"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_drop = 1.0 - residual / residual_median
    relative_drop = relative_drop.where(residual_median.gt(0))
    absolute_drop = (
        pd.to_numeric(frame["expected_current"], errors="coerce")
        - pd.to_numeric(frame["string_current"], errors="coerce")
    )

    relative_ok = relative_drop.ge(config.minimum_relative_drop)
    absolute_ok = absolute_drop.ge(config.minimum_absolute_drop)
    if config.effect_gate_mode == "all":
        relative_side = relative_ok if config.minimum_relative_drop > 0 else True
        absolute_side = absolute_ok if config.minimum_absolute_drop > 0 else True
        eligible = relative_side & absolute_side
    elif config.minimum_relative_drop <= 0 and config.minimum_absolute_drop <= 0:
        eligible = pd.Series(True, index=frame.index)
    else:
        relative_side = (
            relative_ok
            if config.minimum_relative_drop > 0
            else pd.Series(False, index=frame.index)
        )
        absolute_side = (
            absolute_ok
            if config.minimum_absolute_drop > 0
            else pd.Series(False, index=frame.index)
        )
        eligible = relative_side | absolute_side
    return relative_drop, absolute_drop, eligible.fillna(False)


def _add_segmentation(
    frame: pd.DataFrame,
    config: PVLOFV17Config,
) -> tuple[pd.DataFrame, pd.Series]:
    result = frame.copy()
    relative_drop, absolute_drop, physical_ok = _physical_effect(result, config)
    result["pvlof_v17_relative_drop"] = relative_drop.astype(np.float32)
    result["pvlof_v17_absolute_drop"] = absolute_drop.astype(np.float32)
    result["pvlof_v17_physical_effect_eligible"] = physical_ok.astype(np.int8)

    current = pd.to_numeric(result["string_current"], errors="coerce")
    residual = pd.to_numeric(result["residual_ratio"], errors="coerce")
    score = pd.to_numeric(result["pvlof_score"], errors="coerce")
    valid = (
        result["v2_eligible"].fillna(False).astype(bool)
        & result["response_known"].fillna(False).astype(bool)
        & current.gt(0)
        & residual.notna()
        & score.notna()
    )

    row_count = len(result)
    structural = np.zeros(row_count, dtype=np.int8)
    candidate = np.zeros(row_count, dtype=np.int8)
    detected = np.zeros(row_count, dtype=np.int8)
    no_contrast = np.zeros(row_count, dtype=np.int8)
    reference_reliable = np.zeros(row_count, dtype=np.int8)
    segment_index = np.zeros(row_count, dtype=np.int16)
    segment_count = np.zeros(row_count, dtype=np.int16)
    candidate_size = np.zeros(row_count, dtype=np.int16)
    reference_size = np.zeros(row_count, dtype=np.int16)
    reference_residual = np.full(row_count, np.nan, dtype=np.float32)
    objective = np.full(row_count, np.nan, dtype=np.float32)
    sse_gain = np.full(row_count, np.nan, dtype=np.float32)

    residual_values = residual.to_numpy(dtype=float)
    valid_values = valid.to_numpy(dtype=bool)
    physical_values = physical_ok.to_numpy(dtype=bool)
    minimum_total = config.minimum_reference_strings + config.minimum_candidate_strings

    for _, positions in result.groupby(
        ["plant_id", "device_no", "event_time"], observed=True, sort=False
    ).indices.items():
        group_indexes = np.asarray(positions, dtype=np.int64)
        usable = group_indexes[valid_values[group_indexes]]
        if len(usable) < minimum_total:
            continue
        order = np.argsort(residual_values[usable], kind="stable")
        ordered_indexes = usable[order]
        ordered_values = residual_values[ordered_indexes]
        segments = penalized_segments(
            ordered_values,
            penalty=config.segmentation_penalty,
            minimum_segment_size=config.minimum_segment_strings,
        )
        if not segments:
            continue

        prefix = np.concatenate(([0.0], np.cumsum(ordered_values)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(ordered_values * ordered_values)))
        segmented_sse = sum(
            _segment_sse(prefix, prefix_sq, start, end) for start, end in segments
        )
        total_sse = _segment_sse(prefix, prefix_sq, 0, len(ordered_values))
        group_objective = segmented_sse + config.segmentation_penalty * (len(segments) - 1)
        objective[group_indexes] = group_objective
        sse_gain[group_indexes] = total_sse - segmented_sse
        segment_count[group_indexes] = len(segments)
        for number, (start, end) in enumerate(segments, 1):
            segment_index[ordered_indexes[start:end]] = number

        if len(segments) == 1:
            no_contrast[group_indexes] = 1
            continue

        eligible_references = [
            number
            for number, (start, end) in enumerate(segments)
            if end - start >= config.minimum_reference_strings
        ]
        if not eligible_references:
            continue
        # Tiny high outlier segments are ignored; choose the highest segment
        # that still has enough members to be a credible healthy reference.
        reference_number = eligible_references[-1]
        reference_start, reference_end = segments[reference_number]
        if reference_number == 0:
            continue
        reference_indexes = ordered_indexes[reference_start:reference_end]
        lower_end = segments[reference_number - 1][1]
        lower_indexes = ordered_indexes[:lower_end]
        if len(lower_indexes) < config.minimum_candidate_strings:
            continue

        reference_reliable[group_indexes] = 1
        reference_size[group_indexes] = len(reference_indexes)
        reference_residual[group_indexes] = float(
            np.median(residual_values[reference_indexes])
        )
        structural[lower_indexes] = 1
        physical_candidates = lower_indexes[physical_values[lower_indexes]]
        candidate_size[group_indexes] = len(physical_candidates)
        if len(physical_candidates) < config.minimum_candidate_strings:
            continue
        candidate[physical_candidates] = 1
        detected[group_indexes] = 1

    result["pvlof_v17_segmentation_structural_member"] = structural
    result["pvlof_v17_segmentation_raw_candidate"] = candidate
    result["pvlof_v17_segmentation_detected"] = detected
    result["pvlof_v17_no_internal_contrast"] = no_contrast
    result["pvlof_v17_reference_reliable"] = reference_reliable
    result["pvlof_v17_segment_index"] = segment_index
    result["pvlof_v17_segment_count"] = segment_count
    result["pvlof_v17_segmentation_candidate_size"] = candidate_size
    result["pvlof_v17_segmentation_reference_size"] = reference_size
    result["pvlof_v17_segmentation_reference_residual"] = reference_residual
    result["pvlof_v17_segmentation_objective"] = objective
    result["pvlof_v17_segmentation_sse_gain"] = sse_gain
    return result, valid


def _memory_state(
    frame: pd.DataFrame,
    *,
    raw: pd.Series,
    valid: pd.Series,
    config: PVLOFV17Config,
) -> dict[str, np.ndarray]:
    row_count = len(frame)
    entry_values = np.zeros(row_count, dtype=np.int32)
    normal_values = np.zeros(row_count, dtype=np.int32)
    active_values = np.zeros(row_count, dtype=np.int8)
    alert_values = np.zeros(row_count, dtype=np.int8)
    clear_values = np.zeros(row_count, dtype=np.int8)
    reactivated_values = np.zeros(row_count, dtype=np.int8)
    raw_values = raw.to_numpy(dtype=bool)
    valid_values = valid.to_numpy(dtype=bool)
    time_values = frame["event_time"].to_numpy(dtype="datetime64[ns]")
    expected = np.timedelta64(config.expected_interval_minutes, "m")

    for _, positions in frame.groupby(KEYS, observed=True, sort=False).indices.items():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        raw_streak = 0
        normal_streak = 0
        memory_active = False

        for index in indexes:
            current_time = time_values[index]
            contiguous = (
                previous_time is not None and current_time - previous_time == expected
            )
            clear_code = 0
            if previous_time is not None and not contiguous:
                if memory_active:
                    clear_code = 2
                raw_streak = 0
                normal_streak = 0
                memory_active = False

            is_valid = bool(valid_values[index])
            is_raw = bool(raw_values[index]) and is_valid
            memory_alert = False
            reactivated = False
            if not is_valid:
                if memory_active:
                    clear_code = 3
                raw_streak = 0
                normal_streak = 0
                memory_active = False
            elif not memory_active:
                normal_streak = 0
                raw_streak = raw_streak + 1 if is_raw and contiguous else int(is_raw)
                if raw_streak >= config.entry_consecutive:
                    memory_active = True
                    memory_alert = True
            elif is_raw:
                reactivated = normal_streak > 0
                raw_streak = raw_streak + 1 if contiguous else 1
                normal_streak = 0
                memory_alert = True
            else:
                raw_streak = 0
                normal_streak += 1
                if normal_streak >= config.recovery_consecutive:
                    clear_code = 1
                    normal_streak = 0
                    memory_active = False

            entry_values[index] = raw_streak
            normal_values[index] = normal_streak
            active_values[index] = int(memory_active)
            alert_values[index] = int(memory_alert)
            clear_values[index] = clear_code
            reactivated_values[index] = int(reactivated and memory_alert)
            previous_time = current_time

    return {
        "entry_streak": entry_values,
        "normal_streak": normal_values,
        "memory_active": active_values,
        "memory_alert": alert_values,
        "memory_clear_code": clear_values,
        "memory_reactivated_alert": reactivated_values,
    }


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(0, index=frame.index, dtype="int32")
    return pd.to_numeric(frame[name], errors="coerce").fillna(0).astype("int32")


def apply_pvlof_v17(frame: pd.DataFrame, config: PVLOFV17Config) -> pd.DataFrame:
    """Add v1.7 columns to a fully memory-applied v1.6 points table."""

    required = {
        "plant_id",
        "device_no",
        "event_time",
        "string_no",
        "v2_eligible",
        "response_known",
        "string_current",
        "expected_current",
        "residual_ratio",
        "residual_median",
        "pvlof_score",
        "pvlof_v16_raw_anomaly",
        "pvlof_v16_alert",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PVLOF v1.7 input is missing columns: {missing}")

    result = frame.copy()
    result["plant_id"] = result["plant_id"].astype(str)
    result["device_no"] = result["device_no"].astype(str)
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    result["string_no"] = pd.to_numeric(result["string_no"], errors="raise").astype("Int64")
    result = result.sort_values([*KEYS, "event_time"]).reset_index(drop=True)
    result, valid = _add_segmentation(result, config)

    segmentation_raw = result["pvlof_v17_segmentation_raw_candidate"].astype(bool)
    state = _memory_state(result, raw=segmentation_raw, valid=valid, config=config)
    for name, values in state.items():
        result[f"pvlof_v17_segmentation_{name}"] = values
    result["pvlof_v17_segmentation_strict_alert"] = (
        segmentation_raw
        & result["pvlof_v17_segmentation_entry_streak"].ge(config.entry_consecutive)
    ).astype(np.int8)

    v16_raw = result["pvlof_v16_raw_anomaly"].fillna(False).astype(bool)
    v16_alert = result["pvlof_v16_alert"].fillna(False).astype(bool)
    v16_active = _numeric_column(result, "pvlof_v16_memory_active").astype(bool)
    v16_reactivated = _numeric_column(
        result, "pvlof_v16_memory_reactivated_alert"
    ).astype(bool)
    v16_clear = _numeric_column(result, "pvlof_v16_memory_clear_code")
    v16_entry = _numeric_column(result, "pvlof_v16_entry_streak")
    v16_normal = _numeric_column(result, "pvlof_v16_normal_streak")

    segmentation_alert = result["pvlof_v17_segmentation_memory_alert"].astype(bool)
    segmentation_active = result["pvlof_v17_segmentation_memory_active"].astype(bool)
    segmentation_reactivated = result[
        "pvlof_v17_segmentation_memory_reactivated_alert"
    ].astype(bool)
    segmentation_clear = result[
        "pvlof_v17_segmentation_memory_clear_code"
    ].astype("int32")
    aggregate_active = v16_active | segmentation_active
    aggregate_clear = pd.concat([v16_clear, segmentation_clear], axis=1).max(axis=1)
    aggregate_clear = aggregate_clear.where(~aggregate_active, 0).astype(np.int8)

    result["pvlof_v17_raw_anomaly"] = (v16_raw | segmentation_raw).astype(np.int8)
    result["pvlof_v17_valid_point"] = valid.astype(np.int8)
    result["pvlof_v17_entry_streak"] = pd.concat(
        [v16_entry, result["pvlof_v17_segmentation_entry_streak"]], axis=1
    ).max(axis=1).astype(np.int32)
    result["pvlof_v17_normal_streak"] = pd.concat(
        [v16_normal, result["pvlof_v17_segmentation_normal_streak"]], axis=1
    ).max(axis=1).astype(np.int32)
    result["pvlof_v17_memory_active"] = aggregate_active.astype(np.int8)
    result["pvlof_v17_memory_clear_code"] = aggregate_clear
    result["pvlof_v17_memory_reactivated_alert"] = (
        v16_reactivated | segmentation_reactivated
    ).astype(np.int8)
    result["pvlof_v17_segmentation_alert"] = segmentation_alert.astype(np.int8)
    result["pvlof_v17_alert"] = (v16_alert | segmentation_alert).astype(np.int8)
    result["pvlof_v17_added_alert"] = (
        result["pvlof_v17_alert"].astype(bool) & ~v16_alert
    ).astype(np.int8)
    return result.sort_values(
        ["event_time", "plant_id", "device_no", "string_no"]
    ).reset_index(drop=True)
