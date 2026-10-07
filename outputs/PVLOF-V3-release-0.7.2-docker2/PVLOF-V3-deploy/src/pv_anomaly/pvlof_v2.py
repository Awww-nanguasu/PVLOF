"""Virtual-irradiance conditioned PVLOF (PVLOF-V2).

This module deliberately keeps the first-version PVLOF untouched.  PVLOF-V2
uses other inverters at the same plant as a leave-one-inverter-out reference:
the reference is a dimensionless virtual irradiance index, and LOF is applied
to string current residuals rather than raw current values.  A second,
collective-low branch catches a group of strings that are all low together.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof import _lof_1d


CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")


@dataclass(frozen=True)
class PVLOFV2Calibration:
    """Serializable V2 calibration and robust response estimates."""

    version: str = "pvlof-v2"
    n_neighbors: int = 5
    lof_quantile: float = 0.995
    lof_threshold: float = 2.0
    residual_quantile: float = 0.01
    residual_threshold: float = 0.70
    collective_gap_quantile: float = 0.995
    collective_gap_threshold: float = 0.15
    minimum_collective_gap: float = 0.10
    minimum_collective_strings: int = 2
    maximum_collective_fraction: float = 0.60
    collective_overlap_threshold: float = 0.50
    minimum_peer_devices: int = 5
    minimum_strings: int = 4
    minimum_consecutive: int = 2
    expected_interval_minutes: int = 5
    zero_current_threshold: float = 0.0
    minimum_virtual_irradiance: float = 0.05
    minimum_isolated_relative_drop: float = 0.0
    minimum_isolated_absolute_drop: float = 0.0
    isolated_effect_gate_mode: str = "all"
    distance_floor: float = 0.01
    device_scales: dict[str, float] = field(default_factory=dict)
    string_responses: dict[str, float] = field(default_factory=dict)
    configured_strings: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFV2Calibration":
        return cls(**dict(payload))


def _string_columns(frame: pd.DataFrame) -> list[tuple[int, str]]:
    columns: list[tuple[int, str]] = []
    for column in frame.columns:
        match = CURRENT_PATTERN.match(column)
        if match:
            columns.append((int(match.group(1)), column))
    columns.sort()
    if not columns:
        raise ValueError("No string_current_XX columns were found")
    return columns


def _key(plant: Any, device: Any) -> str:
    return f"{plant}|{device}"


def _string_key(plant: Any, device: Any, string_no: int) -> str:
    return f"{plant}|{device}|{string_no:02d}"


def _normalise_source(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    required = {"event_time", "device_no"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing PVLOF-V2 source columns: {missing}")
    source = frame.copy()
    source["event_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    source["device_no"] = source["device_no"].astype(str)
    if "plant_id" not in source.columns:
        source["plant_id"] = "unknown"
    source["plant_id"] = source["plant_id"].astype(str)
    source = source.drop_duplicates(["plant_id", "device_no", "event_time"], keep="last")
    source = source.sort_values(["plant_id", "event_time", "device_no"]).reset_index(drop=True)
    return source, _string_columns(source)


def _configured_mask(
    source: pd.DataFrame,
    strings: list[tuple[int, str]],
    currents: np.ndarray,
    configured_strings: dict[str, list[int]] | None = None,
) -> np.ndarray:
    configured = np.isfinite(currents)
    if configured_strings:
        allowed = {
            key: frozenset(int(value) for value in values)
            for key, values in configured_strings.items()
        }
        keys = [
            _key(plant, device)
            for plant, device in source[["plant_id", "device_no"]].itertuples(
                index=False, name=None
            )
        ]
        for column_index, (number, _) in enumerate(strings):
            configured[:, column_index] &= np.fromiter(
                (number in allowed.get(key, ()) for key in keys),
                dtype=bool,
                count=len(keys),
            )
        return configured
    if "main_string_count" not in source.columns:
        return configured
    counts = pd.to_numeric(source["main_string_count"], errors="coerce").to_numpy(float)
    for column_index, (number, _) in enumerate(strings):
        configured[:, column_index] &= np.isfinite(counts) & (counts >= number)
    return configured


def _device_medians(
    source: pd.DataFrame,
    strings: list[tuple[int, str]],
    *,
    zero_current_threshold: float,
    configured_strings: dict[str, list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    currents = source[[column for _, column in strings]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    configured = _configured_mask(source, strings, currents, configured_strings)
    positive = configured & np.isfinite(currents) & (currents > zero_current_threshold)
    usable = np.where(positive, currents, np.nan)
    count = np.isfinite(usable).sum(axis=1)
    safe = np.where((count > 0)[:, None], usable, np.nan)
    medians = np.full(len(source), np.nan, dtype=float)
    valid_rows = count > 0
    if valid_rows.any():
        medians[valid_rows] = np.nanmedian(safe[valid_rows], axis=1)
    return currents, configured, medians


def _virtual_context(
    source: pd.DataFrame,
    strings: list[tuple[int, str]],
    device_scales: dict[str, float],
    *,
    minimum_peer_devices: int,
    zero_current_threshold: float,
    configured_strings: dict[str, list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-source-row virtual irradiance and peer metadata."""
    _, _, medians = _device_medians(
        source,
        strings,
        zero_current_threshold=zero_current_threshold,
        configured_strings=configured_strings,
    )
    row_table = source[["plant_id", "event_time", "device_no"]].copy()
    row_table["device_median"] = medians
    pivot = row_table.pivot_table(
        index=["plant_id", "event_time"],
        columns="device_no",
        values="device_median",
        aggfunc="median",
    )
    # The key is plant|device; construct the scale matrix explicitly.
    scale_matrix = np.empty(pivot.shape, dtype=float)
    for col_index, device in enumerate(pivot.columns):
        scale_matrix[:, col_index] = [
            device_scales.get(_key(plant, device), np.nan)
            for plant in pivot.index.get_level_values("plant_id")
        ]
    scale_matrix[~np.isfinite(scale_matrix) | (scale_matrix <= 0)] = np.nan
    normalised = pivot.to_numpy(float) / scale_matrix
    virtual_by_device = np.full_like(normalised, np.nan, dtype=float)
    peer_count_by_device = np.zeros_like(normalised, dtype=np.int16)
    for col_index in range(normalised.shape[1]):
        peers = normalised.copy()
        peers[:, col_index] = np.nan
        peer_count = np.isfinite(peers).sum(axis=1)
        peer_count_by_device[:, col_index] = peer_count.astype(np.int16)
        valid = peer_count >= minimum_peer_devices
        valid_peer_rows = peer_count > 0
        if valid_peer_rows.any():
            virtual_by_device[valid_peer_rows, col_index] = np.nanmedian(
                peers[valid_peer_rows], axis=1
            )
        virtual_by_device[~valid, col_index] = np.nan

    index_to_row = {
        (str(plant), timestamp, str(device)): (virtual_by_device[row, col], peer_count_by_device[row, col])
        for row, (plant, timestamp) in enumerate(pivot.index)
        for col, device in enumerate(pivot.columns)
    }
    virtual = np.full(len(source), np.nan, dtype=float)
    peer_count = np.zeros(len(source), dtype=np.int16)
    for row_index, item in enumerate(
        source[["plant_id", "event_time", "device_no"]].itertuples(index=False, name=None)
    ):
        value = index_to_row.get((str(item[0]), item[1], str(item[2])))
        if value is not None:
            virtual[row_index], peer_count[row_index] = value
    return virtual, peer_count, medians, currents_to_valid(
        source, strings, configured_strings=configured_strings
    )


def currents_to_valid(
    source: pd.DataFrame,
    strings: list[tuple[int, str]],
    *,
    configured_strings: dict[str, list[int]] | None = None,
) -> np.ndarray:
    currents = source[[column for _, column in strings]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    return _configured_mask(source, strings, currents, configured_strings)


def _fit_device_scales(
    source: pd.DataFrame,
    strings: list[tuple[int, str]],
    *,
    zero_current_threshold: float,
    configured_strings: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    _, _, medians = _device_medians(
        source,
        strings,
        zero_current_threshold=zero_current_threshold,
        configured_strings=configured_strings,
    )
    table = source[["plant_id", "device_no"]].copy()
    table["device_median"] = medians
    scales: dict[str, float] = {}
    plant_values: dict[str, list[float]] = {}
    for (plant, device), group in table.groupby(["plant_id", "device_no"], observed=True):
        values = group["device_median"].dropna().to_numpy(float)
        values = values[np.isfinite(values) & (values > zero_current_threshold)]
        scale = float(np.quantile(values, 0.95)) if len(values) else np.nan
        if np.isfinite(scale) and scale > 0:
            scales[_key(plant, device)] = scale
            plant_values.setdefault(str(plant), []).append(scale)
    for plant, values in plant_values.items():
        fallback = float(np.median(values))
        for key in [key for key in scales if key.startswith(f"{plant}|")]:
            if not np.isfinite(scales[key]) or scales[key] <= 0:
                scales[key] = fallback
    return scales


def _fit_string_responses(
    source: pd.DataFrame,
    strings: list[tuple[int, str]],
    device_scales: dict[str, float],
    *,
    minimum_peer_devices: int,
    zero_current_threshold: float,
    minimum_virtual_irradiance: float,
    configured_strings: dict[str, list[int]] | None = None,
    virtual_override: pd.DataFrame | None = None,
) -> dict[str, float]:
    virtual, _, _, configured = _virtual_context(
        source,
        strings,
        device_scales,
        minimum_peer_devices=minimum_peer_devices,
        zero_current_threshold=zero_current_threshold,
        configured_strings=configured_strings,
    )
    virtual = _apply_virtual_override(source, virtual, virtual_override)
    currents = source[[column for _, column in strings]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    responses: dict[str, float] = {}
    valid_virtual = np.isfinite(virtual) & (virtual >= minimum_virtual_irradiance)
    for column_index, (number, _) in enumerate(strings):
        valid = (
            valid_virtual
            & configured[:, column_index]
            & np.isfinite(currents[:, column_index])
            & (currents[:, column_index] > zero_current_threshold)
        )
        if not valid.any():
            continue
        temporary = pd.DataFrame(
            {
                "plant_id": source["plant_id"].to_numpy()[valid],
                "device_no": source["device_no"].to_numpy()[valid],
                "response": currents[valid, column_index] / virtual[valid],
            }
        )
        for (plant, device), group in temporary.groupby(
            ["plant_id", "device_no"], observed=True
        ):
            values = group["response"].to_numpy(float)
            values = values[np.isfinite(values) & (values > 0)]
            if len(values):
                responses[_string_key(plant, device, number)] = float(np.median(values))
    return responses


def _apply_virtual_override(
    source: pd.DataFrame,
    virtual: np.ndarray,
    override: pd.DataFrame | None,
) -> np.ndarray:
    """Align an optional conditioned virtual irradiance by source identity."""
    if override is None:
        return virtual
    value_column = "conditioned_virtual_irradiance"
    keys = ["plant_id", "device_no", "event_time"]
    required = set(keys + [value_column])
    missing = sorted(required - set(override.columns))
    if missing:
        raise ValueError(f"Virtual irradiance override is missing columns: {missing}")
    right = override[keys + [value_column]].copy()
    right["plant_id"] = right["plant_id"].astype(str)
    right["device_no"] = right["device_no"].astype(str)
    right["event_time"] = pd.to_datetime(right["event_time"], errors="raise", utc=True)
    if right.duplicated(keys).any():
        raise ValueError("Virtual irradiance override contains duplicate source keys")
    left = source[keys].copy()
    left["_row_order"] = np.arange(len(left))
    aligned = left.merge(right, on=keys, how="left", validate="one_to_one").sort_values(
        "_row_order"
    )
    values = pd.to_numeric(aligned[value_column], errors="coerce").to_numpy(float)
    result = np.asarray(virtual, dtype=float).copy()
    available = np.isfinite(values)
    result[available] = values[available]
    return result


def _apply_peer_count_override(
    source: pd.DataFrame,
    peer_count: np.ndarray,
    override: pd.DataFrame | None,
) -> np.ndarray:
    """Optionally replace effective peer support for an external context."""
    if override is None or "conditioned_peer_count" not in override.columns:
        return peer_count
    keys = ["plant_id", "device_no", "event_time"]
    right = override[keys + ["conditioned_peer_count"]].copy()
    right["plant_id"] = right["plant_id"].astype(str)
    right["device_no"] = right["device_no"].astype(str)
    right["event_time"] = pd.to_datetime(right["event_time"], errors="raise", utc=True)
    if right.duplicated(keys).any():
        raise ValueError("Peer-count override contains duplicate source keys")
    left = source[keys].copy()
    left["_row_order"] = np.arange(len(left))
    aligned = left.merge(right, on=keys, how="left", validate="one_to_one").sort_values(
        "_row_order"
    )
    values = pd.to_numeric(aligned["conditioned_peer_count"], errors="coerce").to_numpy(float)
    result = np.asarray(peer_count, dtype=float).copy()
    available = np.isfinite(values)
    result[available] = values[available]
    return result


def _score_features(
    source: pd.DataFrame,
    calibration: PVLOFV2Calibration,
    *,
    max_score_rows: int | None = None,
    virtual_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source, strings = _normalise_source(source)
    currents, configured, device_medians = _device_medians(
        source,
        strings,
        zero_current_threshold=calibration.zero_current_threshold,
        configured_strings=calibration.configured_strings,
    )
    virtual, peer_count, _, _ = _virtual_context(
        source,
        strings,
        calibration.device_scales,
        minimum_peer_devices=calibration.minimum_peer_devices,
        zero_current_threshold=calibration.zero_current_threshold,
        configured_strings=calibration.configured_strings,
    )
    virtual = _apply_virtual_override(source, virtual, virtual_override)
    peer_count = _apply_peer_count_override(source, peer_count, virtual_override)
    responses = np.full_like(currents, np.nan, dtype=float)
    plant_values = source["plant_id"].to_numpy()
    device_values = source["device_no"].to_numpy()
    for column_index, (number, _) in enumerate(strings):
        responses[:, column_index] = [
            calibration.string_responses.get(_string_key(plant, device, number), np.nan)
            for plant, device in zip(plant_values, device_values, strict=True)
        ]
    expected = responses * virtual[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = currents / expected
    positive = configured & np.isfinite(currents) & (currents > calibration.zero_current_threshold)
    response_known = np.isfinite(responses) & (responses > 0)
    row_positive_count = positive.sum(axis=1)
    row_eligible = (
        np.isfinite(virtual)
        & (virtual >= calibration.minimum_virtual_irradiance)
        & (peer_count >= calibration.minimum_peer_devices)
        & (row_positive_count >= calibration.minimum_strings)
    )
    # A directional reference for the isolated branch.  This is deliberately
    # computed per inverter-time point, after virtual-irradiance conditioning,
    # so it is not a fixed current threshold.  A string is considered
    # directionally low only when its residual is below the contemporaneous
    # median of the other usable strings.
    usable_residual = np.where(
        positive & response_known & np.isfinite(residual), residual, np.nan
    )
    residual_median = np.full(len(source), np.nan, dtype=float)
    median_rows = np.flatnonzero(np.isfinite(usable_residual).any(axis=1))
    if len(median_rows):
        residual_median[median_rows] = np.nanmedian(
            usable_residual[median_rows], axis=1
        )
    scores = np.full_like(currents, np.nan, dtype=float)
    group_member = np.zeros_like(configured, dtype=bool)
    group_size = np.zeros(len(source), dtype=np.int16)
    group_median = np.full(len(source), np.nan, dtype=float)
    largest_gap = np.full(len(source), np.nan, dtype=float)
    score_indexes = np.flatnonzero(row_eligible)
    if max_score_rows is not None and len(score_indexes) > max_score_rows:
        score_indexes = score_indexes[:max_score_rows]
    max_fraction = min(max(calibration.maximum_collective_fraction, 0.1), 0.95)
    for row_index in score_indexes:
        indexes = np.flatnonzero(
            positive[row_index] & response_known[row_index] & np.isfinite(residual[row_index])
        )
        if len(indexes) < max(calibration.minimum_strings, calibration.n_neighbors + 1):
            continue
        values = residual[row_index, indexes]
        scores[row_index, indexes] = _lof_1d(
            values,
            n_neighbors=calibration.n_neighbors,
            distance_floor=calibration.distance_floor,
        )
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        minimum_group = calibration.minimum_collective_strings
        maximum_group = min(int(np.floor(len(indexes) * max_fraction)), len(indexes) - 1)
        if maximum_group < minimum_group:
            continue
        gaps = sorted_values[1:] - sorted_values[:-1]
        # Identify the physically dominant separation first.  If the lower
        # side contains more than the configured minority fraction, reject
        # the collective candidate instead of selecting a smaller internal
        # gap and manufacturing a minority group.
        all_cuts = np.arange(minimum_group, len(indexes))
        all_candidate_gaps = gaps[all_cuts - 1]
        best = int(all_cuts[int(np.nanargmax(all_candidate_gaps))])
        if best > maximum_group:
            continue
        largest_gap[row_index] = float(gaps[best - 1])
        low_indexes = indexes[order[:best]]
        group_member[row_index, low_indexes] = True
        group_size[row_index] = best
        group_median[row_index] = float(np.median(values[order[:best]]))
    rows, columns = np.nonzero(configured)
    result = pd.DataFrame(
        {
            "event_time": source["event_time"].to_numpy()[rows],
            "plant_id": source["plant_id"].to_numpy()[rows],
            "device_no": source["device_no"].to_numpy()[rows],
            "string_no": np.asarray([number for number, _ in strings], dtype=np.int16)[columns],
            "string_current": currents[rows, columns].astype(np.float32),
            "virtual_irradiance": virtual[rows].astype(np.float32),
            "peer_device_count": peer_count[rows].astype(np.int16),
            "device_median_current": device_medians[rows].astype(np.float32),
            "expected_current": expected[rows, columns].astype(np.float32),
            "residual_ratio": residual[rows, columns].astype(np.float32),
            "residual_median": residual_median[rows].astype(np.float32),
            "response_known": response_known[rows, columns],
            "v2_eligible": row_eligible[rows],
            "pvlof_score": scores[rows, columns].astype(np.float32),
            "collective_gap": largest_gap[rows].astype(np.float32),
            "collective_group_size": group_size[rows].astype(np.int16),
            "collective_group_median": group_median[rows].astype(np.float32),
            "collective_group_member": group_member[rows, columns],
            "zero_current_alert": (
                configured[rows, columns]
                & (currents[rows, columns] <= calibration.zero_current_threshold)
            ).astype(np.int8),
        }
    )
    return result.sort_values(["event_time", "plant_id", "device_no", "string_no"]).reset_index(
        drop=True
    )


def fit_pvlof_v2_calibration(
    frame: pd.DataFrame,
    *,
    n_neighbors: int = 5,
    lof_quantile: float = 0.995,
    residual_quantile: float = 0.01,
    collective_gap_quantile: float = 0.995,
    minimum_collective_gap: float = 0.10,
    minimum_collective_strings: int = 2,
    maximum_collective_fraction: float = 0.60,
    collective_overlap_threshold: float = 0.50,
    minimum_peer_devices: int = 5,
    minimum_strings: int = 4,
    minimum_consecutive: int = 2,
    expected_interval_minutes: int = 5,
    zero_current_threshold: float = 0.0,
    minimum_virtual_irradiance: float = 0.05,
    minimum_isolated_relative_drop: float = 0.0,
    minimum_isolated_absolute_drop: float = 0.0,
    isolated_effect_gate_mode: str = "all",
    distance_floor: float = 0.01,
    max_score_rows: int = 100_000,
    configured_strings: dict[str, list[int]] | None = None,
    version: str = "pvlof-v2",
    virtual_override: pd.DataFrame | None = None,
) -> tuple[PVLOFV2Calibration, dict[str, Any]]:
    """Fit V2 scales, string responses and robust alert thresholds."""
    if not 0.5 < lof_quantile < 1 or not 0 < residual_quantile < 0.5:
        raise ValueError("quantiles are invalid")
    if minimum_peer_devices < 1 or minimum_strings < 2:
        raise ValueError("minimum peer/string counts are invalid")
    if isolated_effect_gate_mode not in {"all", "any"}:
        raise ValueError("isolated_effect_gate_mode must be 'all' or 'any'")
    source, strings = _normalise_source(frame)
    scales = _fit_device_scales(
        source,
        strings,
        zero_current_threshold=zero_current_threshold,
        configured_strings=configured_strings,
    )
    responses = _fit_string_responses(
        source,
        strings,
        scales,
        minimum_peer_devices=minimum_peer_devices,
        zero_current_threshold=zero_current_threshold,
        minimum_virtual_irradiance=minimum_virtual_irradiance,
        configured_strings=configured_strings,
        virtual_override=virtual_override,
    )
    provisional = PVLOFV2Calibration(
        version=version,
        n_neighbors=n_neighbors,
        lof_quantile=lof_quantile,
        lof_threshold=2.0,
        residual_quantile=residual_quantile,
        residual_threshold=0.70,
        collective_gap_quantile=collective_gap_quantile,
        collective_gap_threshold=minimum_collective_gap,
        minimum_collective_gap=minimum_collective_gap,
        minimum_collective_strings=minimum_collective_strings,
        maximum_collective_fraction=maximum_collective_fraction,
        collective_overlap_threshold=collective_overlap_threshold,
        minimum_peer_devices=minimum_peer_devices,
        minimum_strings=minimum_strings,
        minimum_consecutive=minimum_consecutive,
        expected_interval_minutes=expected_interval_minutes,
        zero_current_threshold=zero_current_threshold,
        minimum_virtual_irradiance=minimum_virtual_irradiance,
        minimum_isolated_relative_drop=minimum_isolated_relative_drop,
        minimum_isolated_absolute_drop=minimum_isolated_absolute_drop,
        isolated_effect_gate_mode=isolated_effect_gate_mode,
        distance_floor=distance_floor,
        device_scales=scales,
        string_responses=responses,
        configured_strings=configured_strings or {},
    )
    scored = _score_features(
        source,
        provisional,
        max_score_rows=max_score_rows,
        virtual_override=virtual_override,
    )
    eligible = scored[scored["v2_eligible"] & scored["response_known"]]
    residual_values = pd.to_numeric(eligible["residual_ratio"], errors="coerce").to_numpy(float)
    residual_values = residual_values[np.isfinite(residual_values) & (residual_values > 0)]
    score_values = pd.to_numeric(eligible["pvlof_score"], errors="coerce").to_numpy(float)
    score_values = score_values[np.isfinite(score_values)]
    gap_values = pd.to_numeric(eligible["collective_gap"], errors="coerce").to_numpy(float)
    gap_values = gap_values[np.isfinite(gap_values)]
    if not len(residual_values) or not len(score_values):
        raise ValueError("No eligible PVLOF-V2 calibration samples remain")
    residual_threshold = float(np.quantile(residual_values, residual_quantile))
    lof_threshold = float(np.quantile(score_values, lof_quantile))
    gap_threshold = max(
        minimum_collective_gap,
        float(np.quantile(gap_values, collective_gap_quantile)) if len(gap_values) else minimum_collective_gap,
    )
    calibration = PVLOFV2Calibration(
        **{
            **provisional.to_dict(),
            "lof_threshold": lof_threshold,
            "residual_threshold": residual_threshold,
            "collective_gap_threshold": gap_threshold,
        }
    )
    return calibration, {
        "version": calibration.version,
        "wide_samples": len(source),
        "plants": int(source["plant_id"].nunique()),
        "devices": int(source[["plant_id", "device_no"]].drop_duplicates().shape[0]),
        "device_scale_count": len(scales),
        "string_response_count": len(responses),
        "channel_inventory_device_count": len(configured_strings or {}),
        "maximum_collective_fraction": maximum_collective_fraction,
        "collective_continuity_metric": "jaccard",
        "collective_overlap_threshold": collective_overlap_threshold,
        "minimum_isolated_relative_drop": minimum_isolated_relative_drop,
        "minimum_isolated_absolute_drop": minimum_isolated_absolute_drop,
        "isolated_effect_gate_mode": isolated_effect_gate_mode,
        "scored_samples": len(scored),
        "eligible_samples": len(eligible),
        "residual_samples": len(residual_values),
        "score_samples": len(score_values),
        "thresholds": {
            "lof": lof_threshold,
            "residual": residual_threshold,
            "collective_gap": gap_threshold,
        },
        "virtual_irradiance_coverage": float(scored["virtual_irradiance"].notna().mean()),
    }


def _add_branch_consecutive(
    frame: pd.DataFrame,
    *,
    raw_column: str,
    output_column: str,
    calibration: PVLOFV2Calibration,
) -> pd.DataFrame:
    result = frame.sort_values(["plant_id", "device_no", "string_no", "event_time"]).reset_index(
        drop=True
    ).copy()
    raw = result[raw_column].astype(bool).to_numpy()
    times = result["event_time"].to_numpy(dtype="datetime64[ns]")
    consecutive = np.zeros(len(result), dtype=np.int32)
    expected = np.timedelta64(calibration.expected_interval_minutes, "m")
    groups = result.groupby(["plant_id", "device_no", "string_no"], observed=True).indices
    for positions in groups.values():
        indexes = np.asarray(positions, dtype=np.int64)
        previous: np.datetime64 | None = None
        run = 0
        for index in indexes:
            contiguous = previous is not None and times[index] - previous == expected
            if not contiguous:
                run = 0
            run = run + 1 if raw[index] else 0
            consecutive[index] = run
            previous = times[index]
    result[output_column.replace("_alert", "_consecutive")] = consecutive
    result[output_column] = (
        raw & (consecutive >= calibration.minimum_consecutive)
    ).astype(np.int8)
    return result


def _add_collective_event_continuity(
    frame: pd.DataFrame,
    calibration: PVLOFV2Calibration,
) -> pd.DataFrame:
    """Confirm a collective event at device level, then alert current members.

    Continuity uses Jaccard similarity so that a small group cannot expand
    into a much larger dusk/noise group while inheriting the former run.
    """
    result = frame.copy()
    device_keys = ["plant_id", "device_no", "event_time"]
    event_rows: list[dict[str, Any]] = []
    for key, group in result.groupby(device_keys, observed=True, sort=False):
        members = frozenset(
            int(value)
            for value in group.loc[group["collective_raw_alert"].astype(bool), "string_no"]
        )
        event_rows.append(
            {
                "plant_id": key[0],
                "device_no": key[1],
                "event_time": key[2],
                "_collective_members": members,
                "collective_event_raw": int(
                    len(members) >= calibration.minimum_collective_strings
                ),
            }
        )
    events = pd.DataFrame(event_rows)
    if events.empty:
        result["collective_event_raw"] = np.int8(0)
        result["collective_overlap"] = np.float32(np.nan)
        result["collective_event_consecutive"] = np.int32(0)
        result["collective_event_alert"] = np.int8(0)
        result["collective_event_group_size"] = np.int16(0)
        result["collective_member_alert"] = np.int8(0)
        return result

    events = events.sort_values(["plant_id", "device_no", "event_time"]).reset_index(drop=True)
    events["collective_overlap"] = np.nan
    events["collective_event_consecutive"] = 0
    events["collective_event_alert"] = 0
    events["collective_event_group_size"] = events["_collective_members"].map(len).astype(np.int16)
    for _, positions in events.groupby(["plant_id", "device_no"], observed=True).indices.items():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        previous_members: frozenset[int] = frozenset()
        previous_raw = False
        run = 0
        for index in indexes:
            current_time = events.at[index, "event_time"]
            current_members = events.at[index, "_collective_members"]
            current_raw = bool(events.at[index, "collective_event_raw"])
            contiguous = (
                previous_time is not None
                and current_time - previous_time
                == np.timedelta64(calibration.expected_interval_minutes, "m")
            )
            overlap = np.nan
            if current_raw and previous_raw and contiguous:
                overlap = len(current_members & previous_members) / max(
                    len(current_members | previous_members), 1
                )
                continues = overlap >= calibration.collective_overlap_threshold
            else:
                continues = False
            if current_raw:
                run = run + 1 if continues else 1
            else:
                run = 0
            events.at[index, "collective_overlap"] = overlap
            events.at[index, "collective_event_consecutive"] = run
            events.at[index, "collective_event_alert"] = int(
                current_raw and run >= calibration.minimum_consecutive
            )
            previous_time = current_time
            previous_members = current_members
            previous_raw = current_raw

    metadata = events.drop(columns="_collective_members")
    result = result.merge(metadata, on=device_keys, how="left", validate="many_to_one")
    result["collective_member_alert"] = (
        result["collective_event_alert"].astype(bool)
        & result["collective_raw_alert"].astype(bool)
    ).astype(np.int8)
    return result


def _add_consecutive(frame: pd.DataFrame, calibration: PVLOFV2Calibration) -> pd.DataFrame:
    frame = frame.copy()
    # Backward-compatible for callers/tests that construct the scored frame
    # manually and predate the directional isolated branch.
    if "isolated_directional_raw_alert" not in frame.columns:
        frame["isolated_directional_raw_alert"] = frame["isolated_raw_alert"]
    result = _add_branch_consecutive(
        frame,
        raw_column="isolated_raw_alert",
        output_column="isolated_alert",
        calibration=calibration,
    )
    directional = _add_branch_consecutive(
        frame,
        raw_column="isolated_directional_raw_alert",
        output_column="isolated_directional_alert",
        calibration=calibration,
    )[[
        "plant_id", "device_no", "string_no", "event_time",
        "isolated_directional_consecutive", "isolated_directional_alert",
    ]]
    # Carry the legacy mixed-branch result across the second stable sort.
    legacy_keys = ["plant_id", "device_no", "string_no", "event_time"]
    legacy = _add_branch_consecutive(
        frame.assign(
            pvlof_v2_raw_alert=(
                frame["isolated_raw_alert"].astype(bool)
                | frame["collective_raw_alert"].astype(bool)
            ).astype(np.int8)
        ),
        raw_column="pvlof_v2_raw_alert",
        output_column="pvlof_v2_legacy_alert",
        calibration=calibration,
    )[legacy_keys + ["pvlof_v2_legacy_consecutive", "pvlof_v2_legacy_alert"]]
    result = result.merge(legacy, on=legacy_keys, how="left", validate="one_to_one")
    result = result.merge(directional, on=legacy_keys, how="left", validate="one_to_one")
    result = _add_collective_event_continuity(result, calibration)
    result["pvlof_v2_raw_alert"] = (
        result["isolated_raw_alert"].astype(bool)
        | result["collective_raw_alert"].astype(bool)
    ).astype(np.int8)
    result["pvlof_v2_consecutive"] = np.maximum(
        result["pvlof_v2_legacy_consecutive"], result["collective_event_consecutive"]
    ).astype(np.int32)
    result["pvlof_v2_alert"] = (
        result["pvlof_v2_legacy_alert"].astype(bool)
        | result["collective_member_alert"].astype(bool)
    ).astype(np.int8)
    # The isolated-modified result is monotonic with the current group version:
    # it may add directionally-low LOF strings, but it can never remove a
    # previously confirmed group or legacy mixed-branch alert.
    result["pvlof_v2_iso_mod_alert"] = (
        result["pvlof_v2_alert"].astype(bool)
        | result["isolated_directional_alert"].astype(bool)
    ).astype(np.int8)
    result["pvlof_v2_iso_mod_consecutive"] = np.maximum(
        result["pvlof_v2_consecutive"],
        result["isolated_directional_consecutive"],
    ).astype(np.int32)
    isolated_final = result["isolated_alert"].astype(bool)
    collective_final = result["collective_member_alert"].astype(bool)
    mixed_final = (
        result["pvlof_v2_legacy_alert"].astype(bool)
        & ~isolated_final
        & ~collective_final
    )
    result["alert_reason"] = np.select(
        [isolated_final & collective_final, isolated_final, collective_final, mixed_final],
        ["both", "isolated", "collective", "mixed_persistence"],
        default="",
    )
    result["combined_alert"] = (
        result["zero_current_alert"].astype(bool) | result["pvlof_v2_alert"].astype(bool)
    ).astype(np.int8)
    result["combined_iso_mod_alert"] = (
        result["zero_current_alert"].astype(bool)
        | result["pvlof_v2_iso_mod_alert"].astype(bool)
    ).astype(np.int8)
    return result.sort_values(["event_time", "plant_id", "device_no", "string_no"]).reset_index(
        drop=True
    )


def apply_pvlof_v2(
    frame: pd.DataFrame,
    calibration: PVLOFV2Calibration,
    *,
    virtual_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score rows and apply isolated, collective and persistence rules."""
    scored = _score_features(frame, calibration, virtual_override=virtual_override)
    score = pd.to_numeric(scored["pvlof_score"], errors="coerce")
    residual = pd.to_numeric(scored["residual_ratio"], errors="coerce")
    residual_median = pd.to_numeric(scored["residual_median"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_drop = 1.0 - residual / residual_median
    relative_drop = relative_drop.where(residual_median.gt(0))
    expected_current = pd.to_numeric(scored["expected_current"], errors="coerce")
    string_current = pd.to_numeric(scored["string_current"], errors="coerce")
    absolute_drop = expected_current - string_current
    relative_effect_eligible = relative_drop.ge(
        calibration.minimum_isolated_relative_drop
    )
    if calibration.minimum_isolated_absolute_drop > 0:
        absolute_effect_eligible = absolute_drop.ge(
            calibration.minimum_isolated_absolute_drop
        )
    else:
        # Zero means that the new absolute gate is disabled.  This preserves
        # byte-for-byte decision compatibility for pre-v1.4 calibrations,
        # including rows where expected_current is unavailable.
        absolute_effect_eligible = pd.Series(True, index=scored.index)
    gate_mode = calibration.isolated_effect_gate_mode
    if gate_mode == "all":
        # Historical v1.4/v1.5 behaviour: every enabled effect gate must pass.
        effect_eligible = relative_effect_eligible & absolute_effect_eligible
    elif gate_mode == "any":
        # Hybrid v1.6 behaviour. A zero threshold disables that side instead
        # of turning the OR expression into an unconditional pass.
        relative_any = (
            relative_effect_eligible
            if calibration.minimum_isolated_relative_drop > 0
            else pd.Series(False, index=scored.index)
        )
        absolute_any = (
            absolute_effect_eligible
            if calibration.minimum_isolated_absolute_drop > 0
            else pd.Series(False, index=scored.index)
        )
        if (
            calibration.minimum_isolated_relative_drop <= 0
            and calibration.minimum_isolated_absolute_drop <= 0
        ):
            effect_eligible = pd.Series(True, index=scored.index)
        else:
            effect_eligible = relative_any | absolute_any
    else:
        raise ValueError(
            "isolated_effect_gate_mode must be 'all' or 'any', "
            f"got {gate_mode!r}"
        )
    scored["isolated_relative_drop"] = relative_drop.astype(np.float32)
    scored["isolated_absolute_drop"] = absolute_drop.astype(np.float32)
    scored["isolated_relative_effect_eligible"] = (
        relative_effect_eligible.fillna(False).astype(np.int8)
    )
    scored["isolated_absolute_effect_eligible"] = (
        absolute_effect_eligible.fillna(False).astype(np.int8)
    )
    scored["isolated_effect_eligible"] = effect_eligible.fillna(False).astype(np.int8)
    isolated = (
        scored["v2_eligible"].astype(bool)
        & scored["response_known"].astype(bool)
        & scored["string_current"].gt(calibration.zero_current_threshold)
        & score.ge(calibration.lof_threshold)
        & residual.le(calibration.residual_threshold)
    )
    # Improved isolated branch: LOF remains the anomaly detector; residual is
    # used only for direction (below the current inverter-time median), not as
    # a fixed severity gate.  This preserves the collective branch unchanged.
    isolated_directional = (
        scored["v2_eligible"].astype(bool)
        & scored["response_known"].astype(bool)
        & scored["string_current"].gt(calibration.zero_current_threshold)
        & score.ge(calibration.lof_threshold)
        & residual.lt(residual_median)
        & effect_eligible
    )
    collective = (
        scored["v2_eligible"].astype(bool)
        & scored["response_known"].astype(bool)
        & scored["collective_group_member"].astype(bool)
        & scored["collective_gap"].ge(calibration.collective_gap_threshold)
        & scored["collective_group_median"].le(calibration.residual_threshold)
        & scored["string_current"].gt(calibration.zero_current_threshold)
    )
    scored["isolated_raw_alert"] = isolated.astype(np.int8)
    scored["isolated_directional_raw_alert"] = isolated_directional.astype(np.int8)
    scored["collective_raw_alert"] = collective.astype(np.int8)
    return _add_consecutive(scored, calibration)


def collapse_pvlof_v2_events(
    frame: pd.DataFrame,
    *,
    alert_column: str = "pvlof_v2_alert",
    expected_interval_minutes: int = 5,
) -> pd.DataFrame:
    """Collapse V2 string alerts into plant/device/string events."""
    required = {"plant_id", "device_no", "string_no", "event_time", alert_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing PVLOF-V2 event columns: {missing}")
    alerts = frame[frame[alert_column].astype(bool)].copy()
    columns = [
        "event_id", "plant_id", "device_no", "string_no", "start_time", "end_time",
        "points", "duration_minutes", "maximum_score", "minimum_residual", "alert_reasons",
    ]
    if alerts.empty:
        return pd.DataFrame(columns=columns)
    alerts = alerts.sort_values(
        ["plant_id", "device_no", "string_no", "event_time"]
    ).reset_index(drop=True)
    expected = pd.Timedelta(minutes=expected_interval_minutes)
    new_event = (
        alerts["plant_id"].ne(alerts["plant_id"].shift())
        | alerts["device_no"].ne(alerts["device_no"].shift())
        | alerts["string_no"].ne(alerts["string_no"].shift())
        | alerts["event_time"].sub(alerts["event_time"].shift()).ne(expected)
    )
    alerts["_event_number"] = new_event.cumsum()
    aggregations: dict[str, tuple[str, str]] = {
        "start_time": ("event_time", "min"),
        "end_time": ("event_time", "max"),
        "points": ("event_time", "size"),
        "maximum_score": ("pvlof_score", "max"),
        "minimum_residual": ("residual_ratio", "min"),
        "alert_reasons": ("alert_reason", lambda values: ",".join(sorted({str(x) for x in values if str(x)}))),
    }
    events = (
        alerts.groupby(["plant_id", "device_no", "string_no", "_event_number"], observed=True)
        .agg(**aggregations)
        .reset_index()
        .drop(columns="_event_number")
    )
    events["duration_minutes"] = (
        events["end_time"] - events["start_time"]
    ).dt.total_seconds() / 60 + expected_interval_minutes
    events.insert(0, "event_id", [f"pvlof-v2-{index:06d}" for index in range(1, len(events) + 1)])
    return events[columns]


def save_calibration(calibration: PVLOFV2Calibration, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> PVLOFV2Calibration:
    return PVLOFV2Calibration.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
