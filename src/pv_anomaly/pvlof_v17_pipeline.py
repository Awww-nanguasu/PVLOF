"""Shared in-memory PVLOF v1.6/v1.7 alarm-window evaluation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_events import reconstruct_pvlof_events
from pv_anomaly.pvlof_v12 import (
    PVLOFV12WeatherCalibration,
    build_conditioned_virtual_context,
)
from pv_anomaly.pvlof_v16 import (
    PVLOFV16MemoryConfig,
    apply_confirmed_anomaly_memory_v16,
)
from pv_anomaly.pvlof_v17 import PVLOFV17Config, apply_pvlof_v17
from pv_anomaly.pvlof_v2 import PVLOFV2Calibration, apply_pvlof_v2
from pv_anomaly.pvlof_v2_hier import (
    PVLOFV2HierCalibration,
    apply_hierarchical_isolated,
)


DEVICE_TIME_KEYS = ["plant_id", "device_no", "event_time"]
STRING_KEYS = [*DEVICE_TIME_KEYS, "string_no"]
COMPARISON_LABELS = [
    "PVLOF_V1_6_HYBRID_GATE",
    "PVLOF_V1_7_PENALIZED_SEGMENTATION",
]
COMPARISON_COLUMNS = [
    "row",
    "event_id",
    "plant_id",
    "device_no",
    "raise_time_local",
    "end_time_local",
    *COMPARISON_LABELS,
    "comparison_case",
    "alert_time_points",
    "duration_minutes",
]


def score_conditioned_context(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
    base_calibration: PVLOFV2Calibration,
    weather_calibration: PVLOFV12WeatherCalibration,
    hierarchy: PVLOFV2HierCalibration,
    *,
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the common V1.6/V1.7 score table from wide alarm context."""

    context, context_report = build_conditioned_virtual_context(
        frame,
        base_calibration,
        weather_calibration,
        weather,
    )
    scored = apply_pvlof_v2(frame, base_calibration, virtual_override=context)
    scored = apply_hierarchical_isolated(scored, hierarchy, timezone=timezone)
    metadata_columns = [
        "plant_id",
        "device_no",
        "event_time",
        "raw_virtual_irradiance",
        "raw_peer_device_count",
        "forecast_ghi",
        "forecast_source_time",
        "forecast_virtual_irradiance",
        "forecast_virtual_capped",
        "virtual_peer_weight",
        "conditioned_virtual_irradiance",
        "forecast_available",
        "forecast_offset_minutes",
    ]
    metadata = context[metadata_columns]
    scored = scored.merge(
        metadata,
        on=DEVICE_TIME_KEYS,
        how="left",
        validate="many_to_one",
    )
    report = {
        "conditioned_context": context_report,
        "wide_context_rows": int(len(frame)),
        "wide_context_devices": int(
            frame[["plant_id", "device_no"]].drop_duplicates().shape[0]
        ),
        "scored_string_rows": int(len(scored)),
        "scored_device_time_rows": int(
            scored[DEVICE_TIME_KEYS].drop_duplicates().shape[0]
        ),
    }
    return scored, report


def _normalise_target_keys(alarm_points: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(DEVICE_TIME_KEYS) - set(alarm_points.columns))
    if missing:
        raise ValueError(f"Alarm target input is missing columns: {missing}")
    keys = alarm_points[DEVICE_TIME_KEYS].copy()
    keys["plant_id"] = keys["plant_id"].astype(str)
    keys["device_no"] = keys["device_no"].astype(str).str.strip()
    keys["event_time"] = pd.to_datetime(
        keys["event_time"], errors="raise", utc=True
    )
    return keys.drop_duplicates().reset_index(drop=True)


def select_alarm_targets(
    frame: pd.DataFrame,
    alarm_points: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep alarmed device-time rows after full peer-context scoring."""

    source = frame.copy()
    source["plant_id"] = source["plant_id"].astype(str)
    source["device_no"] = source["device_no"].astype(str).str.strip()
    source["event_time"] = pd.to_datetime(
        source["event_time"], errors="raise", utc=True
    )
    if alarm_points is None:
        return source, {
            "filter_applied": False,
            "target_device_time_rows": int(
                source[DEVICE_TIME_KEYS].drop_duplicates().shape[0]
            ),
            "matched_device_time_rows": int(
                source[DEVICE_TIME_KEYS].drop_duplicates().shape[0]
            ),
            "matched_string_rows": int(len(source)),
        }

    targets = _normalise_target_keys(alarm_points)
    selected = source.merge(
        targets,
        on=DEVICE_TIME_KEYS,
        how="inner",
        validate="many_to_one",
    )
    return selected, {
        "filter_applied": True,
        "target_device_time_rows": int(len(targets)),
        "matched_device_time_rows": int(
            selected[DEVICE_TIME_KEYS].drop_duplicates().shape[0]
        ),
        "matched_string_rows": int(len(selected)),
    }


def _v17_event_compatibility_frame(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        *STRING_KEYS,
        "pvlof_v17_alert",
        "isolated_directional_raw_alert",
        "isolated_hier_raw_alert",
        "isolated_raw_alert",
        "collective_raw_alert",
        "isolated_directional_alert",
        "isolated_hier_strict_alert",
        "isolated_alert",
        "collective_member_alert",
        "pvlof_v2_legacy_alert",
        "isolated_directional_consecutive",
        "isolated_hier_strict_consecutive",
        "isolated_consecutive",
        "collective_event_consecutive",
        "pvlof_v17_raw_anomaly",
        "pvlof_v17_entry_streak",
        "pvlof_v17_normal_streak",
        "pvlof_v17_memory_active",
        "pvlof_v17_memory_reactivated_alert",
        "pvlof_v17_memory_clear_code",
    ]
    result = frame[[column for column in candidates if column in frame]].copy()
    for suffix in (
        "raw_anomaly",
        "entry_streak",
        "normal_streak",
        "memory_active",
        "memory_reactivated_alert",
        "memory_clear_code",
    ):
        result[f"pvlof_v16_{suffix}"] = result[f"pvlof_v17_{suffix}"]
    return result


def _mark_v17_branch(
    evidence: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for frame in (evidence, events):
        if "confirmation_branch" in frame:
            frame.loc[
                frame["confirmation_branch"].eq("final_union"),
                "confirmation_branch",
            ] = "penalized_segmentation"
    return evidence, events


def _numbers(values: pd.Series) -> str:
    numbers = {
        int(float(item))
        for value in values.dropna()
        for item in str(value).split(",")
        if item.strip()
    }
    return ",".join(f"{number:02d}" for number in sorted(numbers))


def _confirmed_points(evidence: pd.DataFrame, label: str) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=[*DEVICE_TIME_KEYS, label])
    selected = evidence[
        evidence["raw_evidence"].fillna(False).astype(bool)
        | evidence["final_alert"].fillna(False).astype(bool)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=[*DEVICE_TIME_KEYS, label])
    selected["event_time"] = pd.to_datetime(
        selected["event_time"], errors="raise", utc=True
    )
    return (
        selected.groupby(DEVICE_TIME_KEYS, observed=True)["string_no"]
        .agg(_numbers)
        .rename(label)
        .reset_index()
    )


def _comparison_case(frame: pd.DataFrame) -> pd.Series:
    left = frame[COMPARISON_LABELS[0]].ne("")
    right = frame[COMPARISON_LABELS[1]].ne("")
    result = pd.Series("both_same", index=frame.index, dtype="string")
    result.loc[left & ~right] = "v1_6_only"
    result.loc[~left & right] = "v1_7_only"
    result.loc[
        left
        & right
        & frame[COMPARISON_LABELS[0]].ne(frame[COMPARISON_LABELS[1]])
    ] = "both_different"
    return result


def build_customer_event_comparison(
    v16_evidence: pd.DataFrame,
    v17_evidence: pd.DataFrame,
    *,
    timezone: str = "Asia/Shanghai",
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """Build a compact comparison with confirmed candidates backfilled."""

    left = _confirmed_points(v16_evidence, COMPARISON_LABELS[0])
    right = _confirmed_points(v17_evidence, COMPARISON_LABELS[1])
    points = left.merge(
        right,
        on=DEVICE_TIME_KEYS,
        how="outer",
        validate="one_to_one",
    )
    if points.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    points[COMPARISON_LABELS] = points[COMPARISON_LABELS].astype("string").fillna("")
    points = points.sort_values(DEVICE_TIME_KEYS).reset_index(drop=True)
    expected = pd.Timedelta(minutes=interval_minutes)
    points["_event"] = (
        points["plant_id"].ne(points["plant_id"].shift())
        | points["device_no"].ne(points["device_no"].shift())
        | points["event_time"].sub(points["event_time"].shift()).ne(expected)
    ).cumsum()
    points["event_id"] = points["_event"].map({
        value: f"pvlof-v16-v17-{number:06d}"
        for number, value in enumerate(points["_event"].unique(), 1)
    })
    events = (
        points.groupby(
            ["event_id", "plant_id", "device_no", "_event"], observed=True
        )
        .agg(
            raise_time=("event_time", "min"),
            end_time=("event_time", "max"),
            PVLOF_V1_6_HYBRID_GATE=(COMPARISON_LABELS[0], _numbers),
            PVLOF_V1_7_PENALIZED_SEGMENTATION=(COMPARISON_LABELS[1], _numbers),
            alert_time_points=("event_time", "size"),
        )
        .reset_index()
        .drop(columns="_event")
    )
    events["comparison_case"] = _comparison_case(events)
    events["raise_time_local"] = (
        events["raise_time"]
        .dt.tz_convert(timezone)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    events["end_time_local"] = (
        events["end_time"]
        .dt.tz_convert(timezone)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    events["duration_minutes"] = (
        (events["end_time"] - events["raise_time"]).dt.total_seconds() / 60
        + interval_minutes
    ).astype(int)
    events = events.sort_values(
        ["plant_id", "device_no", "raise_time"]
    ).reset_index(drop=True)
    events.insert(0, "row", range(1, len(events) + 1))
    return events[COMPARISON_COLUMNS]


def build_v17_additions(
    v16_points: pd.DataFrame,
    v17_points: pd.DataFrame,
    *,
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    diagnostics = [
        "string_current",
        "expected_current",
        "residual_ratio",
        "residual_median",
        "pvlof_score",
        "pvlof_v17_relative_drop",
        "pvlof_v17_absolute_drop",
        "pvlof_v17_physical_effect_eligible",
        "pvlof_v17_segmentation_structural_member",
        "pvlof_v17_segmentation_raw_candidate",
        "pvlof_v17_segmentation_memory_reactivated_alert",
        "pvlof_v17_segment_index",
        "pvlof_v17_segment_count",
        "pvlof_v17_segmentation_candidate_size",
        "pvlof_v17_segmentation_reference_size",
        "pvlof_v17_segmentation_reference_residual",
        "pvlof_v17_segmentation_objective",
        "pvlof_v17_segmentation_sse_gain",
    ]
    diagnostics = [column for column in diagnostics if column in v17_points]
    left = v16_points[[*STRING_KEYS, "pvlof_v16_alert"]].rename(
        columns={"pvlof_v16_alert": "_v16_alert"}
    )
    right = v17_points[[*STRING_KEYS, "pvlof_v17_alert", *diagnostics]].rename(
        columns={"pvlof_v17_alert": "_v17_alert"}
    )
    merged = right.merge(left, on=STRING_KEYS, how="left", validate="one_to_one")
    additions = merged[
        merged["_v17_alert"].fillna(False).astype(bool)
        & ~merged["_v16_alert"].fillna(False).astype(bool)
    ].copy()
    additions["addition_reason"] = "penalized_segmentation"
    reactivation_column = "pvlof_v17_segmentation_memory_reactivated_alert"
    if reactivation_column in additions:
        reactivated = additions[reactivation_column].fillna(False).astype(bool)
        additions.loc[reactivated, "addition_reason"] = (
            "segmentation_memory_reactivation"
        )
    additions["event_time_local"] = (
        pd.to_datetime(additions["event_time"], errors="raise", utc=True)
        .dt.tz_convert(timezone)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    additions["string_no"] = additions["string_no"].map(
        lambda value: f"{int(value):02d}" if pd.notna(value) else ""
    )
    columns = [
        "plant_id",
        "device_no",
        "event_time_local",
        "string_no",
        "addition_reason",
        *diagnostics,
    ]
    return additions[columns].sort_values(
        ["plant_id", "device_no", "event_time_local", "string_no"]
    ).reset_index(drop=True)


def apply_versions_to_context(
    scored: pd.DataFrame,
    v16_config: PVLOFV16MemoryConfig,
    v17_config: PVLOFV17Config,
    *,
    alarm_points: pd.DataFrame | None = None,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    """Apply both versions in memory, then select alarm_device rows."""

    v16_full = apply_confirmed_anomaly_memory_v16(scored, v16_config)
    v17_full = apply_pvlof_v17(v16_full, v17_config)
    v16_target, target_report = select_alarm_targets(v16_full, alarm_points)
    target_keys = v16_target[STRING_KEYS].drop_duplicates()
    v17_target = v17_full.merge(
        target_keys,
        on=STRING_KEYS,
        how="inner",
        validate="one_to_one",
    )

    v16_evidence, v16_events, v16_unconfirmed = reconstruct_pvlof_events(
        v16_target,
        version=v16_config.version,
        final_alert_column="pvlof_v16_alert",
        expected_interval_minutes=v16_config.expected_interval_minutes,
        timezone=timezone,
        memory_prefix="pvlof_v16",
    )
    v17_evidence, v17_events, v17_unconfirmed = reconstruct_pvlof_events(
        _v17_event_compatibility_frame(v17_target),
        version=v17_config.version,
        final_alert_column="pvlof_v17_alert",
        expected_interval_minutes=v17_config.expected_interval_minutes,
        timezone=timezone,
        memory_prefix="pvlof_v16",
    )
    v17_evidence, v17_events = _mark_v17_branch(v17_evidence, v17_events)
    comparison = build_customer_event_comparison(
        v16_evidence,
        v17_evidence,
        timezone=timezone,
        interval_minutes=v17_config.expected_interval_minutes,
    )
    additions = build_v17_additions(v16_target, v17_target, timezone=timezone)
    return {
        "v16_full_points": v16_full,
        "v17_full_points": v17_full,
        "v16_target_points": v16_target,
        "v17_target_points": v17_target,
        "v16_evidence": v16_evidence,
        "v17_evidence": v17_evidence,
        "v16_events": v16_events,
        "v17_events": v17_events,
        "v16_unconfirmed": v16_unconfirmed,
        "v17_unconfirmed": v17_unconfirmed,
        "comparison_events": comparison,
        "v17_additions": additions,
        "target_report": target_report,
    }


__all__ = [
    "apply_versions_to_context",
    "build_customer_event_comparison",
    "build_v17_additions",
    "score_conditioned_context",
    "select_alarm_targets",
]
