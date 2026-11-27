"""Hierarchical LOF thresholds for the isolated PVLOF-V2 branch.

This module intentionally sits on top of :mod:`pv_anomaly.pvlof_v2`.  The
existing PVLOF-V2 and PVLOF_V2_iso_mod columns are never rewritten.  It only
selects a context-specific LOF threshold and adds a new isolated branch whose
result is monotonic with the existing result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_v2 import (
    PVLOFV2Calibration,
    apply_pvlof_v2,
)


@dataclass(frozen=True)
class PVLOFV2HierCalibration:
    """Serializable hierarchical isolated-branch calibration."""

    version: str = "pvlof-v2-hier-v1"
    lof_quantile: float = 0.995
    minimum_device_samples: int = 10_000
    minimum_device_days: int = 20
    minimum_plant_count_samples: int = 20_000
    minimum_plant_count_days: int = 20
    minimum_count_samples: int = 20_000
    minimum_count_days: int = 20
    shrinkage_k: float = 5_000.0
    minimum_consecutive: int = 2
    expected_interval_minutes: int = 5
    global_threshold: float = 6.0
    device_thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    plant_string_count_thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    string_count_thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFV2HierCalibration":
        return cls(**dict(payload))


def save_hier_calibration(calibration: PVLOFV2HierCalibration, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_hier_calibration(path: str | Path) -> PVLOFV2HierCalibration:
    return PVLOFV2HierCalibration.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def exclude_alarm_windows(
    frame: pd.DataFrame,
    events_path: str | Path,
    *,
    buffer_minutes: int = 10,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove alarm intervals and a surrounding buffer from calibration data."""

    events = pd.read_parquet(events_path)
    required = {"plant_id", "device_no"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Alarm events are missing columns: {missing}")
    start_column = "effective_start_time" if "effective_start_time" in events else "raise_time"
    end_column = "effective_end_time" if "effective_end_time" in events else "end_time"
    events[start_column] = pd.to_datetime(events[start_column], errors="coerce", utc=True)
    events[end_column] = pd.to_datetime(events[end_column], errors="coerce", utc=True)
    source = frame.copy()
    source["event_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    source["_excluded_alarm"] = False
    interval_count = 0
    for (plant, device), group in events.groupby(["plant_id", "device_no"], observed=True):
        mask_source = source["plant_id"].astype(str).eq(str(plant)) & source[
            "device_no"
        ].astype(str).eq(str(device))
        if not mask_source.any():
            continue
        for start, end in group[[start_column, end_column]].dropna().itertuples(
            index=False, name=None
        ):
            start = start - pd.Timedelta(minutes=buffer_minutes)
            end = end + pd.Timedelta(minutes=buffer_minutes)
            source.loc[
                mask_source & source["event_time"].between(start, end),
                "_excluded_alarm",
            ] = True
            interval_count += 1
    excluded = int(source["_excluded_alarm"].sum())
    return (
        source.loc[~source["_excluded_alarm"]].drop(columns="_excluded_alarm").reset_index(
            drop=True
        ),
        {"alarm_intervals": interval_count, "excluded_rows": excluded},
    )


def _prepare_scored(scored: pd.DataFrame, timezone: str) -> pd.DataFrame:
    required = {
        "plant_id",
        "device_no",
        "event_time",
        "pvlof_score",
        "residual_ratio",
        "residual_median",
        "v2_eligible",
        "response_known",
        "string_current",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Scored PVLOF-V2 frame is missing columns: {missing}")
    result = scored.copy()
    result["plant_id"] = result["plant_id"].astype(str)
    result["device_no"] = result["device_no"].astype(str)
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    result["event_date"] = result["event_time"].dt.tz_convert(timezone).dt.strftime("%Y-%m-%d")
    keys = ["plant_id", "device_no", "event_time"]
    # Scores are available only for usable strings.  This count is the
    # configuration context used by the fallback hierarchy.
    score_count = result["pvlof_score"].notna().groupby(
        [result[key] for key in keys], observed=True
    ).transform("sum")
    result["usable_string_count"] = score_count.astype("Int64")
    effect_eligible = (
        result["isolated_effect_eligible"].fillna(False).astype(bool)
        if "isolated_effect_eligible" in result.columns
        else pd.Series(True, index=result.index)
    )
    result["_directional_normal"] = (
        result["v2_eligible"].astype(bool)
        & result["response_known"].astype(bool)
        & pd.to_numeric(result["string_current"], errors="coerce").gt(0)
        & pd.to_numeric(result["pvlof_score"], errors="coerce").notna()
        & pd.to_numeric(result["residual_ratio"], errors="coerce").notna()
        & pd.to_numeric(result["residual_median"], errors="coerce").notna()
        & pd.to_numeric(result["residual_ratio"], errors="coerce").lt(
            pd.to_numeric(result["residual_median"], errors="coerce")
        )
        & effect_eligible
    )
    return result


def _threshold_record(
    values: pd.Series,
    dates: pd.Series,
    *,
    quantile: float,
    minimum_samples: int,
    minimum_days: int,
    parent_threshold: float,
    shrinkage_k: float,
    level: str,
) -> dict[str, Any] | None:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    samples = int(len(values))
    days = int(pd.Series(dates).nunique())
    if samples < minimum_samples or days < minimum_days:
        return None
    raw = float(values.quantile(quantile))
    weight = float(samples / (samples + max(shrinkage_k, 0.0)))
    effective = float(weight * raw + (1.0 - weight) * parent_threshold)
    return {
        "raw_threshold": raw,
        "threshold": effective,
        "samples": samples,
        "days": days,
        "weight": weight,
        "parent_threshold": float(parent_threshold),
        "level": level,
    }


def _fit_group_records(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    quantile: float,
    minimum_samples: int,
    minimum_days: int,
    parent_lookup: Any,
    shrinkage_k: float,
    level: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key, group in frame.groupby(group_columns, observed=True, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_string = "|".join(str(value) for value in key_tuple)
        parent = float(parent_lookup(group, key_tuple))
        record = _threshold_record(
            group["pvlof_score"],
            group["event_date"],
            quantile=quantile,
            minimum_samples=minimum_samples,
            minimum_days=minimum_days,
            parent_threshold=parent,
            shrinkage_k=shrinkage_k,
            level=level,
        )
        if record is not None:
            records[key_string] = record
    return records


def fit_hierarchical_calibration(
    scored: pd.DataFrame,
    *,
    quantile: float = 0.995,
    minimum_device_samples: int = 10_000,
    minimum_device_days: int = 20,
    minimum_plant_count_samples: int = 20_000,
    minimum_plant_count_days: int = 20,
    minimum_count_samples: int = 20_000,
    minimum_count_days: int = 20,
    shrinkage_k: float = 5_000.0,
    minimum_consecutive: int = 2,
    expected_interval_minutes: int = 5,
    timezone: str = "Asia/Shanghai",
    version: str = "pvlof-v2-hier-v1",
) -> tuple[PVLOFV2HierCalibration, dict[str, Any]]:
    """Fit a hierarchical upper-tail threshold on directional LOF scores."""

    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be between 0.5 and 1")
    prepared = _prepare_scored(scored, timezone)
    eligible = prepared[prepared["_directional_normal"]].copy()
    if eligible.empty:
        raise ValueError("No directional eligible samples remain for hierarchical calibration")

    score_values = pd.to_numeric(eligible["pvlof_score"], errors="coerce")
    global_threshold = float(score_values.quantile(quantile))

    count_records = _fit_group_records(
        eligible,
        ["usable_string_count"],
        quantile=quantile,
        minimum_samples=minimum_count_samples,
        minimum_days=minimum_count_days,
        parent_lookup=lambda _group, _key: global_threshold,
        shrinkage_k=shrinkage_k,
        level="string_count",
    )

    def count_parent(group: pd.DataFrame, key: tuple[Any, ...]) -> float:
        record = count_records.get(str(key[1] if len(key) > 1 else key[0]))
        return float(record["threshold"] if record else global_threshold)

    plant_count_records = _fit_group_records(
        eligible,
        ["plant_id", "usable_string_count"],
        quantile=quantile,
        minimum_samples=minimum_plant_count_samples,
        minimum_days=minimum_plant_count_days,
        parent_lookup=count_parent,
        shrinkage_k=shrinkage_k,
        level="plant_string_count",
    )

    def device_parent(group: pd.DataFrame, _key: tuple[Any, ...]) -> float:
        plant = str(group["plant_id"].iloc[0])
        count = str(group["usable_string_count"].iloc[0])
        parent = plant_count_records.get(f"{plant}|{count}")
        if parent:
            return float(parent["threshold"])
        parent = count_records.get(count)
        return float(parent["threshold"] if parent else global_threshold)

    device_records = _fit_group_records(
        eligible,
        ["plant_id", "device_no"],
        quantile=quantile,
        minimum_samples=minimum_device_samples,
        minimum_days=minimum_device_days,
        parent_lookup=device_parent,
        shrinkage_k=shrinkage_k,
        level="device",
    )
    calibration = PVLOFV2HierCalibration(
        version=version,
        lof_quantile=quantile,
        minimum_device_samples=minimum_device_samples,
        minimum_device_days=minimum_device_days,
        minimum_plant_count_samples=minimum_plant_count_samples,
        minimum_plant_count_days=minimum_plant_count_days,
        minimum_count_samples=minimum_count_samples,
        minimum_count_days=minimum_count_days,
        shrinkage_k=shrinkage_k,
        minimum_consecutive=minimum_consecutive,
        expected_interval_minutes=expected_interval_minutes,
        global_threshold=global_threshold,
        device_thresholds=device_records,
        plant_string_count_thresholds=plant_count_records,
        string_count_thresholds=count_records,
    )
    report = {
        "version": calibration.version,
        "directional_samples": int(len(eligible)),
        "plants": int(eligible["plant_id"].nunique()),
        "devices": int(eligible[["plant_id", "device_no"]].drop_duplicates().shape[0]),
        "global_threshold": global_threshold,
        "device_threshold_count": len(device_records),
        "plant_string_count_threshold_count": len(plant_count_records),
        "string_count_threshold_count": len(count_records),
        "fallback_coverage": {
            "device": len(device_records),
            "plant_string_count": len(plant_count_records),
            "string_count": len(count_records),
            "global": 1,
        },
    }
    return calibration, report


def _add_hierarchical_consecutive(
    frame: pd.DataFrame,
    calibration: PVLOFV2HierCalibration,
) -> pd.DataFrame:
    result = frame.sort_values(
        ["plant_id", "device_no", "string_no", "event_time"]
    ).reset_index(drop=True).copy()
    result["isolated_hier_consecutive"] = 0
    result["isolated_hier_state"] = 0
    expected = np.timedelta64(calibration.expected_interval_minutes, "m")
    for _, positions in result.groupby(
        ["plant_id", "device_no", "string_no"], observed=True
    ).indices.items():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        active = False
        run = 0
        for index in indexes:
            current_time = result.at[index, "event_time"]
            direction = bool(result.at[index, "isolated_hier_directional_low"])
            entry = bool(result.at[index, "isolated_hier_raw_alert"])
            contiguous = previous_time is not None and current_time - previous_time == expected
            if entry:
                run = run + 1 if active and contiguous else 1
                active = True
            elif active and direction and contiguous:
                # A confirmed candidate may continue while the point remains
                # directionally low, even if its current LOF falls below the
                # entry threshold.
                run += 1
            else:
                active = False
                run = 0
            result.at[index, "isolated_hier_consecutive"] = run
            result.at[index, "isolated_hier_state"] = int(active)
            previous_time = current_time
    result["isolated_hier_alert"] = (
        result["isolated_hier_state"].astype(bool)
        & result["isolated_hier_consecutive"].ge(calibration.minimum_consecutive)
    ).astype(np.int8)
    return result


def _add_hierarchical_strict_consecutive(
    frame: pd.DataFrame,
    calibration: PVLOFV2HierCalibration,
) -> pd.DataFrame:
    """Require every consecutive point to pass the full LOF entry rule.

    This deliberately does not carry state on ``directional_low`` alone.  A
    point whose LOF has fallen below its context threshold cannot confirm or
    extend a strict isolated event.
    """

    result = frame.sort_values(
        ["plant_id", "device_no", "string_no", "event_time"]
    ).reset_index(drop=True).copy()
    result["isolated_hier_strict_consecutive"] = 0
    expected = np.timedelta64(calibration.expected_interval_minutes, "m")
    for _, positions in result.groupby(
        ["plant_id", "device_no", "string_no"], observed=True
    ).indices.items():
        indexes = np.asarray(positions, dtype=np.int64)
        previous_time: np.datetime64 | None = None
        run = 0
        for index in indexes:
            current_time = result.at[index, "event_time"]
            raw = bool(result.at[index, "isolated_hier_raw_alert"])
            contiguous = previous_time is not None and current_time - previous_time == expected
            if raw:
                run = run + 1 if contiguous else 1
            else:
                run = 0
            result.at[index, "isolated_hier_strict_consecutive"] = run
            previous_time = current_time
    result["isolated_hier_strict_alert"] = result[
        "isolated_hier_strict_consecutive"
    ].ge(calibration.minimum_consecutive).astype(np.int8)
    return result


def apply_hierarchical_isolated(
    scored: pd.DataFrame,
    calibration: PVLOFV2HierCalibration,
    *,
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Apply hierarchical isolated detection on an existing V2 score frame."""

    result = _prepare_scored(scored, timezone)

    def lookup(row: pd.Series) -> tuple[float, str, int]:
        device_key = f"{row['plant_id']}|{row['device_no']}"
        record = calibration.device_thresholds.get(device_key)
        if record:
            return float(record["threshold"]), "device", int(record["samples"])
        plant_count_key = f"{row['plant_id']}|{int(row['usable_string_count'])}"
        record = calibration.plant_string_count_thresholds.get(plant_count_key)
        if record:
            return float(record["threshold"]), "plant_string_count", int(record["samples"])
        count_key = str(int(row["usable_string_count"]))
        record = calibration.string_count_thresholds.get(count_key)
        if record:
            return float(record["threshold"]), "string_count", int(record["samples"])
        return calibration.global_threshold, "global", 0

    selected = result.apply(lookup, axis=1, result_type="expand")
    selected.columns = [
        "isolated_hier_lof_threshold",
        "isolated_hier_threshold_level",
        "isolated_hier_threshold_samples",
    ]
    result = pd.concat([result, selected], axis=1)
    score = pd.to_numeric(result["pvlof_score"], errors="coerce")
    result["isolated_hier_directional_low"] = (
        result["_directional_normal"].astype(bool)
    ).astype(np.int8)
    result["isolated_hier_raw_alert"] = (
        result["_directional_normal"].astype(bool)
        & score.ge(result["isolated_hier_lof_threshold"])
    ).astype(np.int8)
    result = _add_hierarchical_consecutive(result, calibration)
    result = _add_hierarchical_strict_consecutive(result, calibration)
    if "pvlof_v2_iso_mod_alert" not in result:
        raise ValueError("Base V2 frame is missing pvlof_v2_iso_mod_alert")
    result["pvlof_v2_hier_alert"] = (
        result["pvlof_v2_iso_mod_alert"].astype(bool)
        | result["isolated_hier_alert"].astype(bool)
    ).astype(np.int8)
    result["pvlof_v2_hier_strict_alert"] = (
        result["pvlof_v2_iso_mod_alert"].astype(bool)
        | result["isolated_hier_strict_alert"].astype(bool)
    ).astype(np.int8)
    if "zero_current_alert" in result:
        result["combined_hier_alert"] = (
            result["zero_current_alert"].astype(bool)
            | result["pvlof_v2_hier_alert"].astype(bool)
        ).astype(np.int8)
        result["combined_hier_strict_alert"] = (
            result["zero_current_alert"].astype(bool)
            | result["pvlof_v2_hier_strict_alert"].astype(bool)
        ).astype(np.int8)
    result = result.drop(columns=["event_date", "_directional_normal"], errors="ignore")
    return result.sort_values(
        ["event_time", "plant_id", "device_no", "string_no"]
    ).reset_index(drop=True)


def score_and_apply_hierarchical(
    frame: pd.DataFrame,
    base_calibration: PVLOFV2Calibration,
    hierarchical_calibration: PVLOFV2HierCalibration,
    *,
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Score with unchanged V2, then add only the hierarchical branch."""

    base = apply_pvlof_v2(frame, base_calibration)
    return apply_hierarchical_isolated(base, hierarchical_calibration, timezone=timezone)


__all__ = [
    "PVLOFV2HierCalibration",
    "apply_hierarchical_isolated",
    "exclude_alarm_windows",
    "fit_hierarchical_calibration",
    "load_hier_calibration",
    "save_hier_calibration",
    "score_and_apply_hierarchical",
]
