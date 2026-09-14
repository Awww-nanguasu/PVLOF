"""Build auditable, time-split candidate-normal power-prediction datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml


DEVICE_COLUMNS = [
    "event_time",
    "plant_id",
    "device_no",
    "active_power",
    "dc_power",
    "total_power",
    "rated_power",
    "device_temperature",
    "status_code",
    "main_string_count",
    "valid_current_string_count",
    "string_overall_status",
]

WEATHER_COLUMNS = [
    "time",
    "plant_id",
    "energy",
    "sensor_ghi",
    "sensor_temperature",
    "sensor_humidity",
    "sensor_wind_speed",
    "forecast_ghi",
    "forecast_temperature",
    "forecast_humidity",
    "forecast_wind_speed",
    "forecast_cloud_cover",
    "forecast_pressure",
    "forecast_rain",
    "forecast_visibility",
    "forecast_weather_code",
    "forecast_wind_direction",
]

NUMERIC_FIELDS = [
    "active_power",
    "dc_power",
    "total_power",
    "rated_power",
    "device_temperature",
    "status_code",
]

REQUIRED_NUMERIC_FIELDS = [
    "active_power",
    "dc_power",
    "rated_power",
    "device_temperature",
    "status_code",
]


@dataclass(frozen=True)
class TrainingConfig:
    timezone_name: str
    device_root: Path | tuple[Path, ...]
    weather_root: Path | tuple[Path, ...]
    aligned_output: Path
    transformer_output: Path
    audit_output: Path
    train_end: date
    validation_end: date
    test_end: date
    running_status_code: int
    interval_minutes: int
    maximum_power_ratio: float
    minimum_temperature: float
    maximum_temperature: float
    weather_tolerance_minutes: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        inputs, outputs = raw["inputs"], raw["outputs"]
        splits, quality = raw["splits"], raw["quality"]

        def input_paths(value: str | list[str]) -> Path | tuple[Path, ...]:
            if isinstance(value, list):
                if not value:
                    raise ValueError("Training input path list cannot be empty")
                return tuple(Path(item) for item in value)
            return Path(value)

        return cls(
            timezone_name=str(raw["timezone"]),
            device_root=input_paths(inputs["device"]),
            weather_root=input_paths(inputs["weather"]),
            aligned_output=Path(outputs["aligned"]),
            transformer_output=Path(outputs["transformer"]),
            audit_output=Path(outputs["audit"]),
            train_end=date.fromisoformat(str(splits["train_end_exclusive"])),
            validation_end=date.fromisoformat(str(splits["validation_end_exclusive"])),
            test_end=date.fromisoformat(str(splits["test_end_exclusive"])),
            running_status_code=int(quality["running_status_code"]),
            interval_minutes=int(quality["expected_interval_minutes"]),
            maximum_power_ratio=float(quality["maximum_power_to_rated_ratio"]),
            minimum_temperature=float(quality["minimum_device_temperature_exclusive"]),
            maximum_temperature=float(quality["maximum_device_temperature"]),
            weather_tolerance_minutes=int(quality["weather_tolerance_minutes"]),
        )


def _read_parts(roots: Path | tuple[Path, ...], columns: list[str]) -> pd.DataFrame:
    selected_roots = (roots,) if isinstance(roots, Path) else roots
    frames = []
    text_or_time = {"event_time", "time", "device_no"}
    for root in selected_roots:
        files = ds.dataset(root, format="parquet", partitioning="hive").files
        for path in files:
            part = pd.read_parquet(path, columns=columns)
            for column in set(columns) - text_or_time:
                part[column] = pd.to_numeric(part[column], errors="coerce").astype(
                    "float64"
                )
            frames.append(part)
    if not frames:
        raise FileNotFoundError(f"No Parquet files found under {selected_roots}")
    return pd.concat(frames, ignore_index=True)


def _split_for_dates(local_dates: pd.Series, config: TrainingConfig) -> pd.Series:
    values = np.select(
        [
            local_dates < config.train_end,
            local_dates < config.validation_end,
            local_dates < config.test_end,
        ],
        ["train", "validation", "test"],
        default="outside",
    )
    return pd.Series(values, index=local_dates.index, dtype="string")


def add_quality_and_targets(frame: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Add transparent quality flags, next-step target and leakage-safe split labels."""
    if "plant_id" in frame:
        plant = (
            pd.to_numeric(frame["plant_id"], errors="raise")
            .astype("Int64")
            .astype("string")
        )
        frame["device_key"] = plant + "::" + frame["device_no"].astype("string")
        identity = "device_key"
    else:
        identity = "device_no"
    frame = frame.sort_values([identity, "event_time"]).reset_index(drop=True)
    for field in NUMERIC_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["active_power_ratio"] = frame["active_power"] / frame["rated_power"]
    frame["dc_power_ratio"] = frame["dc_power"] / frame["rated_power"]
    frame["quality_running"] = frame["status_code"].eq(config.running_status_code)
    frame["quality_positive_power"] = frame["active_power"].gt(0) & frame["dc_power"].ge(0)
    frame["quality_power_range"] = (
        frame["rated_power"].gt(0)
        & frame["active_power_ratio"].ge(0)
        & frame["active_power_ratio"].le(config.maximum_power_ratio)
    )
    frame["quality_temperature"] = frame["device_temperature"].gt(
        config.minimum_temperature
    ) & frame["device_temperature"].le(config.maximum_temperature)
    frame["quality_finite"] = np.isfinite(frame[REQUIRED_NUMERIC_FIELDS]).all(axis=1)
    base_quality = frame[
        [
            "quality_running",
            "quality_positive_power",
            "quality_power_range",
            "quality_temperature",
            "quality_finite",
        ]
    ].all(axis=1)

    groups = frame.groupby(identity, observed=True, sort=False)
    frame["target_time"] = groups["event_time"].shift(-1)
    frame["target_active_power"] = groups["active_power"].shift(-1)
    target_quality = base_quality.astype("boolean").groupby(
        frame[identity], observed=True
    ).shift(-1)
    interval = frame["target_time"] - frame["event_time"]
    frame["quality_next_interval"] = interval.eq(
        pd.Timedelta(minutes=config.interval_minutes)
    )

    local_current = frame["event_time"].dt.tz_convert(config.timezone_name).dt.date
    local_target = frame["target_time"].dt.tz_convert(config.timezone_name).dt.date
    frame["split"] = _split_for_dates(local_current, config)
    target_split = _split_for_dates(local_target, config)
    frame["quality_same_split"] = frame["split"].eq(target_split)
    frame["candidate_normal"] = (
        base_quality
        & target_quality.fillna(False)
        & frame["quality_next_interval"]
        & frame["quality_same_split"]
        & frame["split"].ne("outside")
    )
    return frame


def _align_weather(device: pd.DataFrame, weather: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    device["event_time"] = pd.to_datetime(device["event_time"], errors="coerce", utc=True)
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce", utc=True)
    device["plant_id"] = pd.to_numeric(device["plant_id"], errors="raise").astype("int64")
    weather["plant_id"] = pd.to_numeric(weather["plant_id"], errors="raise").astype("int64")
    weather = weather.rename(columns={"time": "weather_time"})
    aligned = pd.merge_asof(
        device.sort_values(["event_time", "plant_id"]),
        weather.sort_values(["weather_time", "plant_id"]),
        left_on="event_time",
        right_on="weather_time",
        by="plant_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=config.weather_tolerance_minutes),
        allow_exact_matches=True,
    )
    aligned["weather_age_minutes"] = (
        aligned["event_time"] - aligned["weather_time"]
    ).dt.total_seconds() / 60
    aligned["weather_matched"] = aligned["weather_time"].notna()
    return aligned


def _add_time_features(frame: pd.DataFrame, timezone_name: str) -> None:
    local = frame["event_time"].dt.tz_convert(timezone_name)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    day_of_year = local.dt.dayofyear
    frame["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    frame["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    frame["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    frame["local_date"] = local.dt.date.astype(str)


def _ensure_empty_outputs(config: TrainingConfig) -> None:
    targets = [config.aligned_output, config.transformer_output]
    for target in targets:
        if target.exists() and any(target.rglob("*.parquet")):
            raise FileExistsError(f"Output already contains Parquet files: {target}")


def _write_outputs(frame: pd.DataFrame, config: TrainingConfig) -> dict[str, int]:
    _ensure_empty_outputs(config)
    aligned_counts: dict[str, int] = {}
    for local_date, subset in frame.groupby("local_date", sort=True):
        partition = config.aligned_output / f"date={local_date}"
        partition.mkdir(parents=True, exist_ok=True)
        subset.drop(columns=["local_date"]).to_parquet(
            partition / "part-00000.parquet",
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        aligned_counts[str(local_date)] = len(subset)

    config.transformer_output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        subset = frame[frame["candidate_normal"] & frame["split"].eq(split)].copy()
        subset.drop(columns=["local_date"]).to_parquet(
            config.transformer_output / f"{split}.parquet",
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
    return aligned_counts


def build_training_data(config: TrainingConfig) -> dict[str, Any]:
    device = _read_parts(config.device_root, DEVICE_COLUMNS)
    weather = _read_parts(config.weather_root, WEATHER_COLUMNS)
    duplicate_device = int(device.duplicated(["event_time", "plant_id", "device_no"]).sum())
    duplicate_weather = int(weather.duplicated(["time", "plant_id"]).sum())
    if duplicate_device or duplicate_weather:
        raise ValueError(
            f"Duplicate keys prevent safe alignment: device={duplicate_device}, weather={duplicate_weather}"
        )

    frame = _align_weather(device, weather, config)
    _add_time_features(frame, config.timezone_name)
    frame = add_quality_and_targets(frame, config)
    aligned_counts = _write_outputs(frame, config)

    split_counts = {}
    for split in ("train", "validation", "test"):
        mask = frame["split"].eq(split)
        split_counts[split] = {
            "all_rows": int(mask.sum()),
            "candidate_normal_rows": int((mask & frame["candidate_normal"]).sum()),
        }
    quality_failures = {
        column: int((~frame[column]).sum())
        for column in [
            "quality_running",
            "quality_positive_power",
            "quality_power_range",
            "quality_temperature",
            "quality_finite",
            "quality_next_interval",
            "quality_same_split",
        ]
    }
    per_plant: dict[str, Any] = {}
    for plant_id, plant_frame in frame.groupby("plant_id", observed=True):
        candidate = plant_frame["candidate_normal"]
        candidate_frame = plant_frame[candidate]
        per_plant[str(int(plant_id))] = {
            "all_rows": len(plant_frame),
            "devices": int(plant_frame["device_key"].nunique()),
            "candidate_normal_rows": int(candidate.sum()),
            "candidate_by_split": {
                str(split): int(count)
                for split, count in candidate_frame["split"].value_counts().items()
            },
            "numeric_nulls": {
                field: int(plant_frame[field].isna().sum()) for field in NUMERIC_FIELDS
            },
            "weather_matched_percent": round(
                float(plant_frame["weather_matched"].mean() * 100), 4
            ),
            "candidate_forecast_ghi_present_percent": (
                round(float(candidate_frame["forecast_ghi"].notna().mean() * 100), 4)
                if len(candidate_frame)
                else None
            ),
            "candidate_sensor_ghi_present_percent": (
                round(float(candidate_frame["sensor_ghi"].notna().mean() * 100), 4)
                if len(candidate_frame)
                else None
            ),
        }
    report = {
        "definition": {
            "target": "next 5-minute active_power for the same device",
            "candidate_normal_excludes_low_current_count": True,
            "optional_nullable_fields": ["total_power"],
            "weather_join": "backward as-of by plant_id with 15-minute tolerance",
            "forecast_leakage_warning": (
                "Forecast fields are retained but must not be model features until their issue-time "
                "and backfill semantics are confirmed."
            ),
        },
        "input": {
            "device_rows": len(device),
            "weather_rows": len(weather),
            "device_roots": [
                str(path)
                for path in (
                    (config.device_root,)
                    if isinstance(config.device_root, Path)
                    else config.device_root
                )
            ],
            "weather_roots": [
                str(path)
                for path in (
                    (config.weather_root,)
                    if isinstance(config.weather_root, Path)
                    else config.weather_root
                )
            ],
            "device_schema_normalized_during_processing": True,
        },
        "output": {
            "aligned_rows": len(frame),
            "aligned_date_partitions": len(aligned_counts),
            "candidate_normal_rows": int(frame["candidate_normal"].sum()),
            "split_counts": split_counts,
        },
        "quality_failures": quality_failures,
        "per_plant": per_plant,
        "weather": {
            "matched_percent": round(float(frame["weather_matched"].mean() * 100), 4),
            "sensor_ghi_null_percent": round(float(frame["sensor_ghi"].isna().mean() * 100), 4),
            "forecast_ghi_null_percent": round(float(frame["forecast_ghi"].isna().mean() * 100), 4),
        },
        "weak_status_reference": {
            "string_overall_status_not_1_rows": int(frame["string_overall_status"].ne(1).sum()),
            "used_as_hard_filter": False,
        },
        "aligned_rows_by_date": aligned_counts,
    }
    config.audit_output.parent.mkdir(parents=True, exist_ok=True)
    config.audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
