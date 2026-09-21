"""Build comparable current-time and plus-15-minute forecast-GHI features."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")
VARIANTS = ("current", "plus15")


def read_weather_forecast(
    roots: str | Path | Sequence[str | Path],
) -> pd.DataFrame:
    selected_roots = (
        [roots]
        if isinstance(roots, (str, Path))
        else list(roots)
    )
    if not selected_roots:
        raise ValueError("At least one weather directory is required")
    frames = []
    for root in selected_roots:
        source = Path(root)
        files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No weather Parquet files found under {source}")
        for path in files:
            part = pd.read_parquet(path, columns=["time", "plant_id", "forecast_ghi"])
            part["forecast_ghi"] = pd.to_numeric(
                part["forecast_ghi"], errors="coerce"
            ).astype("float64")
            frames.append(part)
    if not frames:
        raise FileNotFoundError(f"No weather Parquet files found under {selected_roots}")
    weather = pd.concat(frames, ignore_index=True)
    weather["time"] = pd.to_datetime(weather["time"], errors="raise", utc=True)
    weather["plant_id"] = pd.to_numeric(weather["plant_id"], errors="raise").astype("int64")
    if weather.duplicated(["time", "plant_id"]).any():
        raise ValueError("Weather forecast contains duplicate time + plant_id keys")
    return weather


def align_forecast_ghi(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    variant: str,
    tolerance_minutes: int = 15,
) -> pd.DataFrame:
    """Align the latest current slot or the slot interpreted as valid 15 minutes later."""
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported forecast variant: {variant}")
    left = frame.copy()
    left["event_time"] = pd.to_datetime(left["event_time"], errors="raise", utc=True)
    left["plant_id"] = pd.to_numeric(left["plant_id"], errors="raise").astype("int64")
    left["_row_order"] = np.arange(len(left))

    right = weather[["time", "plant_id", "forecast_ghi"]].copy()
    right = right.rename(
        columns={
            "time": "forecast_ghi_source_time",
            "forecast_ghi": "_forecast_ghi_aligned",
        }
    )
    shift = 15 if variant == "plus15" else 0
    right["_forecast_lookup_time"] = right["forecast_ghi_source_time"] - pd.Timedelta(
        minutes=shift
    )
    aligned = pd.merge_asof(
        left.sort_values(["event_time", "plant_id"]),
        right.sort_values(["_forecast_lookup_time", "plant_id"]),
        left_on="event_time",
        right_on="_forecast_lookup_time",
        by="plant_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
        allow_exact_matches=True,
    )
    aligned = aligned.sort_values("_row_order").drop(
        columns=["_row_order", "_forecast_lookup_time"]
    )
    if len(aligned) != len(frame):
        raise AssertionError("Forecast alignment changed the number of modeling rows")
    aligned["forecast_ghi_available"] = aligned["_forecast_ghi_aligned"].notna().astype("float32")
    aligned["forecast_ghi_offset_minutes"] = (
        aligned["forecast_ghi_source_time"] - aligned["event_time"]
    ).dt.total_seconds() / 60
    return aligned


def impute_forecast_from_training_slots(
    frames: dict[str, pd.DataFrame],
    *,
    timezone: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Impute missing GHI by train-only minute-of-day medians and add availability."""

    def minute_of_day(frame: pd.DataFrame) -> pd.Series:
        local = frame["event_time"].dt.tz_convert(timezone)
        return local.dt.hour * 60 + local.dt.minute

    train = frames["train"]
    train_minutes = minute_of_day(train)
    train_values = train["_forecast_ghi_aligned"]
    slot_medians = train_values.groupby(train_minutes).median()
    global_median = float(train_values.median())
    if not np.isfinite(global_median):
        raise ValueError("Training forecast_ghi contains no finite values")

    completed: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "imputation": "train-only local-minute median, then train global median",
        "train_global_median": global_median,
        "train_time_slots": int(slot_medians.notna().sum()),
        "splits": {},
    }
    for split, source in frames.items():
        result = source.copy()
        fallback = minute_of_day(result).map(slot_medians).fillna(global_median)
        result["forecast_ghi_feature"] = (
            result["_forecast_ghi_aligned"].fillna(fallback).astype("float32")
        )
        result = result.drop(columns=["_forecast_ghi_aligned"])
        if not np.isfinite(result["forecast_ghi_feature"]).all():
            raise ValueError(f"Non-finite forecast GHI remains in {split}")
        available = result["forecast_ghi_available"].astype(bool)
        report["splits"][split] = {
            "rows": len(result),
            "available_rows": int(available.sum()),
            "available_percent": round(float(available.mean() * 100), 4),
            "source_offset_minutes_min": (
                float(result.loc[available, "forecast_ghi_offset_minutes"].min())
                if available.any()
                else None
            ),
            "source_offset_minutes_max": (
                float(result.loc[available, "forecast_ghi_offset_minutes"].max())
                if available.any()
                else None
            ),
        }
        completed[split] = result
    return completed, report


def build_forecast_variants(
    *,
    base_directory: str | Path,
    weather_directory: str | Path | Sequence[str | Path],
    output_root: str | Path,
    report_path: str | Path,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    base = Path(base_directory)
    output = Path(output_root)
    targets = {variant: output / f"transformer_forecast_{variant}" for variant in VARIANTS}
    for target in targets.values():
        if target.exists() and any(target.glob("*.parquet")):
            raise FileExistsError(f"Forecast variant output already exists: {target}")

    base_frames = {split: pd.read_parquet(base / f"{split}.parquet") for split in SPLITS}
    weather = read_weather_forecast(weather_directory)
    report: dict[str, Any] = {
        "definition": {
            "current": "latest weather slot at or before event_time",
            "plus15": (
                "weather source time shifted back 15 minutes before as-of alignment; "
                "experimental assumption that the plus-15-minute value was available"
            ),
            "same_modeling_rows_across_variants": True,
            "timezone": timezone,
        },
        "weather_directories": [
            str(path)
            for path in (
                [weather_directory]
                if isinstance(weather_directory, (str, Path))
                else weather_directory
            )
        ],
        "variants": {},
    }
    for variant in VARIANTS:
        aligned = {
            split: align_forecast_ghi(frame, weather, variant=variant)
            for split, frame in base_frames.items()
        }
        completed, variant_report = impute_forecast_from_training_slots(
            aligned,
            timezone=timezone,
        )
        target = targets[variant]
        target.mkdir(parents=True, exist_ok=True)
        for split, frame in completed.items():
            frame.to_parquet(target / f"{split}.parquet", index=False)
        variant_report["output_directory"] = str(target)
        report["variants"][variant] = variant_report

    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
