"""Data-quality and algorithm-feasibility audit for exported ES samples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pv_anomaly.settings import DataSettings


def load_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix != ".json":
        raise ValueError("Supported inputs are .json, .jsonl, .ndjson and .csv")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "hits" in payload:
        hits = payload.get("hits", {}).get("hits", [])
        records = [item.get("_source", {}) for item in hits]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("JSON must be an array or an Elasticsearch search response")
    return pd.json_normalize(records)


def _first_existing(columns: list[str], aliases: list[str]) -> str | None:
    return next((name for name in aliases if name in columns), None)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def audit_dataframe(frame: pd.DataFrame, config: DataSettings) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    resolved = {
        logical: _first_existing(columns, aliases) for logical, aliases in config.fields.items()
    }
    string_currents = sorted(
        column
        for column in columns
        if any(re.search(pattern, column, flags=re.IGNORECASE) for pattern in config.string_current_patterns)
    )
    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": len(columns),
        "column_names": columns,
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "null_percent": {
            column: round(float(value), 4)
            for column, value in (frame.isna().mean() * 100).sort_values(ascending=False).items()
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "resolved_fields": resolved,
        "string_current_fields": string_currents,
    }

    timestamp_name = resolved.get("timestamp")
    if timestamp_name:
        timestamps = pd.to_datetime(frame[timestamp_name], errors="coerce", utc=True)
        valid = timestamps.dropna().sort_values()
        positive_deltas = valid.diff().dropna().dt.total_seconds()
        positive_deltas = positive_deltas[positive_deltas > 0]
        report["timestamp"] = {
            "field": timestamp_name,
            "invalid_or_missing": int(timestamps.isna().sum()),
            "start_utc": valid.min().isoformat() if not valid.empty else None,
            "end_utc": valid.max().isoformat() if not valid.empty else None,
            "median_interval_seconds": (
                float(positive_deltas.median()) if not positive_deltas.empty else None
            ),
            "most_common_interval_seconds": (
                float(positive_deltas.mode().iloc[0]) if not positive_deltas.empty else None
            ),
        }
    else:
        report["timestamp"] = {"field": None, "error": "No timestamp alias was found"}

    numeric = frame.select_dtypes(include="number")
    report["numeric_summary"] = {
        column: {
            key: _json_value(value)
            for key, value in numeric[column].describe(percentiles=[0.05, 0.5, 0.95]).items()
        }
        for column in numeric.columns
    }
    has_power = resolved.get("power") is not None
    has_weather = any(
        resolved.get(name) for name in ("irradiance", "module_temperature", "ambient_temperature")
    )
    comparison_keys = [
        name for name in ("station_id", "inverter_id", "mppt_id") if resolved.get(name)
    ]
    report["feasibility"] = {
        "transformer_power_prediction": {
            "candidate": bool(timestamp_name and has_power and has_weather),
            "requires": "timestamp, power, weather/history fields, sufficient normal history",
        },
        "ewma_residual_detection": {
            "candidate": bool(timestamp_name and has_power),
            "requires": "actual power, model prediction, confirmed sampling interval",
        },
        "pvlof_string_localization": {
            "candidate": bool(timestamp_name and len(string_currents) >= 2 and comparison_keys),
            "comparison_keys_found": comparison_keys,
            "requires": "multiple string currents and station/inverter/MPPT topology",
        },
    }
    return report


def audit_file(path: str | Path, config_path: str | Path) -> dict[str, Any]:
    return audit_dataframe(load_records(path), DataSettings.from_yaml(config_path))

