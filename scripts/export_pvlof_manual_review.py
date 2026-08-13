"""Export a wide inverter-time CSV for manual Baseline/PVLOF review."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_PATTERN = re.compile(r"^string_current_(\d{2})$")
PREDICTION_KEYS = ["plant_id", "device_no", "event_time"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    return pd.concat(
        [pd.read_parquet(file, columns=columns) for file in files],
        ignore_index=True,
    )


def _normalize_source(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "alarm_event_id",
        "plant_id",
        "device_no",
        "event_time",
        "main_string_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Alarm production currents are missing columns: {missing}")
    result = frame.copy()
    result["alarm_event_id"] = result["alarm_event_id"].astype(str)
    result["plant_id"] = pd.to_numeric(result["plant_id"], errors="raise").astype(int)
    result["device_no"] = result["device_no"].astype(str).str.strip()
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    result["main_string_count"] = pd.to_numeric(
        result["main_string_count"], errors="coerce"
    ).astype("Int64")
    duplicate = result.duplicated(
        ["alarm_event_id", "plant_id", "device_no", "event_time"]
    )
    if duplicate.any():
        result = result.loc[~duplicate].copy()
    return result.reset_index(drop=True)


def _current_columns(frame: pd.DataFrame) -> list[tuple[int, str]]:
    columns: list[tuple[int, str]] = []
    for column in frame.columns:
        match = CURRENT_PATTERN.match(column)
        if match:
            columns.append((int(match.group(1)), column))
    columns.sort()
    if not columns:
        raise ValueError("No string_current_XX columns were found")
    return columns


def _blank_unconfigured_currents(
    frame: pd.DataFrame,
    current_columns: list[tuple[int, str]],
) -> pd.DataFrame:
    result = frame.copy()
    main_count = result["main_string_count"]
    for string_no, column in current_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        unconfigured = main_count.notna() & main_count.lt(string_no)
        result.loc[unconfigured, column] = np.nan
    return result


def _read_events(path: str | Path, timezone: str) -> pd.DataFrame:
    events = _read_parquet(path)
    required = {"alarm_event_id", "raise_time", "end_time"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Cleaned alarm events are missing columns: {missing}")
    if "classification" in events.columns:
        events = events[events["classification"].eq("complete")].copy()
    events["alarm_event_id"] = events["alarm_event_id"].astype(str)
    events["raise_time_local"] = pd.to_datetime(
        events["raise_time"], errors="raise", utc=True
    ).dt.tz_convert(timezone)
    events["end_time_local"] = pd.to_datetime(
        events["end_time"], errors="raise", utc=True
    ).dt.tz_convert(timezone)
    return events[
        ["alarm_event_id", "raise_time_local", "end_time_local"]
    ].drop_duplicates("alarm_event_id")


def _read_pvlof_alerts(
    path: str | Path,
    *,
    prediction_column: str = "pvlof_alert",
    output_column: str = "_pvlof_strings",
) -> pd.DataFrame:
    columns = [
        "plant_id",
        "device_no",
        "event_time",
        "string_no",
        "string_current",
        prediction_column,
    ]
    predictions = _read_parquet(path, columns=columns)
    predictions["plant_id"] = pd.to_numeric(
        predictions["plant_id"], errors="raise"
    ).astype(int)
    predictions["device_no"] = predictions["device_no"].astype(str).str.strip()
    predictions["event_time"] = pd.to_datetime(
        predictions["event_time"], errors="raise", utc=True
    )
    predictions["string_no"] = pd.to_numeric(
        predictions["string_no"], errors="raise"
    ).astype(int)
    predictions["string_current"] = pd.to_numeric(
        predictions["string_current"], errors="coerce"
    )
    predictions = predictions[
        predictions[prediction_column].astype(bool)
        & predictions["string_current"].gt(0)
    ].copy()
    if predictions.empty:
        return pd.DataFrame(columns=PREDICTION_KEYS + [output_column])
    return (
        predictions.groupby(PREDICTION_KEYS, observed=True)["string_no"]
        .agg(lambda values: tuple(sorted(set(int(value) for value in values))))
        .rename(output_column)
        .reset_index()
    )


def _format_pvlof_strings(values: object, main_string_count: object) -> str:
    if not isinstance(values, tuple):
        return "FALSE"
    maximum = int(main_string_count) if pd.notna(main_string_count) else None
    valid = [value for value in values if maximum is None or value <= maximum]
    return ",".join(f"{value:02d}" for value in valid) if valid else "FALSE"


def export_manual_review(
    currents: str | Path,
    predictions: str | Path | None,
    alarm_events: str | Path,
    output: str | Path,
    *,
    report: str | Path | None = None,
    timezone: str = "Asia/Shanghai",
    prediction_column: str = "pvlof_alert",
    predictions_v1: str | Path | None = None,
    predictions_v2: str | Path | None = None,
    predictions_v2_mod: str | Path | None = None,
    predictions_v2_iso_mod: str | Path | None = None,
    predictions_v2_hier: str | Path | None = None,
    predictions_v2_hier_strict: str | Path | None = None,
) -> dict[str, Any]:
    source = _normalize_source(_read_parquet(currents))
    numbered_currents = _current_columns(source)
    source = _blank_unconfigured_currents(source, numbered_currents)
    source = source.merge(_read_events(alarm_events, timezone), on="alarm_event_id")
    source["event_time_local"] = source["event_time"].dt.tz_convert(timezone)
    source["Baseline"] = "TRUE"
    comparison_mode = any(
        path is not None
        for path in (
            predictions_v1,
            predictions_v2,
            predictions_v2_mod,
            predictions_v2_iso_mod,
            predictions_v2_hier,
            predictions_v2_hier_strict,
        )
    )
    prediction_definitions: dict[str, str] = {}
    if comparison_mode:
        if predictions_v1 is None or predictions_v2 is None:
            raise ValueError("Both predictions_v1 and predictions_v2 are required")
        source = source.merge(
            _read_pvlof_alerts(
                predictions_v1,
                prediction_column="pvlof_alert",
                output_column="_pvlof_v1_strings",
            ),
            on=PREDICTION_KEYS,
            how="left",
        )
        source = source.merge(
            _read_pvlof_alerts(
                predictions_v2,
                prediction_column="pvlof_v2_alert",
                output_column="_pvlof_v2_strings",
            ),
            on=PREDICTION_KEYS,
            how="left",
        )
        prediction_sources = [
            ("PVLOF_V1", "_pvlof_v1_strings"),
            ("PVLOF_V2", "_pvlof_v2_strings"),
        ]
        if predictions_v2_mod is not None:
            source = source.merge(
                _read_pvlof_alerts(
                    predictions_v2_mod,
                    prediction_column="pvlof_v2_alert",
                    output_column="_pvlof_v2_mod_strings",
                ),
                on=PREDICTION_KEYS,
                how="left",
            )
            prediction_sources.append(("PVLOF_V2_mod", "_pvlof_v2_mod_strings"))
        if predictions_v2_iso_mod is not None:
            source = source.merge(
                _read_pvlof_alerts(
                    predictions_v2_iso_mod,
                    prediction_column="pvlof_v2_iso_mod_alert",
                    output_column="_pvlof_v2_iso_mod_strings",
                ),
                on=PREDICTION_KEYS,
                how="left",
            )
            prediction_sources.append(("PVLOF_V2_iso_mod", "_pvlof_v2_iso_mod_strings"))
        if predictions_v2_hier is not None:
            source = source.merge(
                _read_pvlof_alerts(
                    predictions_v2_hier,
                    prediction_column="pvlof_v2_hier_alert",
                    output_column="_pvlof_v2_hier_strings",
                ),
                on=PREDICTION_KEYS,
                how="left",
            )
            prediction_sources.append(("PVLOF_V2_hier", "_pvlof_v2_hier_strings"))
        if predictions_v2_hier_strict is not None:
            source = source.merge(
                _read_pvlof_alerts(
                    predictions_v2_hier_strict,
                    prediction_column="pvlof_v2_hier_strict_alert",
                    output_column="_pvlof_v2_hier_strict_strings",
                ),
                on=PREDICTION_KEYS,
                how="left",
            )
            prediction_sources.append(
                ("PVLOF_V2_hier_strict", "_pvlof_v2_hier_strict_strings")
            )
        for version, values_column in prediction_sources:
            source[version] = [
                _format_pvlof_strings(values, main_count)
                for values, main_count in zip(
                    source[values_column], source["main_string_count"], strict=True
                )
            ]
        result_columns = [column for column, _ in prediction_sources]
        prediction_definitions = {
            "PVLOF_V1": "nonzero final pvlof_alert string numbers, otherwise FALSE",
            "PVLOF_V2": "nonzero final pvlof_v2_alert string numbers, otherwise FALSE",
        }
        if predictions_v2_mod is not None:
            prediction_definitions["PVLOF_V2_mod"] = (
                "group-level-continuity pvlof_v2_alert string numbers, otherwise FALSE"
            )
        if predictions_v2_iso_mod is not None:
            prediction_definitions["PVLOF_V2_iso_mod"] = (
                "group-continuity plus directional-isolated LOF string numbers, otherwise FALSE"
            )
        if predictions_v2_hier is not None:
            prediction_definitions["PVLOF_V2_hier"] = (
                "hierarchical-threshold isolated LOF plus PVLOF_V2_iso_mod string numbers, otherwise FALSE"
            )
        if predictions_v2_hier_strict is not None:
            prediction_definitions["PVLOF_V2_hier_strict"] = (
                "hierarchical-threshold isolated LOF with strict consecutive points plus PVLOF_V2_iso_mod string numbers, otherwise FALSE"
            )
    else:
        if predictions is None:
            raise ValueError("predictions is required outside V1/V2 comparison mode")
        source = source.merge(
            _read_pvlof_alerts(predictions, prediction_column=prediction_column),
            on=PREDICTION_KEYS,
            how="left",
        )
        source["PVLOF"] = [
            _format_pvlof_strings(values, main_count)
            for values, main_count in zip(
                source["_pvlof_strings"], source["main_string_count"], strict=True
            )
        ]
        result_columns = ["PVLOF"]
        prediction_definitions = {
            "PVLOF": f"nonzero final {prediction_column} string numbers, otherwise FALSE"
        }
    source["manual_label"] = ""
    source["manual_note"] = ""
    source = source.sort_values(
        ["plant_id", "device_no", "alarm_event_id", "event_time"]
    ).reset_index(drop=True)
    source.insert(0, "row", np.arange(1, len(source) + 1, dtype=np.int64))

    current_names = [column for _, column in numbered_currents]
    output_columns = [
        "row",
        "alarm_event_id",
        "plant_id",
        "device_no",
        "raise_time_local",
        "end_time_local",
        "event_time_local",
        *current_names,
        "Baseline",
        *result_columns,
        "manual_label",
        "manual_note",
    ]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    current_headers = {
        column: f"{string_no:02d}" for string_no, column in numbered_currents
    }
    source[output_columns].rename(columns=current_headers).to_csv(
        destination, index=False, encoding="utf-8-sig"
    )

    current_values = source[current_names].to_numpy(dtype=np.float64)
    prediction_counts = {
        column: {
            "true_rows": int(source[column].ne("FALSE").sum()),
            "false_rows": int(source[column].eq("FALSE").sum()),
        }
        for column in result_columns
    }
    summary = {
        "currents": str(currents),
        "predictions": (
            {
                "v1": str(predictions_v1),
                "v2": str(predictions_v2),
                **(
                    {"v2_mod": str(predictions_v2_mod)}
                    if predictions_v2_mod is not None
                    else {}
                ),
            }
            if comparison_mode
            else str(predictions)
        ),
        "alarm_events": str(alarm_events),
        "output": str(destination),
        "rows": int(len(source)),
        "events": int(source["alarm_event_id"].nunique()),
        "plants": int(source["plant_id"].nunique()),
        "devices": int(source["device_no"].nunique()),
        "current_columns": len(current_names),
        "baseline_true_rows": int(len(source)),
        "prediction_counts": prediction_counts,
        "zero_current_values": int(np.sum(np.isfinite(current_values) & (current_values <= 0))),
        "baseline_definition": "inside a cleaned device_alarm 101001 interval",
        "prediction_definitions": prediction_definitions,
    }
    report_path = Path(report) if report else destination.with_suffix(".summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--currents",
        default=(
            "data/processed/pvlof/alarm_windows_v1/"
            "alarm_production_currents.parquet"
        ),
    )
    parser.add_argument(
        "--predictions",
        default="artifacts/models/pvlof/alarm_windows_v1/pvlof_points.parquet",
    )
    parser.add_argument(
        "--alarm-events",
        default="artifacts/reports/pvlof_cleaning_v2/cleaned_alarm_events.parquet",
    )
    parser.add_argument(
        "--output",
        default="artifacts/reports/pvlof_manual_review_wide_v2.csv",
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--prediction-column", default="pvlof_alert")
    parser.add_argument("--predictions-v1")
    parser.add_argument("--predictions-v2")
    parser.add_argument("--predictions-v2-mod")
    parser.add_argument("--predictions-v2-iso-mod")
    parser.add_argument("--predictions-v2-hier")
    parser.add_argument("--predictions-v2-hier-strict")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    export_manual_review(
        args.currents,
        args.predictions,
        args.alarm_events,
        args.output,
        report=args.report,
        timezone=args.timezone,
        prediction_column=args.prediction_column,
        predictions_v1=args.predictions_v1,
        predictions_v2=args.predictions_v2,
        predictions_v2_mod=args.predictions_v2_mod,
        predictions_v2_iso_mod=args.predictions_v2_iso_mod,
        predictions_v2_hier=args.predictions_v2_hier,
        predictions_v2_hier_strict=args.predictions_v2_hier_strict,
    )


if __name__ == "__main__":
    main()
