"""Align PVLOF predictions with inverter alarm events."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pv_anomaly.alarm_time import alarm_time_grid
from pv_anomaly.ewma_audit import binary_metrics


STRING_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")
DEFAULT_ALARM_CODES = ("101001", "101002")


def parse_alarm_string(value: object) -> tuple[int, ...]:
    """Parse a CSV alarm_string value into sorted, unique string numbers."""

    if value is None or pd.isna(value):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    numbers = {
        int(match.group(1))
        for match in STRING_NUMBER_PATTERN.finditer(text)
        if 1 <= int(match.group(1)) <= 100
    }
    return tuple(sorted(numbers))


def load_device_manifest(path: str | Path) -> pd.DataFrame:
    """Load ``{station_name: {devices: [...]}}`` into a device lookup table."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stations = payload.get("stations", payload)
    rows: list[dict[str, str]] = []
    for station_name, value in stations.items():
        devices = value.get("devices", value) if isinstance(value, dict) else value
        if not isinstance(devices, list):
            raise ValueError(f"Manifest devices for {station_name!r} must be a list")
        for device_no in devices:
            rows.append({"station_name": str(station_name), "device_no": str(device_no)})
    manifest = pd.DataFrame(rows).drop_duplicates()
    if manifest.empty:
        raise ValueError("Device manifest is empty")
    duplicates = manifest.duplicated("device_no", keep=False)
    if duplicates.any():
        examples = manifest.loc[duplicates].to_dict("records")[:10]
        raise ValueError(f"A device_no belongs to multiple stations: {examples}")
    return manifest.sort_values(["station_name", "device_no"]).reset_index(drop=True)


def _epoch_milliseconds(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)


def _filter_interval(
    frame: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
    timezone: str,
) -> pd.DataFrame:
    result = frame.copy()
    if start is None and end is None:
        return result
    start_time = _bound_to_utc(start, timezone) if start else None
    end_time = _bound_to_utc(end, timezone) if end else None
    mask = pd.Series(True, index=result.index)
    if start_time is not None:
        mask &= result["end_time"] >= start_time
    if end_time is not None:
        mask &= result["raise_time"] < end_time
    return result.loc[mask].reset_index(drop=True)


def _bound_to_utc(value: str, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    return timestamp.tz_convert("UTC")


def read_alarm_events(
    path: str | Path,
    manifest: pd.DataFrame,
    *,
    alarm_codes: Iterable[str] = DEFAULT_ALARM_CODES,
    start: str | None = None,
    end: str | None = None,
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read alarm rows, attach station validation, and expand string numbers."""

    source = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {"id", "alarm_code", "alarm_name", "alarm_string", "device_no", "station_name"}
    required |= {"raise_time", "end_time"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Alarm CSV is missing columns: {missing}")
    source["device_no"] = source["device_no"].astype(str)
    source["station_name"] = source["station_name"].astype(str)
    source["alarm_code"] = source["alarm_code"].astype(str)
    selected_codes = {str(code) for code in alarm_codes}
    source = source[source["alarm_code"].isin(selected_codes)].copy()
    source["raise_time"] = _epoch_milliseconds(source["raise_time"])
    source["end_time"] = _epoch_milliseconds(source["end_time"])
    source = source.dropna(subset=["raise_time", "end_time"])
    source = source[source["end_time"] >= source["raise_time"]]
    source = _filter_interval(source, start=start, end=end, timezone=timezone)
    source = source.merge(
        manifest.rename(columns={"station_name": "manifest_station_name"}),
        on="device_no",
        how="left",
    )
    source["device_station_match"] = source["manifest_station_name"].eq(
        source["station_name"]
    )
    source["device_known"] = source["manifest_station_name"].notna()

    records: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        string_numbers = parse_alarm_string(row.alarm_string)
        scope = "string" if string_numbers else "device"
        for string_no in string_numbers or (pd.NA,):
            records.append(
                {
                    "alarm_event_id": str(row.id),
                    "station_name": row.station_name,
                    "device_no": row.device_no,
                    "alarm_code": row.alarm_code,
                    "alarm_name": row.alarm_name,
                    "alarm_string": row.alarm_string,
                    "string_no": string_no,
                    "label_scope": scope,
                    "raise_time": row.raise_time,
                    "end_time": row.end_time,
                    "device_known": bool(row.device_known),
                    "device_station_match": bool(row.device_station_match),
                }
            )
    events = pd.DataFrame(records)
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "alarm_event_id",
                "station_name",
                "device_no",
                "alarm_code",
                "alarm_name",
                "alarm_string",
                "string_no",
                "label_scope",
                "raise_time",
                "end_time",
                "device_known",
                "device_station_match",
            ]
        )
    else:
        events["string_no"] = pd.to_numeric(events["string_no"], errors="coerce").astype("Int64")
        events = events.sort_values(
            ["station_name", "device_no", "raise_time", "alarm_event_id", "string_no"]
        ).reset_index(drop=True)
    report = {
        "path": str(path),
        "alarm_codes": sorted(selected_codes),
        "timezone": timezone,
        "source_rows": int(len(source)),
        "expanded_event_rows": int(len(events)),
        "unique_alarm_ids": int(events["alarm_event_id"].nunique()) if len(events) else 0,
        "string_scope_rows": int((events["label_scope"] == "string").sum()) if len(events) else 0,
        "device_scope_rows": int((events["label_scope"] == "device").sum()) if len(events) else 0,
        "unknown_devices": int((~events["device_known"]).sum()) if len(events) else 0,
        "station_mismatches": int(
            (events["device_known"] & ~events["device_station_match"]).sum()
        )
        if len(events)
        else 0,
        "time_start": events["raise_time"].min().isoformat() if len(events) else None,
        "time_end": events["end_time"].max().isoformat() if len(events) else None,
    }
    return events, report


def expand_alarm_points(
    events: pd.DataFrame,
    *,
    interval_minutes: int = 5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand alarm intervals to timestamps on the five-minute grid."""

    if interval_minutes < 1:
        raise ValueError("interval_minutes must be positive")
    rows: list[dict[str, Any]] = []
    events_without_grid_points = 0
    for row in events.itertuples(index=False):
        grid = alarm_time_grid(
            row.raise_time,
            row.end_time,
            interval_minutes=interval_minutes,
        )
        if grid.empty:
            events_without_grid_points += 1
            continue
        for timestamp in grid:
            rows.append(
                {
                    "alarm_event_id": row.alarm_event_id,
                    "station_name": row.station_name,
                    "device_no": row.device_no,
                    "alarm_code": row.alarm_code,
                    "alarm_name": row.alarm_name,
                    "string_no": row.string_no,
                    "label_scope": row.label_scope,
                    "event_time": timestamp,
                }
            )
    points = pd.DataFrame(rows)
    if points.empty:
        points = pd.DataFrame(
            columns=[
                "alarm_event_id",
                "station_name",
                "device_no",
                "alarm_code",
                "alarm_name",
                "string_no",
                "label_scope",
                "event_time",
            ]
        )
    else:
        points["string_no"] = pd.to_numeric(points["string_no"], errors="coerce").astype("Int64")
        points = points.drop_duplicates(
            ["station_name", "device_no", "string_no", "event_time", "alarm_code"]
        ).sort_values(["event_time", "station_name", "device_no", "string_no"])
        points = points.reset_index(drop=True)
    return points, {
        "label_points": int(len(points)),
        "events_without_grid_points": events_without_grid_points,
    }


def _read_parquet(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def read_pvlof_points(
    path: str | Path,
    manifest: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read PVLOF point output and attach validated station names."""

    points = _read_parquet(path)
    required = {"event_time", "device_no", "string_no"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"PVLOF output is missing columns: {missing}")
    points["event_time"] = pd.to_datetime(points["event_time"], errors="raise", utc=True)
    points["device_no"] = points["device_no"].astype(str)
    if start is not None:
        points = points[points["event_time"] >= _bound_to_utc(start, timezone)]
    if end is not None:
        points = points[points["event_time"] < _bound_to_utc(end, timezone)]
    points = points.merge(manifest, on="device_no", how="left", suffixes=("", "_manifest"))
    unknown_devices = points["station_name"].isna()
    if "station_name" in points.columns and "station_name_manifest" in points.columns:
        points["station_name_match"] = points["station_name"].eq(
            points["station_name_manifest"]
        )
        points["station_name"] = points["station_name"].fillna(points["station_name_manifest"])
        points = points.drop(columns=["station_name_manifest"])
    else:
        points["station_name_match"] = True
    points["string_no"] = pd.to_numeric(points["string_no"], errors="coerce").astype("Int64")
    points = points.dropna(subset=["station_name", "string_no"])
    points = points.sort_values(["event_time", "station_name", "device_no", "string_no"])
    points = points.reset_index(drop=True)
    prediction_columns = [
        column
        for column in ("combined_alert", "pvlof_alert", "zero_current_alert")
        if column in points.columns
    ]
    if not prediction_columns:
        raise ValueError("PVLOF output has no supported alert columns")
    return points, {
        "path": str(path),
        "timezone": timezone,
        "rows": int(len(points)),
        "devices": int(points["device_no"].nunique()) if len(points) else 0,
        "station_mismatches": int((~points["station_name_match"]).sum())
        if len(points)
        else 0,
        "unknown_devices_removed": int(unknown_devices.sum()),
        "prediction_columns": prediction_columns,
        "time_start": points["event_time"].min().isoformat() if len(points) else None,
        "time_end": points["event_time"].max().isoformat() if len(points) else None,
    }


def restrict_alarm_events_to_predictions(
    alarm_events: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only alarm events from station/device pairs covered by PVLOF output."""

    coverage = predictions[["station_name", "device_no"]].drop_duplicates()
    covered = alarm_events.merge(
        coverage.assign(_pvlof_covered=True),
        on=["station_name", "device_no"],
        how="left",
    )
    keep = covered["_pvlof_covered"].astype("boolean").fillna(False).astype(bool)
    result = alarm_events.loc[keep.to_numpy()].reset_index(drop=True)
    report = {
        "input_events": int(len(alarm_events)),
        "covered_events": int(len(result)),
        "excluded_events": int((~keep).sum()),
        "pvlof_stations": int(coverage["station_name"].nunique()),
        "pvlof_devices": int(len(coverage)),
        "excluded_stations": sorted(
            set(alarm_events.loc[~keep.to_numpy(), "station_name"].astype(str))
        ),
    }
    return result, report


def _key_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].astype({column: str for column in columns if column != "string_no"})


def _point_metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    prediction_column: str,
    scope: str,
) -> dict[str, Any]:
    if scope == "string":
        keys = ["station_name", "device_no", "string_no", "event_time"]
        predicted = predictions[keys + [prediction_column]].copy()
        predicted = predicted.groupby(keys, as_index=False, dropna=False)[prediction_column].max()
        truth = labels[labels["label_scope"] == "string"].copy()
        truth = truth.rename(columns={"event_time": "event_time"})
        truth = truth[keys].drop_duplicates()
    elif scope == "device":
        keys = ["station_name", "device_no", "event_time"]
        predicted = predictions[keys + [prediction_column]].copy()
        predicted = predicted.groupby(keys, as_index=False, dropna=False)[prediction_column].max()
        truth = labels[labels["label_scope"] == "device"].copy()
        truth = truth[keys].drop_duplicates()
    else:
        raise ValueError(f"Unsupported point scope: {scope}")
    predicted["predicted"] = predicted[prediction_column].astype(bool)
    predicted = predicted[keys + ["predicted"]]
    predicted["label"] = False
    truth_keys = (
        pd.MultiIndex.from_frame(truth[keys])
        if len(truth)
        else pd.MultiIndex.from_arrays([[]] * len(keys))
    )
    predicted_keys = (
        pd.MultiIndex.from_frame(predicted[keys])
        if len(predicted)
        else pd.MultiIndex.from_arrays([[]] * len(keys))
    )
    predicted_lookup = pd.Series(predicted["predicted"].to_numpy(), index=predicted_keys)
    label_lookup = pd.Series(True, index=truth_keys)
    union = predicted_lookup.index.union(label_lookup.index)
    actual = pd.Series(union.isin(label_lookup.index), index=union)
    guess = pd.Series(
        predicted_lookup.reindex(union, fill_value=False).astype(bool).to_numpy(),
        index=union,
    )
    result = binary_metrics(actual, guess)
    result["scope"] = scope
    result["prediction_column"] = prediction_column
    result["label_points"] = int(len(truth))
    result["prediction_points"] = int(len(predicted))
    return result


def _collapse_prediction_events(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
    interval_minutes: int,
) -> pd.DataFrame:
    source = predictions[predictions[prediction_column].astype(bool)].copy()
    if source.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "station_name",
                "device_no",
                "string_no",
                "start_time",
                "end_time",
                "points",
            ]
        )
    source = source.sort_values(["station_name", "device_no", "string_no", "event_time"])
    expected = pd.Timedelta(minutes=interval_minutes)
    group_columns = ["station_name", "device_no", "string_no"]
    source["_new_event"] = (
        source[group_columns].ne(source[group_columns].shift()).any(axis=1)
        | source["event_time"].sub(source["event_time"].shift()).ne(expected)
    )
    source["_event_number"] = source["_new_event"].cumsum()
    events = (
        source.groupby(group_columns + ["_event_number"], observed=True)
        .agg(
            start_time=("event_time", "min"),
            end_time=("event_time", "max"),
            points=("event_time", "size"),
        )
        .reset_index()
        .drop(columns=["_event_number"])
    )
    events.insert(0, "event_id", [f"pred-{index:06d}" for index in range(1, len(events) + 1)])
    return events


def _label_events_for_scope(events: pd.DataFrame, scope: str) -> pd.DataFrame:
    result = events[events["label_scope"] == scope].copy()
    return result[
        [
            "alarm_event_id",
            "station_name",
            "device_no",
            "string_no",
            "raise_time",
            "end_time",
            "alarm_code",
            "alarm_name",
        ]
    ].rename(columns={"alarm_event_id": "event_id", "raise_time": "start_time"})


def _event_metrics(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    prediction_column: str,
    label_scope: str,
    interval_minutes: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    predicted_events = _collapse_prediction_events(
        predictions,
        prediction_column=prediction_column,
        interval_minutes=interval_minutes,
    )
    label_events = _label_events_for_scope(labels, label_scope)
    matched_pred: list[str] = []
    matched_label: list[str] = []
    matches: list[dict[str, Any]] = []
    for predicted in predicted_events.itertuples(index=False):
        candidates = label_events[
            (label_events["station_name"] == predicted.station_name)
            & (label_events["device_no"] == predicted.device_no)
            & (
                (label_scope == "device")
                | (label_events["string_no"] == predicted.string_no)
            )
            & (
                label_events["start_time"]
                <= predicted.end_time + pd.Timedelta(minutes=interval_minutes)
            )
            & (label_events["end_time"] >= predicted.start_time)
        ]
        for label in candidates.itertuples(index=False):
            matched_pred.append(predicted.event_id)
            matched_label.append(label.event_id)
            matches.append(
                {
                    "prediction_event_id": predicted.event_id,
                    "label_event_id": label.event_id,
                    "station_name": predicted.station_name,
                    "device_no": predicted.device_no,
                    "string_no": predicted.string_no if label_scope == "string" else pd.NA,
                    "alarm_code": label.alarm_code,
                    "alarm_name": label.alarm_name,
                }
            )
    predicted_hits = len(set(matched_pred))
    label_hits = len(set(matched_label))
    predicted_count = len(predicted_events)
    label_count = len(label_events)
    precision = predicted_hits / predicted_count if predicted_count else 0.0
    recall = label_hits / label_count if label_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "prediction_column": prediction_column,
        "label_scope": label_scope,
        "predicted_events": predicted_count,
        "label_events": label_count,
        "predicted_events_hit": predicted_hits,
        "label_events_hit": label_hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, pd.DataFrame(matches)


def evaluate_pvlof(
    predictions: pd.DataFrame,
    alarm_events: pd.DataFrame,
    alarm_points: pd.DataFrame,
    *,
    interval_minutes: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute point/event metrics for string and device-scoped alarm labels."""

    prediction_columns = [
        column
        for column in ("combined_alert", "pvlof_alert", "zero_current_alert")
        if column in predictions.columns
    ]
    point_metrics: list[dict[str, Any]] = []
    event_metrics: list[dict[str, Any]] = []
    point_metrics_by_alarm_code: list[dict[str, Any]] = []
    event_metrics_by_alarm_code: list[dict[str, Any]] = []
    match_frames: list[pd.DataFrame] = []
    for prediction_column in prediction_columns:
        for scope in ("string", "device"):
            point_metrics.append(
                _point_metrics(
                    predictions,
                    alarm_points,
                    prediction_column=prediction_column,
                    scope=scope,
                )
            )
            metrics, matches = _event_metrics(
                predictions,
                alarm_events,
                prediction_column=prediction_column,
                label_scope=scope,
                interval_minutes=interval_minutes,
            )
            event_metrics.append(metrics)
            if not matches.empty:
                matches["prediction_column"] = prediction_column
                matches["label_scope"] = scope
                match_frames.append(matches)
        for alarm_code in sorted(alarm_events["alarm_code"].astype(str).unique()):
            code_events = alarm_events[alarm_events["alarm_code"].astype(str) == alarm_code]
            code_points = alarm_points[alarm_points["alarm_code"].astype(str) == alarm_code]
            for scope in ("string", "device"):
                if not (code_events["label_scope"] == scope).any():
                    continue
                point_metric = _point_metrics(
                    predictions,
                    code_points,
                    prediction_column=prediction_column,
                    scope=scope,
                )
                point_metric["alarm_code"] = alarm_code
                point_metrics_by_alarm_code.append(point_metric)
                event_metric, code_matches = _event_metrics(
                    predictions,
                    code_events,
                    prediction_column=prediction_column,
                    label_scope=scope,
                    interval_minutes=interval_minutes,
                )
                event_metric["alarm_code"] = alarm_code
                event_metrics_by_alarm_code.append(event_metric)
                if not code_matches.empty:
                    code_matches["prediction_column"] = prediction_column
                    code_matches["label_scope"] = scope
                    code_matches["alarm_code_filter"] = alarm_code
                    match_frames.append(code_matches)
    matches = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    summary = {
        "alarm_events": int(len(alarm_events)),
        "alarm_points": int(len(alarm_points)),
        "prediction_points": int(len(predictions)),
        "point_metrics": point_metrics,
        "event_metrics": event_metrics,
        "point_metrics_by_alarm_code": point_metrics_by_alarm_code,
        "event_metrics_by_alarm_code": event_metrics_by_alarm_code,
        "note": (
            "String-scope metrics require alarm_string. Device-scope metrics cover alarms such as "
            "101001 that do not identify a specific string."
        ),
    }
    return summary, matches
