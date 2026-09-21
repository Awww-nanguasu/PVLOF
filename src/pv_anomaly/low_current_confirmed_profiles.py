"""Confirmed-event inputs for the second low-current profile baseline.

The comparison workbook is the formal-event ledger. Exact per-string event
boundaries come from the V1.7 Improved V2 replay outputs. Raw candidates are
joined only to enrich severity and morphology; they never create profile events.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pv_anomaly.low_current_profiles import LowCurrentProfileConfig


STRING_KEYS = ["plant_id", "device_no", "string_no"]
POINT_KEYS = [*STRING_KEYS, "event_time"]
FALSE_VALUES = {"", "FALSE", "NONE", "NAN", "NULL", "0"}
MORPHOLOGY_ORDER = {"isolated": 0, "group": 1, "gradient_group": 2}


@dataclass(frozen=True)
class ConfirmedLowCurrentProfileConfig:
    """Configuration for profiles whose population is confirmed V2 events."""

    version: str = "low-current-profile-v0.2-confirmed-events"
    source_algorithm_version: str = (
        "pvlof-v1.7-improved-v2-internal-member-gate"
    )
    confirmed_workbook_sheet: str = "pvlof_58d_confirmed_events"
    target_version_column: str = "PVLOF_V1_7_IMPROVED_V2"
    source_event_filename: str = "pvlof_v17_improved_v2_events.parquet"
    source_evidence_filename: str = (
        "pvlof_v17_improved_v2_evidence_points.parquet"
    )
    source_unconfirmed_filename: str = (
        "pvlof_v17_improved_v2_unconfirmed_candidates.parquet"
    )
    candidate_points_filename: str = "low_current_candidate_points.parquet"
    valid_day_coverage_filename: str = "low_current_valid_day_coverage.parquet"
    timezone: str = "Asia/Shanghai"
    interval_minutes: int = 5
    minimum_valid_points_per_day: int = 3
    history_window_days: int = 30
    minimum_valid_history_days: int = 14
    minimum_prior_alarm_days_long_term: int = 3
    minimum_prior_alarm_day_ratio_long_term: float = 0.0
    fixed_time_bin_minutes: int = 60
    minimum_fixed_time_alarm_days: int = 3
    minimum_fixed_time_concentration: float = 0.60
    strict_workbook_match: bool = True
    provisional_thresholds: bool = True

    def __post_init__(self) -> None:
        if self.interval_minutes < 1:
            raise ValueError("interval_minutes must be positive")
        if self.minimum_valid_points_per_day < 1:
            raise ValueError("minimum_valid_points_per_day must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> "ConfirmedLowCurrentProfileConfig":
        return cls(**dict(payload))

    def history_config(self) -> LowCurrentProfileConfig:
        """Return the compatible causal-history configuration."""

        return LowCurrentProfileConfig(
            version=self.version,
            source_algorithm_version=self.source_algorithm_version,
            timezone=self.timezone,
            interval_minutes=self.interval_minutes,
            minimum_valid_points_per_day=self.minimum_valid_points_per_day,
            history_window_days=self.history_window_days,
            minimum_valid_history_days=self.minimum_valid_history_days,
            minimum_prior_alarm_days_long_term=(
                self.minimum_prior_alarm_days_long_term
            ),
            minimum_prior_alarm_day_ratio_long_term=(
                self.minimum_prior_alarm_day_ratio_long_term
            ),
            fixed_time_bin_minutes=self.fixed_time_bin_minutes,
            minimum_fixed_time_alarm_days=self.minimum_fixed_time_alarm_days,
            minimum_fixed_time_concentration=self.minimum_fixed_time_concentration,
            provisional_thresholds=self.provisional_thresholds,
        )


def load_confirmed_config(
    path: str | Path,
) -> ConfirmedLowCurrentProfileConfig:
    return ConfirmedLowCurrentProfileConfig.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def read_parquet_file(path: str | Path) -> pd.DataFrame:
    """Read the physical file without Hive partition-column inference."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return pq.ParquetFile(source).read().to_pandas()


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, source: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _identifiers(series: pd.Series) -> pd.Series:
    return series.map(_identifier).astype("string")


def parse_string_members(value: Any) -> list[int]:
    """Parse workbook string lists such as ``01,02,11`` or ``FALSE``."""

    if pd.isna(value):
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)] if int(value) > 0 else []
    if isinstance(value, (float, np.floating)):
        return [int(value)] if float(value).is_integer() and value > 0 else []
    text = str(value).strip()
    if text.upper() in FALSE_VALUES:
        return []
    members: set[int] = set()
    for token in re.split(r"[,，;；\s]+", text):
        if not token:
            continue
        try:
            number = int(float(token))
        except ValueError as error:
            raise ValueError(f"Invalid string member {token!r} in {value!r}") from error
        if number < 1:
            raise ValueError(f"String numbers must be positive: {value!r}")
        members.add(number)
    return sorted(members)


def format_string_members(values: Iterable[Any]) -> str:
    members = sorted({int(value) for value in values if pd.notna(value)})
    return ",".join(f"{number:02d}" for number in members)


def _normalise_result_cell(value: Any) -> str:
    return format_string_members(parse_string_members(value))


def _local_series_to_utc(series: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise"
        )
    else:
        parsed = parsed.dt.tz_convert(timezone)
    return parsed.dt.tz_convert("UTC")


def _utc_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _join_unique(values: Iterable[Any]) -> str:
    return ",".join(sorted({_identifier(value) for value in values if _identifier(value)}))


def _highest_morphology(values: Iterable[Any]) -> str:
    present = [_identifier(value) for value in values if _identifier(value)]
    if not present:
        return "isolated"
    return max(present, key=lambda value: MORPHOLOGY_ORDER.get(value, -1))


def normalise_confirmed_workbook_frame(
    frame: pd.DataFrame,
    config: ConfirmedLowCurrentProfileConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select V2 formal events and explode their workbook string memberships."""

    required = {
        "plant_id",
        "device_no",
        "raise_time_local",
        "end_time_local",
        config.target_version_column,
    }
    _require_columns(frame, required, source="confirmed-event workbook")
    source = frame.copy().reset_index(drop=True)
    source.insert(0, "source_excel_row", source.index + 2)
    source["plant_id"] = _identifiers(source["plant_id"])
    source["device_no"] = _identifiers(source["device_no"])
    result_list_columns = [
        column
        for column in source.columns
        if str(column).upper().startswith("PVLOF_")
        or str(column).lower()
        in {"baseline", "v2_added_strings", "v2_removed_strings"}
    ]
    for column in result_list_columns:
        source[column] = source[column].map(_normalise_result_cell).astype("string")
    for column in (
        "comparison_case",
        "v2_vs_improved_case",
        "manual_label",
        "manual_strings",
        "review_note",
    ):
        if column in source:
            source[column] = _identifiers(source[column])
    if "event_id" in source:
        supplied_ids = _identifiers(source["event_id"])
    else:
        supplied_ids = pd.Series("", index=source.index, dtype="string")
    generated = source["source_excel_row"].map(
        lambda value: f"confirmed-workbook-{int(value):06d}"
    )
    source["comparison_event_id"] = supplied_ids.mask(
        supplied_ids.eq(""), generated
    )
    if source["comparison_event_id"].duplicated().any():
        duplicates = source.loc[
            source["comparison_event_id"].duplicated(False),
            "comparison_event_id",
        ].tolist()
        raise ValueError(f"Workbook event IDs are not unique: {duplicates[:10]}")

    source["raise_time"] = _local_series_to_utc(
        source["raise_time_local"], config.timezone
    )
    source["end_time"] = _local_series_to_utc(
        source["end_time_local"], config.timezone
    )
    invalid_keys = source[["plant_id", "device_no"]].eq("").any(axis=1)
    invalid_times = source[["raise_time", "end_time"]].isna().any(axis=1)
    reversed_times = source["end_time"].lt(source["raise_time"])
    if invalid_keys.any() or invalid_times.any() or reversed_times.any():
        raise ValueError(
            "Confirmed-event workbook contains invalid identifiers or time ranges: "
            f"blank_keys={int(invalid_keys.sum())}, "
            f"invalid_times={int(invalid_times.sum())}, "
            f"reversed_times={int(reversed_times.sum())}"
        )
    source["raise_time_local"] = (
        source["raise_time"]
        .dt.tz_convert(config.timezone)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .astype("string")
    )
    source["end_time_local"] = (
        source["end_time"]
        .dt.tz_convert(config.timezone)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .astype("string")
    )

    source["target_member_numbers"] = source[
        config.target_version_column
    ].map(parse_string_members)
    source["target_member_strings"] = source["target_member_numbers"].map(
        format_string_members
    )
    selected = source[source["target_member_numbers"].map(bool)].copy()
    selected["target_member_count"] = selected["target_member_numbers"].map(len)

    membership_rows: list[dict[str, Any]] = []
    passthrough = [
        "row",
        "comparison_case",
        "v2_vs_improved_case",
        "manual_label",
        "manual_strings",
        "review_note",
    ]
    for record in selected.to_dict(orient="records"):
        for string_no in record["target_member_numbers"]:
            membership = {
                "membership_key": (
                    f"{record['comparison_event_id']}::string-{int(string_no):02d}"
                ),
                "comparison_event_id": record["comparison_event_id"],
                "source_excel_row": int(record["source_excel_row"]),
                "plant_id": record["plant_id"],
                "device_no": record["device_no"],
                "string_no": int(string_no),
                "workbook_raise_time": record["raise_time"],
                "workbook_end_time": record["end_time"],
                "workbook_raise_time_local": str(record["raise_time_local"]),
                "workbook_end_time_local": str(record["end_time_local"]),
            }
            for column in passthrough:
                if column in record:
                    membership[f"workbook_{column}"] = record[column]
            membership_rows.append(membership)

    memberships = pd.DataFrame(membership_rows)
    if memberships.empty:
        raise ValueError(
            f"Workbook has no events in {config.target_version_column!r}"
        )
    memberships["string_no"] = memberships["string_no"].astype("Int64")
    if memberships["membership_key"].duplicated().any():
        raise ValueError("Workbook contains duplicate event/string memberships")
    return selected.reset_index(drop=True), memberships.reset_index(drop=True)


def load_confirmed_workbook(
    path: str | Path,
    config: ConfirmedLowCurrentProfileConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_excel(
        source,
        sheet_name=config.confirmed_workbook_sheet,
        engine="openpyxl",
    )
    selected, memberships = normalise_confirmed_workbook_frame(frame, config)
    return selected, memberships, int(len(frame))


def _normalise_source_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["plant_id"] = _identifiers(result["plant_id"])
    result["device_no"] = _identifiers(result["device_no"])
    result["string_no"] = pd.to_numeric(
        result["string_no"], errors="raise"
    ).astype("Int64")
    result["event_id"] = _identifiers(result["event_id"])
    result["source_event_key"] = (
        result["plant_id"].astype(str) + "::" + result["event_id"].astype(str)
    )
    return result


def load_replay_confirmed_tables(
    input_root: str | Path,
    config: ConfirmedLowCurrentProfileConfig,
    *,
    plant_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Read exact confirmed events, evidence, and auxiliary candidate runs."""

    root = Path(input_root)
    requested = [str(value) for value in plant_ids or []]
    directories = (
        [root / f"plant_id={plant_id}" for plant_id in requested]
        if requested
        else sorted(root.glob("plant_id=*"))
    )
    if not directories:
        raise FileNotFoundError(f"No plant_id=* directories found under {root}")

    event_frames: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    unconfirmed_frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        event_path = directory / config.source_event_filename
        evidence_path = directory / config.source_evidence_filename
        unconfirmed_path = directory / config.source_unconfirmed_filename
        events = read_parquet_file(event_path)
        evidence = read_parquet_file(evidence_path)
        unconfirmed = read_parquet_file(unconfirmed_path)
        _require_columns(
            events,
            [
                "version",
                "event_id",
                *STRING_KEYS,
                "candidate_start_time",
                "alert_confirm_time",
                "last_anomaly_time",
                "event_end_time",
                "evidence_points",
                "final_alert_points",
            ],
            source=str(event_path),
        )
        _require_columns(
            evidence,
            ["version", "event_id", *POINT_KEYS, "raw_evidence", "final_alert"],
            source=str(evidence_path),
        )
        if not unconfirmed.empty:
            _require_columns(
                unconfirmed,
                [*STRING_KEYS, "candidate_start_time", "candidate_end_time"],
                source=str(unconfirmed_path),
            )
        for label, table in (("events", events), ("evidence", evidence)):
            versions = set(table["version"].dropna().astype(str))
            unexpected = versions - {config.source_algorithm_version}
            if unexpected:
                raise ValueError(
                    f"{directory} {label} contain unexpected versions: {unexpected}"
                )
        events = _normalise_source_keys(events)
        evidence = _normalise_source_keys(evidence)
        for column in (
            "candidate_start_time",
            "alert_confirm_time",
            "last_anomaly_time",
            "clear_time",
            "event_end_time",
        ):
            if column in events:
                events[column] = _utc_series(events[column])
        evidence["event_time"] = _utc_series(evidence["event_time"])
        evidence["raw_evidence"] = evidence["raw_evidence"].fillna(False).astype(bool)
        evidence["final_alert"] = evidence["final_alert"].fillna(False).astype(bool)
        event_frames.append(events)
        evidence_frames.append(evidence)

        if not unconfirmed.empty:
            unconfirmed["plant_id"] = _identifiers(unconfirmed["plant_id"])
            unconfirmed["device_no"] = _identifiers(unconfirmed["device_no"])
            unconfirmed["string_no"] = pd.to_numeric(
                unconfirmed["string_no"], errors="raise"
            ).astype("Int64")
            for column in ("candidate_start_time", "candidate_end_time"):
                unconfirmed[column] = _utc_series(unconfirmed[column])
            unconfirmed_frames.append(unconfirmed)
        audits.append(
            {
                "plant_directory": str(directory),
                "confirmed_events": int(len(events)),
                "evidence_rows": int(len(evidence)),
                "unconfirmed_candidate_runs": int(len(unconfirmed)),
            }
        )

    all_events = pd.concat(event_frames, ignore_index=True)
    all_evidence = pd.concat(evidence_frames, ignore_index=True)
    all_unconfirmed = (
        pd.concat(unconfirmed_frames, ignore_index=True)
        if unconfirmed_frames
        else pd.DataFrame()
    )
    if all_events["source_event_key"].duplicated().any():
        raise ValueError("Replay event IDs are not unique within plants")
    return all_events, all_evidence, all_unconfirmed, audits


def build_workbook_crosswalk(
    source_events: pd.DataFrame,
    evidence: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Map workbook device intervals to exact source string-event IDs.

    A source string event may legitimately intersect several workbook rows.
    It is retained once in ``confirmed_events`` and keeps all matching workbook
    IDs in the audit columns.
    """

    signal = evidence[evidence["raw_evidence"] | evidence["final_alert"]].copy()
    signal = signal.reset_index(drop=True)
    signal["_evidence_row_id"] = signal.index
    joined = signal.merge(
        memberships,
        on=STRING_KEYS,
        how="left",
        validate="many_to_many",
    )
    in_window = joined["event_time"].between(
        joined["workbook_raise_time"],
        joined["workbook_end_time"],
        inclusive="both",
    )
    matched = joined[in_window.fillna(False)].copy()
    matched = matched.drop_duplicates(
        ["_evidence_row_id", "membership_key"]
    )

    ambiguous_evidence = int(
        matched.groupby("_evidence_row_id", observed=True)["membership_key"]
        .nunique()
        .gt(1)
        .sum()
    )
    matched_evidence_ids = set(matched["_evidence_row_id"].astype(int))
    unmatched_evidence_rows = int(len(signal) - len(matched_evidence_ids))
    matched_memberships = set(matched["membership_key"].dropna().astype(str))
    unmatched_memberships = int(
        (~memberships["membership_key"].astype(str).isin(matched_memberships)).sum()
    )

    detail_columns = [
        "source_event_key",
        "event_id",
        "plant_id",
        "device_no",
        "string_no",
        "membership_key",
        "comparison_event_id",
        "source_excel_row",
        "workbook_raise_time",
        "workbook_end_time",
    ]
    optional_columns = [
        column
        for column in memberships.columns
        if column.startswith("workbook_") and column not in detail_columns
    ]
    if matched.empty:
        mapping_detail = pd.DataFrame(
            columns=[*detail_columns, *optional_columns, "matched_evidence_points"]
        )
    else:
        grouped = matched.groupby(
            [*detail_columns, *optional_columns],
            observed=True,
            dropna=False,
            as_index=False,
        )
        mapping_detail = grouped.agg(
            matched_evidence_points=("_evidence_row_id", "nunique"),
            first_matched_evidence_time=("event_time", "min"),
            last_matched_evidence_time=("event_time", "max"),
        )

    if mapping_detail.empty:
        event_mapping = pd.DataFrame(
            columns=[
                "source_event_key",
                "comparison_event_ids",
                "source_workbook_rows",
                "workbook_event_count",
                "matched_workbook_evidence_points",
            ]
        )
    else:
        event_mapping = (
            mapping_detail.groupby("source_event_key", observed=True, as_index=False)
            .agg(
                comparison_event_ids=("comparison_event_id", _join_unique),
                source_workbook_rows=("source_excel_row", _join_unique),
                workbook_event_count=("comparison_event_id", "nunique"),
                matched_workbook_evidence_points=(
                    "matched_evidence_points",
                    "sum",
                ),
            )
        )

    confirmed_events = source_events.merge(
        event_mapping,
        on="source_event_key",
        how="left",
        validate="one_to_one",
    )
    mapped_mask = confirmed_events["workbook_event_count"].notna()
    unmatched_source_events = int((~mapped_mask).sum())
    confirmed_events = confirmed_events[mapped_mask].copy()
    confirmed_events["workbook_event_count"] = confirmed_events[
        "workbook_event_count"
    ].astype("int16")
    confirmed_events["matched_workbook_evidence_points"] = confirmed_events[
        "matched_workbook_evidence_points"
    ].astype("int32")

    audit = {
        "source_confirmed_events": int(len(source_events)),
        "source_signal_evidence_rows": int(len(signal)),
        "workbook_memberships": int(len(memberships)),
        "matched_source_events": int(len(confirmed_events)),
        "mapping_rows": int(len(mapping_detail)),
        "source_events_spanning_multiple_workbook_rows": int(
            confirmed_events["workbook_event_count"].gt(1).sum()
        ),
        "unmatched_source_events": unmatched_source_events,
        "unmatched_evidence_rows": unmatched_evidence_rows,
        "unmatched_workbook_memberships": unmatched_memberships,
        "ambiguous_evidence_rows": ambiguous_evidence,
    }
    blocking = {
        key: audit[key]
        for key in (
            "unmatched_source_events",
            "unmatched_evidence_rows",
            "unmatched_workbook_memberships",
            "ambiguous_evidence_rows",
        )
        if audit[key]
    }
    if strict and blocking:
        raise ValueError(f"Confirmed-event workbook crosswalk is incomplete: {blocking}")
    return (
        confirmed_events.reset_index(drop=True),
        mapping_detail.reset_index(drop=True),
        audit,
    )


def _normalise_candidate_points(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, POINT_KEYS, source="candidate-point profile input")
    result = frame.copy()
    result["plant_id"] = _identifiers(result["plant_id"])
    result["device_no"] = _identifiers(result["device_no"])
    result["string_no"] = pd.to_numeric(
        result["string_no"], errors="raise"
    ).astype("Int64")
    result["event_time"] = _utc_series(result["event_time"])
    duplicate = result.duplicated(POINT_KEYS, keep=False)
    if duplicate.any():
        raise ValueError(
            "Candidate-point profile input contains duplicate keys: "
            f"{int(duplicate.sum())} rows"
        )
    return result


def attach_candidate_metrics(
    confirmed_events: pd.DataFrame,
    evidence: pd.DataFrame,
    candidate_points: pd.DataFrame,
    *,
    interval_minutes: int,
    timezone: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enrich formal events with raw-candidate metrics without changing scope."""

    candidates = _normalise_candidate_points(candidate_points)
    raw_evidence = evidence[evidence["raw_evidence"]].copy()
    evidence_keys = raw_evidence[
        ["source_event_key", *POINT_KEYS]
    ].drop_duplicates()
    joined = evidence_keys.merge(
        candidates,
        on=POINT_KEYS,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    joined["candidate_metric_matched"] = joined["_merge"].eq("both")
    unmatched_candidate_metrics = int((~joined["candidate_metric_matched"]).sum())

    aggregations: dict[str, tuple[str, Any]] = {
        "candidate_metric_points": ("candidate_metric_matched", "sum"),
    }
    definitions = {
        "valid_candidate_points": ("valid_point", "sum"),
        "device_time_candidate_count_max": (
            "device_time_candidate_count",
            "max",
        ),
        "minimum_string_current": ("string_current", "min"),
        "median_string_current": ("string_current", "median"),
        "median_expected_current": ("expected_current", "median"),
        "minimum_residual_ratio": ("residual_ratio", "min"),
        "median_residual_ratio": ("residual_ratio", "median"),
        "maximum_relative_drop": ("relative_drop", "max"),
        "median_relative_drop": ("relative_drop", "median"),
        "maximum_absolute_drop": ("absolute_drop", "max"),
        "median_absolute_drop": ("absolute_drop", "median"),
        "maximum_internal_relative_drop": ("internal_relative_drop", "max"),
        "maximum_internal_absolute_drop": ("internal_absolute_drop", "max"),
        "maximum_group_relative_drop": ("group_relative_drop", "max"),
        "median_conditioned_virtual_irradiance": (
            "conditioned_virtual_irradiance",
            "median",
        ),
    }
    for output, (input_column, operation) in definitions.items():
        if input_column in joined:
            aggregations[output] = (input_column, operation)
    if "source_branch" in joined:
        aggregations["source_branches"] = ("source_branch", _join_unique)
    if "point_morphology" in joined:
        aggregations["event_morphology"] = (
            "point_morphology",
            _highest_morphology,
        )
    metric_rows = (
        joined.groupby("source_event_key", observed=True, as_index=False)
        .agg(**aggregations)
    )
    events = confirmed_events.merge(
        metric_rows,
        on="source_event_key",
        how="left",
        validate="one_to_one",
    )

    events["candidate_start_time"] = _utc_series(events["candidate_start_time"])
    events["candidate_end_time"] = _utc_series(events["last_anomaly_time"])
    events["candidate_window_end_time"] = (
        events["candidate_end_time"] + pd.Timedelta(minutes=interval_minutes)
    )
    events["candidate_span_minutes"] = (
        events["candidate_end_time"] - events["candidate_start_time"]
    ).dt.total_seconds().div(60)
    events["candidate_duration_minutes"] = (
        events["candidate_span_minutes"] + interval_minutes
    )
    events["formal_event_lifecycle_minutes"] = (
        _utc_series(events["event_end_time"]) - events["candidate_start_time"]
    ).dt.total_seconds().div(60)
    events["candidate_points"] = pd.to_numeric(
        events["evidence_points"], errors="coerce"
    ).fillna(0).astype("int32")
    events["candidate_metric_points"] = pd.to_numeric(
        events["candidate_metric_points"], errors="coerce"
    ).fillna(0).astype("int32")
    events["has_final_alert"] = pd.to_numeric(
        events["final_alert_points"], errors="coerce"
    ).fillna(0).gt(0).astype("int8")
    events["meets_three_point_rule"] = events["candidate_points"].ge(3).astype("int8")
    if "valid_candidate_points" in events:
        events["valid_candidate_points"] = pd.to_numeric(
            events["valid_candidate_points"], errors="coerce"
        ).fillna(0).astype("int32")
        events["all_candidates_valid"] = (
            events["candidate_metric_points"].gt(0)
            & events["valid_candidate_points"].eq(events["candidate_metric_points"])
        ).astype("int8")
    else:
        events["valid_candidate_points"] = 0
        events["all_candidates_valid"] = 0
    if "source_branches" not in events:
        events["source_branches"] = events.get("confirmation_branch", "")
    elif "confirmation_branch" in events:
        events["source_branches"] = events["source_branches"].fillna(
            events["confirmation_branch"]
        )
    else:
        events["source_branches"] = events["source_branches"].fillna("")
    if "event_morphology" not in events:
        events["event_morphology"] = "isolated"

    events = events.sort_values(
        [*STRING_KEYS, "candidate_start_time"]
    ).reset_index(drop=True)
    events.insert(
        0,
        "string_event_id",
        [f"confirmed-low-current-{number:08d}" for number in range(1, len(events) + 1)],
    )
    for column in (
        "candidate_start_time",
        "alert_confirm_time",
        "candidate_end_time",
        "clear_time",
        "event_end_time",
        "candidate_window_end_time",
    ):
        if column in events:
            events[f"{column}_local"] = (
                _utc_series(events[column])
                .dt.tz_convert(timezone)
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )
    local_start = events["candidate_start_time"].dt.tz_convert(timezone)
    events["event_local_date"] = local_start.dt.strftime("%Y-%m-%d")
    events["event_start_hour"] = local_start.dt.hour.astype("int8")

    audit = {
        "confirmed_events_enriched": int(len(events)),
        "raw_evidence_points": int(len(evidence_keys)),
        "candidate_metric_points_matched": int(
            joined["candidate_metric_matched"].sum()
        ),
        "candidate_metric_points_unmatched": unmatched_candidate_metrics,
        "events_without_candidate_metrics": int(
            events["candidate_metric_points"].eq(0).sum()
        ),
    }
    return events, audit


def prepare_unconfirmed_runs(
    unconfirmed: pd.DataFrame,
    *,
    timezone: str,
    interval_minutes: int,
) -> pd.DataFrame:
    """Format rejected candidate runs as an explicitly non-profile table."""

    if unconfirmed.empty:
        return unconfirmed.copy()
    result = unconfirmed.copy()
    result = result.sort_values(
        [*STRING_KEYS, "candidate_start_time"]
    ).reset_index(drop=True)
    result.insert(
        0,
        "unconfirmed_run_id",
        [f"unconfirmed-candidate-{number:08d}" for number in range(1, len(result) + 1)],
    )
    result["candidate_span_minutes"] = (
        result["candidate_end_time"] - result["candidate_start_time"]
    ).dt.total_seconds().div(60)
    result["candidate_duration_minutes"] = (
        result["candidate_span_minutes"] + interval_minutes
    )
    for column in ("candidate_start_time", "candidate_end_time"):
        result[f"{column}_local"] = (
            _utc_series(result[column])
            .dt.tz_convert(timezone)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
    result["profile_population"] = "excluded_unconfirmed_candidate"
    result["exclusion_reason"] = (
        "Not present in the confirmed V1.7 Improved V2 event population"
    )
    return result


__all__ = [
    "ConfirmedLowCurrentProfileConfig",
    "attach_candidate_metrics",
    "build_workbook_crosswalk",
    "format_string_members",
    "load_confirmed_config",
    "load_confirmed_workbook",
    "load_replay_confirmed_tables",
    "normalise_confirmed_workbook_frame",
    "parse_string_members",
    "prepare_unconfirmed_runs",
    "read_parquet_file",
]
