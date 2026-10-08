"""Evidence-aware PVLOF event reconstruction for v1.4 and v1.5."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


KEYS = ["plant_id", "device_no", "string_no"]
CLEAR_REASONS = {1: "recovered", 2: "data_gap", 3: "not_evaluable"}
EVIDENCE_COLUMNS = [
    "version", "event_id", "plant_id", "device_no", "string_no", "event_time",
    "point_state", "raw_evidence", "final_alert", "entry_streak", "normal_streak",
    "memory_active", "valid_point", "unknown_streak", "memory_suspended",
    "memory_resumed_alert",
    "confirmation_branch",
]
EVENT_COLUMNS = [
    "version", "event_id", "plant_id", "device_no", "string_no",
    "candidate_start_time", "alert_confirm_time", "last_anomaly_time", "clear_time",
    "event_end_time", "evidence_points", "final_alert_points",
    "recovery_observation_points", "detection_delay_minutes", "confirmation_branch",
    "clear_reason",
]
UNCONFIRMED_COLUMNS = [
    "version", "plant_id", "device_no", "string_no", "candidate_start_time",
    "candidate_end_time", "candidate_points", "maximum_entry_streak", "end_reason",
]


def _boolean_union(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame:
            result |= frame[column].fillna(False).astype(bool)
    return result


def _confirmation_branch(row: pd.Series) -> str:
    definitions = [
        ("isolated_hier_strict_alert", "hierarchical_strict"),
        ("isolated_directional_alert", "base_directional"),
        ("isolated_alert", "legacy_isolated"),
        ("collective_member_alert", "collective"),
        ("pvlof_v2_legacy_alert", "legacy_mixed"),
    ]
    branches = [
        label for column, label in definitions
        if column in row.index and bool(pd.notna(row[column]) and row[column])
    ]
    return ",".join(branches) if branches else "final_union"


def _entry_streak(frame: pd.DataFrame) -> pd.Series:
    columns = [
        column for column in (
            "pvlof_v16_entry_streak",
            "pvlof_v15_entry_streak",
            "isolated_directional_consecutive",
            "isolated_hier_strict_consecutive",
            "isolated_consecutive",
            "collective_event_consecutive",
        ) if column in frame
    ]
    if not columns:
        return pd.Series(0, index=frame.index, dtype="int32")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return values.max(axis=1).astype("int32")


def reconstruct_pvlof_events(
    frame: pd.DataFrame,
    *,
    version: str,
    final_alert_column: str,
    expected_interval_minutes: int = 5,
    timezone: str = "Asia/Shanghai",
    use_v15_memory: bool = False,
    memory_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return confirmed evidence points, string events and unconfirmed runs."""

    required = {*KEYS, "event_time", final_alert_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PVLOF evidence input is missing columns: {missing}")
    source = frame.copy()
    source["plant_id"] = source["plant_id"].astype(str)
    source["device_no"] = source["device_no"].astype(str)
    source["string_no"] = pd.to_numeric(source["string_no"], errors="raise").astype("Int64")
    source["event_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    source = source.sort_values([*KEYS, "event_time"]).reset_index(drop=True)

    if memory_prefix is None and use_v15_memory:
        memory_prefix = "pvlof_v15"
    raw_columns = [
        "pvlof_v16_raw_anomaly",
        "pvlof_v15_raw_anomaly",
        "isolated_directional_raw_alert",
        "isolated_hier_raw_alert",
        "isolated_raw_alert",
        "collective_raw_alert",
    ]
    source["_raw_evidence"] = _boolean_union(source, raw_columns)
    source["_final_alert"] = source[final_alert_column].fillna(False).astype(bool)
    memory_active_column = (
        f"{memory_prefix}_memory_active" if memory_prefix else None
    )
    memory_clear_column = (
        f"{memory_prefix}_memory_clear_code" if memory_prefix else None
    )
    memory_normal_column = (
        f"{memory_prefix}_normal_streak" if memory_prefix else None
    )
    memory_reactivated_column = (
        f"{memory_prefix}_memory_reactivated_alert" if memory_prefix else None
    )
    memory_valid_column = (
        f"{memory_prefix}_valid_point" if memory_prefix else None
    )
    memory_unknown_column = (
        f"{memory_prefix}_unknown_streak" if memory_prefix else None
    )
    memory_suspended_column = (
        f"{memory_prefix}_memory_suspended" if memory_prefix else None
    )
    memory_resumed_column = (
        f"{memory_prefix}_memory_resumed_alert" if memory_prefix else None
    )
    source["_memory_active"] = (
        source[memory_active_column].fillna(False).astype(bool)
        if memory_active_column and memory_active_column in source
        else False
    )
    source["_clear_code"] = (
        pd.to_numeric(source[memory_clear_column], errors="coerce").fillna(0).astype("int8")
        if memory_clear_column and memory_clear_column in source
        else np.int8(0)
    )
    source["_entry_streak"] = _entry_streak(source)
    source["_normal_streak"] = (
        pd.to_numeric(source[memory_normal_column], errors="coerce").fillna(0).astype("int32")
        if memory_normal_column and memory_normal_column in source
        else np.int32(0)
    )
    source["_valid_point"] = (
        source[memory_valid_column].fillna(False).astype(bool)
        if memory_valid_column and memory_valid_column in source
        else True
    )
    source["_unknown_streak"] = (
        pd.to_numeric(source[memory_unknown_column], errors="coerce").fillna(0).astype("int32")
        if memory_unknown_column and memory_unknown_column in source
        else np.int32(0)
    )
    source["_memory_suspended"] = (
        source[memory_suspended_column].fillna(False).astype(bool)
        if memory_suspended_column and memory_suspended_column in source
        else (
            source["_memory_active"]
            & ~source["_valid_point"]
            & source["_clear_code"].eq(0)
        )
    )
    source["_memory_resumed"] = (
        source[memory_resumed_column].fillna(False).astype(bool)
        if memory_resumed_column and memory_resumed_column in source
        else False
    )
    relevant = source[
        source["_raw_evidence"]
        | source["_final_alert"]
        | source["_memory_active"]
        | source["_clear_code"].gt(0)
    ].copy()

    evidence_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    unconfirmed_rows: list[dict[str, Any]] = []
    event_number = 0
    expected = pd.Timedelta(minutes=expected_interval_minutes)

    for key, group in relevant.groupby(KEYS, observed=True, sort=False):
        group = group.sort_values("event_time").copy()
        prior_clear = group["_clear_code"].shift(fill_value=0).gt(0)
        source_gap = group["event_time"].diff().ne(expected)
        preserved_gap = (
            source_gap
            & group["_memory_active"]
            & group["_clear_code"].eq(0)
        )
        new_segment = (source_gap & ~preserved_gap) | prior_clear
        group["_segment"] = new_segment.cumsum()
        for _, segment in group.groupby("_segment", observed=True, sort=False):
            if not segment["_final_alert"].any():
                raw_segment = segment[segment["_raw_evidence"]]
                if raw_segment.empty:
                    continue
                unconfirmed_rows.append({
                    "version": version,
                    "plant_id": str(key[0]),
                    "device_no": str(key[1]),
                    "string_no": int(key[2]),
                    "candidate_start_time": raw_segment["event_time"].min(),
                    "candidate_end_time": raw_segment["event_time"].max(),
                    "candidate_points": int(len(raw_segment)),
                    "maximum_entry_streak": int(raw_segment["_entry_streak"].max()),
                    "end_reason": "not_confirmed",
                })
                continue

            event_number += 1
            event_id = f"{version}-{event_number:08d}"
            confirm_index = segment.index[segment["_final_alert"]][0]
            confirm_time = source.at[confirm_index, "event_time"]
            before_confirmation = segment[
                segment["event_time"].le(confirm_time) & segment["_raw_evidence"]
            ]
            candidate_start = (
                before_confirmation["event_time"].min()
                if not before_confirmation.empty else confirm_time
            )
            branch = _confirmation_branch(source.loc[confirm_index])
            last_anomaly = segment.loc[segment["_raw_evidence"], "event_time"].max()
            clear_rows = segment[segment["_clear_code"].gt(0)]
            clear_time = clear_rows["event_time"].iloc[0] if not clear_rows.empty else pd.NaT
            clear_reason = (
                CLEAR_REASONS.get(int(clear_rows["_clear_code"].iloc[0]), "unknown")
                if not clear_rows.empty
                else "end_of_data" if bool(segment["_memory_active"].iloc[-1])
                else "evidence_ended"
            )

            recovery_points = 0
            for index, row in segment.iterrows():
                timestamp = row["event_time"]
                raw = bool(row["_raw_evidence"])
                final = bool(row["_final_alert"])
                clear_code = int(row["_clear_code"])
                if timestamp < confirm_time:
                    state = "candidate"
                elif timestamp == confirm_time:
                    state = "confirmed"
                elif bool(
                    row.get(memory_reactivated_column, 0)
                    if memory_reactivated_column else False
                ):
                    state = "memory_reactivation"
                elif bool(row["_memory_resumed"]):
                    state = "data_quality_resumed"
                elif raw:
                    state = "active_anomaly"
                elif clear_code:
                    state = CLEAR_REASONS.get(clear_code, "cleared")
                    recovery_points += int(clear_code == 1)
                elif bool(row["_memory_suspended"]):
                    state = "data_quality_suspended"
                elif bool(row["_memory_active"]):
                    state = "recovery_pending"
                    recovery_points += int(bool(row["_valid_point"]))
                elif final:
                    state = "active_alert"
                else:
                    continue
                evidence_rows.append({
                    "version": version,
                    "event_id": event_id,
                    "plant_id": str(key[0]),
                    "device_no": str(key[1]),
                    "string_no": int(key[2]),
                    "event_time": timestamp,
                    "point_state": state,
                    "raw_evidence": int(raw),
                    "final_alert": int(final),
                    "entry_streak": int(row["_entry_streak"]),
                    "normal_streak": int(row["_normal_streak"]),
                    "memory_active": int(bool(row["_memory_active"])),
                    "valid_point": int(bool(row["_valid_point"])),
                    "unknown_streak": int(row["_unknown_streak"]),
                    "memory_suspended": int(bool(row["_memory_suspended"])),
                    "memory_resumed_alert": int(bool(row["_memory_resumed"])),
                    "confirmation_branch": branch,
                })

            event_rows.append({
                "version": version,
                "event_id": event_id,
                "plant_id": str(key[0]),
                "device_no": str(key[1]),
                "string_no": int(key[2]),
                "candidate_start_time": candidate_start,
                "alert_confirm_time": confirm_time,
                "last_anomaly_time": last_anomaly,
                "clear_time": clear_time,
                "event_end_time": clear_time if pd.notna(clear_time) else segment["event_time"].max(),
                "evidence_points": int(segment["_raw_evidence"].sum()),
                "final_alert_points": int(segment["_final_alert"].sum()),
                "recovery_observation_points": recovery_points,
                "detection_delay_minutes": float(
                    (confirm_time - candidate_start).total_seconds() / 60
                ),
                "confirmation_branch": branch,
                "clear_reason": clear_reason,
            })

    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS)
    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    unconfirmed = pd.DataFrame(unconfirmed_rows, columns=UNCONFIRMED_COLUMNS)
    for output in (evidence, events, unconfirmed):
        for column in [
            "event_time", "candidate_start_time", "candidate_end_time",
            "alert_confirm_time", "last_anomaly_time", "clear_time", "event_end_time",
        ]:
            if column in output:
                output[f"{column}_local"] = pd.to_datetime(
                    output[column], errors="coerce", utc=True
                ).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    return evidence, events, unconfirmed
