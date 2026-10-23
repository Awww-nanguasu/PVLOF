"""Current PVLOF alarm membership and independent string duration accounting.

The selected combined per-string confirmation flags drive the current formal
view. Original legacy-union flags remain available for audit. Candidate
evidence is never backfilled into the realtime formal view. Normal recovery
points keep the string's episode linkage but contribute no undercurrent
duration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


KEYS = ["plant_id", "device_no", "string_no"]
TIME_KEYS = ["plant_id", "device_no", "event_time"]
DEFAULT_PREFIX = "pvlof_v17_improved"
STATE_SUFFIXES = [
    "raw_anomaly", "alert", "combined_memory_alert", "valid_point", "memory_active",
    "memory_clear_code",
]
OPTIONAL_STATE_SUFFIXES = [
    "unknown_streak", "memory_suspended", "memory_resumed_alert",
]
STATE_COLUMNS = [
    *KEYS, "event_time", "string_episode_id", "string_event_id", "point_state",
    "raw_candidate", "final_alert", "memory_active", "candidate_start_time",
    "alert_confirm_time", "cumulative_anomaly_points", "cumulative_anomaly_minutes",
    "cumulative_final_alert_points", "reactivated_alert", "resumed_alert",
    "unknown_streak", "memory_suspended", "reset_reason",
    "source_final_alert", "source_alert_without_string_confirmation",
]
EVENT_COLUMNS = [
    *KEYS, "string_event_id", "candidate_start_time", "alert_confirm_time",
    "last_anomaly_time", "clear_time", "event_end_time", "anomaly_points",
    "anomaly_duration_minutes", "final_alert_points", "final_alert_duration_minutes",
    "recovery_observation_points", "reactivations", "clear_reason",
]


def input_columns(prefix: str = DEFAULT_PREFIX) -> list[str]:
    return [*KEYS, "event_time", *(f"{prefix}_{name}" for name in STATE_SUFFIXES)]


def optional_input_columns(prefix: str = DEFAULT_PREFIX) -> list[str]:
    return [f"{prefix}_{name}" for name in OPTIONAL_STATE_SUFFIXES]


def _id(kind: str, *values: Any) -> str:
    payload = json.dumps([str(value) for value in values], ensure_ascii=False)
    return f"{kind}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _flag(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").fillna(0)
    if not values.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only 0/1 flags")
    return values.astype(bool)


def _local_times(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    result = frame.copy()
    for column in list(result.columns):
        if column == "event_time" or column.endswith("_time"):
            result[f"{column}_local"] = pd.to_datetime(
                result[column], utc=True, errors="raise"
            ).dt.tz_convert(timezone).dt.strftime("%Y-%m-%d %H:%M:%S")
    return result


def build_string_alarm_states(
    frame: pd.DataFrame,
    *,
    prefix: str = DEFAULT_PREFIX,
    interval_minutes: int = 5,
    entry_consecutive: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read existing full-point flags; return sparse states and string lifecycles.

    A candidate run interrupted before confirmation starts a new episode.
    Once confirmed, the same episode survives recovery-pending points and
    bounded data-quality suspension. Only detector-emitted timeout clear codes
    close a confirmed episode; missing/invalid points never count as recovery.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if entry_consecutive <= 0:
        raise ValueError("entry_consecutive must be positive")
    missing = sorted(set(input_columns(prefix)) - set(frame.columns))
    if missing:
        raise ValueError(f"Full-point state input is missing columns: {missing}")
    selected = [
        *input_columns(prefix),
        *(column for column in optional_input_columns(prefix) if column in frame),
    ]
    source = frame[selected].copy()
    if source[[*KEYS, "event_time"]].isna().any().any():
        raise ValueError("String/time keys must not be missing")
    source["plant_id"] = source["plant_id"].astype(str)
    source["device_no"] = source["device_no"].astype(str)
    if source[["plant_id", "device_no"]].apply(lambda s: s.str.strip().eq("")).any().any():
        raise ValueError("Plant/device identifiers must not be blank")
    numbers = pd.to_numeric(source["string_no"], errors="raise")
    if not (numbers.gt(0) & numbers.mod(1).eq(0)).all():
        raise ValueError("string_no must be a positive integer")
    source["string_no"] = numbers.astype("int64")
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    if source.duplicated([*KEYS, "event_time"]).any():
        raise ValueError("Duplicate plant/device/string/time rows are not allowed")
    source = source.sort_values([*KEYS, "event_time"]).reset_index(drop=True)
    valid = _flag(source[f"{prefix}_valid_point"], "valid_point")
    raw = _flag(source[f"{prefix}_raw_anomaly"], "raw_anomaly") & valid
    source_final = _flag(source[f"{prefix}_alert"], "source_alert")
    final = _flag(source[f"{prefix}_combined_memory_alert"], "combined_memory_alert")
    active = _flag(source[f"{prefix}_memory_active"], "memory_active")
    clear = pd.to_numeric(source[f"{prefix}_memory_clear_code"], errors="raise").fillna(0)
    if not clear.isin([0, 1, 2, 3]).all():
        raise ValueError("memory_clear_code must be 0, 1, 2 or 3")
    unknown_column = f"{prefix}_unknown_streak"
    unknown = (
        pd.to_numeric(source[unknown_column], errors="raise").fillna(0).astype("int32")
        if unknown_column in source
        else pd.Series(0, index=source.index, dtype="int32")
    )
    suspended_column = f"{prefix}_memory_suspended"
    suspended = (
        _flag(source[suspended_column], "memory_suspended")
        if suspended_column in source
        else (~valid & active & clear.eq(0))
    )
    resumed_column = f"{prefix}_memory_resumed_alert"
    resumed = (
        _flag(source[resumed_column], "memory_resumed_alert")
        if resumed_column in source
        else pd.Series(False, index=source.index)
    )
    if (final & (~raw | ~active)).any():
        raise ValueError("A formal alert has no valid current candidate evidence")
    if (final & ~source_final).any():
        raise ValueError("Combined string confirmation is missing from the source final union")
    source["_valid"] = valid
    source["_raw"] = raw
    source["_final"] = final
    source["_active"] = active
    source["_clear"] = clear.astype(int)
    source["_unknown"] = unknown
    source["_suspended"] = suspended
    source["_resumed"] = resumed
    source["_source_final"] = source_final
    source_times = source.groupby(KEYS, observed=True)["event_time"].shift()
    source["_source_gap"] = (
        source_times.notna()
        & source["event_time"].sub(source_times).ne(pd.Timedelta(minutes=interval_minutes))
    )
    # Process only evidence, remembered recovery and reset rows in Python.
    # The full input is still checked, so normal rows and duplicate keys cannot
    # be silently mistaken for consecutive candidate evidence.
    source = source[raw | final | active | clear.gt(0) | source_final]
    columns = [
        *KEYS, "event_time", "_valid", "_raw", "_final", "_active", "_clear",
        "_unknown", "_suspended", "_resumed", "_source_gap", "_source_final",
    ]
    states: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    expected = pd.Timedelta(minutes=interval_minutes)
    for key, group in source.groupby(KEYS, observed=True, sort=False):
        episode: dict[str, Any] | None = None
        previous_sparse_time: pd.Timestamp | None = None

        def close(reason: str, clear_time: Any = pd.NaT) -> None:
            nonlocal episode
            if episode is not None and pd.notna(episode["alert_confirm_time"]):
                events.append({
                    **dict(zip(KEYS, key)),
                    "string_event_id": episode["id"],
                    "candidate_start_time": episode["start"],
                    "alert_confirm_time": episode["alert_confirm_time"],
                    "last_anomaly_time": episode["last_anomaly"],
                    "clear_time": clear_time,
                    "event_end_time": (
                        clear_time if reason == "recovered" else episode["last_observed"]
                    ),
                    "anomaly_points": episode["raw_count"],
                    "anomaly_duration_minutes": episode["raw_count"] * interval_minutes,
                    "final_alert_points": episode["final_count"],
                    "final_alert_duration_minutes": episode["final_count"] * interval_minutes,
                    "recovery_observation_points": episode["normal_count"],
                    "reactivations": episode["reactivations"],
                    "clear_reason": reason,
                })
            episode = None

        for row in group[columns].itertuples(index=False, name=None):
            (
                _, _, _, timestamp, is_valid, is_raw, is_final, is_active, clear_code,
                unknown_streak, is_suspended, is_resumed, source_gap, original_final,
            ) = row
            reset_reason = ""
            sparse_gap = (
                previous_sparse_time is not None
                and timestamp - previous_sparse_time != expected
            )
            if clear_code == 2:
                reset_reason = "data_gap"
                close(reset_reason, timestamp)
            elif not is_active and (source_gap or sparse_gap):
                # An unconfirmed candidate cannot bridge a missing timestamp,
                # an invalid point or an ordinary valid-normal interruption.
                reset_reason = "data_gap" if source_gap else "candidate_interrupted"
                close(reset_reason, timestamp)
            if clear_code == 3:
                close("not_evaluable", timestamp)
                reset_reason = "not_evaluable"
            elif not is_valid and is_active:
                reset_reason = "data_quality_suspended"
            elif not is_valid:
                close("not_evaluable", timestamp)
                reset_reason = "not_evaluable"

            if is_raw and episode is None:
                episode = {
                    "id": _id("pvlof-string", *key, timestamp.isoformat()),
                    "start": timestamp, "alert_confirm_time": pd.NaT,
                    "last_anomaly": timestamp, "last_observed": timestamp,
                    "raw_count": 0, "final_count": 0, "normal_count": 0,
                    "pending_normal": False, "reactivations": 0,
                }
            reactivated = False
            if episode is not None:
                episode["last_observed"] = timestamp
                if is_raw:
                    episode["raw_count"] += 1
                    episode["last_anomaly"] = timestamp
                if is_final:
                    reactivated = (
                        pd.notna(episode["alert_confirm_time"]) and episode["pending_normal"]
                    )
                    if pd.isna(episode["alert_confirm_time"]):
                        if episode["raw_count"] < entry_consecutive:
                            raise ValueError(
                                f"Formal string confirmation without {entry_consecutive} earlier "
                                f"candidate evidence points for {key} at {timestamp}; use the "
                                "complete full_points timeline and its matching config"
                            )
                        episode["alert_confirm_time"] = timestamp
                    episode["final_count"] += 1
                    episode["reactivations"] += int(reactivated)
                if not is_raw and is_valid:
                    episode["normal_count"] += 1
                if is_valid:
                    episode["pending_normal"] = not is_raw
            elif is_active:
                raise ValueError(
                    f"Remembered state without earlier candidate evidence for {key} at "
                    f"{timestamp}; use the complete full_points timeline"
                )

            confirmed = episode is not None and pd.notna(episode["alert_confirm_time"])
            if is_suspended:
                state = "data_quality_suspended"
            elif not is_valid:
                state = "not_evaluable"
            elif is_final:
                state = "memory_reactivation" if reactivated else (
                    "data_quality_resumed" if is_resumed else (
                        "confirmed" if episode["alert_confirm_time"] == timestamp
                        else "active_anomaly"
                    )
                )
            elif is_raw:
                state = "candidate"
            elif clear_code == 1 and not is_active:
                state = "recovered"
            elif is_active:
                state = "recovery_pending"
            elif original_final:
                state = "source_alert_unconfirmed_by_string"
            else:
                state = reset_reason or "normal"
            states.append({
                **dict(zip(KEYS, key)), "event_time": timestamp,
                "string_episode_id": episode["id"] if episode else "",
                "string_event_id": episode["id"] if confirmed else "",
                "point_state": state, "raw_candidate": int(is_raw),
                "final_alert": int(is_final), "memory_active": int(is_active),
                "candidate_start_time": episode["start"] if episode else pd.NaT,
                "alert_confirm_time": episode["alert_confirm_time"] if confirmed else pd.NaT,
                "cumulative_anomaly_points": episode["raw_count"] if episode else 0,
                "cumulative_anomaly_minutes": (
                    episode["raw_count"] * interval_minutes if episode else 0
                ),
                "cumulative_final_alert_points": episode["final_count"] if episode else 0,
                "reactivated_alert": int(reactivated),
                "resumed_alert": int(is_resumed),
                "unknown_streak": int(unknown_streak),
                "memory_suspended": int(is_suspended),
                "reset_reason": reset_reason,
                "source_final_alert": int(original_final),
                "source_alert_without_string_confirmation": int(original_final and not is_final),
            })
            if clear_code == 1 and not is_active:
                close("recovered", timestamp)
            elif episode is not None and not is_active and not is_raw:
                close("source_memory_reset", timestamp)
            previous_sparse_time = timestamp
        close("end_of_data")

    state_frame = pd.DataFrame(states, columns=STATE_COLUMNS)
    event_frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return state_frame, event_frame


def _numbers(values: Any) -> str:
    return ",".join(f"{int(value):02d}" for value in sorted(set(values)))


def _mapping(frame: pd.DataFrame, value_column: str) -> str:
    return json.dumps({
        f"{int(row.string_no):02d}": getattr(row, value_column)
        for row in frame.itertuples(index=False)
    }, ensure_ascii=False, sort_keys=True)


def build_device_alarm_points(states: pd.DataFrame) -> pd.DataFrame:
    """Both sets are CURRENT sets, with no union over earlier timestamps."""
    rows = []
    for key, group in states.groupby(TIME_KEYS, observed=True, sort=True):
        raw = group[group["raw_candidate"].eq(1)]
        formal = group[group["final_alert"].eq(1)]
        pending = raw[raw["final_alert"].eq(0)]
        recovery = group[group["point_state"].eq("recovery_pending")]
        rows.append({
            **dict(zip(TIME_KEYS, key)),
            "candidate_strings": _numbers(raw["string_no"]),
            "alert_strings": _numbers(formal["string_no"]),
            "unconfirmed_candidate_strings": _numbers(pending["string_no"]),
            "recovery_pending_strings": _numbers(recovery["string_no"]),
            "member_cumulative_anomaly_minutes": _mapping(formal, "cumulative_anomaly_minutes"),
            "candidate_cumulative_anomaly_minutes": _mapping(raw, "cumulative_anomaly_minutes"),
            "string_event_ids": _mapping(formal, "string_event_id"),
            "candidate_episode_ids": _mapping(raw, "string_episode_id"),
            "max_member_cumulative_anomaly_minutes": (
                int(formal["cumulative_anomaly_minutes"].max()) if not formal.empty else 0
            ),
        })
    columns = [
        *TIME_KEYS, "candidate_strings", "alert_strings", "unconfirmed_candidate_strings",
        "recovery_pending_strings", "member_cumulative_anomaly_minutes",
        "candidate_cumulative_anomaly_minutes", "string_event_ids", "candidate_episode_ids",
        "max_member_cumulative_anomaly_minutes",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_member_segments(
    points: pd.DataFrame,
    *,
    members_column: str,
    interval_minutes: int = 5,
    kind: str = "formal",
) -> pd.DataFrame:
    """Maximal consecutive runs with exactly the same current member set."""
    columns = [
        "row", "event_id", "plant_id", "device_no", "raise_time", "end_time",
        members_column, "time_points", "duration_minutes", "segment_duration_minutes",
        "member_cumulative_anomaly_minutes", "max_member_cumulative_anomaly_minutes",
        "string_event_ids", "candidate_cumulative_anomaly_minutes", "candidate_episode_ids",
    ]
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    selected = points[points[members_column].ne("")].sort_values(TIME_KEYS).copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    link_column = "string_event_ids" if kind == "formal" else "candidate_episode_ids"
    changed = (
        selected["plant_id"].ne(selected["plant_id"].shift())
        | selected["device_no"].ne(selected["device_no"].shift())
        | selected["event_time"].diff().ne(pd.Timedelta(minutes=interval_minutes))
        | selected[members_column].ne(selected[members_column].shift())
        | selected[link_column].ne(selected[link_column].shift())
    )
    selected["_segment"] = changed.cumsum()
    rows = []
    for _, group in selected.groupby("_segment", observed=True, sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        rows.append({
            "row": len(rows) + 1,
            "event_id": _id(
                f"pvlof-{kind}", first["plant_id"], first["device_no"],
                first["event_time"].isoformat(), first[members_column], first[link_column],
            ),
            "plant_id": first["plant_id"], "device_no": first["device_no"],
            "raise_time": first["event_time"], "end_time": last["event_time"],
            members_column: first[members_column], "time_points": len(group),
            "duration_minutes": len(group) * interval_minutes,
            "segment_duration_minutes": len(group) * interval_minutes,
            **{name: last[name] for name in columns[10:]},
        })
    return pd.DataFrame(rows, columns=columns)


def build_pvlof_alarm_views(
    frame: pd.DataFrame,
    *,
    prefix: str = DEFAULT_PREFIX,
    interval_minutes: int = 5,
    entry_consecutive: int = 3,
    timezone: str = "Asia/Shanghai",
) -> dict[str, pd.DataFrame]:
    states, events = build_string_alarm_states(
        frame, prefix=prefix, interval_minutes=interval_minutes,
        entry_consecutive=entry_consecutive,
    )
    device = build_device_alarm_points(states)
    formal = device[device["alert_strings"].ne("")].copy()
    candidates = device[device["candidate_strings"].ne("")].copy()
    return {
        name: _local_times(output, timezone)
        for name, output in {
            "string_states": states,
            "string_events": events,
            "final_alert_points": formal,
            "raw_candidate_points": candidates,
            "confirmed_events": build_member_segments(
                formal, members_column="alert_strings", interval_minutes=interval_minutes,
            ),
            "candidate_segments": build_member_segments(
                candidates, members_column="candidate_strings", interval_minutes=interval_minutes,
                kind="candidate",
            ),
        }.items()
    }
