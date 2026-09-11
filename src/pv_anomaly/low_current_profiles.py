"""Historical profile features for low-current string events.

Event-level features are causal: an event can only see observations and alarms
strictly before its own start. The separate retrospective string table uses the
complete requested period and is intended for review/reporting, not live labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRING_KEYS = ["plant_id", "device_no", "string_no"]
CLASS_NAMES_ZH = {
    "insufficient_history": "历史不足",
    "long_term_recurrent": "长期重复型",
    "sudden_new": "突发型",
    "no_event": "无低电流事件",
}


@dataclass(frozen=True)
class LowCurrentProfileConfig:
    """Configuration for the first interpretable profile baseline."""

    version: str = "low-current-profile-v0.1"
    source_algorithm_version: str = (
        "pvlof-v1.7-improved-v2-internal-member-gate"
    )
    source_points_filename: str = "pvlof_v17_improved_v2_full_points.parquet"
    raw_candidate_column: str = "pvlof_v17_improved_raw_anomaly"
    final_alert_column: str = "pvlof_v17_improved_alert"
    valid_point_column: str = "pvlof_v17_improved_valid_point"
    timezone: str = "Asia/Shanghai"
    interval_minutes: int = 5
    entry_consecutive: int = 3
    minimum_valid_points_per_day: int = 3
    history_window_days: int = 30
    minimum_valid_history_days: int = 14
    minimum_prior_alarm_days_long_term: int = 3
    minimum_prior_alarm_day_ratio_long_term: float = 0.0
    fixed_time_bin_minutes: int = 60
    minimum_fixed_time_alarm_days: int = 3
    minimum_fixed_time_concentration: float = 0.60
    batch_size: int = 250_000
    provisional_thresholds: bool = True

    def __post_init__(self) -> None:
        positive = {
            "interval_minutes": self.interval_minutes,
            "entry_consecutive": self.entry_consecutive,
            "minimum_valid_points_per_day": self.minimum_valid_points_per_day,
            "history_window_days": self.history_window_days,
            "minimum_valid_history_days": self.minimum_valid_history_days,
            "minimum_prior_alarm_days_long_term": (
                self.minimum_prior_alarm_days_long_term
            ),
            "fixed_time_bin_minutes": self.fixed_time_bin_minutes,
            "minimum_fixed_time_alarm_days": self.minimum_fixed_time_alarm_days,
            "batch_size": self.batch_size,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"Profile configuration values must be positive: {invalid}")
        if 1_440 % self.fixed_time_bin_minutes:
            raise ValueError("fixed_time_bin_minutes must divide one day exactly")
        for name, value in (
            (
                "minimum_prior_alarm_day_ratio_long_term",
                self.minimum_prior_alarm_day_ratio_long_term,
            ),
            (
                "minimum_fixed_time_concentration",
                self.minimum_fixed_time_concentration,
            ),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LowCurrentProfileConfig":
        return cls(**dict(payload))


def load_config(path: str | Path) -> LowCurrentProfileConfig:
    return LowCurrentProfileConfig.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def save_config(config: LowCurrentProfileConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_valid_day_coverage(
    daily_counts: pd.DataFrame,
    *,
    minimum_valid_points_per_day: int,
) -> pd.DataFrame:
    """Consolidate streamed valid-point counts into physical string-days."""

    required = {*STRING_KEYS, "local_date", "valid_point_count"}
    missing = sorted(required - set(daily_counts.columns))
    if missing:
        raise ValueError(f"Valid-day counts are missing columns: {missing}")
    if minimum_valid_points_per_day < 1:
        raise ValueError("minimum_valid_points_per_day must be positive")

    source = daily_counts.copy()
    source["plant_id"] = source["plant_id"].astype("string")
    source["device_no"] = source["device_no"].astype("string")
    source["string_no"] = pd.to_numeric(
        source["string_no"], errors="raise"
    ).astype("Int64")
    source["local_date"] = pd.to_datetime(
        source["local_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    source["valid_point_count"] = pd.to_numeric(
        source["valid_point_count"], errors="raise"
    )
    coverage = (
        source.groupby([*STRING_KEYS, "local_date"], observed=True, as_index=False)[
            "valid_point_count"
        ]
        .sum()
        .sort_values([*STRING_KEYS, "local_date"])
        .reset_index(drop=True)
    )
    coverage["is_valid_day"] = coverage["valid_point_count"].ge(
        minimum_valid_points_per_day
    ).astype("int8")
    return coverage


def _normalise_events(
    events: pd.DataFrame,
    *,
    timezone: str,
) -> pd.DataFrame:
    required = {*STRING_KEYS, "candidate_start_time", "candidate_end_time"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"String events are missing columns: {missing}")
    result = events.copy()
    result["plant_id"] = result["plant_id"].astype("string")
    result["device_no"] = result["device_no"].astype("string")
    result["string_no"] = pd.to_numeric(
        result["string_no"], errors="raise"
    ).astype("Int64")
    for column in ("candidate_start_time", "candidate_end_time"):
        result[column] = pd.to_datetime(result[column], errors="raise", utc=True)
    local_start = result["candidate_start_time"].dt.tz_convert(timezone)
    result["_local_date"] = local_start.dt.date
    result["_start_minute"] = local_start.dt.hour * 60 + local_start.dt.minute
    return result.sort_values(
        [*STRING_KEYS, "candidate_start_time"]
    ).reset_index(drop=True)


def _coverage_date_map(coverage: pd.DataFrame) -> dict[tuple[str, str, int], list[date]]:
    required = {*STRING_KEYS, "local_date", "is_valid_day"}
    missing = sorted(required - set(coverage.columns))
    if missing:
        raise ValueError(f"Valid-day coverage is missing columns: {missing}")
    selected = coverage[coverage["is_valid_day"].fillna(False).astype(bool)].copy()
    selected["local_date"] = pd.to_datetime(
        selected["local_date"], errors="raise"
    ).dt.date
    result: dict[tuple[str, str, int], list[date]] = {}
    for key, group in selected.groupby(STRING_KEYS, observed=True, sort=False):
        normalised = (str(key[0]), str(key[1]), int(key[2]))
        result[normalised] = sorted(set(group["local_date"].tolist()))
    return result


def _time_bin(minute_of_day: int, bin_minutes: int) -> int:
    return int(minute_of_day // bin_minutes)


def _time_pattern(
    events: pd.DataFrame,
    *,
    bin_minutes: int,
) -> tuple[int | None, float]:
    if events.empty:
        return None, float("nan")
    per_day_bin = {
        (local_date, _time_bin(int(start_minute), bin_minutes))
        for local_date, start_minute in zip(
            events["_local_date"],
            events["_start_minute"],
            strict=True,
        )
    }
    if not per_day_bin:
        return None, float("nan")
    counts: dict[int, int] = {}
    for _, bin_number in per_day_bin:
        counts[bin_number] = counts.get(bin_number, 0) + 1
    dominant = min(
        (bin_number for bin_number, count in counts.items() if count == max(counts.values())),
        default=None,
    )
    alarm_days = len({day for day, _ in per_day_bin})
    concentration = counts[dominant] / alarm_days if dominant is not None else float("nan")
    return dominant, float(concentration)


def _classify(
    *,
    valid_days: int,
    alarm_days: int,
    alarm_day_ratio: float,
    config: LowCurrentProfileConfig,
) -> tuple[str, str]:
    if valid_days < config.minimum_valid_history_days:
        code = "insufficient_history"
        reason = (
            f"valid_days={valid_days} < "
            f"{config.minimum_valid_history_days}"
        )
        return code, reason
    if (
        alarm_days >= config.minimum_prior_alarm_days_long_term
        and alarm_day_ratio >= config.minimum_prior_alarm_day_ratio_long_term
    ):
        code = "long_term_recurrent"
        reason = (
            f"alarm_days={alarm_days} >= "
            f"{config.minimum_prior_alarm_days_long_term}; "
            f"alarm_day_ratio={alarm_day_ratio:.4f}"
        )
        return code, reason
    code = "sudden_new"
    reason = (
        f"alarm_days={alarm_days} < "
        f"{config.minimum_prior_alarm_days_long_term}; "
        f"valid_days={valid_days}"
    )
    return code, reason


def _window_label(bin_number: int | None, bin_minutes: int) -> str:
    if bin_number is None:
        return ""
    start = bin_number * bin_minutes
    end = start + bin_minutes
    return (
        f"{start // 60:02d}:{start % 60:02d}-"
        f"{(end // 60) % 24:02d}:{end % 60:02d}"
    )


def attach_historical_features(
    events: pd.DataFrame,
    valid_day_coverage: pd.DataFrame,
    config: LowCurrentProfileConfig,
) -> pd.DataFrame:
    """Attach causal history features and provisional labels to string events."""

    source = _normalise_events(events, timezone=config.timezone)
    if source.empty:
        return source.drop(columns=["_local_date", "_start_minute"], errors="ignore")
    coverage_map = _coverage_date_map(valid_day_coverage)
    feature_rows: list[dict[str, Any]] = []

    for raw_key, raw_group in source.groupby(STRING_KEYS, observed=True, sort=False):
        key = (str(raw_key[0]), str(raw_key[1]), int(raw_key[2]))
        group = raw_group.sort_values("candidate_start_time").reset_index()
        valid_dates = coverage_map.get(key, [])

        for position, row in group.iterrows():
            current_start = row["candidate_start_time"]
            current_date = row["_local_date"]
            prior = group.iloc[:position]
            window_start = current_start - pd.Timedelta(
                days=config.history_window_days
            )
            prior_window = prior[prior["candidate_start_time"].ge(window_start)]
            prior_7d = prior[
                prior["candidate_start_time"].ge(
                    current_start - pd.Timedelta(days=7)
                )
            ]

            prior_alarm_dates_all = set(prior["_local_date"].tolist())
            prior_alarm_dates_window = set(prior_window["_local_date"].tolist())
            prior_alarm_dates_7d = set(prior_7d["_local_date"].tolist())
            first_valid_date = current_date - timedelta(
                days=config.history_window_days
            )
            valid_dates_window = [
                value
                for value in valid_dates
                if first_valid_date <= value < current_date
            ]
            valid_days = len(valid_dates_window)
            alarm_days = len(prior_alarm_dates_window)
            alarm_day_ratio = alarm_days / valid_days if valid_days else float("nan")

            dominant_bin, time_concentration = _time_pattern(
                prior_window,
                bin_minutes=config.fixed_time_bin_minutes,
            )
            current_bin = _time_bin(
                int(row["_start_minute"]),
                config.fixed_time_bin_minutes,
            )
            prior_fixed_time = bool(
                alarm_days >= config.minimum_fixed_time_alarm_days
                and pd.notna(time_concentration)
                and time_concentration >= config.minimum_fixed_time_concentration
            )
            current_matches = bool(
                dominant_bin is not None and current_bin == dominant_bin
            )
            class_ratio = alarm_day_ratio if pd.notna(alarm_day_ratio) else 0.0
            class_code, class_reason = _classify(
                valid_days=valid_days,
                alarm_days=alarm_days,
                alarm_day_ratio=class_ratio,
                config=config,
            )
            days_since_previous = (
                (
                    current_start - prior.iloc[-1]["candidate_start_time"]
                ).total_seconds()
                / 86_400
                if not prior.empty
                else float("nan")
            )
            feature_rows.append(
                {
                    "_row_index": int(row["index"]),
                    "prior_event_count_all": int(len(prior)),
                    "prior_alarm_days_all": int(len(prior_alarm_dates_all)),
                    "prior_event_count_7d": int(len(prior_7d)),
                    "prior_alarm_days_7d": int(len(prior_alarm_dates_7d)),
                    "prior_event_count_window": int(len(prior_window)),
                    "prior_alarm_days_window": alarm_days,
                    "prior_valid_days_window": valid_days,
                    "prior_alarm_day_ratio_window": alarm_day_ratio,
                    "days_since_previous_event": days_since_previous,
                    "is_first_seen": int(prior.empty),
                    "prior_dominant_time_bin": (
                        int(dominant_bin) if dominant_bin is not None else pd.NA
                    ),
                    "prior_dominant_time_window": _window_label(
                        dominant_bin,
                        config.fixed_time_bin_minutes,
                    ),
                    "prior_time_concentration": time_concentration,
                    "prior_fixed_time_pattern": int(prior_fixed_time),
                    "current_matches_dominant_time": int(current_matches),
                    "fixed_time_tag": int(prior_fixed_time and current_matches),
                    "profile_class_code": class_code,
                    "profile_class": CLASS_NAMES_ZH[class_code],
                    "profile_class_reason": class_reason,
                }
            )

    features = pd.DataFrame(feature_rows).set_index("_row_index")
    result = source.join(features, how="left")
    result["manual_profile_class"] = ""
    result["manual_fixed_time"] = ""
    result["manual_review_note"] = ""
    return result.drop(columns=["_local_date", "_start_minute"]).reset_index(drop=True)


def _safe_stat(group: pd.DataFrame, column: str, operation: str) -> float:
    if column not in group:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce")
    if not values.notna().any():
        return float("nan")
    return float(getattr(values, operation)())


def _median_alarm_interval_days(alarm_dates: set[date]) -> float:
    ordered = sorted(alarm_dates)
    if len(ordered) < 2:
        return float("nan")
    intervals = [
        (right - left).days
        for left, right in zip(ordered[:-1], ordered[1:], strict=True)
    ]
    return float(np.median(intervals))


def _count_flag(group: pd.DataFrame, column: str) -> int:
    if column not in group:
        return 0
    return int(
        pd.to_numeric(group[column], errors="coerce")
        .fillna(0)
        .astype(bool)
        .sum()
    )


def build_string_profiles(
    event_profiles: pd.DataFrame,
    valid_day_coverage: pd.DataFrame,
    config: LowCurrentProfileConfig,
) -> pd.DataFrame:
    """Build one retrospective profile for every string with an event."""

    source = _normalise_events(event_profiles, timezone=config.timezone)
    if source.empty:
        return pd.DataFrame()
    coverage_map = _coverage_date_map(valid_day_coverage)
    rows: list[dict[str, Any]] = []

    for raw_key, group in source.groupby(STRING_KEYS, observed=True, sort=False):
        key = (str(raw_key[0]), str(raw_key[1]), int(raw_key[2]))
        valid_dates = coverage_map.get(key, [])
        alarm_dates = set(group["_local_date"].tolist())
        valid_days = len(valid_dates)
        alarm_days = len(alarm_dates)
        alarm_day_ratio = alarm_days / valid_days if valid_days else float("nan")
        dominant_bin, time_concentration = _time_pattern(
            group,
            bin_minutes=config.fixed_time_bin_minutes,
        )
        class_ratio = alarm_day_ratio if pd.notna(alarm_day_ratio) else 0.0
        class_code, class_reason = _classify(
            valid_days=valid_days,
            alarm_days=alarm_days,
            alarm_day_ratio=class_ratio,
            config=config,
        )
        fixed_time = bool(
            alarm_days >= config.minimum_fixed_time_alarm_days
            and pd.notna(time_concentration)
            and time_concentration >= config.minimum_fixed_time_concentration
        )
        morphology = group.get(
            "event_morphology",
            pd.Series("isolated", index=group.index),
        ).fillna("isolated")
        source_branches = group.get(
            "source_branches",
            pd.Series("", index=group.index),
        )
        rows.append(
            {
                "plant_id": key[0],
                "device_no": key[1],
                "string_no": key[2],
                "history_start_local": (
                    min(valid_dates).isoformat() if valid_dates else ""
                ),
                "history_end_local": (
                    max(valid_dates).isoformat() if valid_dates else ""
                ),
                "valid_days": valid_days,
                "event_count": int(len(group)),
                "alarm_days": alarm_days,
                "alarm_day_ratio": alarm_day_ratio,
                "first_event_time_local": group.iloc[0].get(
                    "candidate_start_time_local",
                    "",
                ),
                "last_event_time_local": group.iloc[-1].get(
                    "candidate_start_time_local",
                    "",
                ),
                "median_days_between_alarm_dates": _median_alarm_interval_days(
                    alarm_dates
                ),
                "median_event_duration_minutes": _safe_stat(
                    group,
                    "candidate_duration_minutes",
                    "median",
                ),
                "maximum_event_duration_minutes": _safe_stat(
                    group,
                    "candidate_duration_minutes",
                    "max",
                ),
                "median_maximum_relative_drop": _safe_stat(
                    group,
                    "maximum_relative_drop",
                    "median",
                ),
                "maximum_relative_drop": _safe_stat(
                    group,
                    "maximum_relative_drop",
                    "max",
                ),
                "median_maximum_absolute_drop": _safe_stat(
                    group,
                    "maximum_absolute_drop",
                    "median",
                ),
                "maximum_absolute_drop": _safe_stat(
                    group,
                    "maximum_absolute_drop",
                    "max",
                ),
                "three_point_event_count": _count_flag(
                    group,
                    "meets_three_point_rule",
                ),
                "final_alert_event_count": _count_flag(
                    group,
                    "has_final_alert",
                ),
                "isolated_event_count": int(morphology.eq("isolated").sum()),
                "group_event_count": int(morphology.eq("group").sum()),
                "gradient_group_event_count": int(
                    morphology.eq("gradient_group").sum()
                ),
                "dominant_time_bin": (
                    int(dominant_bin) if dominant_bin is not None else pd.NA
                ),
                "dominant_time_window": _window_label(
                    dominant_bin,
                    config.fixed_time_bin_minutes,
                ),
                "time_concentration": time_concentration,
                "fixed_time_tag": int(fixed_time),
                "profile_class_code": class_code,
                "profile_class": CLASS_NAMES_ZH[class_code],
                "profile_class_reason": class_reason,
                "source_branches": ",".join(
                    sorted(
                        {
                            item
                            for value in source_branches
                            for item in str(value).split(",")
                            if item
                        }
                    )
                ),
                "manual_profile_class": "",
                "manual_fixed_time": "",
                "manual_review_note": "",
            }
        )
    return pd.DataFrame(rows).sort_values(STRING_KEYS).reset_index(drop=True)


__all__ = [
    "CLASS_NAMES_ZH",
    "LowCurrentProfileConfig",
    "attach_historical_features",
    "build_string_profiles",
    "build_valid_day_coverage",
    "load_config",
    "save_config",
]
