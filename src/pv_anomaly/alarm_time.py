"""Time-grid helpers for aligning alarm intervals to telemetry samples."""

from __future__ import annotations

import pandas as pd


def alarm_time_grid(
    raise_time: object,
    end_time: object,
    *,
    interval_minutes: int = 5,
) -> pd.DatetimeIndex:
    """Return the inclusive telemetry grid covered by an alarm interval.

    The sample at ``floor(raise_time)`` covers the beginning of an alarm. Both
    boundaries are therefore floored. This prevents alarms containing seconds
    or milliseconds from losing their first relevant five-minute sample.
    """

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    start = pd.Timestamp(raise_time)
    end = pd.Timestamp(end_time)
    if pd.isna(start) or pd.isna(end):
        return pd.DatetimeIndex([])
    frequency = f"{interval_minutes}min"
    start = start.floor(frequency)
    end = end.floor(frequency)
    if end < start:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start, end=end, freq=frequency)
