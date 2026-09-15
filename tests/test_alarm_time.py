import pandas as pd

from pv_anomaly.alarm_time import alarm_time_grid


def test_alarm_grid_floors_raise_and_end_and_is_inclusive():
    result = alarm_time_grid(
        "2026-04-09 13:40:05.961+08:00",
        "2026-04-09 13:45:05.641+08:00",
    )

    assert result.tolist() == [
        pd.Timestamp("2026-04-09 13:40:00+08:00"),
        pd.Timestamp("2026-04-09 13:45:00+08:00"),
    ]


def test_alarm_grid_returns_one_point_inside_same_bucket():
    result = alarm_time_grid(
        "2026-04-09 13:40:05.961+08:00",
        "2026-04-09 13:44:59.999+08:00",
    )

    assert result.tolist() == [pd.Timestamp("2026-04-09 13:40:00+08:00")]
