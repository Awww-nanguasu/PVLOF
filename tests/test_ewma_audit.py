import pandas as pd

from pv_anomaly.ewma_audit import (
    add_weak_label,
    binary_metrics,
    collapse_alert_events,
    match_event_tables,
)


def test_binary_metrics():
    result = binary_metrics(pd.Series([True, True, False, False]), pd.Series([True, False, True, False]))
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_collapse_alert_events_and_event_matching():
    frame = pd.DataFrame(
        {
            "device_no": ["a", "a", "a", "a", "a"],
            "target_time": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
            "ewma_alert": [0, 1, 1, 0, 1],
        }
    )
    events = collapse_alert_events(frame, alert_column="ewma_alert")
    assert len(events) == 2
    assert events["points"].tolist() == [2, 1]
    assert events["duration_minutes"].tolist() == [10.0, 5.0]
    assert match_event_tables(events, events)["f1"] == 1.0


def test_add_weak_label_aligns_at_target_time():
    alerts = pd.DataFrame(
        {
            "device_no": ["a", "a"],
            "target_time": pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
            "ewma_alert": [0, 1],
        }
    )
    labels = pd.DataFrame(
        {
            "device_no": ["a", "a"],
            "event_time": pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
            "string_overall_status": [1, 2],
        }
    )
    merged, report = add_weak_label(alerts, labels)
    assert merged["weak_current_label"].tolist() == [False, True]
    assert report["matched_rows"] == 2
    assert report["positive_labels_after_alignment"] == 1


def test_add_weak_label_accepts_zero_current_status_when_requested():
    alerts = pd.DataFrame(
        {
            "device_no": ["a"],
            "target_time": pd.to_datetime(["2026-01-01"], utc=True),
            "ewma_alert": [1],
        }
    )
    labels = pd.DataFrame(
        {
            "device_no": ["a"],
            "event_time": pd.to_datetime(["2026-01-01"], utc=True),
            "string_overall_status": [3],
        }
    )
    merged, report = add_weak_label(alerts, labels, label_statuses=(2, 3, 4))
    assert merged["weak_current_label"].tolist() == [True]
    assert report["positive_labels_after_alignment"] == 1
