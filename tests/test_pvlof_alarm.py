import pandas as pd

from pv_anomaly.pvlof_alarm import (
    evaluate_pvlof,
    expand_alarm_points,
    load_device_manifest,
    parse_alarm_string,
    read_alarm_events,
    restrict_alarm_events_to_predictions,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_name": ["拓斯达", "拓斯达"],
            "device_no": ["device-a", "device-b"],
        }
    )


def test_parse_alarm_string():
    assert parse_alarm_string("3") == (3,)
    assert parse_alarm_string("3, 10,11") == (3, 10, 11)
    assert parse_alarm_string("") == ()


def test_read_alarm_events_and_expand(tmp_path):
    alarm_path = tmp_path / "alarms.csv"
    pd.DataFrame(
        [
            {
                "id": "zero-1",
                "alarm_code": "101002",
                "alarm_name": "组串3电流为零",
                "alarm_string": "3",
                "device_no": "device-a",
                "station_name": "拓斯达",
                "raise_time": "1780279200000",
                "end_time": "1780279800000",
            },
            {
                "id": "low-1",
                "alarm_code": "101001",
                "alarm_name": "组串电流偏低",
                "alarm_string": "",
                "device_no": "device-a",
                "station_name": "拓斯达",
                "raise_time": "1780279800000",
                "end_time": "1780280400000",
            },
        ]
    ).to_csv(alarm_path, index=False, encoding="utf-8-sig")
    events, report = read_alarm_events(alarm_path, _manifest())
    points, point_report = expand_alarm_points(events)
    assert len(events) == 2
    assert report["string_scope_rows"] == 1
    assert report["device_scope_rows"] == 1
    assert point_report["label_points"] == 6
    assert set(points["label_scope"]) == {"string", "device"}


def test_evaluate_string_level_prediction_against_zero_alarm():
    times = pd.date_range("2026-06-01 10:00", periods=3, freq="5min", tz="UTC")
    predictions = pd.DataFrame(
        {
            "station_name": ["拓斯达"] * 3,
            "device_no": ["device-a"] * 3,
            "string_no": [3] * 3,
            "event_time": times,
            "combined_alert": [1, 1, 0],
            "pvlof_alert": [1, 1, 0],
            "zero_current_alert": [1, 1, 0],
        }
    )
    events = pd.DataFrame(
        {
            "alarm_event_id": ["zero-1"],
            "station_name": ["拓斯达"],
            "device_no": ["device-a"],
            "alarm_code": ["101002"],
            "alarm_name": ["组串3电流为零"],
            "alarm_string": ["3"],
            "string_no": pd.Series([3], dtype="Int64"),
            "label_scope": ["string"],
            "raise_time": [times[0]],
            "end_time": [times[-1]],
        }
    )
    label_points, _ = expand_alarm_points(events)
    summary, matches = evaluate_pvlof(predictions, events, label_points)
    string_point = next(
        row
        for row in summary["point_metrics"]
        if row["prediction_column"] == "combined_alert" and row["scope"] == "string"
    )
    string_event = next(
        row
        for row in summary["event_metrics"]
        if row["prediction_column"] == "combined_alert" and row["label_scope"] == "string"
    )
    assert string_point["positive_labels"] == 3
    assert string_point["true_positive"] == 2
    assert string_event["predicted_events_hit"] == 1
    assert string_event["label_events_hit"] == 1
    assert len(matches) >= 1


def test_alarm_events_are_restricted_to_pvlof_coverage():
    events = pd.DataFrame(
        {
            "station_name": ["拓斯达", "益成工业园一期"],
            "device_no": ["device-a", "device-b"],
        }
    )
    predictions = pd.DataFrame(
        {
            "station_name": ["拓斯达"],
            "device_no": ["device-a"],
        }
    )
    covered, report = restrict_alarm_events_to_predictions(events, predictions)
    assert len(covered) == 1
    assert report["excluded_events"] == 1
    assert report["excluded_stations"] == ["益成工业园一期"]


def test_manifest_file_has_three_stations():
    manifest = load_device_manifest("configs/test_station_devices.json")
    assert manifest["station_name"].nunique() == 3
    assert len(manifest) == 74
