import pandas as pd

from pv_anomaly.pvlof_v2 import (
    PVLOFV2Calibration,
    _add_consecutive,
    apply_pvlof_v2,
    fit_pvlof_v2_calibration,
)


def _frame(rows: list[dict[str, list[float]]]) -> pd.DataFrame:
    # The input helper is intentionally explicit: one row per device/time.
    records: list[dict[str, object]] = []
    devices = sorted({device for row in rows for device in row})
    for row_index, row in enumerate(rows):
        for device in devices:
            currents = row[device]
            record: dict[str, object] = {
                "event_time": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(minutes=5 * row_index),
                "plant_id": 234,
                "device_no": device,
                "main_string_count": len(currents),
            }
            for string_index, current in enumerate(currents, start=1):
                record[f"string_current_{string_index:02d}"] = current
            records.append(record)
    return pd.DataFrame(records)


def test_virtual_irradiance_uses_other_devices_and_common_drop_is_not_alerted():
    normal = [[10, 10, 10, 10, 10], [12, 12, 12, 12, 12], [8, 8, 8, 8, 8]]
    frame = _frame([{"a": row, "b": [2 * x for x in row], "c": [1.5 * x for x in row]} for row in normal])
    calibration, _ = fit_pvlof_v2_calibration(
        frame,
        n_neighbors=2,
        minimum_peer_devices=2,
        minimum_strings=4,
        max_score_rows=1000,
    )
    common_drop = _frame([
        {"a": [5, 5, 5, 5, 5], "b": [10, 10, 10, 10, 10], "c": [7.5, 7.5, 7.5, 7.5, 7.5]},
    ])
    scored = apply_pvlof_v2(common_drop, calibration)
    assert scored["virtual_irradiance"].notna().all()
    assert not scored["pvlof_v2_alert"].any()


def test_isolated_and_collective_nonzero_low_current_are_detected():
    normal = [[10, 10, 10, 10, 10], [12, 12, 12, 12, 12], [8, 8, 8, 8, 8]]
    training = _frame([{"a": row, "b": [2 * x for x in row], "c": [1.5 * x for x in row]} for row in normal])
    calibration, _ = fit_pvlof_v2_calibration(
        training,
        n_neighbors=2,
        minimum_peer_devices=2,
        minimum_strings=4,
        max_score_rows=1000,
    )
    target_rows = [
        {"a": [10, 10, 10, 10, 10], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
        {"a": [10, 10, 10, 10, 10], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
        {"a": [10, 10, 10, 10, 3], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
        {"a": [10, 10, 10, 10, 3], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
    ]
    scored = apply_pvlof_v2(_frame(target_rows), calibration)
    target = scored[scored["device_no"].eq("a")]
    assert target.loc[target["string_no"].eq(5), "pvlof_v2_alert"].tolist()[-2:] == [1, 1]
    assert target.loc[target["string_no"].eq(5), "alert_reason"].iloc[-1] in {"isolated", "both"}

    collective_rows = [
        {"a": [10, 10, 5, 5, 10], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
        {"a": [10, 10, 5, 5, 10], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
    ]
    collective = apply_pvlof_v2(_frame(collective_rows), calibration)
    low = collective[(collective["device_no"].eq("a")) & (collective["string_no"].isin([3, 4]))]
    assert low["collective_raw_alert"].any()


def test_zero_current_is_not_pvlof_v2_alert():
    training = _frame([
        {"a": [10, 10, 10, 10, 10], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
        {"a": [12, 12, 12, 12, 12], "b": [24, 24, 24, 24, 24], "c": [18, 18, 18, 18, 18]},
    ])
    calibration, _ = fit_pvlof_v2_calibration(
        training, n_neighbors=2, minimum_peer_devices=2, minimum_strings=4, max_score_rows=1000
    )
    scored = apply_pvlof_v2(_frame([
        {"a": [10, 10, 10, 10, 0], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
        {"a": [10, 10, 10, 10, 0], "b": [20, 20, 20, 20, 20], "c": [15, 15, 15, 15, 15]},
    ]), calibration)
    zero = scored[scored["device_no"].eq("a") & scored["string_no"].eq(5)]
    assert zero["zero_current_alert"].eq(1).all()
    assert not zero["pvlof_v2_alert"].any()


def test_collective_event_alerts_new_members_after_group_is_confirmed():
    training = _frame([
        {"a": [10] * 6, "b": [20] * 6, "c": [15] * 6},
        {"a": [12] * 6, "b": [24] * 6, "c": [18] * 6},
        {"a": [8] * 6, "b": [16] * 6, "c": [12] * 6},
    ])
    calibration, _ = fit_pvlof_v2_calibration(
        training,
        n_neighbors=2,
        minimum_peer_devices=2,
        minimum_strings=4,
        max_score_rows=1000,
    )
    scored = apply_pvlof_v2(_frame([
        {"a": [10, 10, 5, 5, 10, 10], "b": [20] * 6, "c": [15] * 6},
        {"a": [10, 10, 5, 5, 7, 10], "b": [20] * 6, "c": [15] * 6},
    ]), calibration)
    target = scored[scored["device_no"].eq("a")]
    second = target[target["event_time"].eq(target["event_time"].max())]
    assert second["collective_event_alert"].iloc[0] == 1
    assert second.loc[second["string_no"].eq(5), "collective_member_alert"].iloc[0] == 1
    assert second.loc[second["string_no"].eq(5), "pvlof_v2_alert"].iloc[0] == 1
    assert second.loc[second["string_no"].eq(5), "collective_event_consecutive"].iloc[0] == 2


def test_group_continuity_modification_preserves_legacy_mixed_branch_alerts():
    times = pd.date_range("2026-06-01", periods=2, freq="5min", tz="UTC")
    frame = pd.DataFrame([
        {
            "plant_id": "234",
            "device_no": "a",
            "string_no": string_no,
            "event_time": timestamp,
            "isolated_raw_alert": int(point == 0 and string_no == 1),
            "collective_raw_alert": int(point == 1),
            "zero_current_alert": 0,
        }
        for point, timestamp in enumerate(times)
        for string_no in (1, 2)
    ])
    scored = _add_consecutive(
        frame,
        PVLOFV2Calibration(minimum_collective_strings=2, minimum_consecutive=2),
    )
    target = scored[
        scored["event_time"].eq(times[1]) & scored["string_no"].eq(1)
    ].iloc[0]
    assert target["pvlof_v2_legacy_alert"] == 1
    assert target["pvlof_v2_alert"] == 1
    assert target["alert_reason"] == "mixed_persistence"
