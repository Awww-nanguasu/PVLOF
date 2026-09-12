import pandas as pd

from pv_anomaly.pvlof_events import reconstruct_pvlof_events


def _base_frame(raw, final):
    size = len(raw)
    return pd.DataFrame({
        "plant_id": ["234"] * size,
        "device_no": ["dev-a"] * size,
        "string_no": [1] * size,
        "event_time": pd.date_range("2026-06-01", periods=size, freq="5min", tz="UTC"),
        "isolated_directional_raw_alert": raw,
        "isolated_directional_alert": final,
        "isolated_hier_raw_alert": [0] * size,
        "isolated_hier_strict_alert": [0] * size,
        "pvlof_v2_hier_strict_alert": final,
        "isolated_directional_consecutive": [
            min(index + 1, 3) if value else 0
            for index, value in enumerate(raw)
        ],
    })


def test_v14_confirmed_event_backfills_candidate_evidence():
    evidence, events, unconfirmed = reconstruct_pvlof_events(
        _base_frame([1, 1, 1, 0], [0, 0, 1, 0]),
        version="v1.4",
        final_alert_column="pvlof_v2_hier_strict_alert",
    )
    assert evidence["point_state"].tolist() == ["candidate", "candidate", "confirmed"]
    assert events.loc[0, "evidence_points"] == 3
    assert events.loc[0, "final_alert_points"] == 1
    assert events.loc[0, "detection_delay_minutes"] == 10
    assert events.loc[0, "confirmation_branch"] == "base_directional"
    assert unconfirmed.empty


def test_unconfirmed_candidates_are_kept_out_of_formal_events():
    evidence, events, unconfirmed = reconstruct_pvlof_events(
        _base_frame([1, 1, 0, 0], [0, 0, 0, 0]),
        version="v1.4",
        final_alert_column="pvlof_v2_hier_strict_alert",
    )
    assert evidence.empty
    assert events.empty
    assert unconfirmed.loc[0, "candidate_points"] == 2


def test_v15_memory_reactivation_and_recovery_share_one_event():
    frame = _base_frame(
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
    )
    frame["pvlof_v15_raw_anomaly"] = frame["isolated_directional_raw_alert"]
    frame["pvlof_v15_alert"] = frame["pvlof_v2_hier_strict_alert"]
    frame["pvlof_v15_memory_active"] = [0, 0, 1, 1, 1, 1, 1, 0]
    frame["pvlof_v15_memory_reactivated_alert"] = [0, 0, 0, 0, 1, 0, 0, 0]
    frame["pvlof_v15_entry_streak"] = [1, 2, 3, 3, 3, 3, 3, 0]
    frame["pvlof_v15_normal_streak"] = [0, 0, 0, 1, 0, 1, 2, 0]
    frame["pvlof_v15_memory_clear_code"] = [0, 0, 0, 0, 0, 0, 0, 1]
    evidence, events, _ = reconstruct_pvlof_events(
        frame,
        version="v1.5",
        final_alert_column="pvlof_v15_alert",
        use_v15_memory=True,
    )
    assert len(events) == 1
    assert evidence["point_state"].tolist() == [
        "candidate", "candidate", "confirmed", "recovery_pending",
        "memory_reactivation", "recovery_pending", "recovery_pending", "recovered",
    ]
    assert events.loc[0, "evidence_points"] == 4
    assert events.loc[0, "final_alert_points"] == 2
    assert events.loc[0, "clear_reason"] == "recovered"
