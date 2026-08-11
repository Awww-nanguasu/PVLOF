import pandas as pd
import pytest

from pv_anomaly.pvlof_v12_persistent import (
    apply_persistent_collective,
    collapse_persistent_events,
    fit_persistent_calibration,
)


def _scored(periods: int = 12) -> pd.DataFrame:
    rows = []
    times = pd.date_range("2026-06-01", periods=periods, freq="5min", tz="UTC")
    for timestamp in times:
        for string_no in range(1, 9):
            member = string_no <= 3
            rows.append(
                {
                    "event_time": timestamp,
                    "plant_id": "234",
                    "device_no": "dev-a",
                    "string_no": string_no,
                    "residual_ratio": 0.9 if member else 1.0,
                    "string_current": 9.0 if member else 10.0,
                    "response_known": True,
                    "collective_group_member": member,
                }
            )
    return pd.DataFrame(rows)


def test_persistent_collective_requires_six_continuous_points():
    frame = _scored()
    calibration, report = fit_persistent_calibration(
        frame,
        deficit_quantile=0.5,
        minimum_deficit=0.03,
        minimum_device_samples=1,
        minimum_device_days=1,
        minimum_plant_count_samples=1,
        minimum_plant_count_days=1,
        minimum_count_samples=1,
        minimum_count_days=1,
        shrinkage_k=1,
        minimum_consecutive=6,
    )
    # Only points with a complete six-point stable window calibrate the threshold.
    assert report["structural_candidate_rows"] == 12
    assert report["candidate_rows"] == 7
    scored = apply_persistent_collective(frame, calibration)
    device_time = scored[
        ["plant_id", "device_no", "event_time", "persistent_event_alert"]
    ].drop_duplicates()
    assert device_time["persistent_event_alert"].sum() == 7
    first_confirmed = device_time.loc[
        device_time["persistent_event_alert"].astype(bool), "event_time"
    ].min()
    assert first_confirmed == pd.Timestamp("2026-06-01 00:25:00+00:00")
    assert scored.loc[
        scored["string_no"].eq(1), "persistent_collective_member_alert"
    ].sum() == 7
    assert scored.loc[
        scored["string_no"].eq(4), "persistent_collective_member_alert"
    ].sum() == 0

    events = collapse_persistent_events(scored)
    assert len(events) == 3
    assert set(events["string_no"]) == {1, 2, 3}


def test_missing_five_minute_point_breaks_persistence():
    frame = _scored(periods=8)
    frame = frame[frame["event_time"].ne(pd.Timestamp("2026-06-01 00:15:00+00:00"))]
    calibration, _ = fit_persistent_calibration(
        frame,
        deficit_quantile=0.5,
        minimum_device_samples=1,
        minimum_device_days=1,
        minimum_plant_count_samples=1,
        minimum_plant_count_days=1,
        minimum_count_samples=1,
        minimum_count_days=1,
        shrinkage_k=1,
    )
    scored = apply_persistent_collective(frame, calibration)
    assert not scored["persistent_event_alert"].astype(bool).any()


def test_threshold_is_applied_to_window_median_not_each_point():
    frame = _scored(periods=6)
    # One point is weaker than the final threshold, but the stable six-point
    # window remains a persistent mild deficit.
    mask = frame["event_time"].eq(pd.Timestamp("2026-06-01 00:10:00+00:00")) & frame["string_no"].le(3)
    frame.loc[mask, "residual_ratio"] = 0.94
    calibration, _ = fit_persistent_calibration(
        _scored(periods=12), deficit_quantile=0.5, minimum_device_samples=1,
        minimum_device_days=1, minimum_plant_count_samples=1,
        minimum_plant_count_days=1, minimum_count_samples=1,
        minimum_count_days=1, shrinkage_k=1,
    )
    scored = apply_persistent_collective(frame, calibration)
    final = scored[scored["event_time"].eq(pd.Timestamp("2026-06-01 00:25:00+00:00"))]
    assert final["persistent_window_median_deficit"].iloc[0] == pytest.approx(0.1)
    assert final["persistent_event_alert"].iloc[0] == 1


def test_robust_calibration_caps_extreme_unlabelled_tail():
    frame = _scored(periods=40)
    extreme_times = frame["event_time"].drop_duplicates().tail(8)
    mask = frame["event_time"].isin(extreme_times) & frame["string_no"].le(3)
    frame.loc[mask, "residual_ratio"] = 0.2
    calibration, _ = fit_persistent_calibration(
        frame, deficit_quantile=0.995, minimum_device_samples=1,
        minimum_device_days=1, minimum_plant_count_samples=1,
        minimum_plant_count_days=1, minimum_count_samples=1,
        minimum_count_days=1, shrinkage_k=1,
    )
    assert calibration.global_threshold < 0.2


def test_parallel_branch_does_not_modify_existing_strict_alert():
    frame = _scored(periods=6)
    frame["pvlof_v2_hier_strict_alert"] = (frame["string_no"] == 8).astype("int8")
    original = frame["pvlof_v2_hier_strict_alert"].copy()
    calibration, _ = fit_persistent_calibration(
        _scored(periods=12), deficit_quantile=0.5, minimum_device_samples=1,
        minimum_device_days=1, minimum_plant_count_samples=1,
        minimum_plant_count_days=1, minimum_count_samples=1,
        minimum_count_days=1, shrinkage_k=1,
    )
    scored = apply_persistent_collective(frame, calibration)
    assert scored["pvlof_v2_hier_strict_alert"].reset_index(drop=True).equals(
        original.reset_index(drop=True)
    )
    assert scored.loc[scored["string_no"].eq(8), "pvlof_v12_combined_alert"].eq(1).all()


def test_persistent_jaccard_resets_when_two_members_expand_to_five():
    rows = []
    times = pd.date_range("2026-06-01", periods=6, freq="5min", tz="UTC")
    for point, timestamp in enumerate(times):
        for string_no in range(1, 11):
            member = string_no <= (2 if point < 5 else 5)
            rows.append({
                "event_time": timestamp, "plant_id": "234", "device_no": "dev-a",
                "string_no": string_no, "residual_ratio": 0.7 if member else 1.0,
                "string_current": 7.0 if member else 10.0, "response_known": True,
                "collective_group_member": member,
            })
    frame = pd.DataFrame(rows)
    calibration, _ = fit_persistent_calibration(
        _scored(periods=12), deficit_quantile=0.5, minimum_device_samples=1,
        minimum_device_days=1, minimum_plant_count_samples=1,
        minimum_plant_count_days=1, minimum_count_samples=1,
        minimum_count_days=1, shrinkage_k=1,
    )
    scored = apply_persistent_collective(frame, calibration)
    final = scored[scored["event_time"].eq(times[-1])]
    assert final["persistent_membership_overlap"].iloc[0] == pytest.approx(2 / 5)
    assert final["persistent_consecutive"].iloc[0] == 1
    assert not final["persistent_event_alert"].astype(bool).any()
