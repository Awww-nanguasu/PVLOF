from __future__ import annotations

import json
import sys

import pandas as pd

from pv_anomaly.low_current_confirmed_profiles import (
    ConfirmedLowCurrentProfileConfig,
)
from scripts.build_low_current_confirmed_profiles_58d import main


def test_build_confirmed_profile_outputs(tmp_path, monkeypatch) -> None:
    config = ConfirmedLowCurrentProfileConfig(
        source_algorithm_version="test-v2",
        timezone="UTC",
        minimum_valid_history_days=1,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False), encoding="utf-8"
    )

    workbook_path = tmp_path / "confirmed.xlsx"
    workbook = pd.DataFrame(
        {
            "event_id": ["comparison-1"],
            "plant_id": ["234"],
            "device_no": ["device-a"],
            "raise_time_local": ["2026-06-01 00:00:00"],
            "end_time_local": ["2026-06-01 00:10:00"],
            "PVLOF_V1_7_IMPROVED_V2": ["01"],
            "manual_label": [""],
            "manual_strings": [""],
            "review_note": [""],
        }
    )
    workbook.to_excel(
        workbook_path,
        sheet_name=config.confirmed_workbook_sheet,
        index=False,
        engine="openpyxl",
    )

    replay_directory = tmp_path / "replay" / "plant_id=234"
    replay_directory.mkdir(parents=True)
    start = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    source_events = pd.DataFrame(
        {
            "version": ["test-v2"],
            "event_id": ["event-1"],
            "plant_id": ["234"],
            "device_no": ["device-a"],
            "string_no": pd.Series([1], dtype="Int64"),
            "candidate_start_time": [start],
            "alert_confirm_time": [start + pd.Timedelta(minutes=10)],
            "last_anomaly_time": [start + pd.Timedelta(minutes=10)],
            "clear_time": [start + pd.Timedelta(minutes=15)],
            "event_end_time": [start + pd.Timedelta(minutes=15)],
            "evidence_points": [3],
            "final_alert_points": [1],
            "recovery_observation_points": [1],
            "detection_delay_minutes": [10.0],
            "confirmation_branch": ["segmentation"],
            "clear_reason": ["recovered"],
        }
    )
    source_events.to_parquet(
        replay_directory / config.source_event_filename, index=False
    )
    times = pd.date_range(start, periods=3, freq="5min")
    evidence = pd.DataFrame(
        {
            "version": ["test-v2"] * 3,
            "event_id": ["event-1"] * 3,
            "plant_id": ["234"] * 3,
            "device_no": ["device-a"] * 3,
            "string_no": pd.Series([1, 1, 1], dtype="Int64"),
            "event_time": times,
            "point_state": ["candidate", "candidate", "active_anomaly"],
            "raw_evidence": [True, True, True],
            "final_alert": [False, False, True],
        }
    )
    evidence.to_parquet(
        replay_directory / config.source_evidence_filename, index=False
    )
    unconfirmed = pd.DataFrame(
        {
            "version": ["test-v2"],
            "plant_id": ["234"],
            "device_no": ["device-a"],
            "string_no": pd.Series([2], dtype="Int64"),
            "candidate_start_time": [start],
            "candidate_end_time": [start + pd.Timedelta(minutes=5)],
            "candidate_points": [2],
            "maximum_entry_streak": [2],
            "end_reason": ["not_confirmed"],
        }
    )
    unconfirmed.to_parquet(
        replay_directory / config.source_unconfirmed_filename, index=False
    )

    candidate_directory = tmp_path / "candidate-profile"
    candidate_directory.mkdir()
    candidates = pd.DataFrame(
        {
            "plant_id": ["234"] * 3,
            "device_no": ["device-a"] * 3,
            "string_no": pd.Series([1, 1, 1], dtype="Int64"),
            "event_time": times,
            "valid_point": [True, True, True],
            "string_current": [8.0, 7.0, 6.0],
            "expected_current": [10.0, 10.0, 10.0],
            "relative_drop": [0.2, 0.3, 0.4],
            "absolute_drop": [2.0, 3.0, 4.0],
            "source_branch": ["segmentation"] * 3,
            "point_morphology": ["gradient_group"] * 3,
        }
    )
    candidates.to_parquet(
        candidate_directory / config.candidate_points_filename, index=False
    )
    coverage = pd.DataFrame(
        {
            "plant_id": ["234"],
            "device_no": ["device-a"],
            "string_no": pd.Series([1], dtype="Int64"),
            "local_date": ["2026-05-31"],
            "valid_point_count": [10],
            "is_valid_day": [1],
        }
    )
    coverage.to_parquet(
        candidate_directory / config.valid_day_coverage_filename, index=False
    )

    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_low_current_confirmed_profiles_58d.py",
            "--confirmed-events-workbook",
            str(workbook_path),
            "--input-root",
            str(tmp_path / "replay"),
            "--candidate-profile-directory",
            str(candidate_directory),
            "--config",
            str(config_path),
            "--output-directory",
            str(output),
            "--skip-workbook",
        ],
    )
    main()

    events = pd.read_parquet(
        output / "low_current_confirmed_string_events.parquet"
    )
    profiles = pd.read_parquet(
        output / "low_current_confirmed_event_profiles.parquet"
    )
    rejected = pd.read_parquet(
        output / "low_current_unconfirmed_candidate_runs.parquet"
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert len(events) == 1
    assert len(profiles) == 1
    assert events.loc[0, "source_event_key"] == "234::event-1"
    assert events.loc[0, "maximum_relative_drop"] == 0.4
    assert profiles.loc[0, "profile_class_code"] == "sudden_new"
    assert len(rejected) == 1
    assert summary["counts"]["exact_confirmed_string_events"] == 1
    assert summary["contract"]["candidates_are_event_filter"] is False
    assert summary["contract"]["unconfirmed_candidates_affect_classification"] is False
