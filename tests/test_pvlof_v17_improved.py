import numpy as np
import pandas as pd

from pv_anomaly.pvlof_v17_improved import (
    PVLOFV17ImprovedConfig,
    apply_pvlof_v17_improved,
)


def _points(
    currents,
    *,
    periods=3,
    expected_current=14.0,
    expected_currents=None,
    string_numbers=None,
):
    rows = []
    currents = np.asarray(currents, dtype=float)
    if expected_currents is None:
        expected_currents = np.full(len(currents), expected_current, dtype=float)
    else:
        expected_currents = np.asarray(expected_currents, dtype=float)
    if string_numbers is None:
        string_numbers = range(1, len(currents) + 1)
    residuals = currents / expected_currents
    residual_median = float(np.median(residuals))
    for event_time in pd.date_range(
        "2026-06-01", periods=periods, freq="5min", tz="UTC"
    ):
        for string_no, current, expected, residual in zip(
            string_numbers, currents, expected_currents, residuals
        ):
            rows.append({
                "plant_id": "234",
                "device_no": "dev-a",
                "event_time": event_time,
                "string_no": string_no,
                "v2_eligible": 1,
                "response_known": 1,
                "string_current": current,
                "expected_current": expected,
                "residual_ratio": residual,
                "residual_median": residual_median,
                "pvlof_score": 1.0,
                "pvlof_v16_raw_anomaly": 0,
                "pvlof_v16_alert": 0,
                "pvlof_v16_memory_active": 0,
                "pvlof_v16_memory_reactivated_alert": 0,
                "pvlof_v16_memory_clear_code": 0,
                "pvlof_v16_entry_streak": 0,
                "pvlof_v16_normal_streak": 0,
            })
    return pd.DataFrame(rows)


def _candidate_numbers(result):
    return set(result.loc[
        result["pvlof_v17_improved_segmentation_raw_candidate"].eq(1),
        "string_no",
    ].astype(int))


def test_improved_v17_merges_fragmented_high_reference_segments():
    frame = _points([4.9, 5.0, 5.1, 9.8, 9.9, 10.0, 13.3, 13.4, 14.7, 14.8])
    result = apply_pvlof_v17_improved(frame, PVLOFV17ImprovedConfig())
    first = result[result["event_time"].eq(result["event_time"].min())]

    assert first["pvlof_v17_improved_reference_reliable"].eq(1).all()
    assert first["pvlof_v17_improved_segmentation_reference_size"].iloc[0] >= 5
    assert first.loc[
        first["string_no"].isin([1, 2, 3]),
        "pvlof_v17_improved_segmentation_raw_candidate",
    ].eq(1).all()


def test_improved_v17_blocks_weak_group_even_when_external_gap_is_large():
    residuals = [
        0.770, 0.790, 0.805, 0.815, 0.825, 0.830, 0.835, 0.840,
        0.845, 0.850, 0.855, 0.865, 0.880, 0.900, 0.915, 0.932,
    ]
    frame = _points(np.asarray(residuals) * 12.0, expected_current=12.0)
    result = apply_pvlof_v17_improved(frame, PVLOFV17ImprovedConfig())

    assert result["pvlof_v17_improved_group_effect_eligible"].eq(0).all()
    assert result["pvlof_v17_improved_segmentation_raw_candidate"].eq(0).all()


def test_v3_judges_structural_members_against_reference_median(monkeypatch):
    """A broad weak lower-side union must not mask its strong members in V3."""
    string_numbers = [4, 5, 6, 1, 2, 10, 9, 11, 7, 3, 13, 14, 16, 17, 19, 20]
    residuals = np.asarray([
        0.4704, 0.4730, 0.4756, 0.6474,
        0.8338, 0.8601, 0.8716, 0.8788, 0.8892,
        0.9300, 0.9320, 0.9340, 0.9360, 0.9380, 0.9400, 0.9420,
    ])
    frame = _points(
        residuals * 12.0,
        expected_current=12.0,
        string_numbers=string_numbers,
    )
    # Skip recomputing the frozen original branch; it is not part of this
    # regression and V2/V3 both consume the same structural segmentation.
    frame["pvlof_v17_segmentation_raw_candidate"] = 0

    def fixed_segments(values, *, penalty, minimum_segment_size):
        assert len(values) == 16
        return [(0, 3), (3, 5), (5, 9), (9, 16)]

    monkeypatch.setattr(
        "pv_anomaly.pvlof_v17_improved.penalized_segments",
        fixed_segments,
    )

    v2 = apply_pvlof_v17_improved(frame, PVLOFV17ImprovedConfig())
    v3 = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v3-reference-member-gate",
            algorithm_variant="reference_member_gate_v3",
        ),
    )

    assert v2["pvlof_v17_improved_group_effect_eligible"].eq(0).all()
    assert _candidate_numbers(v2) == set()

    assert v3["pvlof_v17_improved_group_effect_eligible"].eq(0).all()
    assert v3["pvlof_v17_improved_group_gate_applied"].eq(0).all()
    assert set(v3.loc[
        v3["pvlof_v17_improved_segmentation_reference_member_gate_pass"].eq(1),
        "string_no",
    ].astype(int)) == {1, 4, 5, 6}
    assert _candidate_numbers(v3) == {1, 4, 5, 6}


def test_v5_keeps_lowest_core_and_rejects_unsupported_middle_segment(monkeypatch):
    string_numbers = [1, 21, 19, 2, 20, 3, 7, 8, 9, 13, 4, 14, 10, 11, 17, 16]
    residuals = np.asarray([
        0.940, 0.942, 0.950, 0.952, 0.960, 0.968,
        1.060, 1.080, 1.090, 1.100,
        1.200, 1.220,
        1.400, 1.500, 1.600, 1.700,
    ])
    frame = _points(
        residuals * 5.0,
        periods=1,
        expected_current=5.0,
        string_numbers=string_numbers,
    )
    frame["pvlof_v17_segmentation_raw_candidate"] = 0
    frame.loc[
        frame["string_no"].isin([7, 8, 9, 13]),
        "pvlof_v16_raw_anomaly",
    ] = 1

    def fixed_segments(values, *, penalty, minimum_segment_size):
        assert len(values) == 16
        return [(0, 6), (6, 10), (10, 12), (12, 16)]

    monkeypatch.setattr(
        "pv_anomaly.pvlof_v17_improved.penalized_segments",
        fixed_segments,
    )

    v4 = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v4-combined-candidate-memory",
            algorithm_variant="combined_candidate_memory_v4",
        ),
    )
    v5 = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v5-lowest-core-expansion",
            algorithm_variant="lowest_core_expansion_v5",
        ),
    )

    assert _candidate_numbers(v4) == {1, 2, 3, 7, 8, 9, 13, 19, 20, 21}
    assert _candidate_numbers(v5) == {1, 2, 3, 19, 20, 21}
    assert set(v5.loc[
        v5["pvlof_v17_improved_combined_raw_candidate"].eq(1),
        "string_no",
    ].astype(int)) == {1, 2, 3, 19, 20, 21}
    assert set(v5.loc[
        v5["pvlof_v17_improved_v16_middle_group_veto"].eq(1),
        "string_no",
    ].astype(int)) == {7, 8, 9, 13}
    assert set(v5.loc[
        v5["pvlof_v17_improved_segmentation_lowest_core_member"].eq(1),
        "string_no",
    ].astype(int)) == {1, 2, 3, 19, 20, 21}
    assert v5[
        "pvlof_v17_improved_segmentation_validated_expansion_raw_candidate"
    ].eq(0).all()
    assert v5[
        "pvlof_v17_improved_segmentation_expansion_segment_count"
    ].eq(0).all()


def test_v5_expands_through_coherently_supported_adjacent_low_segment(monkeypatch):
    residuals = np.asarray([
        0.300, 0.320,
        0.500, 0.520, 0.550, 0.580,
        0.950, 0.980, 1.000, 1.020, 1.050,
    ])
    frame = _points(residuals * 12.0, periods=1, expected_current=12.0)
    frame["pvlof_v17_segmentation_raw_candidate"] = 0

    def fixed_segments(values, *, penalty, minimum_segment_size):
        assert len(values) == 11
        return [(0, 2), (2, 6), (6, 11)]

    monkeypatch.setattr(
        "pv_anomaly.pvlof_v17_improved.penalized_segments",
        fixed_segments,
    )
    result = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v5-lowest-core-expansion",
            algorithm_variant="lowest_core_expansion_v5",
        ),
    )

    assert _candidate_numbers(result) == {1, 2, 3, 4, 5, 6}
    assert set(result.loc[
        result[
            "pvlof_v17_improved_segmentation_validated_expansion_raw_candidate"
        ].eq(1),
        "string_no",
    ].astype(int)) == {3, 4, 5, 6}
    assert result[
        "pvlof_v17_improved_segmentation_expansion_segment_count"
    ].eq(1).all()


def test_improved_v17_rescues_small_legacy_pair_diluted_by_weak_members():
    string_numbers = [14, 13, 11, 17, 19, 12, 16, 10, 20, 18]
    currents = [9.81, 10.73, 10.76, 11.38, 11.54, 11.11, 11.80, 11.51, 11.43, 11.69]
    expecteds = [
        10.953568, 11.446064, 11.209875, 11.665341, 11.639491,
        10.623799, 10.948946, 10.631083, 10.466143, 10.667706,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )

    assert result["pvlof_v17_improved_group_effect_eligible"].eq(0).all()
    assert _candidate_numbers(result) == {13, 14}
    assert set(result.loc[
        result[
            "pvlof_v17_improved_segmentation_small_candidate_rescue_raw_candidate"
        ].eq(1),
        "string_no",
    ].astype(int)) == {13, 14}


def test_improved_v17_uses_partial_next_segment_for_strong_member_only():
    string_numbers = [21, 5, 4, 3, 2, 1, 10, 15, 19, 13, 20, 11, 12, 14, 16]
    currents = [
        5.12, 5.47, 5.58, 5.85, 6.35, 6.99, 9.85, 11.22,
        13.17, 14.48, 14.59, 14.40, 14.94, 14.84, 14.52,
    ]
    expecteds = [
        12.897593, 13.670637, 13.909782, 13.548129, 13.858347,
        13.775694, 13.413177, 12.997246, 14.014457, 14.382939,
        14.374144, 14.001888, 13.912714, 13.810225, 13.314374,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )

    assert _candidate_numbers(result) == {1, 2, 3, 4, 5, 10, 21}
    assert set(result.loc[
        result[
            "pvlof_v17_improved_segmentation_partial_next_segment_raw_candidate"
        ].eq(1),
        "string_no",
    ].astype(int)) == {10}


def test_improved_v17_uses_internal_gate_when_external_forecast_is_low():
    string_numbers = [13, 20, 14, 19, 16, 17, 2, 3, 7, 9, 10, 1, 11, 4, 5, 6]
    currents = [
        7.68, 7.94, 7.56, 7.43, 7.79, 7.95, 8.85, 8.63,
        9.18, 9.63, 10.10, 10.58, 10.40, 10.79, 11.14, 11.49,
    ]
    expecteds = [
        7.318141, 7.262158, 6.896561, 6.728638, 6.768589, 6.861787,
        7.221974, 6.631704, 6.892756, 7.135947, 7.066829, 7.196069,
        7.070908, 7.307551, 7.004474, 6.859780,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )

    assert _candidate_numbers(result) == {13, 14, 16, 17, 19, 20}
    selected = result["string_no"].isin([13, 14, 16, 17, 19, 20])
    assert result.loc[selected, "pvlof_v17_improved_internal_physical_effect_eligible"].eq(1).all()


def test_improved_v17_applies_group_tolerance_without_relaxing_member_gate():
    string_numbers = [6, 5, 4, 1, 11, 10, 9, 17, 7, 20, 16, 2, 3, 13, 14, 19]
    currents = [
        4.57, 4.94, 6.10, 8.14, 8.69, 9.24, 9.99, 10.01,
        10.56, 11.18, 10.53, 11.51, 10.64, 11.96, 11.51, 11.83,
    ]
    expecteds = [
        5.702370, 5.822651, 6.074592, 5.981920, 5.877875, 5.874485,
        5.931941, 5.704039, 5.729782, 6.036858, 5.626565, 6.003453,
        5.512777, 6.083395, 5.732945, 5.593355,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )

    assert result["pvlof_v17_improved_group_effect_eligible"].eq(1).all()
    assert _candidate_numbers(result) == {1, 4, 5, 6, 11}


def test_improved_v17_does_not_expand_an_all_strong_next_segment():
    string_numbers = [1, 2, 3, 7, 9, 5, 21, 20, 4, 19, 11, 10, 15, 14, 13]
    currents = [
        2.24, 2.35, 2.42, 2.66, 3.37, 4.13, 4.07, 5.21,
        5.30, 5.41, 6.45, 7.10, 7.54, 7.96, 8.21,
    ]
    expecteds = [
        8.070323, 8.164168, 7.828997, 7.722178, 8.030823, 8.196454,
        7.890145, 8.007366, 7.928181, 7.746544, 7.952875, 7.844100,
        7.550354, 7.763701, 7.876826,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )

    assert _candidate_numbers(result) == {1, 2, 3, 5, 7, 9, 21}
    assert result[
        "pvlof_v17_improved_segmentation_partial_next_segment_raw_candidate"
    ].eq(0).all()


def test_improved_v17_leaves_single_segment_point_for_temporal_handling():
    string_numbers = [4, 13, 20, 2, 1, 9, 17, 5, 11, 10, 16, 14, 3, 7, 6, 19]
    currents = [
        9.67, 9.73, 9.67, 9.67, 9.65, 9.62, 9.34, 9.55,
        9.68, 9.68, 9.31, 9.50, 9.19, 9.56, 9.55, 9.47,
    ]
    expecteds = [
        9.598557, 9.612468, 9.538933, 9.486151, 9.452125, 9.373153,
        9.013041, 9.200462, 9.287723, 9.282366, 8.890624, 9.058717,
        8.710825, 9.053719, 9.010405, 8.838147,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )

    assert result["pvlof_v17_improved_no_internal_contrast"].eq(1).all()
    assert _candidate_numbers(result) == set()


def test_improved_v17_allows_correct_majority_candidate_group():
    string_numbers = [1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21]
    currents = [
        4.84, 5.17, 5.52, 12.83, 10.25, 11.05, 10.95, 13.35,
        13.25, 10.59, 11.50, 12.92, 13.57, 5.89, 7.97, 6.34,
    ]
    expecteds = [
        11.656163, 11.940763, 11.620643, 11.791109, 11.542122, 11.908609,
        12.099731, 12.116145, 11.769814, 12.179885, 11.469931, 11.328053,
        12.252741, 11.558640, 11.997408, 11.844434,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )
    first = result[result["event_time"].eq(result["event_time"].min())]

    assert first["pvlof_v17_improved_segmentation_candidate_fraction"].iloc[0] > 0.5
    assert first["pvlof_v17_improved_group_effect_eligible"].eq(1).all()
    actual = set(
        first.loc[
            first["pvlof_v17_improved_segmentation_raw_candidate"].eq(1),
            "string_no",
        ].astype(int)
    )
    assert actual == {1, 2, 3, 7, 8, 9, 13, 19, 20, 21}

    v5 = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v5-lowest-core-expansion",
            algorithm_variant="lowest_core_expansion_v5",
        ),
    )
    assert _candidate_numbers(v5) == {1, 2, 3, 7, 8, 9, 13, 19, 20, 21}


def test_improved_v17_rejects_absolute_only_legacy_members_without_segment_support():
    string_numbers = [1, 2, 9, 7, 3, 8, 12, 11, 10, 20, 21, 16, 18, 15, 17, 19]
    currents = [
        4.57, 4.74, 4.87, 4.72, 4.98, 5.04, 11.65, 11.23,
        12.09, 13.49, 13.50, 14.19, 13.49, 13.37, 13.38, 13.66,
    ]
    expecteds = [
        12.187836, 12.474640, 12.504573, 12.043607, 12.488345,
        11.984311, 12.636935, 12.093164, 11.942677, 12.613292,
        12.224320, 12.827065, 12.156022, 11.909730, 11.906477,
        12.065891,
    ]
    result = apply_pvlof_v17_improved(
        _points(
            currents,
            periods=1,
            expected_currents=expecteds,
            string_numbers=string_numbers,
        ),
        PVLOFV17ImprovedConfig(),
    )

    assert _candidate_numbers(result) == {1, 2, 3, 7, 8, 9}
    legacy_extras = result["string_no"].isin([11, 12])
    assert result.loc[
        legacy_extras, "pvlof_v17_segmentation_raw_candidate"
    ].eq(1).all()
    assert result.loc[
        legacy_extras,
        "pvlof_v17_improved_segmentation_original_candidate_accepted",
    ].eq(0).all()


def test_improved_v17_rescues_first_fragmented_reference_case_exactly():
    string_numbers = [1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21]
    currents = [9.52, 9.14, 8.68, 3.55, 5.61, 5.35, 5.59, 3.82,
                2.77, 5.46, 4.83, 3.38, 3.65, 8.10, 6.91, 7.31]
    expecteds = [3.243794, 3.322995, 3.233909, 3.281348, 3.212057, 3.314047,
                 3.367234, 3.371802, 3.275422, 3.389540, 3.191967, 3.152484,
                 3.409815, 3.216654, 3.338759, 3.296188]
    result = apply_pvlof_v17_improved(
        _points(currents, periods=1, expected_currents=expecteds, string_numbers=string_numbers),
        PVLOFV17ImprovedConfig(),
    )
    actual = set(result.loc[
        result["pvlof_v17_improved_segmentation_raw_candidate"].eq(1), "string_no"
    ].astype(int))
    assert actual == {4, 10, 11, 16, 17}

    v5 = apply_pvlof_v17_improved(
        _points(currents, periods=1, expected_currents=expecteds, string_numbers=string_numbers),
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v5-lowest-core-expansion",
            algorithm_variant="lowest_core_expansion_v5",
        ),
    )
    assert _candidate_numbers(v5) == {4, 10, 11, 16, 17}


def test_frozen_dual_branch_rescues_first_fragmented_reference_case_exactly():
    string_numbers = [1, 2, 3, 4, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21]
    currents = [9.52, 9.14, 8.68, 3.55, 5.61, 5.35, 5.59, 3.82,
                2.77, 5.46, 4.83, 3.38, 3.65, 8.10, 6.91, 7.31]
    expecteds = [3.243794, 3.322995, 3.233909, 3.281348, 3.212057, 3.314047,
                 3.367234, 3.371802, 3.275422, 3.389540, 3.191967, 3.152484,
                 3.409815, 3.216654, 3.338759, 3.296188]
    result = apply_pvlof_v17_improved(
        _points(currents, periods=1, expected_currents=expecteds, string_numbers=string_numbers),
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-dual-branch",
            algorithm_variant="dual_branch",
        ),
    )

    assert _candidate_numbers(result) == {4, 10, 11, 16, 17}
    assert result[
        "pvlof_v17_improved_segmentation_partial_next_segment_raw_candidate"
    ].eq(0).all()
    assert result[
        "pvlof_v17_improved_segmentation_small_candidate_rescue_raw_candidate"
    ].eq(0).all()


def test_improved_v17_rescues_second_fragmented_reference_case_exactly():
    string_numbers = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21]
    currents = [9.63, 10.96, 12.41, 13.37, 14.42, 12.64, 5.01, 4.54,
                6.41, 4.80, 10.32, 4.43, 8.56, 5.27, 14.73]
    expecteds = [14.659510, 14.747466, 14.417345, 14.802200, 14.547712,
                 14.273733, 14.900216, 14.805321, 15.305715, 14.696257,
                 13.831119, 14.168592, 14.913590, 15.296354, 13.725071]
    result = apply_pvlof_v17_improved(
        _points(currents, periods=1, expected_currents=expecteds, string_numbers=string_numbers),
        PVLOFV17ImprovedConfig(),
    )
    actual = set(result.loc[
        result["pvlof_v17_improved_segmentation_raw_candidate"].eq(1), "string_no"
    ].astype(int))
    assert actual == {11, 12, 13, 14, 16, 20}

    v5 = apply_pvlof_v17_improved(
        _points(currents, periods=1, expected_currents=expecteds, string_numbers=string_numbers),
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v5-lowest-core-expansion",
            algorithm_variant="lowest_core_expansion_v5",
        ),
    )
    assert _candidate_numbers(v5) == {11, 12, 13, 14, 16, 20}


def test_improved_v17_preserves_same_run_v16_alert():
    frame = _points([13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6], periods=1)
    frame.loc[frame["string_no"].eq(1), "pvlof_v16_alert"] = 1
    result = apply_pvlof_v17_improved(frame, PVLOFV17ImprovedConfig())

    preserved = result.loc[result["string_no"].eq(1)].iloc[0]
    assert preserved["pvlof_v17_improved_alert"] == 1
    assert preserved["pvlof_v17_improved_added_alert"] == 0


def test_v4_combines_v16_and_v3_candidates_before_temporal_confirmation():
    frame = _points([12.0] * 7, periods=3, expected_current=12.0)
    times = sorted(frame["event_time"].unique())

    # The same string is a V1.6 candidate at the first two points. At the
    # third point, a two-string segmented group replaces the V1.6 evidence.
    v16_prefix = frame["event_time"].isin(times[:2]) & frame["string_no"].eq(1)
    frame.loc[v16_prefix, "pvlof_v16_raw_anomaly"] = 1
    third_low = frame["event_time"].eq(times[2]) & frame["string_no"].isin([1, 2])
    frame.loc[third_low, "string_current"] = 3.0
    frame.loc[third_low, "residual_ratio"] = 0.25
    frame["pvlof_v17_segmentation_raw_candidate"] = 0

    v3 = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v3-reference-member-gate",
            algorithm_variant="reference_member_gate_v3",
        ),
    )
    v4 = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v4-combined-candidate-memory",
            algorithm_variant="combined_candidate_memory_v4",
        ),
    )
    v5 = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v5-lowest-core-expansion",
            algorithm_variant="lowest_core_expansion_v5",
        ),
    )

    assert v3["pvlof_v17_improved_segmentation_raw_candidate"].equals(
        v4["pvlof_v17_improved_segmentation_raw_candidate"]
    )
    v3_string_1 = v3[v3["string_no"].eq(1)].sort_values("event_time")
    v4_string_1 = v4[v4["string_no"].eq(1)].sort_values("event_time")
    v4_string_2 = v4[v4["string_no"].eq(2)].sort_values("event_time")
    v5_string_1 = v5[v5["string_no"].eq(1)].sort_values("event_time")

    assert v3_string_1["pvlof_v17_improved_alert"].tolist() == [0, 0, 0]
    assert v4_string_1["pvlof_v17_improved_combined_raw_candidate"].tolist() == [1, 1, 1]
    assert v4_string_1["pvlof_v17_improved_combined_entry_streak"].tolist() == [1, 2, 3]
    assert v4_string_1["pvlof_v17_improved_alert"].tolist() == [0, 0, 1]
    assert v4_string_1["pvlof_v17_improved_combined_bridge_alert"].tolist() == [0, 0, 1]
    assert v4_string_2["pvlof_v17_improved_alert"].tolist() == [0, 0, 0]
    assert v5_string_1["pvlof_v17_improved_combined_raw_candidate"].tolist() == [1, 1, 1]
    assert v5_string_1["pvlof_v17_improved_combined_entry_streak"].tolist() == [1, 2, 3]
    assert v5_string_1["pvlof_v17_improved_alert"].tolist() == [0, 0, 1]


def test_v5_suspends_confirmed_string_during_bounded_invalid_data():
    frame = _points(
        [4.9, 5.0, 5.1, 13.3, 13.4, 13.5, 14.5, 14.6],
        periods=6,
    )
    times = sorted(frame["event_time"].unique())
    missing = (
        frame["string_no"].eq(1)
        & frame["event_time"].isin(times[3:5])
    )
    frame.loc[missing, ["v2_eligible", "response_known"]] = 0

    result = apply_pvlof_v17_improved(
        frame,
        PVLOFV17ImprovedConfig(
            version="pvlof-v1.7-improved-v5-lowest-core-expansion",
            algorithm_variant="lowest_core_expansion_v5",
            maximum_unknown_intervals=2,
        ),
    )
    string_1 = result[result["string_no"].eq(1)].sort_values("event_time")

    assert string_1["pvlof_v17_improved_memory_active"].tolist() == [
        0, 0, 1, 1, 1, 1,
    ]
    assert string_1["pvlof_v17_improved_combined_memory_alert"].tolist() == [
        0, 0, 1, 0, 0, 1,
    ]
    assert string_1["pvlof_v17_improved_memory_suspended"].tolist() == [
        0, 0, 0, 1, 1, 0,
    ]
    assert string_1["pvlof_v17_improved_memory_resumed_alert"].tolist() == [
        0, 0, 0, 0, 0, 1,
    ]
