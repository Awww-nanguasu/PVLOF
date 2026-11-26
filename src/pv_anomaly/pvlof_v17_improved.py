"""Improved PVLOF v1.7 device-internal segmentation branches.

The original v1.7 stays frozen for comparison. V2 segments median-normalised
residuals, merges the high-residual tail into a reference, applies a hard
whole-group drop gate, and then validates members against that reference. V3
keeps the same segmentation and reference construction, but records the
whole-group drop for diagnosis only: every lower-side structural member is
judged directly against the reference median. V4 keeps the V3 candidates but
combines v1.6 and segmentation raw evidence before temporal confirmation. V5
uses the lowest segment as the candidate core and expands consecutively only
across a small boundary or, with a stable reference, coherent external physical
evidence. V1.6 raw candidates in a rejected middle group are vetoed for V5's
combined confirmation. Same-run v1.6 alerts remain in the legacy audit union.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_v17 import (
    PVLOFV17Config,
    _memory_state,
    _numeric_column,
    _physical_effect,
    apply_pvlof_v17,
    penalized_segments,
)


KEYS = ["plant_id", "device_no", "string_no"]
PREFIX = "pvlof_v17_improved"
SEG = f"{PREFIX}_segmentation"


@dataclass(frozen=True)
class PVLOFV17ImprovedConfig:
    version: str = "pvlof-v1.7-improved-v2-internal-member-gate"
    algorithm_variant: str = "internal_member_gate_v2"
    segmentation_penalty: float = 0.01
    minimum_segment_strings: int = 2
    minimum_reference_strings: int = 5
    minimum_candidate_strings: int = 2
    minimum_group_relative_drop: float = 0.20
    group_relative_drop_tolerance: float = 0.0001
    minimum_small_candidate_group_relative_drop: float = 0.10
    small_candidate_rescue_max_strings: int = 2
    minimum_relative_drop: float = 0.20
    minimum_absolute_drop: float = 0.50
    effect_gate_mode: str = "any"
    internal_effect_gate_mode: str = "all"
    minimum_original_segment_support: float = 0.80
    minimum_core_segment_support: float = 0.80
    minimum_expansion_segment_support: float = 0.80
    maximum_core_expansion_boundary_gap: float = 0.05
    maximum_external_expansion_reference_segments: int = 1
    entry_consecutive: int = 3
    recovery_consecutive: int = 3
    expected_interval_minutes: int = 5
    maximum_unknown_intervals: int = 0

    def __post_init__(self) -> None:
        if self.algorithm_variant not in {
            "dual_branch",
            "internal_member_gate_v2",
            "reference_member_gate_v3",
            "combined_candidate_memory_v4",
            "lowest_core_expansion_v5",
        }:
            raise ValueError(
                "algorithm_variant must be 'dual_branch', "
                "'internal_member_gate_v2', 'reference_member_gate_v3', "
                "'combined_candidate_memory_v4', or 'lowest_core_expansion_v5'"
            )
        if self.segmentation_penalty <= 0:
            raise ValueError("segmentation_penalty must be positive")
        if self.minimum_segment_strings < 1:
            raise ValueError("minimum_segment_strings must be positive")
        if self.minimum_reference_strings < self.minimum_segment_strings:
            raise ValueError("minimum_reference_strings must be >= minimum_segment_strings")
        if self.minimum_candidate_strings < 1:
            raise ValueError("minimum_candidate_strings must be positive")
        if not 0 <= self.minimum_group_relative_drop < 1:
            raise ValueError("minimum_group_relative_drop must be in [0, 1)")
        if not 0 <= self.group_relative_drop_tolerance < 1:
            raise ValueError("group_relative_drop_tolerance must be in [0, 1)")
        if not 0 <= self.minimum_small_candidate_group_relative_drop < 1:
            raise ValueError(
                "minimum_small_candidate_group_relative_drop must be in [0, 1)"
            )
        if self.small_candidate_rescue_max_strings < self.minimum_candidate_strings:
            raise ValueError(
                "small_candidate_rescue_max_strings must be >= minimum_candidate_strings"
            )
        if not 0 <= self.minimum_relative_drop < 1:
            raise ValueError("minimum_relative_drop must be in [0, 1)")
        if self.minimum_absolute_drop < 0:
            raise ValueError("minimum_absolute_drop must be non-negative")
        if self.effect_gate_mode not in {"all", "any"}:
            raise ValueError("effect_gate_mode must be 'all' or 'any'")
        if self.internal_effect_gate_mode not in {"all", "any"}:
            raise ValueError("internal_effect_gate_mode must be 'all' or 'any'")
        if not 0 < self.minimum_original_segment_support <= 1:
            raise ValueError("minimum_original_segment_support must be in (0, 1]")
        if not 0 < self.minimum_core_segment_support <= 1:
            raise ValueError("minimum_core_segment_support must be in (0, 1]")
        if not 0 < self.minimum_expansion_segment_support <= 1:
            raise ValueError("minimum_expansion_segment_support must be in (0, 1]")
        if self.maximum_core_expansion_boundary_gap < 0:
            raise ValueError("maximum_core_expansion_boundary_gap must be non-negative")
        if self.maximum_external_expansion_reference_segments < 1:
            raise ValueError(
                "maximum_external_expansion_reference_segments must be positive"
            )
        if min(self.entry_consecutive, self.recovery_consecutive) < 1:
            raise ValueError("entry/recovery consecutive counts must be positive")
        if self.expected_interval_minutes < 1:
            raise ValueError("expected_interval_minutes must be positive")
        if self.maximum_unknown_intervals < 0:
            raise ValueError("maximum_unknown_intervals must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PVLOFV17ImprovedConfig":
        return cls(**dict(payload))


def save_config(config: PVLOFV17ImprovedConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> PVLOFV17ImprovedConfig:
    return PVLOFV17ImprovedConfig.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _effect_eligible(
    relative: np.ndarray,
    absolute: np.ndarray,
    config: PVLOFV17ImprovedConfig,
) -> np.ndarray:
    relative_enabled = config.minimum_relative_drop > 0
    absolute_enabled = config.minimum_absolute_drop > 0
    relative_ok = relative >= config.minimum_relative_drop
    absolute_ok = absolute >= config.minimum_absolute_drop
    if config.internal_effect_gate_mode == "all":
        relative_side = relative_ok if relative_enabled else np.ones_like(relative_ok)
        absolute_side = absolute_ok if absolute_enabled else np.ones_like(absolute_ok)
        return relative_side & absolute_side
    if not relative_enabled and not absolute_enabled:
        return np.ones_like(relative_ok)
    relative_side = relative_ok if relative_enabled else np.zeros_like(relative_ok)
    absolute_side = absolute_ok if absolute_enabled else np.zeros_like(absolute_ok)
    return relative_side | absolute_side


def _segment_dual_branch(
    frame: pd.DataFrame,
    config: PVLOFV17ImprovedConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the frozen pre-V2 improved dual-branch candidate logic.

    The historical branch accepts original V1.7 candidates only when the
    device-level group drop is material.  In parallel, it finds the strongest
    penalized-segmentation boundary and rescues members below that boundary
    when the original external physical gate passes.  It deliberately does
    not use the later internal member gate, partial-segment expansion, coherent
    support filtering, or the small-candidate rescue.
    """

    result = frame.copy()
    external_relative, external_absolute, original_physical = _physical_effect(
        result, config
    )
    current = pd.to_numeric(result["string_current"], errors="coerce")
    expected = pd.to_numeric(result["expected_current"], errors="coerce")
    residual = pd.to_numeric(result["residual_ratio"], errors="coerce")
    score = pd.to_numeric(result["pvlof_score"], errors="coerce")
    valid = (
        result["v2_eligible"].fillna(False).astype(bool)
        & result["response_known"].fillna(False).astype(bool)
        & current.gt(0)
        & expected.gt(0)
        & residual.gt(0)
        & score.notna()
    )

    n = len(result)
    zeros8 = lambda: np.zeros(n, dtype=np.int8)
    zeros16 = lambda: np.zeros(n, dtype=np.int16)
    nans = lambda: np.full(n, np.nan, dtype=np.float32)
    structural, candidate, detected = zeros8(), zeros8(), zeros8()
    original_accepted = zeros8()
    rescue_structural, rescue_candidate = zeros8(), zeros8()
    partial_next_candidate, small_candidate_rescue = zeros8(), zeros8()
    no_contrast, reference_reliable = zeros8(), zeros8()
    group_ok_values = zeros8()
    external_ok_values = original_physical.to_numpy(dtype=bool).astype(np.int8)
    internal_ok_values = zeros8()
    segment_index, segment_count = zeros16(), zeros16()
    reference_segment_count, structural_size = zeros16(), zeros16()
    rescue_segment_count, rescue_structural_size = zeros16(), zeros16()
    candidate_size, reference_size = zeros16(), zeros16()
    candidate_fraction, normalizer, normalized = nans(), nans(), nans()
    reference_residual, candidate_residual, group_drop_values = nans(), nans(), nans()
    internal_relative_values, internal_expected_values = nans(), nans()
    internal_absolute_values, rescue_boundary_gap = nans(), nans()
    objective, sse_gain = nans(), nans()

    residual_values = residual.to_numpy(dtype=float)
    valid_values = valid.to_numpy(dtype=bool)
    original_raw_values = (
        result["pvlof_v17_segmentation_raw_candidate"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    original_physical_values = original_physical.to_numpy(dtype=bool)
    minimum_total = config.minimum_reference_strings + config.minimum_candidate_strings

    for _, positions in result.groupby(
        ["plant_id", "device_no", "event_time"], observed=True, sort=False
    ).indices.items():
        group = np.asarray(positions, dtype=np.int64)
        usable = group[valid_values[group]]
        if len(usable) < minimum_total:
            continue
        scale = float(np.median(residual_values[usable]))
        if not np.isfinite(scale) or scale <= 0:
            continue
        normalizer[group] = scale
        normalized[usable] = residual_values[usable] / scale
        order = np.argsort(normalized[usable], kind="stable")
        ordered_indexes = usable[order]
        ordered_values = normalized[ordered_indexes].astype(float)
        segments = penalized_segments(
            ordered_values,
            penalty=config.segmentation_penalty,
            minimum_segment_size=config.minimum_segment_strings,
        )
        if not segments:
            continue

        prefix = np.concatenate(([0.0], np.cumsum(ordered_values)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(ordered_values * ordered_values)))

        def sse(start: int, end: int) -> float:
            count = end - start
            total = float(prefix[end] - prefix[start])
            total_sq = float(prefix_sq[end] - prefix_sq[start])
            return max(total_sq - total * total / count, 0.0)

        segmented_sse = sum(sse(start, end) for start, end in segments)
        objective[group] = (
            segmented_sse + config.segmentation_penalty * (len(segments) - 1)
        )
        sse_gain[group] = sse(0, len(ordered_values)) - segmented_sse
        segment_count[group] = len(segments)
        for number, (start, end) in enumerate(segments, 1):
            segment_index[ordered_indexes[start:end]] = number
        if len(segments) == 1:
            no_contrast[group] = 1
            continue

        reference_first = len(segments)
        merged_size = 0
        for number in range(len(segments) - 1, -1, -1):
            start, end = segments[number]
            merged_size += end - start
            reference_first = number
            if merged_size >= config.minimum_reference_strings:
                break
        if merged_size < config.minimum_reference_strings or reference_first == 0:
            continue
        reference_start = segments[reference_first][0]
        reference_indexes = ordered_indexes[reference_start:]
        lower_indexes = ordered_indexes[:reference_start]
        if len(lower_indexes) < config.minimum_candidate_strings:
            continue

        reference_median = float(np.median(residual_values[reference_indexes]))
        lower_median = float(np.median(residual_values[lower_indexes]))
        if not np.isfinite(reference_median) or reference_median <= 0:
            continue
        group_drop = 1.0 - lower_median / reference_median
        group_ok = (
            group_drop + config.group_relative_drop_tolerance
            >= config.minimum_group_relative_drop
        )

        reference_reliable[group] = 1
        reference_segment_count[group] = len(segments) - reference_first
        reference_size[group] = len(reference_indexes)
        structural_size[group] = len(lower_indexes)
        candidate_fraction[group] = len(lower_indexes) / len(usable)
        reference_residual[group] = reference_median
        candidate_residual[group] = lower_median
        group_drop_values[group] = group_drop
        group_ok_values[group] = int(group_ok)
        structural[lower_indexes] = 1
        if not group_ok:
            continue

        accepted_indexes = lower_indexes[original_raw_values[lower_indexes]]
        original_accepted[accepted_indexes] = 1

        eligible_boundaries: list[tuple[float, int, int]] = []
        for boundary in range(1, len(segments)):
            lower_end = segments[boundary - 1][1]
            upper_start = segments[boundary][0]
            if (
                lower_end >= config.minimum_candidate_strings
                and lower_end <= reference_start
                and len(ordered_values) - upper_start >= config.minimum_reference_strings
            ):
                gap = float(ordered_values[upper_start] - ordered_values[lower_end - 1])
                eligible_boundaries.append((gap, boundary, lower_end))

        if eligible_boundaries:
            gap, boundary, rescue_end = max(
                eligible_boundaries,
                key=lambda item: (item[0], -item[1]),
            )
            rescue_indexes = ordered_indexes[:rescue_end]
            rescue_structural[rescue_indexes] = 1
            rescue_segment_count[group] = boundary
            rescue_structural_size[group] = len(rescue_indexes)
            rescue_boundary_gap[group] = gap
            rescued_indexes = rescue_indexes[
                original_physical_values[rescue_indexes]
            ]
        else:
            rescued_indexes = np.asarray([], dtype=np.int64)
        rescue_candidate[rescued_indexes] = 1

        physical_candidates = np.unique(
            np.concatenate((accepted_indexes, rescued_indexes))
        ).astype(np.int64)
        candidate_size[group] = len(physical_candidates)
        if len(physical_candidates) < config.minimum_candidate_strings:
            continue
        candidate[physical_candidates] = 1
        detected[group] = 1

    result[f"{PREFIX}_normalizer"] = normalizer
    result[f"{PREFIX}_normalized_residual"] = normalized
    result[f"{PREFIX}_relative_drop"] = external_relative.astype(np.float32)
    result[f"{PREFIX}_internal_expected_current"] = internal_expected_values
    result[f"{PREFIX}_absolute_drop"] = external_absolute.astype(np.float32)
    result[f"{PREFIX}_internal_relative_drop"] = internal_relative_values
    result[f"{PREFIX}_internal_absolute_drop"] = internal_absolute_values
    result[f"{PREFIX}_external_physical_effect_eligible"] = external_ok_values
    result[f"{PREFIX}_internal_physical_effect_eligible"] = internal_ok_values
    result[f"{PREFIX}_physical_effect_eligible"] = external_ok_values
    result[f"{SEG}_structural_member"] = structural
    result[f"{SEG}_original_candidate_accepted"] = original_accepted
    result[f"{SEG}_rescue_structural_member"] = rescue_structural
    result[f"{SEG}_rescue_raw_candidate"] = rescue_candidate
    result[f"{SEG}_partial_next_segment_raw_candidate"] = partial_next_candidate
    result[f"{SEG}_small_candidate_rescue_raw_candidate"] = small_candidate_rescue
    result[f"{SEG}_raw_candidate"] = candidate
    result[f"{SEG}_detected"] = detected
    result[f"{PREFIX}_no_internal_contrast"] = no_contrast
    result[f"{PREFIX}_reference_reliable"] = reference_reliable
    result[f"{PREFIX}_group_effect_eligible"] = group_ok_values
    result[f"{PREFIX}_segment_index"] = segment_index
    result[f"{PREFIX}_segment_count"] = segment_count
    result[f"{SEG}_reference_segment_count"] = reference_segment_count
    result[f"{SEG}_structural_size"] = structural_size
    result[f"{SEG}_rescue_segment_count"] = rescue_segment_count
    result[f"{SEG}_rescue_structural_size"] = rescue_structural_size
    result[f"{SEG}_rescue_boundary_gap"] = rescue_boundary_gap
    result[f"{SEG}_candidate_size"] = candidate_size
    result[f"{SEG}_reference_size"] = reference_size
    result[f"{SEG}_candidate_fraction"] = candidate_fraction
    result[f"{SEG}_reference_residual"] = reference_residual
    result[f"{SEG}_candidate_residual"] = candidate_residual
    result[f"{SEG}_group_relative_drop"] = group_drop_values
    result[f"{SEG}_objective"] = objective
    result[f"{SEG}_sse_gain"] = sse_gain
    return result, valid


def _segment_internal_member_gate_v2(
    frame: pd.DataFrame,
    config: PVLOFV17ImprovedConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply V2/V3/V4 gates or the V5 lowest-core expansion gate.

    V2 requires the median of every lower-side structural member to be at
    least ``minimum_group_relative_drop`` below the reference median before
    any member can become a candidate. V3 retains that group statistic as a
    diagnostic, but selects each lower-side member solely from its own
    reference-relative physical effect. V5 always starts from the lowest
    segment and expands upward only through a small adjacent boundary or
    coherent external evidence backed by a single-segment reference.
    """
    result = frame.copy()
    external_relative, external_absolute, original_physical = _physical_effect(
        result, config
    )
    current = pd.to_numeric(result["string_current"], errors="coerce")
    expected = pd.to_numeric(result["expected_current"], errors="coerce")
    residual = pd.to_numeric(result["residual_ratio"], errors="coerce")
    score = pd.to_numeric(result["pvlof_score"], errors="coerce")
    valid = (
        result["v2_eligible"].fillna(False).astype(bool)
        & result["response_known"].fillna(False).astype(bool)
        & current.gt(0)
        & expected.gt(0)
        & residual.gt(0)
        & score.notna()
    )

    n = len(result)
    zeros8 = lambda: np.zeros(n, dtype=np.int8)
    zeros16 = lambda: np.zeros(n, dtype=np.int16)
    nans = lambda: np.full(n, np.nan, dtype=np.float32)
    structural, candidate, detected = zeros8(), zeros8(), zeros8()
    original_accepted, rescue_structural, rescue_candidate = zeros8(), zeros8(), zeros8()
    partial_next_candidate, small_candidate_rescue = zeros8(), zeros8()
    reference_member_gate_pass = zeros8()
    lowest_core_member, lowest_core_candidate = zeros8(), zeros8()
    validated_expansion_candidate, expansion_gate_pass = zeros8(), zeros8()
    expansion_external_support_pass, expansion_small_gap_pass = zeros8(), zeros8()
    no_contrast, reference_reliable = zeros8(), zeros8()
    group_ok_values, external_ok_values, internal_ok_values = zeros8(), zeros8(), zeros8()
    segment_index, segment_count = zeros16(), zeros16()
    reference_segment_count, structural_size = zeros16(), zeros16()
    rescue_segment_count, rescue_structural_size = zeros16(), zeros16()
    candidate_size, reference_size = zeros16(), zeros16()
    lowest_core_size, expansion_segment_count, expansion_size = (
        zeros16(), zeros16(), zeros16()
    )
    candidate_fraction, normalizer, normalized = nans(), nans(), nans()
    reference_residual, candidate_residual, group_drop_values = nans(), nans(), nans()
    internal_relative_values, internal_expected_values, internal_absolute_values = nans(), nans(), nans()
    rescue_boundary_gap = nans()
    objective, sse_gain = nans(), nans()

    current_values = current.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    residual_values = residual.to_numpy(dtype=float)
    valid_values = valid.to_numpy(dtype=bool)
    original_raw_values = result["pvlof_v17_segmentation_raw_candidate"].fillna(False).to_numpy(dtype=bool)
    original_physical_values = original_physical.to_numpy(dtype=bool)
    external_absolute_values = external_absolute.to_numpy(dtype=float)
    external_ok_values[:] = original_physical_values.astype(np.int8)
    minimum_total = config.minimum_reference_strings + config.minimum_candidate_strings

    for _, positions in result.groupby(
        ["plant_id", "device_no", "event_time"], observed=True, sort=False
    ).indices.items():
        group = np.asarray(positions, dtype=np.int64)
        usable = group[valid_values[group]]
        if len(usable) < minimum_total:
            continue
        scale = float(np.median(residual_values[usable]))
        if not np.isfinite(scale) or scale <= 0:
            continue
        normalizer[group] = scale
        normalized[usable] = residual_values[usable] / scale
        order = np.argsort(normalized[usable], kind="stable")
        ordered_indexes = usable[order]
        ordered_values = normalized[ordered_indexes].astype(float)
        segments = penalized_segments(
            ordered_values,
            penalty=config.segmentation_penalty,
            minimum_segment_size=config.minimum_segment_strings,
        )
        if not segments:
            continue

        prefix = np.concatenate(([0.0], np.cumsum(ordered_values)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(ordered_values * ordered_values)))

        def sse(start, end):
            count = end - start
            total = float(prefix[end] - prefix[start])
            total_sq = float(prefix_sq[end] - prefix_sq[start])
            return max(total_sq - total * total / count, 0.0)

        segmented_sse = sum(sse(start, end) for start, end in segments)
        objective[group] = segmented_sse + config.segmentation_penalty * (len(segments) - 1)
        sse_gain[group] = sse(0, len(ordered_values)) - segmented_sse
        segment_count[group] = len(segments)
        for number, (start, end) in enumerate(segments, 1):
            segment_index[ordered_indexes[start:end]] = number
        if len(segments) == 1:
            no_contrast[group] = 1
            continue

        reference_first = len(segments)
        merged_size = 0
        for number in range(len(segments) - 1, -1, -1):
            start, end = segments[number]
            merged_size += end - start
            reference_first = number
            if merged_size >= config.minimum_reference_strings:
                break
        if merged_size < config.minimum_reference_strings or reference_first == 0:
            continue
        reference_start = segments[reference_first][0]
        reference_indexes = ordered_indexes[reference_start:]
        lower_indexes = ordered_indexes[:reference_start]
        if len(lower_indexes) < config.minimum_candidate_strings:
            continue

        reference_median = float(np.median(residual_values[reference_indexes]))
        lower_median = float(np.median(residual_values[lower_indexes]))
        if not np.isfinite(reference_median) or reference_median <= 0:
            continue
        group_drop = 1.0 - lower_median / reference_median
        group_ok = (
            group_drop + config.group_relative_drop_tolerance
            >= config.minimum_group_relative_drop
        )

        reference_reliable[group] = 1
        reference_segment_count[group] = len(segments) - reference_first
        reference_size[group] = len(reference_indexes)
        structural_size[group] = len(lower_indexes)
        candidate_fraction[group] = len(lower_indexes) / len(usable)
        reference_residual[group] = reference_median
        candidate_residual[group] = lower_median
        group_drop_values[group] = group_drop
        group_ok_values[group] = int(group_ok)
        structural[lower_indexes] = 1

        member_relative = 1.0 - residual_values[lower_indexes] / reference_median
        member_expected = expected_values[lower_indexes] * reference_median
        member_absolute = member_expected - current_values[lower_indexes]
        internal_relative_values[lower_indexes] = member_relative
        internal_expected_values[lower_indexes] = member_expected
        internal_absolute_values[lower_indexes] = member_absolute
        member_internal_ok = _effect_eligible(member_relative, member_absolute, config)
        internal_ok_values[lower_indexes] = member_internal_ok.astype(np.int8)

        if config.algorithm_variant in {
            "reference_member_gate_v3",
            "combined_candidate_memory_v4",
            "lowest_core_expansion_v5",
        }:
            # Segmentation defines the reference and search region. A broad
            # lower-side median must not veto individually strong members.
            direct_indexes = lower_indexes[member_internal_ok]
            reference_member_gate_pass[direct_indexes] = 1

        if config.algorithm_variant == "lowest_core_expansion_v5":
            # The first (lowest-residual) segment is the conservative core.
            # Higher low-side segments are not candidates merely because they
            # are below the high-side reference: expansion must be contiguous
            # and coherently supported by the external physical gate too.
            core_start, core_end = segments[0]
            core_indexes = ordered_indexes[core_start:core_end]
            lowest_core_member[core_indexes] = 1
            lowest_core_size[group] = len(core_indexes)
            core_supported = core_indexes[
                internal_ok_values[core_indexes].astype(bool)
            ]
            core_support_fraction = len(core_supported) / len(core_indexes)
            if core_support_fraction + config.group_relative_drop_tolerance < (
                config.minimum_core_segment_support
            ):
                continue
            lowest_core_candidate[core_indexes] = 1

            candidate_parts = [core_indexes]
            expanded_segments = 0
            for start, end in segments[1:reference_first]:
                segment_indexes = ordered_indexes[start:end]
                supported = segment_indexes[
                    original_physical_values[segment_indexes]
                ]
                support_fraction = len(supported) / len(segment_indexes)
                external_support_ok = (
                    len(segments) - reference_first
                    <= config.maximum_external_expansion_reference_segments
                    and support_fraction + config.group_relative_drop_tolerance
                    >= config.minimum_expansion_segment_support
                )
                boundary_gap = float(ordered_values[start] - ordered_values[start - 1])
                small_gap_ok = (
                    boundary_gap
                    <= config.maximum_core_expansion_boundary_gap
                    + config.group_relative_drop_tolerance
                )
                if not (external_support_ok or small_gap_ok):
                    break
                if external_support_ok:
                    selected_expansion = supported
                    expansion_external_support_pass[supported] = 1
                else:
                    selected_expansion = segment_indexes[
                        internal_ok_values[segment_indexes].astype(bool)
                    ]
                if not len(selected_expansion):
                    break
                if small_gap_ok:
                    expansion_small_gap_pass[selected_expansion] = 1
                expansion_gate_pass[selected_expansion] = 1
                validated_expansion_candidate[selected_expansion] = 1
                candidate_parts.append(selected_expansion)
                expanded_segments += 1

            selected_indexes = np.unique(
                np.concatenate(candidate_parts)
            ).astype(np.int64)
            expansion_segment_count[group] = expanded_segments
            expansion_size[group] = sum(len(part) for part in candidate_parts[1:])
            candidate_size[group] = len(selected_indexes)
            if len(selected_indexes) >= config.minimum_candidate_strings:
                candidate[selected_indexes] = 1
                detected[group] = 1
            continue

        if config.algorithm_variant in {
            "reference_member_gate_v3",
            "combined_candidate_memory_v4",
        }:
            candidate_size[group] = len(direct_indexes)
            if len(direct_indexes) >= config.minimum_candidate_strings:
                candidate[direct_indexes] = 1
                detected[group] = 1
            continue

        eligible_boundaries = []
        for boundary in range(1, len(segments)):
            lower_end = segments[boundary - 1][1]
            upper_start = segments[boundary][0]
            if (
                lower_end >= config.minimum_candidate_strings
                and lower_end <= reference_start
                and len(ordered_values) - upper_start >= config.minimum_reference_strings
            ):
                gap = float(ordered_values[upper_start] - ordered_values[lower_end - 1])
                eligible_boundaries.append((gap, boundary, lower_end))
        if eligible_boundaries:
            gap, boundary, rescue_end = max(
                eligible_boundaries,
                key=lambda item: (item[0], -item[1]),
            )
            rescue_indexes = ordered_indexes[:rescue_end]
            rescue_structural[rescue_indexes] = 1
            rescue_segment_count[group] = boundary
            rescue_structural_size[group] = len(rescue_indexes)
            rescue_boundary_gap[group] = gap
        else:
            rescue_indexes = np.asarray([], dtype=np.int64)

        if not group_ok:
            original_lower_indexes = lower_indexes[original_raw_values[lower_indexes]]
            small_rescue_ok = (
                config.minimum_candidate_strings <= len(original_lower_indexes)
                <= config.small_candidate_rescue_max_strings
                and group_drop + config.group_relative_drop_tolerance
                >= config.minimum_small_candidate_group_relative_drop
                and np.all(
                    external_absolute_values[original_lower_indexes]
                    >= config.minimum_absolute_drop
                )
                and len(np.unique(segment_index[original_lower_indexes])) == 1
            )
            if small_rescue_ok:
                small_candidate_rescue[original_lower_indexes] = 1
                candidate[original_lower_indexes] = 1
                candidate_size[group] = len(original_lower_indexes)
                detected[group] = 1
            continue

        rescued_indexes = rescue_indexes[
            internal_ok_values[rescue_indexes].astype(bool)
        ]
        rescue_candidate[rescued_indexes] = 1

        partial_indexes = np.asarray([], dtype=np.int64)
        if eligible_boundaries and boundary < reference_first:
            next_start, next_end = segments[boundary]
            next_indexes = ordered_indexes[next_start:next_end]
            next_internal = next_indexes[
                internal_ok_values[next_indexes].astype(bool)
            ]
            if 0 < len(next_internal) < len(next_indexes):
                partial_indexes = next_internal
                partial_next_candidate[partial_indexes] = 1

        coherent_original_parts = []
        for start, end in segments[:reference_first]:
            segment_indexes = ordered_indexes[start:end]
            supported = segment_indexes[
                original_raw_values[segment_indexes]
                & (external_absolute_values[segment_indexes] >= config.minimum_absolute_drop)
            ]
            support_fraction = len(supported) / len(segment_indexes)
            if support_fraction + config.group_relative_drop_tolerance >= (
                config.minimum_original_segment_support
            ):
                coherent_original_parts.append(supported)
        accepted_indexes = (
            np.concatenate(coherent_original_parts).astype(np.int64)
            if coherent_original_parts
            else np.asarray([], dtype=np.int64)
        )
        original_accepted[accepted_indexes] = 1
        physical_candidates = np.unique(
            np.concatenate((accepted_indexes, rescued_indexes, partial_indexes))
        ).astype(np.int64)
        candidate_size[group] = len(physical_candidates)
        if len(physical_candidates) < config.minimum_candidate_strings:
            continue
        candidate[physical_candidates] = 1
        detected[group] = 1

    result[f"{PREFIX}_normalizer"] = normalizer
    result[f"{PREFIX}_normalized_residual"] = normalized
    result[f"{PREFIX}_relative_drop"] = external_relative.astype(np.float32)
    result[f"{PREFIX}_internal_expected_current"] = internal_expected_values
    result[f"{PREFIX}_absolute_drop"] = external_absolute.astype(np.float32)
    result[f"{PREFIX}_internal_relative_drop"] = internal_relative_values
    result[f"{PREFIX}_internal_absolute_drop"] = internal_absolute_values
    result[f"{PREFIX}_external_physical_effect_eligible"] = external_ok_values
    result[f"{PREFIX}_internal_physical_effect_eligible"] = internal_ok_values
    result[f"{PREFIX}_physical_effect_eligible"] = internal_ok_values
    result[f"{SEG}_structural_member"] = structural
    result[f"{SEG}_original_candidate_accepted"] = original_accepted
    result[f"{SEG}_rescue_structural_member"] = rescue_structural
    result[f"{SEG}_rescue_raw_candidate"] = rescue_candidate
    result[f"{SEG}_partial_next_segment_raw_candidate"] = partial_next_candidate
    result[f"{SEG}_small_candidate_rescue_raw_candidate"] = small_candidate_rescue
    result[f"{SEG}_reference_member_gate_pass"] = reference_member_gate_pass
    result[f"{SEG}_lowest_core_member"] = lowest_core_member
    result[f"{SEG}_lowest_core_raw_candidate"] = lowest_core_candidate
    result[f"{SEG}_validated_expansion_raw_candidate"] = (
        validated_expansion_candidate
    )
    result[f"{SEG}_expansion_gate_pass"] = expansion_gate_pass
    result[f"{SEG}_expansion_external_support_pass"] = (
        expansion_external_support_pass
    )
    result[f"{SEG}_expansion_small_gap_pass"] = expansion_small_gap_pass
    result[f"{SEG}_raw_candidate"] = candidate
    result[f"{SEG}_detected"] = detected
    result[f"{PREFIX}_no_internal_contrast"] = no_contrast
    result[f"{PREFIX}_reference_reliable"] = reference_reliable
    result[f"{PREFIX}_group_effect_eligible"] = group_ok_values
    result[f"{PREFIX}_segment_index"] = segment_index
    result[f"{PREFIX}_segment_count"] = segment_count
    result[f"{SEG}_reference_segment_count"] = reference_segment_count
    result[f"{SEG}_structural_size"] = structural_size
    result[f"{SEG}_rescue_segment_count"] = rescue_segment_count
    result[f"{SEG}_rescue_structural_size"] = rescue_structural_size
    result[f"{SEG}_rescue_boundary_gap"] = rescue_boundary_gap
    result[f"{SEG}_candidate_size"] = candidate_size
    result[f"{SEG}_reference_size"] = reference_size
    result[f"{SEG}_lowest_core_size"] = lowest_core_size
    result[f"{SEG}_expansion_segment_count"] = expansion_segment_count
    result[f"{SEG}_expansion_size"] = expansion_size
    result[f"{SEG}_candidate_fraction"] = candidate_fraction
    result[f"{SEG}_reference_residual"] = reference_residual
    result[f"{SEG}_candidate_residual"] = candidate_residual
    result[f"{SEG}_group_relative_drop"] = group_drop_values
    result[f"{SEG}_objective"] = objective
    result[f"{SEG}_sse_gain"] = sse_gain
    return result, valid


def apply_pvlof_v17_improved(frame: pd.DataFrame, config: PVLOFV17ImprovedConfig):
    required = {
        "plant_id", "device_no", "event_time", "string_no", "v2_eligible",
        "response_known", "string_current", "expected_current", "residual_ratio",
        "residual_median", "pvlof_score", "pvlof_v16_raw_anomaly", "pvlof_v16_alert",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PVLOF v1.7 improved input is missing columns: {missing}")
    result = frame.copy()
    if "pvlof_v17_segmentation_raw_candidate" not in result:
        original_config = PVLOFV17Config(
            version="pvlof-v1.7-penalized-segmentation",
            segmentation_penalty=config.segmentation_penalty,
            minimum_segment_strings=config.minimum_segment_strings,
            minimum_reference_strings=config.minimum_reference_strings,
            minimum_candidate_strings=config.minimum_candidate_strings,
            minimum_relative_drop=config.minimum_relative_drop,
            minimum_absolute_drop=config.minimum_absolute_drop,
            effect_gate_mode=config.effect_gate_mode,
            entry_consecutive=config.entry_consecutive,
            recovery_consecutive=config.recovery_consecutive,
            expected_interval_minutes=config.expected_interval_minutes,
            maximum_unknown_intervals=config.maximum_unknown_intervals,
        )
        result = apply_pvlof_v17(result, original_config)
    result["plant_id"] = result["plant_id"].astype(str)
    result["device_no"] = result["device_no"].astype(str)
    result["event_time"] = pd.to_datetime(result["event_time"], errors="raise", utc=True)
    result["string_no"] = pd.to_numeric(result["string_no"], errors="raise").astype("Int64")
    result = result.sort_values([*KEYS, "event_time"]).reset_index(drop=True)
    if config.algorithm_variant == "dual_branch":
        result, valid = _segment_dual_branch(result, config)
    else:
        result, valid = _segment_internal_member_gate_v2(result, config)

    reference_member_column = f"{SEG}_reference_member_gate_pass"
    if reference_member_column not in result:
        result[reference_member_column] = np.int8(0)
    reference_member_variants = {
        "reference_member_gate_v3",
        "combined_candidate_memory_v4",
        "lowest_core_expansion_v5",
    }
    result[f"{PREFIX}_group_gate_applied"] = np.int8(
        config.algorithm_variant not in reference_member_variants
    )

    segmentation_raw = result[f"{SEG}_raw_candidate"].astype(bool)
    state = _memory_state(result, raw=segmentation_raw, valid=valid, config=config)
    for name, values in state.items():
        result[f"{SEG}_{name}"] = values
    result[f"{SEG}_strict_alert"] = (
        segmentation_raw & result[f"{SEG}_entry_streak"].ge(config.entry_consecutive)
    ).astype(np.int8)

    v16_raw = result["pvlof_v16_raw_anomaly"].fillna(False).astype(bool)
    v16_alert = result["pvlof_v16_alert"].fillna(False).astype(bool)
    v16_active = _numeric_column(result, "pvlof_v16_memory_active").astype(bool)
    v16_reactivated = _numeric_column(result, "pvlof_v16_memory_reactivated_alert").astype(bool)
    v16_clear = _numeric_column(result, "pvlof_v16_memory_clear_code")
    v16_entry = _numeric_column(result, "pvlof_v16_entry_streak")
    v16_normal = _numeric_column(result, "pvlof_v16_normal_streak")
    seg_alert = result[f"{SEG}_memory_alert"].astype(bool)
    seg_active = result[f"{SEG}_memory_active"].astype(bool)
    seg_reactivated = result[f"{SEG}_memory_reactivated_alert"].astype(bool)
    seg_clear = result[f"{SEG}_memory_clear_code"].astype("int32")
    aggregate_active = v16_active | seg_active
    aggregate_clear = pd.concat([v16_clear, seg_clear], axis=1).max(axis=1)
    aggregate_clear = aggregate_clear.where(~aggregate_active, 0).astype(np.int8)

    v16_middle_group_veto = pd.Series(False, index=result.index)
    if config.algorithm_variant == "lowest_core_expansion_v5":
        structural = result[f"{SEG}_structural_member"].astype(bool)
        core = result[f"{SEG}_lowest_core_member"].astype(bool)
        expanded = result[f"{SEG}_validated_expansion_raw_candidate"].astype(bool)
        unvalidated_middle = structural & ~core & ~expanded
        v16_middle_group_veto = v16_raw & unvalidated_middle
        combined_raw = segmentation_raw | (v16_raw & ~unvalidated_middle)
    else:
        combined_raw = v16_raw | segmentation_raw
    result[f"{PREFIX}_v16_middle_group_veto"] = (
        v16_middle_group_veto.astype(np.int8)
    )
    result[f"{PREFIX}_raw_anomaly"] = combined_raw.astype(np.int8)
    result[f"{SEG}_alert"] = seg_alert.astype(np.int8)

    if config.algorithm_variant in {
        "combined_candidate_memory_v4",
        "lowest_core_expansion_v5",
    }:
        current = pd.to_numeric(result["string_current"], errors="coerce")
        score = pd.to_numeric(result["pvlof_score"], errors="coerce")
        combined_valid = (
            result["v2_eligible"].fillna(False).astype(bool)
            & result["response_known"].fillna(False).astype(bool)
            & current.gt(0)
            & score.notna()
        )
        combined_state = _memory_state(
            result,
            raw=combined_raw,
            valid=combined_valid,
            config=config,
        )
        for name, values in combined_state.items():
            result[f"{PREFIX}_combined_{name}"] = values

        combined_alert = result[f"{PREFIX}_combined_memory_alert"].astype(bool)
        final_alert = v16_alert | combined_alert
        result[f"{PREFIX}_combined_raw_candidate"] = combined_raw.astype(np.int8)
        result[f"{PREFIX}_combined_bridge_alert"] = (
            combined_alert & ~v16_alert & ~seg_alert
        ).astype(np.int8)
        result[f"{PREFIX}_valid_point"] = combined_valid.astype(np.int8)
        result[f"{PREFIX}_entry_streak"] = result[
            f"{PREFIX}_combined_entry_streak"
        ].astype(np.int32)
        result[f"{PREFIX}_normal_streak"] = result[
            f"{PREFIX}_combined_normal_streak"
        ].astype(np.int32)
        result[f"{PREFIX}_memory_active"] = result[
            f"{PREFIX}_combined_memory_active"
        ].astype(np.int8)
        result[f"{PREFIX}_memory_clear_code"] = result[
            f"{PREFIX}_combined_memory_clear_code"
        ].astype(np.int8)
        result[f"{PREFIX}_memory_reactivated_alert"] = result[
            f"{PREFIX}_combined_memory_reactivated_alert"
        ].astype(np.int8)
        result[f"{PREFIX}_unknown_streak"] = result[
            f"{PREFIX}_combined_unknown_streak"
        ].astype(np.int32)
        result[f"{PREFIX}_memory_suspended"] = result[
            f"{PREFIX}_combined_memory_suspended"
        ].astype(np.int8)
        result[f"{PREFIX}_memory_resumed_alert"] = result[
            f"{PREFIX}_combined_memory_resumed_alert"
        ].astype(np.int8)
    else:
        final_alert = v16_alert | seg_alert
        result[f"{PREFIX}_valid_point"] = valid.astype(np.int8)
        result[f"{PREFIX}_entry_streak"] = pd.concat(
            [v16_entry, result[f"{SEG}_entry_streak"]], axis=1
        ).max(axis=1).astype(np.int32)
        result[f"{PREFIX}_normal_streak"] = pd.concat(
            [v16_normal, result[f"{SEG}_normal_streak"]], axis=1
        ).max(axis=1).astype(np.int32)
        result[f"{PREFIX}_memory_active"] = aggregate_active.astype(np.int8)
        result[f"{PREFIX}_memory_clear_code"] = aggregate_clear
        result[f"{PREFIX}_memory_reactivated_alert"] = (
            v16_reactivated | seg_reactivated
        ).astype(np.int8)
        result[f"{PREFIX}_unknown_streak"] = result[
            f"{SEG}_unknown_streak"
        ].astype(np.int32)
        result[f"{PREFIX}_memory_suspended"] = result[
            f"{SEG}_memory_suspended"
        ].astype(np.int8)
        result[f"{PREFIX}_memory_resumed_alert"] = result[
            f"{SEG}_memory_resumed_alert"
        ].astype(np.int8)

    result[f"{PREFIX}_alert"] = final_alert.astype(np.int8)
    result[f"{PREFIX}_added_alert"] = (
        result[f"{PREFIX}_alert"].astype(bool) & ~v16_alert
    ).astype(np.int8)
    return result.sort_values(
        ["event_time", "plant_id", "device_no", "string_no"]
    ).reset_index(drop=True)


__all__ = ["PVLOFV17ImprovedConfig", "apply_pvlof_v17_improved", "load_config", "save_config"]
