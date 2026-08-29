"""Write the initial PVLOF v1.7 segmentation configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.pvlof_v17 import PVLOFV17Config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--segmentation-penalty", type=float, default=0.01)
    parser.add_argument("--minimum-segment-strings", type=int, default=2)
    parser.add_argument("--minimum-reference-strings", type=int, default=5)
    parser.add_argument("--minimum-candidate-strings", type=int, default=2)
    parser.add_argument("--minimum-relative-drop", type=float, default=0.20)
    parser.add_argument("--minimum-absolute-drop", type=float, default=0.50)
    parser.add_argument("--effect-gate-mode", choices=("all", "any"), default="any")
    parser.add_argument("--entry-consecutive", type=int, default=3)
    parser.add_argument("--recovery-consecutive", type=int, default=3)
    parser.add_argument("--expected-interval-minutes", type=int, default=5)
    parser.add_argument("--version", default="pvlof-v1.7-penalized-segmentation")
    args = parser.parse_args()

    config = PVLOFV17Config(
        version=args.version,
        segmentation_penalty=args.segmentation_penalty,
        minimum_segment_strings=args.minimum_segment_strings,
        minimum_reference_strings=args.minimum_reference_strings,
        minimum_candidate_strings=args.minimum_candidate_strings,
        minimum_relative_drop=args.minimum_relative_drop,
        minimum_absolute_drop=args.minimum_absolute_drop,
        effect_gate_mode=args.effect_gate_mode,
        entry_consecutive=args.entry_consecutive,
        recovery_consecutive=args.recovery_consecutive,
        expected_interval_minutes=args.expected_interval_minutes,
    )
    save_config(config, args.output)
    report = {
        "version": config.version,
        "algorithm": "penalized_1d_sorted_residual_segmentation",
        "objective": "within_segment_sse + penalty * change_points",
        "physical_effect_rule": (
            "relative_drop >= threshold OR absolute_drop >= threshold"
            if config.effect_gate_mode == "any"
            else "relative_drop >= threshold AND absolute_drop >= threshold"
        ),
        "standalone_physical_rescue": False,
        "whole_device_common_mode_alert": False,
        "v1_6_alerts_preserved": True,
        "parameter_status": "initial_v1_7_defaults_requiring_local_comparison",
        "config": config.to_dict(),
        "output": args.output,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
