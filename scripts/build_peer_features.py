"""Build leave-one-device-out same-plant power features."""

from __future__ import annotations

import argparse
import json

from pv_anomaly.peer_features import build_peer_feature_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        default="data/processed/transformer_forecast_plus15",
    )
    parser.add_argument(
        "--output-directory",
        default="data/processed/transformer_forecast_plus15_peer",
    )
    parser.add_argument(
        "--report",
        default="artifacts/reports/peer_power_features.json",
    )
    parser.add_argument("--minimum-peers", type=int, default=3)
    args = parser.parse_args()
    report = build_peer_feature_dataset(
        input_directory=args.input_directory,
        output_directory=args.output_directory,
        report_path=args.report,
        minimum_peers=args.minimum_peers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
