"""Build aligned and chronologically split candidate-normal training Parquet files."""

from __future__ import annotations

import argparse
import json

from pv_anomaly.training_data import TrainingConfig, build_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    report = build_training_data(TrainingConfig.from_yaml(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

