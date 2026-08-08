"""Calibrate an EWMA underproduction detector on validation residuals."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from pv_anomaly.ewma import fit_ewma_calibration, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        default="artifacts/models/transformer_residual_plus15_peer/residuals/validation.parquet",
    )
    parser.add_argument(
        "--output",
        default="artifacts/models/transformer_residual_plus15_peer/ewma_calibration.json",
    )
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.2)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--minimum-power-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-power-ratio", type=float, default=1.10)
    parser.add_argument("--minimum-consecutive", type=int, default=2)
    parser.add_argument("--interval-minutes", type=int, default=5)
    args = parser.parse_args()
    frame = pd.read_parquet(args.validation)
    calibration, report = fit_ewma_calibration(
        frame,
        lambda_=args.lambda_,
        quantile=args.quantile,
        minimum_power_ratio=args.minimum_power_ratio,
        maximum_power_ratio=args.maximum_power_ratio,
        minimum_consecutive=args.minimum_consecutive,
        expected_interval_minutes=args.interval_minutes,
    )
    save_calibration(calibration, args.output)
    print(
        json.dumps(
            {"calibration": calibration.to_dict(), "report": report},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
