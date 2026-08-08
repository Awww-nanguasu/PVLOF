"""Build current-time and plus-15-minute forecast-GHI modeling datasets."""

from __future__ import annotations

import argparse
import json

from pv_anomaly.forecast_features import build_forecast_variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-directory", default="data/processed/transformer")
    parser.add_argument(
        "--weather-directory",
        action="append",
        help="Weather root; repeat for multiple production plants",
    )
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument(
        "--report",
        default="artifacts/reports/forecast_ghi_variants.json",
    )
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    report = build_forecast_variants(
        base_directory=args.base_directory,
        weather_directory=args.weather_directory or ["data/raw/weather_15min"],
        output_root=args.output_root,
        report_path=args.report,
        timezone=args.timezone,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
