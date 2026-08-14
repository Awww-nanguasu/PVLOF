"""Export configured ES datasets into local date-partitioned Parquet files."""

from __future__ import annotations

import argparse
from datetime import date

from _common import client_from_env
from pv_anomaly.export import ExportConfig, export_dataset, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="Local YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="Exclusive local date")
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--dataset", action="append", help="Dataset name; repeat as needed")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument(
        "--plant-id",
        type=int,
        help="Filter all selected datasets to one production plant",
    )
    parser.add_argument(
        "--manifest",
        help="Output manifest path; defaults to a plant-specific path when filtered",
    )
    args = parser.parse_args()

    config = ExportConfig.from_yaml(args.config)
    selected = args.dataset or list(config.datasets)
    unknown = sorted(set(selected) - set(config.datasets))
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")
    if args.plant_id is not None and args.plant_id <= 0:
        raise SystemExit("--plant-id must be a positive integer")
    manifest = args.manifest or (
        f"artifacts/reports/export_manifest_plant_{args.plant_id}.json"
        if args.plant_id is not None
        else "artifacts/reports/export_manifest.json"
    )
    client = client_from_env()
    results = []
    for name in selected:
        print(f"Exporting {name} ...", flush=True)
        result = export_dataset(
            client,
            config.datasets[name],
            start_date=args.start,
            end_date=args.end,
            timezone_name=config.timezone_name,
            page_size=args.page_size,
            plant_id=args.plant_id,
        )
        results.append(result)
        print(f"Exported {result['exported_rows']} rows for {name}", flush=True)
    write_manifest(results, manifest)
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
