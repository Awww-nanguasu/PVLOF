"""Audit forecast-GHI coverage and current/+15-minute timestamp semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_anomaly.forecast_features import read_weather_forecast
from pv_anomaly.pvlof_io import (
    apply_plant_id_mapping,
    parse_plant_id_mappings,
    read_pvlof_source,
)
from pv_anomaly.pvlof_v2 import load_calibration
from pv_anomaly.pvlof_v12 import (
    build_conditioned_virtual_context,
    fit_weather_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-root", required=True)
    parser.add_argument("--weather", action="append", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="Exclusive local date/time")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--plant-id-map", action="append", default=[], metavar="SOURCE=MODEL")
    parser.add_argument("--base-calibration", required=True)
    parser.add_argument("--minimum-mapping-samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame, device_report = read_pvlof_source(
        args.device_root,
        start=args.start,
        end=args.end,
        timezone=args.timezone,
    )
    mappings = parse_plant_id_mappings(args.plant_id_map)
    if mappings:
        frame, mapping_report = apply_plant_id_mapping(frame, mappings)
    else:
        mapping_report = {
            "requested": {},
            "rows_mapped": 0,
            "rows_unmapped": len(frame),
            "source_plants": sorted(frame["plant_id"].astype(str).unique().tolist()),
            "model_plants": sorted(frame["plant_id"].astype(str).unique().tolist()),
        }

    weather = read_weather_forecast(args.weather)
    weather_rows = len(weather)
    weather_plants_before = sorted(weather["plant_id"].astype(str).unique().tolist())
    if mappings:
        weather["plant_id"] = weather["plant_id"].astype(str).replace(mappings)
        weather = weather.drop_duplicates(["plant_id", "time"], keep="last")

    base = load_calibration(args.base_calibration)
    calibration, semantics = fit_weather_calibration(
        frame,
        base,
        weather,
        candidate_source_offsets_minutes=(0, 15),
        minimum_mapping_samples=args.minimum_mapping_samples,
    )
    _, conditioned = build_conditioned_virtual_context(
        frame, base, calibration, weather
    )
    selected = {
        plant: {
            "source_offset_minutes": record["source_offset_minutes"],
            "interpretation": (
                "current_time" if int(record["source_offset_minutes"]) == 0
                else "forecast_value_interpreted_as_plus_15_minutes"
            ),
            "samples": record["samples"],
            "spearman": record["spearman"],
        }
        for plant, record in calibration.plants.items()
    }
    report = {
        "definition": {
            "offset_0": "latest 15-minute weather slot at or before device time",
            "offset_15": "weather source timestamp shifted back 15 minutes before alignment",
            "selection": "highest Spearman correlation with peer-inverter virtual irradiance",
            "warning": "offset_15 is deployable only if the ES timestamp denotes forecast valid time or the value was already available 15 minutes earlier",
        },
        "device": device_report,
        "weather": {
            "roots": args.weather,
            "rows": weather_rows,
            "source_plants": weather_plants_before,
        },
        "plant_id_mapping": mapping_report,
        "candidate_results": semantics,
        "selected_semantics": selected,
        "conditioned_context": conditioned,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
