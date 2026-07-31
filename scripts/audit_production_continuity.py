"""Audit duplicate rows and timestamp continuity for one production PV plant."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

try:
    from _common import client_from_env
except ModuleNotFoundError:  # pragma: no cover - supports importing from pytest
    from scripts._common import client_from_env


def _utc_iso(value: str, timezone_name: str) -> str:
    local = datetime.fromisoformat(value)
    if local.tzinfo is None:
        local = local.replace(tzinfo=ZoneInfo(timezone_name))
    return local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"event_time is not a string: {value!r}")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _empty_device_summary() -> dict[str, Any]:
    return {
        "rows": 0,
        "first_event": None,
        "last_event": None,
        "duplicate_rows": 0,
        "gap_segments": 0,
        "estimated_missing_points": 0,
        "short_or_irregular_intervals": 0,
    }


def audit_continuity(
    *,
    client: Any,
    index: str,
    plant_id: int,
    start: str,
    end: str,
    timezone_name: str,
    interval_minutes: int,
    page_size: int,
) -> dict[str, Any]:
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be positive")
    if not 1 <= page_size <= 5000:
        raise ValueError("page_size must be between 1 and 5000")

    start_utc = _utc_iso(start, timezone_name)
    end_utc = _utc_iso(end, timezone_name)
    expected_delta = timedelta(minutes=interval_minutes)
    search_after: list[Any] | None = None
    rows_scanned = 0
    pages = 0
    device_summary: dict[str, dict[str, Any]] = defaultdict(_empty_device_summary)
    previous_by_device: dict[str, datetime] = {}
    daily_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    while True:
        body: dict[str, Any] = {
            "size": page_size,
            "track_total_hits": False,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"plant_id": plant_id}},
                        {"range": {"event_time": {"gte": start_utc, "lt": end_utc}}},
                    ]
                }
            },
            "_source": ["plant_id", "device_no", "event_time"],
            "sort": [
                {"event_time": "asc"},
                {"device_no": "asc"},
                {"_id": "asc"},
            ],
        }
        if search_after is not None:
            body["search_after"] = search_after
        response = client.request(
            "GET",
            f"/{quote(index, safe='*,-_')}/_search",
            json_body=body,
        ).body
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break
        pages += 1
        for hit in hits:
            source = hit.get("_source", {})
            device = str(source.get("device_no", ""))
            if not device:
                raise ValueError("A matching document has no device_no")
            event_time = _parse_time(source.get("event_time"))
            summary = device_summary[device]
            summary["rows"] += 1
            if summary["first_event"] is None or event_time.isoformat() < summary["first_event"]:
                summary["first_event"] = event_time.isoformat()
            if summary["last_event"] is None or event_time.isoformat() > summary["last_event"]:
                summary["last_event"] = event_time.isoformat()

            previous = previous_by_device.get(device)
            if previous is not None:
                delta = event_time - previous
                if delta == timedelta(0):
                    summary["duplicate_rows"] += 1
                elif delta > expected_delta:
                    summary["gap_segments"] += 1
                    missing = round(delta / expected_delta) - 1
                    summary["estimated_missing_points"] += max(0, missing)
                elif delta != expected_delta:
                    summary["short_or_irregular_intervals"] += 1
            previous_by_device[device] = event_time
            local_date = event_time.astimezone(ZoneInfo(timezone_name)).date().isoformat()
            daily_counts[local_date][device] += 1
            rows_scanned += 1

        last_sort = hits[-1].get("sort")
        if not isinstance(last_sort, list) or last_sort == search_after:
            raise RuntimeError("search_after did not advance")
        search_after = last_sort
        if len(hits) < page_size:
            break

    return {
        "index": index,
        "plant_id": plant_id,
        "start_local_inclusive": start,
        "end_local_exclusive": end,
        "start_utc_inclusive": start_utc,
        "end_utc_exclusive": end_utc,
        "timezone": timezone_name,
        "expected_interval_minutes": interval_minutes,
        "page_size": page_size,
        "pages": pages,
        "rows_scanned": rows_scanned,
        "unique_devices": len(device_summary),
        "duplicate_rows_total": sum(item["duplicate_rows"] for item in device_summary.values()),
        "gap_segments_total": sum(item["gap_segments"] for item in device_summary.values()),
        "estimated_missing_points_total": sum(
            item["estimated_missing_points"] for item in device_summary.values()
        ),
        "short_or_irregular_intervals_total": sum(
            item["short_or_irregular_intervals"] for item in device_summary.values()
        ),
        "per_device": dict(sorted(device_summary.items())),
        "daily_device_counts": {
            day: dict(sorted(counts.items()))
            for day, counts in sorted(daily_counts.items())
        },
        "read_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="pv_device_data")
    parser.add_argument("--plant-id", type=int, default=234)
    parser.add_argument("--start", default="2026-03-09")
    parser.add_argument("--end", default="2026-07-31", help="Exclusive local date")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=5000)
    parser.add_argument(
        "--output",
        default="artifacts/reports/production_plant_234_continuity.json",
    )
    args = parser.parse_args()
    report = audit_continuity(
        client=client_from_env(),
        index=args.index,
        plant_id=args.plant_id,
        start=args.start,
        end=args.end,
        timezone_name=args.timezone,
        interval_minutes=args.interval_minutes,
        page_size=args.page_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.resolve()}")
    print(json.dumps({key: report[key] for key in (
        "rows_scanned",
        "unique_devices",
        "duplicate_rows_total",
        "gap_segments_total",
        "estimated_missing_points_total",
        "short_or_irregular_intervals_total",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

