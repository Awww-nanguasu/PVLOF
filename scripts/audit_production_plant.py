"""Run a read-only schema and quality audit for one production PV plant."""

from __future__ import annotations

import argparse
from typing import Any

try:
    from _common import client_from_env, write_json
except ModuleNotFoundError:  # pragma: no cover - supports importing from pytest
    from scripts._common import client_from_env, write_json


DEVICE_FIELDS = (
    "device_no",
    "event_time",
    "active_power",
    "rated_power",
    "main_string_count",
    "valid_current_string_count",
    "string_overall_status",
    "string_current_01",
    "string_status_01",
)


def build_audit_query(*, plant_id: int, timezone: str, sample_size: int) -> dict[str, Any]:
    """Build only GET/_search aggregations and a bounded sample request."""
    existence_fields = (
        "event_time",
        "device_no",
        "active_power",
        "rated_power",
        "main_string_count",
        "valid_current_string_count",
        "string_overall_status",
        "string_current_01",
        "string_status_01",
    )
    missing_aggs = {
        f"missing_{field}": {"bool": {"must_not": {"exists": {"field": field}}}}
        for field in existence_fields
    }
    current_missing = {
        f"missing_string_current_{index:02d}": {
            "bool": {
                "must_not": {"exists": {"field": f"string_current_{index:02d}"}}
            }
        }
        for index in range(1, 31)
    }
    query = {"term": {"plant_id": plant_id}}
    return {
        "summary": {
            "size": 0,
            "track_total_hits": True,
            "query": query,
            "aggs": {
                "minimum": {"min": {"field": "event_time"}},
                "maximum": {"max": {"field": "event_time"}},
                "devices": {"cardinality": {"field": "device_no", "precision_threshold": 1000}},
                "status_code": {"terms": {"field": "status_code", "size": 20}},
                "string_overall_status": {
                    "terms": {"field": "string_overall_status", "size": 20}
                },
                "main_string_count": {
                    "terms": {"field": "main_string_count", "size": 50}
                },
                "valid_current_string_count": {
                    "terms": {"field": "valid_current_string_count", "size": 50}
                },
                "days": {
                    "date_histogram": {
                        "field": "event_time",
                        "calendar_interval": "day",
                        "time_zone": timezone,
                        "min_doc_count": 1,
                    },
                    "aggs": {
                        "first_event": {"min": {"field": "event_time"}},
                        "last_event": {"max": {"field": "event_time"}},
                    },
                },
                "device_days": {
                    "terms": {"field": "device_no", "size": 100},
                    "aggs": {
                        "days": {
                            "date_histogram": {
                                "field": "event_time",
                                "calendar_interval": "day",
                                "time_zone": timezone,
                                "min_doc_count": 1,
                            },
                            "aggs": {
                                "first_event": {"min": {"field": "event_time"}},
                                "last_event": {"max": {"field": "event_time"}},
                            },
                        }
                    },
                },
                "missing_fields": {"filters": {"filters": missing_aggs}},
                "missing_string_currents": {"filters": {"filters": current_missing}},
            },
        },
        "devices": {
            "size": max(1, min(sample_size, 1000)),
            "track_total_hits": False,
            "query": query,
            "_source": list(DEVICE_FIELDS),
            "sort": [{"event_time": "asc"}, {"device_no": "asc"}],
        },
    }


def _buckets(aggregation: dict[str, Any]) -> list[dict[str, Any]]:
    return aggregation.get("buckets", []) if isinstance(aggregation, dict) else []


def _extract_summary(response: dict[str, Any], *, plant_id: int, index: str) -> dict[str, Any]:
    hits = response.get("hits", {})
    total = hits.get("total", 0)
    if isinstance(total, dict):
        total = total.get("value", 0)
    aggregations = response.get("aggregations", {})

    def values(name: str) -> dict[str, int]:
        return {
            str(bucket.get("key")): int(bucket.get("doc_count", 0))
            for bucket in _buckets(aggregations.get(name, {}))
        }

    missing = {}
    missing_agg = aggregations.get("missing_fields", {})
    for name, bucket in (missing_agg.get("buckets", {}) or {}).items():
        missing[name.removeprefix("missing_")] = int(bucket.get("doc_count", 0))
    missing_currents = {}
    current_agg = aggregations.get("missing_string_currents", {})
    for name, bucket in (current_agg.get("buckets", {}) or {}).items():
        missing_currents[name.removeprefix("missing_")] = int(bucket.get("doc_count", 0))

    device_days = {}
    for device_bucket in _buckets(aggregations.get("device_days", {})):
        device = str(device_bucket.get("key"))
        device_days[device] = {
            str(day_bucket.get("key_as_string", day_bucket.get("key"))): int(
                day_bucket.get("doc_count", 0)
            )
            for day_bucket in _buckets(device_bucket.get("days", {}))
        }

    return {
        "index": index,
        "plant_id": plant_id,
        "document_count": int(total),
        "minimum": aggregations.get("minimum", {}).get("value_as_string"),
        "maximum": aggregations.get("maximum", {}).get("value_as_string"),
        "unique_devices_estimate": int(
            aggregations.get("devices", {}).get("value", 0)
        ),
        "status_code_counts": values("status_code"),
        "string_overall_status_counts": values("string_overall_status"),
        "main_string_count_counts": values("main_string_count"),
        "valid_current_string_count_counts": values("valid_current_string_count"),
        "daily_document_counts": {
            str(bucket.get("key_as_string", bucket.get("key"))): int(
                bucket.get("doc_count", 0)
            )
            for bucket in _buckets(aggregations.get("days", {}))
        },
        "daily_time_ranges": {
            str(bucket.get("key_as_string", bucket.get("key"))): {
                "first": bucket.get("first_event", {}).get("value_as_string"),
                "last": bucket.get("last_event", {}).get("value_as_string"),
            }
            for bucket in _buckets(aggregations.get("days", {}))
        },
        "per_device_daily_document_counts": device_days,
        "per_device_daily_time_ranges": {
            str(device_bucket.get("key")): {
                str(day_bucket.get("key_as_string", day_bucket.get("key"))): {
                    "first": day_bucket.get("first_event", {}).get("value_as_string"),
                    "last": day_bucket.get("last_event", {}).get("value_as_string"),
                }
                for day_bucket in _buckets(device_bucket.get("days", {}))
            }
            for device_bucket in _buckets(aggregations.get("device_days", {}))
        },
        "missing_required_fields": missing,
        "missing_string_current_fields": missing_currents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="pv_device_data")
    parser.add_argument("--plant-id", type=int, default=234)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--output",
        default="artifacts/reports/production_plant_234_audit.json",
    )
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 1000:
        raise SystemExit("--sample-size must be between 1 and 1000")

    client = client_from_env()
    bodies = build_audit_query(
        plant_id=args.plant_id,
        timezone=args.timezone,
        sample_size=args.sample_size,
    )
    summary_response = client.request(
        "GET",
        f"/{args.index}/_search",
        json_body=bodies["summary"],
    ).body
    sample_response = client.request(
        "GET",
        f"/{args.index}/_search",
        json_body=bodies["devices"],
    ).body
    sample_hits = sample_response.get("hits", {}).get("hits", [])
    report = {
        "summary": _extract_summary(
            summary_response,
            plant_id=args.plant_id,
            index=args.index,
        ),
        "sample_size": len(sample_hits),
        "sample": [hit.get("_source", {}) for hit in sample_hits],
        "read_only": True,
    }
    write_json(report, args.output)


if __name__ == "__main__":
    main()
