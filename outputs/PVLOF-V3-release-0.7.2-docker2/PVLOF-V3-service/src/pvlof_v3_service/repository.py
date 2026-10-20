"""Translate fixed Elasticsearch documents into PVLOF-V3 DataFrames."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

import pandas as pd

from pvlof_v3_service.es_client import ESReadError, ReadOnlyESClient


CURRENT_COLUMNS = [
    "plant_id",
    "device_no",
    "event_time",
    *(f"string_current_{number:02d}" for number in range(1, 31)),
]
CALIBRATION_AUXILIARY_COLUMNS = [
    "main_string_count",
    *(f"string_status_{number:02d}" for number in range(1, 31)),
]
WEATHER_COLUMNS = ["plant_id", "time", "forecast_ghi"]
MAX_EVENT_TIME_TERMS_PER_QUERY = 500


class DataContractError(ValueError):
    """Raised when Elasticsearch data violates the frozen input contract."""


def _iso(timestamp: datetime) -> str:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Elasticsearch query timestamps must be timezone-aware")
    return timestamp.isoformat()


def _report(
    frame: pd.DataFrame,
    *,
    time_column: str,
    requested_start: datetime,
    requested_end: datetime,
) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "minimum_time": (
            frame[time_column].min().isoformat() if not frame.empty else None
        ),
        "maximum_time": (
            frame[time_column].max().isoformat() if not frame.empty else None
        ),
    }


def _total_hits(response: dict[str, Any]) -> int:
    total = (response.get("hits") or {}).get("total", 0)
    if isinstance(total, dict):
        return int(total.get("value", 0))
    return int(total or 0)


def _metric_time(aggregations: dict[str, Any], name: str) -> str | None:
    metric = aggregations.get(name) or {}
    value = metric.get("value_as_string")
    return str(value) if value else None


class PVLOFESRepository:
    """Fixed-index, read-only data source for the production detector."""

    def __init__(self, client: ReadOnlyESClient):
        self.client = client

    @staticmethod
    def _collect_frame(pages: Any) -> pd.DataFrame:
        """Materialize each ES page immediately to avoid one huge dict list."""

        frames: list[pd.DataFrame] = []
        for page in pages:
            if page:
                frames.append(pd.DataFrame.from_records(page))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False, copy=False)

    def discover_current_time_buckets(
        self,
        *,
        plant_id: int,
        start: datetime,
        end: datetime,
        page_size: int = 5000,
    ) -> list[dict[str, Any]]:
        """Return exact current timestamps and document counts via composite agg."""

        size = min(max(int(page_size), 1), 5000)
        after: dict[str, Any] | None = None
        result: list[dict[str, Any]] = []
        while True:
            composite: dict[str, Any] = {
                "size": size,
                "sources": [
                    {"event_time": {"terms": {"field": "event_time"}}}
                ],
            }
            if after is not None:
                composite["after"] = after
            response = self.client.search(
                self.client.settings.device_index,
                {
                    "size": 0,
                    "track_total_hits": False,
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"plant_id": plant_id}},
                                {
                                    "range": {
                                        "event_time": {
                                            "gte": _iso(start),
                                            "lt": _iso(end),
                                        }
                                    }
                                },
                            ]
                        }
                    },
                    "aggs": {"timestamps": {"composite": composite}},
                },
            )
            aggregation = (
                (response.get("aggregations") or {}).get("timestamps") or {}
            )
            buckets = aggregation.get("buckets") or []
            for bucket in buckets:
                key = (bucket.get("key") or {}).get("event_time")
                if key is not None:
                    result.append(
                        {
                            "event_time": key,
                            "documents": int(bucket.get("doc_count", 0)),
                        }
                    )
            next_after = aggregation.get("after_key")
            if not buckets or not isinstance(next_after, dict):
                break
            if next_after == after:
                raise ESReadError(
                    "Composite timestamp aggregation did not advance"
                )
            after = next_after
        return result

    def plan_current_training_sample(
        self,
        *,
        plant_id: int,
        start: datetime,
        end: datetime,
        target_documents: int,
        maximum_timestamps: int,
    ) -> dict[str, Any]:
        """Plan a deterministic timestamp sample retaining all peer devices."""

        if target_documents <= 0 or maximum_timestamps <= 0:
            raise ValueError("training sample limits must be positive")
        buckets = self.discover_current_time_buckets(
            plant_id=plant_id,
            start=start,
            end=end,
        )
        available_documents = sum(item["documents"] for item in buckets)
        available_timestamps = len(buckets)
        maximum_documents_per_timestamp = max(
            (item["documents"] for item in buckets),
            default=0,
        )
        if not buckets or available_documents <= target_documents:
            return {
                "sampled": False,
                "available_documents": available_documents,
                "available_timestamps": available_timestamps,
                "selected_documents": available_documents,
                "selected_timestamps": available_timestamps,
                "maximum_documents_per_timestamp": (
                    maximum_documents_per_timestamp
                ),
                "sample_fraction": 1.0 if available_documents else 0.0,
                "timestamp_values": None,
            }

        timestamp_limit = max(
            1,
            target_documents // max(maximum_documents_per_timestamp, 1),
        )
        timestamp_limit = min(
            timestamp_limit,
            maximum_timestamps,
            available_timestamps,
        )
        if timestamp_limit == 1:
            positions = [available_timestamps // 2]
        else:
            positions = [
                index * (available_timestamps - 1) // (timestamp_limit - 1)
                for index in range(timestamp_limit)
            ]
        positions = sorted(set(positions))
        selected = [buckets[index] for index in positions]
        selected_documents = sum(item["documents"] for item in selected)
        return {
            "sampled": True,
            "available_documents": available_documents,
            "available_timestamps": available_timestamps,
            "selected_documents": selected_documents,
            "selected_timestamps": len(selected),
            "maximum_documents_per_timestamp": (
                maximum_documents_per_timestamp
            ),
            "sample_fraction": (
                selected_documents / available_documents
                if available_documents
                else 0.0
            ),
            "timestamp_values": [item["event_time"] for item in selected],
        }

    def discover_plants(self, *, maximum_plants: int = 10_000) -> list[dict[str, Any]]:
        """List plants that have current documents in the configured index."""

        response = self.client.search(
            self.client.settings.device_index,
            {
                "size": 0,
                "track_total_hits": False,
                "aggs": {
                    "plants": {
                        "terms": {
                            "field": "plant_id",
                            "size": maximum_plants,
                            "order": {"_key": "asc"},
                        },
                        "aggs": {
                            "devices": {
                                "cardinality": {"field": "device_no"}
                            },
                            "minimum_time": {"min": {"field": "event_time"}},
                            "maximum_time": {"max": {"field": "event_time"}},
                        },
                    }
                },
            },
        )
        buckets = (
            (response.get("aggregations") or {})
            .get("plants", {})
            .get("buckets", [])
        )
        plants: list[dict[str, Any]] = []
        for bucket in buckets:
            plants.append(
                {
                    "plant_id": str(bucket.get("key_as_string", bucket.get("key"))),
                    "documents": int(bucket.get("doc_count", 0)),
                    "device_count": int((bucket.get("devices") or {}).get("value", 0)),
                    "minimum_time": (bucket.get("minimum_time") or {}).get(
                        "value_as_string"
                    ),
                    "maximum_time": (bucket.get("maximum_time") or {}).get(
                        "value_as_string"
                    ),
                }
            )
        return plants

    def discover_devices(
        self,
        *,
        plant_id: int,
        start: datetime | None = None,
        end: datetime | None = None,
        maximum_devices: int = 10_000,
    ) -> list[str]:
        filters: list[dict[str, Any]] = [{"term": {"plant_id": plant_id}}]
        if start is not None or end is not None:
            bounds: dict[str, str] = {}
            if start is not None:
                bounds["gte"] = _iso(start)
            if end is not None:
                bounds["lt"] = _iso(end)
            filters.append({"range": {"event_time": bounds}})
        response = self.client.search(
            self.client.settings.device_index,
            {
                "size": 0,
                "track_total_hits": False,
                "query": {"bool": {"filter": filters}},
                "aggs": {
                    "devices": {
                        "terms": {
                            "field": "device_no",
                            "size": maximum_devices,
                            "order": {"_key": "asc"},
                        }
                    }
                },
            },
        )
        buckets = (
            (response.get("aggregations") or {})
            .get("devices", {})
            .get("buckets", [])
        )
        return sorted(str(bucket["key"]) for bucket in buckets)

    def discover_plant_range(self, *, plant_id: int) -> dict[str, Any]:
        """Return current and optional weather coverage for one plant."""

        def coverage(index: str, time_field: str) -> dict[str, Any]:
            response = self.client.search(
                index,
                {
                    "size": 0,
                    "track_total_hits": True,
                    "query": {"term": {"plant_id": plant_id}},
                    "aggs": {
                        "minimum_time": {"min": {"field": time_field}},
                        "maximum_time": {"max": {"field": time_field}},
                    },
                },
            )
            aggregations = response.get("aggregations") or {}
            return {
                "available": _total_hits(response) > 0,
                "documents": _total_hits(response),
                "minimum_time": _metric_time(aggregations, "minimum_time"),
                "maximum_time": _metric_time(aggregations, "maximum_time"),
            }

        current = coverage(self.client.settings.device_index, "event_time")
        current["devices"] = len(self.discover_devices(plant_id=plant_id))
        try:
            weather = coverage(self.client.settings.weather_index, "time")
        except ESReadError as exc:
            # Weather is optional for PVLOF-V3.  The caller receives an
            # explicit unavailable state while current discovery still works.
            weather = {
                "available": False,
                "documents": 0,
                "minimum_time": None,
                "maximum_time": None,
                "error": str(exc)[:500],
            }
        return {
            "plant_id": str(plant_id),
            "current": current,
            "weather": weather,
        }

    def fetch_currents(
        self,
        *,
        plant_id: int,
        device_numbers: list[str],
        start: datetime,
        end: datetime,
        include_calibration_fields: bool = False,
        max_documents: int | None = None,
        event_time_values: list[Any] | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not device_numbers:
            raise ValueError("device_numbers must not be empty")
        base_filters: list[dict[str, Any]] = [
            {"range": {"event_time": {"gte": _iso(start), "lt": _iso(end)}}},
            {"term": {"plant_id": plant_id}},
            {"terms": {"device_no": sorted(set(device_numbers))}},
        ]
        selected_event_times: list[Any] | None = None
        if event_time_values is not None:
            if not event_time_values:
                raise ValueError("event_time_values must not be empty")
            # Preserve the deterministic sample order while preventing the
            # same timestamp from being fetched by more than one batch.
            selected_event_times = list(dict.fromkeys(event_time_values))
        event_time_batches: list[list[Any] | None]
        if selected_event_times is None:
            event_time_batches = [None]
        else:
            event_time_batches = [
                selected_event_times[offset : offset + MAX_EVENT_TIME_TERMS_PER_QUERY]
                for offset in range(
                    0,
                    len(selected_event_times),
                    MAX_EVENT_TIME_TERMS_PER_QUERY,
                )
            ]

        selected_columns = list(CURRENT_COLUMNS)
        if include_calibration_fields:
            selected_columns.extend(CALIBRATION_AUXILIARY_COLUMNS)
        document_limit = int(
            self.client.settings.max_documents
            if max_documents is None
            else max_documents
        )
        if document_limit <= 0:
            raise ValueError("max_documents must be positive")

        def current_pages() -> Iterator[list[dict[str, Any]]]:
            total_documents = 0
            for event_time_batch in event_time_batches:
                remaining_documents = document_limit - total_documents
                if remaining_documents <= 0:
                    raise ESReadError(
                        "Combined Elasticsearch current queries exceeded "
                        f"max_documents={document_limit}"
                    )
                filters = list(base_filters)
                if event_time_batch is not None:
                    filters.append(
                        {"terms": {"event_time": event_time_batch}}
                    )
                query = {"bool": {"filter": filters}}
                for page in self.client.iter_search(
                    self.client.settings.device_index,
                    query=query,
                    source_fields=selected_columns,
                    sort_fields=["event_time", "device_no"],
                    max_documents=remaining_documents,
                ):
                    total_documents += len(page)
                    if total_documents > document_limit:
                        raise ESReadError(
                            "Combined Elasticsearch current queries exceeded "
                            f"max_documents={document_limit}"
                        )
                    if progress_callback is not None:
                        progress_callback(total_documents)
                    yield page

        frame = self._collect_frame(current_pages())
        for column in selected_columns:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame[selected_columns]
        if frame.empty:
            empty_report = _report(
                frame,
                time_column="event_time",
                requested_start=start,
                requested_end=end,
            )
            empty_report["sampled_event_time_count"] = (
                len(selected_event_times)
                if selected_event_times is not None
                else None
            )
            empty_report["event_time_query_batches"] = (
                len(event_time_batches)
                if selected_event_times is not None
                else None
            )
            return frame, empty_report

        missing_keys = frame[["plant_id", "device_no", "event_time"]].isna().any(axis=1)
        if missing_keys.any():
            raise DataContractError(
                f"Device index returned {int(missing_keys.sum())} rows with missing keys"
            )
        frame["plant_id"] = frame["plant_id"].astype(str).str.strip()
        frame["device_no"] = frame["device_no"].astype(str).str.strip()
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
        if frame["event_time"].isna().any():
            raise DataContractError("Device index contains invalid event_time values")
        for column in selected_columns[3:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        keys = ["plant_id", "device_no", "event_time"]
        duplicates = frame.duplicated(keys, keep=False)
        if duplicates.any():
            examples = frame.loc[duplicates, keys].head(5).to_dict("records")
            raise DataContractError(
                "Device index contains duplicate plant/device/time rows; "
                f"examples={examples}"
            )
        frame = frame.sort_values(keys).reset_index(drop=True)
        report = _report(
            frame,
            time_column="event_time",
            requested_start=start,
            requested_end=end,
        )
        report["devices"] = int(frame["device_no"].nunique())
        report["expected_devices"] = len(set(device_numbers))
        report["missing_expected_devices"] = sorted(
            set(device_numbers) - set(frame["device_no"].unique())
        )
        report["calibration_fields_included"] = include_calibration_fields
        report["sampled_event_time_count"] = (
            len(selected_event_times)
            if selected_event_times is not None
            else None
        )
        report["event_time_query_batches"] = (
            len(event_time_batches)
            if selected_event_times is not None
            else None
        )
        return frame, report

    def fetch_weather(
        self,
        *,
        plant_id: int,
        start: datetime,
        end: datetime,
        max_documents: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        query = {
            "bool": {
                "filter": [
                    {"range": {"time": {"gte": _iso(start), "lt": _iso(end)}}},
                    {"term": {"plant_id": plant_id}},
                ]
            }
        }
        frame = self._collect_frame(
            self.client.iter_search(
                self.client.settings.weather_index,
                query=query,
                source_fields=WEATHER_COLUMNS,
                sort_fields=["time"],
                max_documents=max_documents,
            )
        )
        for column in WEATHER_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
        frame = frame[WEATHER_COLUMNS]
        if frame.empty:
            return frame, _report(
                frame,
                time_column="time",
                requested_start=start,
                requested_end=end,
            )

        missing_keys = frame[["plant_id", "time"]].isna().any(axis=1)
        if missing_keys.any():
            raise DataContractError(
                f"Weather index returned {int(missing_keys.sum())} rows with missing keys"
            )
        frame["plant_id"] = frame["plant_id"].astype(str).str.strip()
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        if frame["time"].isna().any():
            raise DataContractError("Weather index contains invalid time values")
        frame["forecast_ghi"] = pd.to_numeric(frame["forecast_ghi"], errors="coerce")
        keys = ["plant_id", "time"]
        duplicates = frame.duplicated(keys, keep=False)
        if duplicates.any():
            examples = frame.loc[duplicates, keys].head(5).to_dict("records")
            raise DataContractError(
                "Weather index contains duplicate plant/time rows; "
                f"examples={examples}"
            )
        frame = frame.sort_values(keys).reset_index(drop=True)
        report = _report(
            frame,
            time_column="time",
            requested_start=start,
            requested_end=end,
        )
        report["forecast_ghi_non_null_rows"] = int(frame["forecast_ghi"].notna().sum())
        return frame, report
