"""Daily independent batches joining Elasticsearch data to frozen PVLOF-V3."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from pv_anomaly.pvlof_v3 import (
    PVLOFV3Detector,
    PVLOF_V3_ALGORITHM_VARIANT,
    PVLOF_V3_VERSION,
)

from pvlof_v3_service import __version__ as service_version
from pvlof_v3_service.config import ServiceSettings
from pvlof_v3_service.es_client import ESReadError, ReadOnlyESClient
from pvlof_v3_service.repository import CURRENT_COLUMNS, PVLOFESRepository
from pvlof_v3_service.serialization import dataframe_records, json_value


class EvaluationInputError(ValueError):
    """Raised when an API evaluation request is outside the service contract."""


class UnsupportedPlantError(EvaluationInputError):
    """Raised when the frozen calibration has no devices for a plant."""


class NoCurrentDataError(EvaluationInputError):
    """Raised when ES has no current documents for the reconstruction window."""


class EvaluationBusyError(RuntimeError):
    """Raised when the configured evaluation concurrency limit is reached."""


STRING_CURRENT_COLUMNS = CURRENT_COLUMNS[3:]
ALERT_EVENT_DETAIL_COLUMNS = [
    "event_id",
    "plant_id",
    "device_no",
    "event_alert_strings",
    "event_time",
    "event_time_local",
    "point_phase",
    "candidate_strings",
    "alert_strings",
    "unconfirmed_candidate_strings",
    "recovery_pending_strings",
    "member_cumulative_anomaly_minutes",
    "candidate_cumulative_anomaly_minutes",
    "max_member_cumulative_anomaly_minutes",
    *STRING_CURRENT_COLUMNS,
]


def _empty_point_values() -> dict[str, Any]:
    return {
        "candidate_strings": "",
        "alert_strings": "",
        "unconfirmed_candidate_strings": "",
        "recovery_pending_strings": "",
        "member_cumulative_anomaly_minutes": "{}",
        "candidate_cumulative_anomaly_minutes": "{}",
        "max_member_cumulative_anomaly_minutes": 0,
    }


def _build_alert_event_details(
    currents: pd.DataFrame,
    candidate_points: pd.DataFrame,
    events: pd.DataFrame,
    *,
    timezone_name: str,
    interval_minutes: int,
    entry_consecutive: int,
) -> pd.DataFrame:
    """Build event-scoped review rows without changing detector decisions.

    Each formal member-segment receives the two preceding evidence slots used
    by the three-point confirmation rule, plus every formal point in the
    segment. Device-time rows may therefore appear under more than one event
    when formal member segments overlap or change membership.
    """

    if currents.empty or events.empty:
        return pd.DataFrame(columns=ALERT_EVENT_DETAIL_COLUMNS)

    source = currents.copy()
    source["plant_id"] = source["plant_id"].astype(str)
    source["device_no"] = source["device_no"].astype(str)
    source["event_time"] = pd.to_datetime(
        source["event_time"], utc=True, errors="raise"
    )

    point_lookup: dict[tuple[str, str, pd.Timestamp], dict[str, Any]] = {}
    if not candidate_points.empty:
        point_source = candidate_points.copy()
        point_source["plant_id"] = point_source["plant_id"].astype(str)
        point_source["device_no"] = point_source["device_no"].astype(str)
        point_source["event_time"] = pd.to_datetime(
            point_source["event_time"], utc=True, errors="raise"
        )
        for record in point_source.to_dict(orient="records"):
            key = (
                str(record["plant_id"]),
                str(record["device_no"]),
                pd.Timestamp(record["event_time"]),
            )
            point_lookup[key] = record

    device_currents = {
        (str(plant_id), str(device_no)): group.sort_values("event_time")
        for (plant_id, device_no), group in source.groupby(
            ["plant_id", "device_no"], observed=True, sort=False
        )
    }
    lookback = pd.Timedelta(
        minutes=max(entry_consecutive - 1, 0) * interval_minutes
    )
    zone = ZoneInfo(timezone_name)
    rows: list[dict[str, Any]] = []
    for event in events.to_dict(orient="records"):
        plant_id = str(event.get("plant_id", ""))
        device_no = str(event.get("device_no", ""))
        device = device_currents.get((plant_id, device_no))
        if device is None or device.empty:
            continue
        raise_time = pd.to_datetime(
            event.get("raise_time"), utc=True, errors="raise"
        )
        end_time = pd.to_datetime(
            event.get("end_time"), utc=True, errors="raise"
        )
        review_start = raise_time - lookback
        selected = device[
            device["event_time"].ge(review_start)
            & device["event_time"].le(end_time)
        ]
        for current in selected.to_dict(orient="records"):
            event_time = pd.Timestamp(current["event_time"])
            point = point_lookup.get(
                (plant_id, device_no, event_time),
                _empty_point_values(),
            )
            if event_time >= raise_time:
                phase = "formal_alert"
            elif str(point.get("candidate_strings") or "").strip():
                phase = "candidate"
            elif str(point.get("recovery_pending_strings") or "").strip():
                phase = "recovery_pending"
            else:
                phase = "pre_alert_observation"
            row = {
                "event_id": event.get("event_id", ""),
                "plant_id": plant_id,
                "device_no": device_no,
                "event_alert_strings": event.get("alert_strings", ""),
                "event_time": event_time,
                "event_time_local": event_time.tz_convert(zone).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "point_phase": phase,
                "candidate_strings": point.get("candidate_strings", ""),
                "alert_strings": point.get("alert_strings", ""),
                "unconfirmed_candidate_strings": point.get(
                    "unconfirmed_candidate_strings", ""
                ),
                "recovery_pending_strings": point.get(
                    "recovery_pending_strings", ""
                ),
                "member_cumulative_anomaly_minutes": point.get(
                    "member_cumulative_anomaly_minutes", "{}"
                ),
                "candidate_cumulative_anomaly_minutes": point.get(
                    "candidate_cumulative_anomaly_minutes", "{}"
                ),
                "max_member_cumulative_anomaly_minutes": point.get(
                    "max_member_cumulative_anomaly_minutes", 0
                ),
            }
            for column in STRING_CURRENT_COLUMNS:
                value = current.get(column)
                row[column] = None if pd.isna(value) else float(value)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=ALERT_EVENT_DETAIL_COLUMNS)
    return pd.DataFrame(rows, columns=ALERT_EVENT_DETAIL_COLUMNS).sort_values(
        ["event_time", "plant_id", "device_no", "event_id"]
    ).reset_index(drop=True)


class EvaluationEngine:
    """Load calibrated devices, query all peers, execute V3, and filter outputs."""

    def __init__(
        self,
        settings: ServiceSettings,
        repository: PVLOFESRepository,
        detector: PVLOFV3Detector,
    ):
        self.settings = settings
        self.repository = repository
        self.detector = detector
        self._slots = threading.BoundedSemaphore(
            settings.max_concurrent_evaluations
        )
        devices: dict[str, list[str]] = {}
        for key in detector.supported_device_keys:
            if "|" not in key:
                continue
            plant_id, device_no = key.split("|", 1)
            devices.setdefault(plant_id, []).append(device_no)
        self._calibrated_devices_by_plant = {
            plant_id: sorted(set(values))
            for plant_id, values in devices.items()
        }

        aliases = {
            str(source): str(calibration)
            for source, calibration in settings.plant_aliases.items()
        }
        unknown_targets = sorted(
            set(aliases.values()) - set(self._calibrated_devices_by_plant)
        )
        if unknown_targets:
            raise ValueError(
                "PVLOF_PLANT_ALIASES references calibration plants absent "
                f"from the frozen model: {','.join(unknown_targets)}"
            )

        if aliases:
            # A configured mapping is also the production allow-list. This
            # prevents calibrated plants absent from the current ES environment
            # from being advertised as queryable API plants.
            source_to_calibration = aliases
        else:
            source_to_calibration = {
                plant_id: plant_id
                for plant_id in self._calibrated_devices_by_plant
            }
        self._source_to_calibration = dict(sorted(source_to_calibration.items()))

    @classmethod
    def from_settings(cls, settings: ServiceSettings) -> "EvaluationEngine":
        client = ReadOnlyESClient(settings.es)
        repository = PVLOFESRepository(client)
        detector = PVLOFV3Detector.from_root(settings.model_root)
        return cls(settings, repository, detector)

    @property
    def supported_plants(self) -> dict[str, int]:
        return {
            source_plant_id: len(
                self._calibrated_devices_by_plant[calibration_plant_id]
            )
            for source_plant_id, calibration_plant_id in (
                self._source_to_calibration.items()
            )
        }

    @property
    def plant_mappings(self) -> dict[str, str]:
        """Return public/source IDs mapped to frozen calibration IDs."""

        return dict(self._source_to_calibration)

    def version_info(self) -> dict[str, Any]:
        config = self.detector.detector_config
        return {
            "service_version": service_version,
            "algorithm_version": PVLOF_V3_VERSION,
            "internal_detector_version": config.version,
            "algorithm_variant": PVLOF_V3_ALGORITHM_VARIANT,
            "supported_plants": self.supported_plants,
            "plant_mappings": self.plant_mappings,
            "interval_minutes": config.expected_interval_minutes,
            "entry_consecutive": config.entry_consecutive,
            "recovery_consecutive": config.recovery_consecutive,
            "execution_mode": "daily_independent_batches",
            "customer_outputs": ["final_alert_points", "confirmed_events"],
            "review_outputs": ["alert_event_details"],
        }

    def readiness(self) -> dict[str, Any]:
        info = self.repository.client.info()
        return {
            "status": "ready",
            **self.version_info(),
            "elasticsearch": {
                "cluster_name": info.get("cluster_name"),
                "version": (info.get("version") or {}).get("number"),
                "device_index": self.settings.es.device_index,
                "weather_index": self.settings.es.weather_index,
            },
        }

    @staticmethod
    def _validate_date_range(start_date: date, end_date: date) -> None:
        if end_date < start_date:
            raise EvaluationInputError("end_date must be on or after start_date")

    def _day_bounds(
        self, local_date: date
    ) -> tuple[datetime, datetime, datetime, datetime]:
        zone = ZoneInfo(self.settings.timezone)
        start_local = datetime.combine(
            local_date,
            datetime.min.time(),
            tzinfo=zone,
        )
        end_local = datetime.combine(
            local_date + timedelta(days=1),
            datetime.min.time(),
            tzinfo=zone,
        )
        return (
            start_local,
            end_local,
            start_local.astimezone(timezone.utc),
            end_local.astimezone(timezone.utc),
        )

    def evaluate(
        self,
        *,
        plant_id: int,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        if not self._slots.acquire(blocking=False):
            raise EvaluationBusyError(
                "The PVLOF evaluator is busy; retry after the active request finishes"
            )
        started = time.perf_counter()
        try:
            return self._evaluate(
                plant_id=plant_id,
                start_date=start_date,
                end_date=end_date,
                started=started,
            )
        finally:
            self._slots.release()

    def _evaluate(
        self,
        *,
        plant_id: int,
        start_date: date,
        end_date: date,
        started: float,
    ) -> dict[str, Any]:
        self._validate_date_range(start_date, end_date)
        source_plant_key = str(plant_id)
        calibration_plant_key = self._source_to_calibration.get(
            source_plant_key
        )
        if calibration_plant_key is None:
            raise UnsupportedPlantError(
                f"plant_id={plant_id} is not configured; supported plants="
                + ",".join(self._source_to_calibration)
            )
        devices = self._calibrated_devices_by_plant[calibration_plant_key]
        day_count = (end_date - start_date).days + 1
        point_frames: list[pd.DataFrame] = []
        event_frames: list[pd.DataFrame] = []
        detail_frames: list[pd.DataFrame] = []
        daily_batches: list[dict[str, Any]] = []
        warnings: list[str] = []
        completed_days = 0
        total_current_rows = 0
        total_weather_rows = 0

        for offset in range(day_count):
            local_date = start_date + timedelta(days=offset)
            batch_started = time.perf_counter()
            start_local, end_local, start_utc, end_utc = self._day_bounds(
                local_date
            )
            currents, current_report = self.repository.fetch_currents(
                plant_id=plant_id,
                device_numbers=devices,
                start=start_utc,
                end=end_utc,
            )
            current_rows = int(current_report.get("rows", len(currents)))
            total_current_rows += current_rows
            if currents.empty:
                message = (
                    f"{local_date.isoformat()}: no calibrated-device current "
                    "data; daily batch skipped"
                )
                warnings.append(message)
                daily_batches.append(
                    {
                        "date": local_date.isoformat(),
                        "start_time_local": start_local.isoformat(),
                        "end_time_local_exclusive": end_local.isoformat(),
                        "status": "no_current_data",
                        "currents": current_report,
                        "weather": {"status": "not_queried", "rows": 0},
                        "outputs": {
                            "final_alert_points": 0,
                            "confirmed_events": 0,
                            "alert_event_details": 0,
                        },
                        "elapsed_ms": round(
                            (time.perf_counter() - batch_started) * 1000,
                            3,
                        ),
                    }
                )
                continue

            detector_currents = currents.copy()
            detector_currents["plant_id"] = calibration_plant_key
            missing_devices = current_report.get("missing_expected_devices", [])
            if missing_devices:
                warnings.append(
                    f"{local_date.isoformat()}: no current documents for "
                    "calibrated devices: " + ",".join(missing_devices)
                )

            margin = timedelta(minutes=self.settings.weather_margin_minutes)
            weather: pd.DataFrame | None = None
            weather_report: dict[str, Any]
            try:
                weather_frame, weather_report = self.repository.fetch_weather(
                    plant_id=plant_id,
                    start=start_utc - margin,
                    end=end_utc + margin,
                )
                weather_rows = int(weather_report.get("rows", len(weather_frame)))
                total_weather_rows += weather_rows
                non_null = int(weather_frame["forecast_ghi"].notna().sum())
                if weather_frame.empty or non_null == 0:
                    weather_report["status"] = "missing_fallback_to_peer"
                    warnings.append(
                        f"{local_date.isoformat()}: forecast_ghi unavailable; "
                        "detector used peer virtual irradiance"
                    )
                else:
                    weather = weather_frame.copy()
                    weather["plant_id"] = calibration_plant_key
                    weather_report["status"] = "available"
            except ESReadError as exc:
                if not self.settings.weather_optional:
                    raise
                weather_report = {
                    "status": "query_failed_fallback_to_peer",
                    "rows": 0,
                    "error": str(exc)[:500],
                }
                warnings.append(
                    f"{local_date.isoformat()}: weather query failed; "
                    "detector used peer virtual irradiance"
                )

            result = self.detector.run(
                detector_currents,
                weather=weather,
                timezone=self.settings.timezone,
                strict_calibration=True,
            )

            points = result.final_alert_points.copy()
            if "plant_id" in points:
                points["plant_id"] = source_plant_key
            if not points.empty:
                point_times = pd.to_datetime(
                    points["event_time"], utc=True, errors="raise"
                )
                points = points[
                    point_times.ge(start_utc) & point_times.lt(end_utc)
                ].copy()
                point_frames.append(points)

            events = result.confirmed_events.copy()
            if "plant_id" in events:
                events["plant_id"] = source_plant_key
            if not events.empty:
                raises = pd.to_datetime(
                    events["raise_time"], utc=True, errors="raise"
                )
                ends = pd.to_datetime(
                    events["end_time"], utc=True, errors="raise"
                )
                events = events[
                    ends.ge(start_utc) & raises.lt(end_utc)
                ].copy()
                event_frames.append(events)

            candidate_points = result.raw_candidate_points.copy()
            if "plant_id" in candidate_points:
                candidate_points["plant_id"] = source_plant_key
            if not candidate_points.empty:
                candidate_times = pd.to_datetime(
                    candidate_points["event_time"], utc=True, errors="raise"
                )
                candidate_points = candidate_points[
                    candidate_times.ge(start_utc)
                    & candidate_times.lt(end_utc)
                ].copy()
            details = _build_alert_event_details(
                currents,
                candidate_points,
                events,
                timezone_name=self.settings.timezone,
                interval_minutes=(
                    self.detector.detector_config.expected_interval_minutes
                ),
                entry_consecutive=(
                    self.detector.detector_config.entry_consecutive
                ),
            )
            if not details.empty:
                detail_frames.append(details)

            completed_days += 1
            daily_batches.append(
                {
                    "date": local_date.isoformat(),
                    "start_time_local": start_local.isoformat(),
                    "end_time_local_exclusive": end_local.isoformat(),
                    "status": "completed",
                    "currents": current_report,
                    "weather": weather_report,
                    "outputs": {
                        "final_alert_points": int(len(points)),
                        "confirmed_events": int(len(events)),
                        "alert_event_details": int(len(details)),
                    },
                    "elapsed_ms": round(
                        (time.perf_counter() - batch_started) * 1000,
                        3,
                    ),
                }
            )

        if completed_days == 0:
            raise NoCurrentDataError(
                "No calibrated-device current data was found for source "
                f"plant_id={plant_id} between {start_date} and {end_date}"
            )

        points = (
            pd.concat(point_frames, ignore_index=True)
            if point_frames
            else pd.DataFrame()
        )
        events = (
            pd.concat(event_frames, ignore_index=True)
            if event_frames
            else pd.DataFrame()
        )
        details = (
            pd.concat(detail_frames, ignore_index=True)
            if detail_frames
            else pd.DataFrame(columns=ALERT_EVENT_DETAIL_COLUMNS)
        )
        if not points.empty:
            points = points.sort_values(
                [
                    column
                    for column in (
                        "event_time",
                        "plant_id",
                        "device_no",
                    )
                    if column in points
                ]
            ).reset_index(drop=True)
        if not events.empty:
            events = events.sort_values(
                [
                    column
                    for column in (
                        "raise_time",
                        "plant_id",
                        "device_no",
                    )
                    if column in events
                ]
            ).reset_index(drop=True)
        if not details.empty:
            details = details.sort_values(
                ["event_time", "plant_id", "device_no", "event_id"]
            ).reset_index(drop=True)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        range_start_local, _, range_start_utc, _ = self._day_bounds(start_date)
        _, range_end_local, _, range_end_utc = self._day_bounds(end_date)
        metadata = {
            "request": {
                "plant_id": plant_id,
                "source_plant_id": int(source_plant_key),
                "calibration_plant_id": int(calibration_plant_key),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "start_time_local": range_start_local.isoformat(),
                "end_time_local_exclusive": range_end_local.isoformat(),
                "start_time_utc": range_start_utc.isoformat(),
                "end_time_utc_exclusive": range_end_utc.isoformat(),
            },
            "execution": {
                "mode": "daily_independent_batches",
                "timezone": self.settings.timezone,
                "requested_days": day_count,
                "completed_days": completed_days,
                "history_lookback_hours": 0,
                "state_initialization": "reset_at_each_local_midnight",
            },
            "algorithm": self.version_info(),
            "data": {
                "current_rows": total_current_rows,
                "weather_rows": total_weather_rows,
                "daily_batches": daily_batches,
            },
            "outputs": {
                "final_alert_points": int(len(points)),
                "confirmed_events": int(len(events)),
                "alert_event_details": int(len(details)),
            },
            "warnings": warnings,
            "elapsed_ms": elapsed_ms,
        }
        return {
            "metadata": json_value(metadata),
            "final_alert_points": dataframe_records(points),
            "confirmed_events": dataframe_records(events),
            "alert_event_details": dataframe_records(details),
        }
