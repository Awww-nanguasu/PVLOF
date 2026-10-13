"""Application service for database discovery, calibration and evaluation."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pvlof_v3_service.calibration import CalibrationManager
from pvlof_v3_service.config import ServiceSettings
from pvlof_v3_service.connections import DataSourceManager
from pvlof_v3_service.engine import EvaluationBusyError, EvaluationEngine
from pvlof_v3_service.es_client import ReadOnlyESClient
from pvlof_v3_service.repository import PVLOFESRepository
from pvlof_v3_service.schemas import DataSourceConnectRequest


class SelfServiceWorkflow:
    """Coordinate ephemeral credentials and persistent per-plant models."""

    def __init__(
        self,
        settings: ServiceSettings,
        template_engine: EvaluationEngine,
    ):
        self.settings = settings
        self.template_engine = template_engine
        self.data_sources = DataSourceManager(settings)
        self.calibrations = CalibrationManager(
            settings,
            template_engine.detector,
        )
        self._evaluation_slots = threading.BoundedSemaphore(
            settings.max_concurrent_evaluations
        )

    def _with_local_times(self, value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        zone = ZoneInfo(self.settings.timezone)
        for name in ("minimum_time", "maximum_time"):
            raw = result.get(name)
            local_name = f"{name}_local"
            if not raw:
                result[local_name] = None
                continue
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                # Elasticsearch date strings without an explicit offset are
                # UTC instants.  This matches repository parsing with
                # pandas.to_datetime(..., utc=True).
                parsed = parsed.replace(tzinfo=timezone.utc)
            result[local_name] = parsed.astimezone(zone).isoformat()
        return result

    def connect(self, request: DataSourceConnectRequest) -> dict[str, Any]:
        return self.data_sources.connect(request).safe_summary()

    def disconnect(self, data_source_id: str) -> bool:
        return self.data_sources.disconnect(data_source_id)

    def plants(self, data_source_id: str) -> dict[str, Any]:
        session = self.data_sources.get(data_source_id)
        repository = PVLOFESRepository(ReadOnlyESClient(session.es))
        plants = [
            self._with_local_times(plant)
            for plant in repository.discover_plants()
        ]
        return {
            "data_source_id": data_source_id,
            "timezone": self.settings.timezone,
            "plants": plants,
        }

    def plant_range(self, data_source_id: str, plant_id: int) -> dict[str, Any]:
        session = self.data_sources.get(data_source_id)
        repository = PVLOFESRepository(ReadOnlyESClient(session.es))
        result = repository.discover_plant_range(plant_id=plant_id)
        result["current"] = self._with_local_times(result["current"])
        result["weather"] = self._with_local_times(result["weather"])
        return {
            "data_source_id": data_source_id,
            "timezone": self.settings.timezone,
            **result,
        }

    def start_calibration(
        self,
        *,
        data_source_id: str,
        plant_id: int,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        session = self.data_sources.get(data_source_id)
        return self.calibrations.submit(
            session=session,
            plant_id=plant_id,
            start=start,
            end=end,
        )

    def calibration(self, calibration_id: str) -> dict[str, Any]:
        return self.calibrations.get_record(calibration_id)

    def list_calibrations(
        self,
        *,
        data_source_id: str,
        plant_id: int | None = None,
    ) -> dict[str, Any]:
        session = self.data_sources.get(data_source_id)
        return {
            "data_source_id": data_source_id,
            "calibrations": self.calibrations.list_records(
                session=session,
                plant_id=plant_id,
            ),
        }

    def evaluate(
        self,
        *,
        data_source_id: str,
        calibration_id: str,
        plant_id: int,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        if not self._evaluation_slots.acquire(blocking=False):
            raise EvaluationBusyError(
                "The PVLOF evaluator is busy; retry after the active request finishes"
            )
        try:
            session = self.data_sources.get(data_source_id)
            detector = self.calibrations.load_detector(
                calibration_id,
                session=session,
                plant_id=plant_id,
            )
            dynamic_settings = replace(
                self.settings,
                es=session.es,
                plant_aliases={},
            )
            repository = PVLOFESRepository(ReadOnlyESClient(session.es))
            engine = EvaluationEngine(
                dynamic_settings,
                repository,
                detector,
            )
            result = engine.evaluate(
                plant_id=plant_id,
                start_date=start_date,
                end_date=end_date,
            )
            result["metadata"]["self_service"] = {
                "data_source_id": data_source_id,
                "calibration_id": calibration_id,
                "calibration_persisted": True,
            }
            return result
        finally:
            self._evaluation_slots.release()


__all__ = ["SelfServiceWorkflow"]
