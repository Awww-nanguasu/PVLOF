"""FastAPI application factory for the read-only PVLOF-V3 service."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from pvlof_v3_service import __version__
from pvlof_v3_service.config import ServiceSettings
from pvlof_v3_service.calibration import (
    CALIBRATION_PIPELINE_VERSION,
    CalibrationInputError,
    CalibrationNotFoundError,
    CalibrationNotReadyError,
)
from pvlof_v3_service.connections import (
    DataSourceInputError,
    DataSourceNotFoundError,
)
from pvlof_v3_service.engine import (
    EvaluationBusyError,
    EvaluationEngine,
    EvaluationInputError,
    NoCurrentDataError,
    UnsupportedPlantError,
)
from pvlof_v3_service.es_client import ESReadError
from pvlof_v3_service.repository import DataContractError
from pvlof_v3_service.schemas import (
    CalibrationRequest,
    DataSourceConnectRequest,
    EvaluationRequest,
)
from pvlof_v3_service.workflow import SelfServiceWorkflow


def create_app(
    settings: ServiceSettings | None = None,
    engine: EvaluationEngine | None = None,
    workflow: SelfServiceWorkflow | None = None,
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    engine = engine or EvaluationEngine.from_settings(settings)
    if workflow is None and isinstance(engine, EvaluationEngine):
        workflow = SelfServiceWorkflow(settings, engine)

    app = FastAPI(
        title="PVLOF-V3 Service",
        version=__version__,
        description=(
            "Read-only Elasticsearch discovery, persistent per-plant calibration "
            "and daily independent date-range evaluation API for PVLOF-V3."
        ),
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.workflow = workflow

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def authorize(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        expected = settings.api_key
        if expected is None:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
            )

    protected = [Depends(authorize)]

    service_capabilities = {
        "default_data_source": True,
        "custom_elasticsearch": settings.allow_custom_data_sources,
        "plant_discovery": True,
        "persistent_self_calibration": True,
        "weather_optional": settings.weather_optional,
        "formal_event_output": True,
        "calendar_date_evaluation": True,
        "daily_independent_batches": True,
        "calibration_pipeline_version": CALIBRATION_PIPELINE_VERSION,
        "training_timestamp_sampling": True,
        "training_target_documents": (
            settings.calibration_target_documents
        ),
        "training_max_timestamps": settings.calibration_max_timestamps,
    }

    def require_workflow() -> SelfServiceWorkflow:
        selected = app.state.workflow
        if selected is None:
            raise HTTPException(
                status_code=503,
                detail="Self-service workflow is unavailable in this process",
            )
        return selected

    @app.get("/healthz", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service_version": __version__}

    @app.get("/readyz", tags=["operations"])
    def ready() -> dict[str, Any]:
        try:
            return {
                **engine.readiness(),
                "service_capabilities": service_capabilities,
            }
        except ESReadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(
        "/api/v1/version",
        tags=["pvlof"],
        dependencies=protected,
    )
    def version() -> dict[str, Any]:
        return {
            **engine.version_info(),
            "service_capabilities": service_capabilities,
        }

    @app.get(
        "/api/v1/plants",
        tags=["pvlof"],
        dependencies=protected,
    )
    def plants() -> dict[str, Any]:
        return {
            "plants": [
                {
                    "plant_id": plant_id,
                    "calibration_plant_id": engine.plant_mappings[plant_id],
                    "calibrated_device_count": count,
                }
                for plant_id, count in engine.supported_plants.items()
            ]
        }

    @app.post(
        "/api/v1/data-sources",
        tags=["data-sources"],
        dependencies=protected,
    )
    def connect_data_source(
        request: DataSourceConnectRequest,
    ) -> dict[str, Any]:
        try:
            return require_workflow().connect(request)
        except DataSourceInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ESReadError as exc:
            code = 401 if exc.status in {401, 403} else 502
            raise HTTPException(status_code=code, detail=str(exc)) from exc

    @app.delete(
        "/api/v1/data-sources/{data_source_id}",
        tags=["data-sources"],
        dependencies=protected,
    )
    def disconnect_data_source(data_source_id: str) -> dict[str, Any]:
        disconnected = require_workflow().disconnect(data_source_id)
        return {
            "data_source_id": data_source_id,
            "disconnected": disconnected,
        }

    @app.get(
        "/api/v1/data-sources/{data_source_id}/plants",
        tags=["data-sources"],
        dependencies=protected,
    )
    def discover_plants(data_source_id: str) -> dict[str, Any]:
        try:
            return require_workflow().plants(data_source_id)
        except DataSourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ESReadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get(
        "/api/v1/data-sources/{data_source_id}/plants/{plant_id}/range",
        tags=["data-sources"],
        dependencies=protected,
    )
    def discover_plant_range(
        data_source_id: str,
        plant_id: int,
    ) -> dict[str, Any]:
        try:
            return require_workflow().plant_range(data_source_id, plant_id)
        except DataSourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ESReadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post(
        "/api/v1/calibrations",
        tags=["calibrations"],
        dependencies=protected,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def start_calibration(request: CalibrationRequest) -> dict[str, Any]:
        try:
            return require_workflow().start_calibration(
                data_source_id=request.data_source_id,
                plant_id=int(request.plant_id),
                start=request.start_datetime,
                end=request.end_datetime,
            )
        except DataSourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CalibrationInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/api/v1/calibrations",
        tags=["calibrations"],
        dependencies=protected,
    )
    def list_calibrations(
        data_source_id: str,
        plant_id: int | None = None,
    ) -> dict[str, Any]:
        try:
            return require_workflow().list_calibrations(
                data_source_id=data_source_id,
                plant_id=plant_id,
            )
        except DataSourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/api/v1/calibrations/{calibration_id}",
        tags=["calibrations"],
        dependencies=protected,
    )
    def calibration_status(calibration_id: str) -> dict[str, Any]:
        try:
            return require_workflow().calibration(calibration_id)
        except CalibrationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/api/v1/evaluations",
        tags=["pvlof"],
        dependencies=protected,
    )
    def evaluate(request: EvaluationRequest) -> dict[str, Any]:
        try:
            if request.uses_persistent_calibration:
                assert request.data_source_id is not None
                assert request.calibration_id is not None
                return require_workflow().evaluate(
                    data_source_id=request.data_source_id,
                    calibration_id=request.calibration_id,
                    plant_id=int(request.plant_id),
                    start_date=request.first_date,
                    end_date=request.last_date,
                )
            return engine.evaluate(
                plant_id=int(request.plant_id),
                start_date=request.first_date,
                end_date=request.last_date,
            )
        except UnsupportedPlantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NoCurrentDataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EvaluationBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (DataSourceNotFoundError, CalibrationNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CalibrationNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (
            EvaluationInputError,
            CalibrationInputError,
            DataContractError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ESReadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
