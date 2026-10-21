"""Persistent, asynchronous self-service calibration for PVLOF-V3."""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from pv_anomaly.pvlof_v12 import (
    build_conditioned_virtual_context,
    fit_weather_calibration,
    save_weather_calibration,
)
from pv_anomaly.pvlof_v16 import save_memory_config
from pv_anomaly.pvlof_v17_improved import save_config as save_detector_config
from pv_anomaly.pvlof_v2 import (
    apply_pvlof_v2,
    fit_pvlof_v2_calibration,
    save_calibration,
)
from pv_anomaly.pvlof_v2_hier import (
    fit_hierarchical_calibration,
    save_hier_calibration,
)
from pv_anomaly.pvlof_v3 import (
    PVLOFV3Detector,
    PVLOFV3ModelPaths,
    PVLOF_V3_VERSION,
)

from pvlof_v3_service.config import ServiceSettings
from pvlof_v3_service.channel_inventory import (
    CHANNEL_INVENTORY_VERSION,
    infer_channel_inventory,
)
from pvlof_v3_service.connections import DataSourceSession
from pvlof_v3_service.es_client import ESReadError, ReadOnlyESClient
from pvlof_v3_service.repository import PVLOFESRepository
from pvlof_v3_service.serialization import json_value


class CalibrationInputError(ValueError):
    """Raised when a requested training range cannot produce a valid model."""


class CalibrationNotFoundError(KeyError):
    """Raised when a persistent calibration identifier is unknown."""


class CalibrationNotReadyError(RuntimeError):
    """Raised when evaluation is attempted before calibration completes."""


CALIBRATION_PIPELINE_VERSION = (
    "pvlof-v3-self-calibration-v3-stratified-timestamp-sampling"
)
COMPATIBLE_CALIBRATION_PIPELINE_VERSIONS = frozenset(
    {
        CALIBRATION_PIPELINE_VERSION,
        "pvlof-v3-self-calibration-v2-relative-current-inventory",
    }
)


def _utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CalibrationInputError(f"{name} must include a UTC offset")
    return value.astimezone(timezone.utc)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample_training_rows(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    """Select deterministic rows spanning the full requested training range."""

    if len(frame) <= maximum_rows:
        return frame.copy()
    positions = np.linspace(0, len(frame) - 1, maximum_rows, dtype=np.int64)
    return frame.iloc[np.unique(positions)].reset_index(drop=True)


class CalibrationManager:
    """Run one-off model fitting jobs and persist reusable artifacts."""

    def __init__(
        self,
        settings: ServiceSettings,
        template_detector: PVLOFV3Detector,
    ):
        self.settings = settings
        self.template_detector = template_detector
        self.root = settings.runtime_root / "calibrations"
        self.root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_calibrations,
            thread_name_prefix="pvlof-calibration",
        )
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._detectors: dict[str, PVLOFV3Detector] = {}
        self._futures: dict[str, Future[None]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for metadata_path in sorted(self.root.glob("*/metadata.json")):
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            calibration_id = str(record.get("calibration_id", ""))
            if calibration_id:
                if record.get("status") in {"queued", "running"}:
                    record.update(
                        {
                            "status": "failed",
                            "stage": "interrupted_by_service_restart",
                            "error_type": "InterruptedCalibration",
                            "error": (
                                "Calibration did not finish before the service "
                                "stopped; submit it again"
                            ),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    _atomic_json(metadata_path, record)
                self._records[calibration_id] = record

    def _metadata_path(self, calibration_id: str) -> Path:
        return self.root / calibration_id / "metadata.json"

    def _model_root(self, calibration_id: str) -> Path:
        return self.root / calibration_id / "model"

    def _save_record(self, record: dict[str, Any]) -> None:
        calibration_id = str(record["calibration_id"])
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._records[calibration_id] = dict(record)
            _atomic_json(self._metadata_path(calibration_id), record)

    def _update(self, calibration_id: str, **changes: Any) -> None:
        with self._lock:
            record = dict(self._records[calibration_id])
        record.update(changes)
        self._save_record(record)

    @staticmethod
    def _same_source(
        stored: dict[str, Any], session: DataSourceSession
    ) -> bool:
        current = session.identity()
        stored_uuid = stored.get("cluster_uuid")
        current_uuid = current.get("cluster_uuid")
        cluster_matches = (
            stored_uuid == current_uuid
            if stored_uuid and current_uuid
            else stored.get("url") == current.get("url")
        )
        return bool(
            cluster_matches
            and stored.get("device_index") == current.get("device_index")
            and stored.get("weather_index") == current.get("weather_index")
        )

    @staticmethod
    def _record_is_compatible(record: dict[str, Any]) -> bool:
        return bool(
            record.get("calibration_pipeline_version")
            in COMPATIBLE_CALIBRATION_PIPELINE_VERSIONS
            and record.get("channel_inventory_version")
            == CHANNEL_INVENTORY_VERSION
        )

    @staticmethod
    def _record_uses_current_pipeline(record: dict[str, Any]) -> bool:
        return bool(
            record.get("calibration_pipeline_version")
            == CALIBRATION_PIPELINE_VERSION
            and record.get("channel_inventory_version")
            == CHANNEL_INVENTORY_VERSION
        )

    def submit(
        self,
        *,
        session: DataSourceSession,
        plant_id: int,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        start_utc = _utc(start, name="start_time")
        end_utc = _utc(end, name="end_time")
        if end_utc <= start_utc:
            raise CalibrationInputError("end_time must be later than start_time")
        duration = end_utc - start_utc
        minimum = timedelta(days=self.settings.minimum_calibration_days)
        maximum = timedelta(days=self.settings.maximum_calibration_days)
        if duration < minimum:
            raise CalibrationInputError(
                "Calibration range must be at least "
                f"{self.settings.minimum_calibration_days} days"
            )
        if duration > maximum:
            raise CalibrationInputError(
                "Calibration range exceeds "
                f"{self.settings.maximum_calibration_days} days"
            )

        for record in self.list_records():
            existing_id = str(record.get("calibration_id", ""))
            existing_paths = PVLOFV3ModelPaths.from_root(
                self._model_root(existing_id)
            )
            same_request = (
                record.get("plant_id") == str(plant_id)
                and record.get("training_start_utc") == start_utc.isoformat()
                and record.get("training_end_utc") == end_utc.isoformat()
                and self._same_source(record.get("data_source", {}), session)
                and self._record_uses_current_pipeline(record)
            )
            if same_request and record.get("status") in {"queued", "running"}:
                return {**record, "reused": True}
            if (
                same_request
                and record.get("status") == "completed"
                and all(path.is_file() for path in existing_paths.as_dict().values())
            ):
                return {**record, "reused": True}

        calibration_id = f"pvlof-cal-{uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "calibration_id": calibration_id,
            "algorithm_version": PVLOF_V3_VERSION,
            "calibration_pipeline_version": CALIBRATION_PIPELINE_VERSION,
            "channel_inventory_version": CHANNEL_INVENTORY_VERSION,
            "status": "queued",
            "progress_percent": 0,
            "stage": "queued",
            "plant_id": str(plant_id),
            "training_start_utc": start_utc.isoformat(),
            "training_end_utc": end_utc.isoformat(),
            "data_source": session.identity(),
            "created_at": now,
            "updated_at": now,
            "credentials_persisted": False,
            "reused": False,
        }
        self._save_record(record)
        future = self._executor.submit(
            self._run,
            calibration_id,
            session,
            plant_id,
            start_utc,
            end_utc,
        )
        with self._lock:
            self._futures[calibration_id] = future
        future.add_done_callback(
            lambda _future, key=calibration_id: self._forget_future(key)
        )
        return self.get_record(calibration_id)

    def _forget_future(self, calibration_id: str) -> None:
        with self._lock:
            self._futures.pop(calibration_id, None)

    def _run(
        self,
        calibration_id: str,
        session: DataSourceSession,
        plant_id: int,
        start_utc: datetime,
        end_utc: datetime,
    ) -> None:
        job_started_at = datetime.now(timezone.utc)
        try:
            self._update(
                calibration_id,
                status="running",
                progress_percent=5,
                stage="discovering_devices",
                started_at=job_started_at.isoformat(),
            )
            repository = PVLOFESRepository(ReadOnlyESClient(session.es))
            devices = repository.discover_devices(
                plant_id=plant_id,
                start=start_utc,
                end=end_utc,
            )
            if not devices:
                raise CalibrationInputError(
                    "No inverter devices were found in the selected training range"
                )

            template = self.template_detector.base_calibration
            minimum_devices = template.minimum_peer_devices + 1
            if len(devices) < minimum_devices:
                raise CalibrationInputError(
                    f"电站 {plant_id} 在所选训练范围内仅发现 {len(devices)} 台"
                    f"逆变器；当前peer模型至少需要 {minimum_devices} 台"
                    f"（目标设备之外还需要 {template.minimum_peer_devices} 台"
                    "参考设备）"
                )

            self._update(
                calibration_id,
                progress_percent=8,
                stage="planning_current_training_sample",
                discovered_devices=len(devices),
            )
            planning_started_at = datetime.now(timezone.utc)
            sample_plan = repository.plan_current_training_sample(
                plant_id=plant_id,
                start=start_utc,
                end=end_utc,
                target_documents=(
                    self.settings.calibration_target_documents
                ),
                maximum_timestamps=(
                    self.settings.calibration_max_timestamps
                ),
            )
            planning_elapsed_seconds = (
                datetime.now(timezone.utc) - planning_started_at
            ).total_seconds()
            maximum_concurrent_documents = int(
                sample_plan.get("maximum_documents_per_timestamp", 0)
            )
            if maximum_concurrent_documents < minimum_devices:
                raise CalibrationInputError(
                    f"电站 {plant_id} 在所选训练范围内同一时间最多只有 "
                    f"{maximum_concurrent_documents} 台设备数据；当前peer模型"
                    f"至少需要 {minimum_devices} 台设备同时有数据"
                )
            timestamp_values = sample_plan.pop("timestamp_values")
            expected_current_rows = int(
                sample_plan.get("selected_documents", 0)
            )
            if expected_current_rows <= 0:
                raise CalibrationInputError(
                    "No current rows were found in the selected training range"
                )

            self._update(
                calibration_id,
                progress_percent=10,
                stage="reading_current_training_data",
                discovered_devices=len(devices),
                current_sampling=sample_plan,
                current_rows_read=0,
                expected_current_rows=expected_current_rows,
                sampling_plan_elapsed_seconds=planning_elapsed_seconds,
            )

            last_read_progress = 9

            def update_read_progress(rows_read: int) -> None:
                nonlocal last_read_progress
                progress = min(
                    24,
                    10
                    + int(
                        14
                        * min(rows_read, expected_current_rows)
                        / expected_current_rows
                    ),
                )
                if progress <= last_read_progress:
                    return
                last_read_progress = progress
                self._update(
                    calibration_id,
                    progress_percent=progress,
                    stage="reading_current_training_data",
                    current_rows_read=rows_read,
                    expected_current_rows=expected_current_rows,
                )

            current_read_started_at = datetime.now(timezone.utc)
            frame, current_report = repository.fetch_currents(
                plant_id=plant_id,
                device_numbers=devices,
                start=start_utc,
                end=end_utc,
                include_calibration_fields=True,
                max_documents=self.settings.calibration_max_documents,
                event_time_values=timestamp_values,
                progress_callback=update_read_progress,
            )
            current_read_elapsed_seconds = (
                datetime.now(timezone.utc) - current_read_started_at
            ).total_seconds()
            if frame.empty:
                raise CalibrationInputError(
                    "No current rows were found in the selected training range"
                )
            current_report["sampling"] = sample_plan

            self._update(
                calibration_id,
                progress_percent=25,
                stage="reading_optional_weather_data",
                current_rows=len(frame),
                current_rows_read=len(frame),
                current_read_elapsed_seconds=current_read_elapsed_seconds,
            )
            weather_error: str | None = None
            try:
                weather, weather_report = repository.fetch_weather(
                    plant_id=plant_id,
                    start=start_utc - timedelta(minutes=30),
                    end=end_utc + timedelta(minutes=30),
                    max_documents=self.settings.calibration_max_documents,
                )
            except ESReadError as exc:
                weather = pd.DataFrame(
                    columns=["plant_id", "time", "forecast_ghi"]
                )
                weather_report = {
                    "rows": 0,
                    "status": "missing_fallback_to_peer",
                }
                weather_error = str(exc)[:500]

            self._update(
                calibration_id,
                progress_percent=32,
                stage="inferring_string_inventory",
                weather_rows=len(weather),
            )
            inventory, inventory_report = infer_channel_inventory(
                frame,
                minimum_evidence_samples=12,
                minimum_evidence_days=3,
                timezone=self.settings.timezone,
            )
            inventory = {
                key: values for key, values in inventory.items() if values
            }
            if not inventory:
                raise CalibrationInputError(
                    "No configured string channels could be inferred"
                )

            common_fit = {
                "n_neighbors": template.n_neighbors,
                "lof_quantile": template.lof_quantile,
                "residual_quantile": template.residual_quantile,
                "collective_gap_quantile": template.collective_gap_quantile,
                "minimum_collective_gap": template.minimum_collective_gap,
                "minimum_collective_strings": template.minimum_collective_strings,
                "maximum_collective_fraction": template.maximum_collective_fraction,
                "collective_overlap_threshold": template.collective_overlap_threshold,
                "minimum_peer_devices": template.minimum_peer_devices,
                "minimum_strings": template.minimum_strings,
                "minimum_consecutive": template.minimum_consecutive,
                "expected_interval_minutes": template.expected_interval_minutes,
                "zero_current_threshold": template.zero_current_threshold,
                "minimum_virtual_irradiance": template.minimum_virtual_irradiance,
                "minimum_isolated_relative_drop": (
                    template.minimum_isolated_relative_drop
                ),
                "minimum_isolated_absolute_drop": (
                    template.minimum_isolated_absolute_drop
                ),
                "isolated_effect_gate_mode": template.isolated_effect_gate_mode,
                "distance_floor": template.distance_floor,
                "max_score_rows": self.settings.calibration_max_score_rows,
                "configured_strings": inventory,
            }

            self._update(
                calibration_id,
                progress_percent=40,
                stage="fitting_initial_peer_model",
            )
            initial_base, initial_fit = fit_pvlof_v2_calibration(
                frame,
                version="pvlof-v3-self-calibration-initial",
                **common_fit,
            )

            self._update(
                calibration_id,
                progress_percent=52,
                stage="fitting_optional_weather_mapping",
            )
            weather_template = self.template_detector.weather_calibration
            weather_calibration, weather_fit = fit_weather_calibration(
                frame,
                initial_base,
                weather,
                candidate_source_offsets_minutes=tuple(
                    weather_template.candidate_source_offsets_minutes
                ),
                tolerance_minutes=weather_template.tolerance_minutes,
                minimum_forecast_ghi=weather_template.minimum_forecast_ghi,
                minimum_mapping_samples=(
                    weather_template.minimum_mapping_samples
                ),
                mapping_bins=weather_template.mapping_bins,
                weak_peer_count=weather_template.weak_peer_count,
                strong_peer_count=weather_template.strong_peer_count,
                weak_virtual_weight=weather_template.weak_virtual_weight,
                strong_virtual_weight=weather_template.strong_virtual_weight,
                maximum_forecast_correction=(
                    weather_template.maximum_forecast_correction
                ),
            )
            context, context_report = build_conditioned_virtual_context(
                frame,
                initial_base,
                weather_calibration,
                weather,
            )

            self._update(
                calibration_id,
                progress_percent=62,
                stage="fitting_conditioned_base_model",
            )
            base, base_fit = fit_pvlof_v2_calibration(
                frame,
                version="pvlof-v3-self-calibrated-base",
                virtual_override=context,
                **common_fit,
            )
            base = replace(
                base,
                version="pvlof-v3-self-calibrated-base",
                minimum_isolated_relative_drop=0.20,
                minimum_isolated_absolute_drop=0.50,
                isolated_effect_gate_mode="any",
            )

            self._update(
                calibration_id,
                progress_percent=75,
                stage="fitting_hierarchical_lof_thresholds",
            )
            hierarchy_source = _sample_training_rows(
                frame,
                self.settings.calibration_max_score_rows,
            )
            keys = ["plant_id", "device_no", "event_time"]
            hierarchy_context = hierarchy_source[keys].merge(
                context,
                on=keys,
                how="left",
                validate="one_to_one",
            )
            hierarchy_scored = apply_pvlof_v2(
                hierarchy_source,
                base,
                virtual_override=hierarchy_context,
            )
            hierarchy_template = self.template_detector.hierarchy_calibration
            hierarchy, hierarchy_fit = fit_hierarchical_calibration(
                hierarchy_scored,
                quantile=hierarchy_template.lof_quantile,
                minimum_device_samples=(
                    hierarchy_template.minimum_device_samples
                ),
                minimum_device_days=hierarchy_template.minimum_device_days,
                minimum_plant_count_samples=(
                    hierarchy_template.minimum_plant_count_samples
                ),
                minimum_plant_count_days=(
                    hierarchy_template.minimum_plant_count_days
                ),
                minimum_count_samples=(
                    hierarchy_template.minimum_count_samples
                ),
                minimum_count_days=hierarchy_template.minimum_count_days,
                shrinkage_k=hierarchy_template.shrinkage_k,
                minimum_consecutive=base.minimum_consecutive,
                expected_interval_minutes=base.expected_interval_minutes,
                timezone=self.settings.timezone,
                version="pvlof-v3-self-calibrated-hierarchy",
            )

            self._update(
                calibration_id,
                progress_percent=90,
                stage="saving_and_validating_artifacts",
            )
            model_root = self._model_root(calibration_id)
            paths = PVLOFV3ModelPaths.from_root(model_root)
            save_calibration(base, paths.base_calibration)
            save_weather_calibration(
                weather_calibration,
                paths.weather_calibration,
            )
            save_hier_calibration(hierarchy, paths.hierarchy_calibration)
            save_memory_config(
                self.template_detector.memory_config,
                paths.memory_config,
            )
            save_detector_config(
                self.template_detector.detector_config,
                paths.detector_config,
            )
            detector = PVLOFV3Detector.load(paths)
            with self._lock:
                self._detectors[calibration_id] = detector

            report = {
                "current": current_report,
                "weather": weather_report,
                "weather_error": weather_error,
                "inventory": inventory_report,
                "initial_fit": initial_fit,
                "weather_fit": weather_fit,
                "conditioned_context": context_report,
                "base_fit": base_fit,
                "hierarchy_fit": hierarchy_fit,
                "hierarchy_training_wide_rows": len(hierarchy_source),
                "model_root": str(model_root),
            }
            _atomic_json(
                self.root / calibration_id / "calibration_report.json",
                report,
            )
            self._update(
                calibration_id,
                status="completed",
                progress_percent=100,
                stage="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=(
                    datetime.now(timezone.utc) - job_started_at
                ).total_seconds(),
                device_count=len(inventory),
                configured_string_count=sum(map(len, inventory.values())),
                current_rows=len(frame),
                current_sampling=sample_plan,
                weather_rows=len(weather),
                weather_mode=(
                    "forecast_ghi_conditioned"
                    if weather_calibration.plants
                    else "peer_only"
                ),
                model_root=str(model_root),
                report_path=str(
                    self.root / calibration_id / "calibration_report.json"
                ),
            )
        except Exception as exc:  # job boundary: persist a concise failure state
            self._update(
                calibration_id,
                status="failed",
                stage="failed",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
                failed_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=(
                    datetime.now(timezone.utc) - job_started_at
                ).total_seconds(),
            )

    def get_record(self, calibration_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(calibration_id)
        if record is None:
            raise CalibrationNotFoundError(
                f"Unknown calibration_id={calibration_id}"
            )
        result = dict(record)
        result["compatible_with_service"] = self._record_is_compatible(result)
        return result

    def list_records(
        self,
        *,
        session: DataSourceSession | None = None,
        plant_id: int | None = None,
        completed_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            records = [dict(record) for record in self._records.values()]
        if plant_id is not None:
            records = [
                record
                for record in records
                if record.get("plant_id") == str(plant_id)
            ]
        if session is not None:
            records = [
                record
                for record in records
                if self._same_source(record.get("data_source", {}), session)
            ]
        if completed_only:
            records = [
                record for record in records if record.get("status") == "completed"
            ]
        for record in records:
            record["compatible_with_service"] = self._record_is_compatible(record)
        return sorted(records, key=lambda item: str(item.get("created_at", "")))

    def load_detector(
        self,
        calibration_id: str,
        *,
        session: DataSourceSession,
        plant_id: int,
    ) -> PVLOFV3Detector:
        record = self.get_record(calibration_id)
        if record.get("status") != "completed":
            raise CalibrationNotReadyError(
                f"calibration_id={calibration_id} status={record.get('status')}"
            )
        if not self._record_is_compatible(record):
            raise CalibrationNotReadyError(
                f"calibration_id={calibration_id} was created by an obsolete "
                "self-calibration pipeline; create a new permanent calibration"
            )
        if record.get("plant_id") != str(plant_id):
            raise CalibrationInputError(
                "The selected calibration belongs to plant_id="
                f"{record.get('plant_id')}, not {plant_id}"
            )
        if not self._same_source(record.get("data_source", {}), session):
            raise CalibrationInputError(
                "The selected calibration belongs to a different ES cluster/index"
            )
        with self._lock:
            cached = self._detectors.get(calibration_id)
        if cached is not None:
            return cached
        try:
            detector = PVLOFV3Detector.from_root(
                self._model_root(calibration_id)
            )
        except (FileNotFoundError, ValueError) as exc:
            raise CalibrationNotReadyError(
                "Persistent calibration artifacts are missing or invalid: "
                f"{exc}"
            ) from exc
        with self._lock:
            self._detectors[calibration_id] = detector
        return detector


__all__ = [
    "CALIBRATION_PIPELINE_VERSION",
    "COMPATIBLE_CALIBRATION_PIPELINE_VERSIONS",
    "CalibrationInputError",
    "CalibrationManager",
    "CalibrationNotFoundError",
    "CalibrationNotReadyError",
]
