"""Ephemeral, secret-safe Elasticsearch connection sessions."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pvlof_v3_service.config import (
    INDEX_PATTERN,
    ESConnectionSettings,
    ServiceSettings,
)
from pvlof_v3_service.es_client import ReadOnlyESClient
from pvlof_v3_service.schemas import DataSourceConnectRequest


class DataSourceInputError(ValueError):
    """Raised when a user-supplied data source is incomplete or unsafe."""


class DataSourceNotFoundError(KeyError):
    """Raised when an in-memory data-source session is absent or expired."""


@dataclass(frozen=True)
class DataSourceSession:
    """Validated ES settings retained only in process memory."""

    data_source_id: str
    es: ESConnectionSettings
    is_default: bool
    cluster_name: str | None
    cluster_uuid: str | None
    es_version: str | None
    created_at: datetime
    expires_at: datetime

    def safe_summary(self) -> dict[str, object]:
        return {
            "data_source_id": self.data_source_id,
            "is_default": self.is_default,
            "profile": (
                "production_environment" if self.is_default else "user_supplied"
            ),
            "url": self.es.url,
            "username": self.es.username,
            "device_index": self.es.device_index,
            "weather_index": self.es.weather_index,
            "verify_certs": self.es.verify_certs,
            "cluster_name": self.cluster_name,
            "cluster_uuid": self.cluster_uuid,
            "elasticsearch_version": self.es_version,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "credentials_persisted": False,
        }

    def identity(self) -> dict[str, str | None]:
        """Stable non-secret identity used to bind calibrations to a source."""

        return {
            "cluster_uuid": self.cluster_uuid,
            "url": self.es.url,
            "device_index": self.es.device_index,
            "weather_index": self.es.weather_index,
        }


class DataSourceManager:
    """Create expiring ES sessions without writing credentials to disk."""

    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self._sessions: dict[str, DataSourceSession] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_index(value: str, *, name: str) -> str:
        value = value.strip()
        if not INDEX_PATTERN.fullmatch(value):
            raise DataSourceInputError(
                f"{name} contains unsupported characters"
            )
        return value

    def _connection_settings(
        self, request: DataSourceConnectRequest
    ) -> tuple[ESConnectionSettings, bool]:
        if request.uses_default:
            return self.settings.es, True
        if not self.settings.allow_custom_data_sources:
            raise DataSourceInputError(
                "Custom data sources are disabled by service policy"
            )

        assert request.url is not None
        assert request.username is not None
        assert request.password is not None
        if request.url.username is not None or request.url.password is not None:
            raise DataSourceInputError(
                "Credentials must not be embedded in the Elasticsearch URL"
            )
        username = request.username.strip()
        password = request.password.get_secret_value()
        if not username or not password:
            raise DataSourceInputError("username and password must not be blank")

        default = self.settings.es
        es = replace(
            default,
            url=str(request.url).rstrip("/"),
            username=username,
            password=password,
            device_index=self._validate_index(
                request.device_index or default.device_index,
                name="device_index",
            ),
            weather_index=self._validate_index(
                request.weather_index or default.weather_index,
                name="weather_index",
            ),
            verify_certs=(
                default.verify_certs
                if request.verify_certs is None
                else request.verify_certs
            ),
            ca_cert=None,
        )
        return es, False

    def connect(self, request: DataSourceConnectRequest) -> DataSourceSession:
        es, is_default = self._connection_settings(request)
        info = ReadOnlyESClient(es).info()
        now = datetime.now(timezone.utc)
        session = DataSourceSession(
            data_source_id=f"ds-{uuid4().hex}",
            es=es,
            is_default=is_default,
            cluster_name=(
                str(info["cluster_name"]) if info.get("cluster_name") else None
            ),
            cluster_uuid=(
                str(info["cluster_uuid"]) if info.get("cluster_uuid") else None
            ),
            es_version=(
                str((info.get("version") or {}).get("number"))
                if (info.get("version") or {}).get("number")
                else None
            ),
            created_at=now,
            expires_at=now
            + timedelta(minutes=self.settings.data_source_ttl_minutes),
        )
        with self._lock:
            self._purge_expired(now)
            self._sessions[session.data_source_id] = session
        return session

    def _purge_expired(self, now: datetime) -> None:
        expired = [
            key
            for key, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for key in expired:
            self._sessions.pop(key, None)

    def get(self, data_source_id: str) -> DataSourceSession:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._purge_expired(now)
            session = self._sessions.get(data_source_id)
        if session is None:
            raise DataSourceNotFoundError(
                "Data-source session was not found or has expired; reconnect first"
            )
        return session

    def disconnect(self, data_source_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(data_source_id, None) is not None


__all__ = [
    "DataSourceInputError",
    "DataSourceManager",
    "DataSourceNotFoundError",
    "DataSourceSession",
]
