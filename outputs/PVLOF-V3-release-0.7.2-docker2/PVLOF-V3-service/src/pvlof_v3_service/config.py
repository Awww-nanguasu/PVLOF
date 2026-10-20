"""Environment-only configuration with no embedded credentials."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


INDEX_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


class ConfigurationError(ValueError):
    """Raised when service configuration is missing or unsafe."""


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("<"):
        raise ConfigurationError(f"Missing configuration: {name}")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _positive_int(name: str, default: int, *, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = f" and no greater than {maximum}" if maximum is not None else ""
        raise ConfigurationError(f"{name} must be positive{suffix}")
    return value


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _index(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not INDEX_PATTERN.fullmatch(value):
        raise ConfigurationError(f"{name} contains unsupported characters")
    return value


def _plant_aliases(name: str = "PVLOF_PLANT_ALIASES") -> dict[str, str]:
    """Parse source-to-calibration aliases such as ``33:234``."""

    raw = os.getenv(name, "").strip()
    if not raw:
        return {}

    aliases: dict[str, str] = {}
    for item in raw.split(","):
        pair = item.strip()
        parts = pair.split(":")
        if len(parts) != 2:
            raise ConfigurationError(
                f"{name} must contain comma-separated source:calibration pairs"
            )
        source_text, calibration_text = (part.strip() for part in parts)
        try:
            source = int(source_text)
            calibration = int(calibration_text)
        except ValueError as exc:
            raise ConfigurationError(
                f"{name} plant identifiers must be integers"
            ) from exc
        if source <= 0 or calibration <= 0:
            raise ConfigurationError(
                f"{name} plant identifiers must be positive"
            )

        source_key = str(source)
        calibration_key = str(calibration)
        previous = aliases.get(source_key)
        if previous is not None and previous != calibration_key:
            raise ConfigurationError(
                f"{name} maps source plant {source_key} more than once"
            )
        aliases[source_key] = calibration_key
    return aliases


@dataclass(frozen=True)
class ESConnectionSettings:
    """Read-only Elasticsearch connection and fixed source indices."""

    url: str
    username: str
    password: str = field(repr=False)
    device_index: str = "pv_device_data"
    weather_index: str = "pv_energy_predict_plant_15min"
    verify_certs: bool = True
    ca_cert: Path | None = None
    timeout_seconds: float = 30.0
    page_size: int = 2000
    max_documents: int = 100000

    def safe_summary(self) -> dict[str, object]:
        return {
            "url": self.url,
            "username": self.username,
            "device_index": self.device_index,
            "weather_index": self.weather_index,
            "verify_certs": self.verify_certs,
            "ca_cert": str(self.ca_cert) if self.ca_cert else None,
            "timeout_seconds": self.timeout_seconds,
            "page_size": self.page_size,
            "max_documents": self.max_documents,
        }


@dataclass(frozen=True)
class ServiceSettings:
    """Complete service policy loaded from environment variables."""

    model_root: Path
    es: ESConnectionSettings
    runtime_root: Path = Path("runtime")
    plant_aliases: dict[str, str] = field(default_factory=dict)
    timezone: str = "Asia/Shanghai"
    weather_margin_minutes: int = 30
    weather_optional: bool = True
    max_concurrent_evaluations: int = 1
    allow_custom_data_sources: bool = False
    data_source_ttl_minutes: int = 480
    minimum_calibration_days: int = 20
    maximum_calibration_days: int = 366
    calibration_max_documents: int = 2_000_000
    calibration_target_documents: int = 300_000
    calibration_max_timestamps: int = 10_000
    calibration_max_score_rows: int = 100_000
    max_concurrent_calibrations: int = 1
    api_key: str | None = field(default=None, repr=False)
    cors_origins: tuple[str, ...] = ()
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    def __post_init__(self) -> None:
        if self.minimum_calibration_days > self.maximum_calibration_days:
            raise ConfigurationError(
                "PVLOF_MINIMUM_CALIBRATION_DAYS must not exceed "
                "PVLOF_MAXIMUM_CALIBRATION_DAYS"
            )
        if self.calibration_target_documents > self.calibration_max_documents:
            raise ConfigurationError(
                "PVLOF_CALIBRATION_TARGET_DOCUMENTS must not exceed "
                "PVLOF_CALIBRATION_MAX_DOCUMENTS"
            )

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "ServiceSettings":
        load_dotenv(dotenv_path=env_file, override=False)

        model_root = Path(_required("PVLOF_MODEL_ROOT")).expanduser().resolve()
        if not model_root.is_dir():
            raise ConfigurationError(f"PVLOF_MODEL_ROOT is not a directory: {model_root}")

        url = _required("ES_URL").rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("ES_URL must be an http(s) URL")

        ca_value = os.getenv("ES_CA_CERT", "").strip()
        ca_cert = Path(ca_value).expanduser().resolve() if ca_value else None
        if ca_cert is not None and not ca_cert.is_file():
            raise ConfigurationError(f"ES_CA_CERT does not exist: {ca_cert}")

        es = ESConnectionSettings(
            url=url,
            username=_required("ES_USERNAME"),
            password=_required("ES_PASSWORD"),
            device_index=_index("ES_DEVICE_INDEX", "pv_device_data"),
            weather_index=_index(
                "ES_WEATHER_INDEX", "pv_energy_predict_plant_15min"
            ),
            verify_certs=_boolean("ES_VERIFY_CERTS", True),
            ca_cert=ca_cert,
            timeout_seconds=_positive_float("ES_TIMEOUT_SECONDS", 30.0),
            page_size=_positive_int("ES_PAGE_SIZE", 5000, maximum=5000),
            max_documents=_positive_int("ES_MAX_DOCUMENTS", 100000),
        )

        origins = tuple(
            item.strip()
            for item in os.getenv("PVLOF_CORS_ORIGINS", "").split(",")
            if item.strip()
        )
        api_key = os.getenv("PVLOF_API_KEY", "").strip() or None
        runtime_root = Path(
            os.getenv("PVLOF_RUNTIME_ROOT", "runtime").strip() or "runtime"
        ).expanduser().resolve()
        return cls(
            model_root=model_root,
            es=es,
            runtime_root=runtime_root,
            plant_aliases=_plant_aliases(),
            timezone=os.getenv("PVLOF_TIMEZONE", "Asia/Shanghai").strip(),
            weather_margin_minutes=_positive_int(
                "PVLOF_WEATHER_MARGIN_MINUTES", 30
            ),
            weather_optional=_boolean("PVLOF_WEATHER_OPTIONAL", True),
            max_concurrent_evaluations=_positive_int(
                "PVLOF_MAX_CONCURRENT_EVALUATIONS", 1, maximum=8
            ),
            allow_custom_data_sources=_boolean(
                "PVLOF_ALLOW_CUSTOM_DATA_SOURCES", False
            ),
            data_source_ttl_minutes=_positive_int(
                "PVLOF_DATA_SOURCE_TTL_MINUTES", 480
            ),
            minimum_calibration_days=_positive_int(
                "PVLOF_MINIMUM_CALIBRATION_DAYS", 20, maximum=366
            ),
            maximum_calibration_days=_positive_int(
                "PVLOF_MAXIMUM_CALIBRATION_DAYS", 366, maximum=3660
            ),
            calibration_max_documents=_positive_int(
                "PVLOF_CALIBRATION_MAX_DOCUMENTS", 2_000_000
            ),
            calibration_target_documents=_positive_int(
                "PVLOF_CALIBRATION_TARGET_DOCUMENTS", 300_000
            ),
            calibration_max_timestamps=_positive_int(
                "PVLOF_CALIBRATION_MAX_TIMESTAMPS", 10_000, maximum=50_000
            ),
            calibration_max_score_rows=_positive_int(
                "PVLOF_CALIBRATION_MAX_SCORE_ROWS", 100_000
            ),
            max_concurrent_calibrations=_positive_int(
                "PVLOF_MAX_CONCURRENT_CALIBRATIONS", 1, maximum=4
            ),
            api_key=api_key,
            cors_origins=origins,
            api_host=os.getenv("PVLOF_API_HOST", "127.0.0.1").strip(),
            api_port=_positive_int("PVLOF_API_PORT", 8000, maximum=65535),
        )

    def safe_summary(self) -> dict[str, object]:
        return {
            "model_root": str(self.model_root),
            "runtime_root": str(self.runtime_root),
            "es": self.es.safe_summary(),
            "plant_aliases": dict(sorted(self.plant_aliases.items())),
            "timezone": self.timezone,
            "weather_margin_minutes": self.weather_margin_minutes,
            "weather_optional": self.weather_optional,
            "max_concurrent_evaluations": self.max_concurrent_evaluations,
            "allow_custom_data_sources": self.allow_custom_data_sources,
            "data_source_ttl_minutes": self.data_source_ttl_minutes,
            "minimum_calibration_days": self.minimum_calibration_days,
            "maximum_calibration_days": self.maximum_calibration_days,
            "calibration_max_documents": self.calibration_max_documents,
            "calibration_target_documents": self.calibration_target_documents,
            "calibration_max_timestamps": self.calibration_max_timestamps,
            "calibration_max_score_rows": self.calibration_max_score_rows,
            "max_concurrent_calibrations": self.max_concurrent_calibrations,
            "api_key_configured": self.api_key is not None,
            "cors_origins": list(self.cors_origins),
            "api_host": self.api_host,
            "api_port": self.api_port,
        }
