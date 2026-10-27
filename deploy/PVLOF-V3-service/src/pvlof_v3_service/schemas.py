"""HTTP request models."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)


class DataSourceConnectRequest(BaseModel):
    """Create an in-memory ES session; an empty body selects the default ES."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl | None = None
    username: str | None = None
    password: SecretStr | None = None
    device_index: str | None = None
    weather_index: str | None = None
    verify_certs: bool | None = None

    @field_validator("username", "device_index", "weather_index", mode="before")
    @classmethod
    def blank_string_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def complete_custom_connection(self) -> "DataSourceConnectRequest":
        custom_values = (self.url, self.username, self.password)
        if any(value is not None for value in custom_values) and not all(
            value is not None for value in custom_values
        ):
            raise ValueError(
                "url, username and password must be supplied together; "
                "omit all three to use the default test environment"
            )
        return self

    @property
    def uses_default(self) -> bool:
        return self.url is None


class CalibrationRequest(BaseModel):
    """Start one persistent plant calibration job."""

    model_config = ConfigDict(extra="forbid")

    data_source_id: str = Field(min_length=1, max_length=100)
    plant_id: PositiveInt
    start_time: AwareDatetime
    end_time: AwareDatetime

    @property
    def start_datetime(self) -> datetime:
        return self.start_time

    @property
    def end_datetime(self) -> datetime:
        return self.end_time


class EvaluationRequest(BaseModel):
    """One inclusive local-date PVLOF evaluation range."""

    model_config = ConfigDict(extra="forbid")

    plant_id: PositiveInt
    data_source_id: str | None = Field(default=None, min_length=1, max_length=100)
    calibration_id: str | None = Field(default=None, min_length=1, max_length=100)
    start_date: date = Field(
        description="First Asia/Shanghai calendar date to evaluate (inclusive)"
    )
    end_date: date = Field(
        description="Last Asia/Shanghai calendar date to evaluate (inclusive)"
    )

    @model_validator(mode="after")
    def paired_self_service_identifiers(self) -> "EvaluationRequest":
        if (self.data_source_id is None) != (self.calibration_id is None):
            raise ValueError(
                "data_source_id and calibration_id must be supplied together"
            )
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self

    @property
    def uses_persistent_calibration(self) -> bool:
        return self.data_source_id is not None

    @property
    def first_date(self) -> date:
        return self.start_date

    @property
    def last_date(self) -> date:
        return self.end_date
