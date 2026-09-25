import pytest
from pydantic import ValidationError

from pvlof_v3_service.schemas import (
    DataSourceConnectRequest,
    EvaluationRequest,
)


def test_empty_data_source_request_selects_default() -> None:
    assert DataSourceConnectRequest().uses_default is True


def test_custom_data_source_requires_complete_credentials() -> None:
    with pytest.raises(ValidationError, match="supplied together"):
        DataSourceConnectRequest(url="http://es.example:9200")


def test_self_service_evaluation_identifiers_are_paired() -> None:
    with pytest.raises(ValidationError, match="supplied together"):
        EvaluationRequest(
            plant_id=33,
            calibration_id="pvlof-cal-1",
            start_date="2026-07-01",
            end_date="2026-07-01",
        )


def test_evaluation_rejects_reversed_date_range() -> None:
    with pytest.raises(ValidationError, match="end_date"):
        EvaluationRequest(
            plant_id=33,
            start_date="2026-07-03",
            end_date="2026-07-01",
        )
