import pandas as pd
import pytest

from pv_anomaly.pvlof_io import (
    apply_plant_id_mapping,
    parse_plant_id_mappings,
    restore_source_plant_ids,
)


def test_plant_id_mapping_preserves_source_and_changes_model_key():
    frame = pd.DataFrame(
        {
            "plant_id": [33, 33, 791],
            "device_no": ["a", "b", "c"],
        }
    )

    result, report = apply_plant_id_mapping(frame, {"33": "234"})

    assert result["source_plant_id"].tolist() == ["33", "33", "791"]
    assert result["plant_id"].tolist() == ["234", "234", "791"]
    assert report["rows_mapped"] == 2
    assert report["rows_unmapped"] == 1
    assert report["model_plants"] == ["234", "791"]


def test_parse_plant_id_mapping_accepts_equals_and_colon():
    assert parse_plant_id_mappings(["33=234", "70:892"]) == {
        "33": "234",
        "70": "892",
    }


def test_parse_plant_id_mapping_rejects_conflicts():
    with pytest.raises(ValueError, match="Conflicting mappings"):
        parse_plant_id_mappings(["33=234", "33=791"])


def test_restore_source_plant_id_keeps_model_id_for_audit():
    source = pd.DataFrame(
        {
            "plant_id": ["33"],
            "device_no": ["a"],
        }
    )
    mapped, _ = apply_plant_id_mapping(source, {"33": "234"})
    scored = pd.DataFrame(
        {
            "plant_id": ["234"],
            "device_no": ["a"],
            "string_no": [1],
        }
    )

    result = restore_source_plant_ids(scored, mapped)

    assert result["plant_id"].tolist() == ["33"]
    assert result["source_plant_id"].tolist() == ["33"]
    assert result["model_plant_id"].tolist() == ["234"]
