from pathlib import Path

import pytest

from pv_anomaly.settings import ConfigurationError, DataSettings, ESSettings


def test_es_settings_loads_without_exposing_password(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ES_URL=https://es.example.test:9200\n"
        "ES_USERNAME=reader\n"
        "ES_PASSWORD=secret-value\n"
        "ES_INDEX=pv-wide\n",
        encoding="utf-8",
    )
    for name in ("ES_URL", "ES_USERNAME", "ES_PASSWORD", "ES_INDEX"):
        monkeypatch.delenv(name, raising=False)
    settings = ESSettings.from_env(env_file)
    assert settings.index == "pv-wide"
    assert "password" not in settings.safe_summary()
    assert "secret-value" not in str(settings.safe_summary())


def test_es_settings_rejects_missing_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("ES_URL", "ES_USERNAME", "ES_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError, match="Missing configuration"):
        ESSettings.from_env(tmp_path / "missing.env")


def test_data_settings_loads_example():
    settings = DataSettings.from_yaml("configs/data.example.yaml")
    assert "active_power" in settings.fields["power"]
    assert settings.string_current_patterns

