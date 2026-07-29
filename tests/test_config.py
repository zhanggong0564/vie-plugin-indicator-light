import pytest
from pydantic import ValidationError

from vie_plugin_indicator_light.config import IndicatorLightConfig


def test_config_defaults(monkeypatch):
    monkeypatch.delenv("INDICATOR_VECTOR_CACHE_ENABLED", raising=False)

    config = IndicatorLightConfig()

    assert config.det_model_path.endswith("det_yolo_v2.onnx")
    assert config.vector_cache_enabled is True
    assert config.allowed_host_values == ()


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("INDICATOR_VECTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("INDICATOR_ALLOWED_HOSTS", "EXAMPLE.COM, api.local ")

    config = IndicatorLightConfig()

    assert config.vector_cache_enabled is False
    assert config.allowed_host_values == ("example.com", "api.local")


def test_config_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("INDICATOR_VECTOR_CACHE_ENABLED", "sometimes")

    with pytest.raises(ValidationError):
        IndicatorLightConfig()
