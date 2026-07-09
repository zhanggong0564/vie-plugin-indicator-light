import pytest

from vie_plugin_indicator_light.config import _env_bool


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "On"])
def test_env_bool_accepts_true_values(monkeypatch, value):
    monkeypatch.setenv("TEST_BOOL", value)

    assert _env_bool("TEST_BOOL", False) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "Off"])
def test_env_bool_accepts_false_values(monkeypatch, value):
    monkeypatch.setenv("TEST_BOOL", value)

    assert _env_bool("TEST_BOOL", True) is False


def test_env_bool_uses_default_when_missing(monkeypatch):
    monkeypatch.delenv("TEST_BOOL", raising=False)

    assert _env_bool("TEST_BOOL", True) is True
    assert _env_bool("TEST_BOOL", False) is False


def test_env_bool_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "sometimes")

    with pytest.raises(ValueError, match="TEST_BOOL"):
        _env_bool("TEST_BOOL", True)
