from pathlib import Path


def test_plugin_wheel_only_declares_framework_dependency():
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'dependencies = ["vie-framework"]' in project
