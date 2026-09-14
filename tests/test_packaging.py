from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_plugin_release_metadata_and_package_scope():
    project_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project_path.open("rb") as project_file:
        project = tomllib.load(project_file)
    assert project["project"]["version"] == "0.1.5"
    assert project["project"]["dependencies"] == [
        "vie-framework",
        "chromadb==1.5.9",
    ]
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "vie_plugin_indicator_light*"
    ]

    converter_path = project_path.parent / "scripts" / "convert_rec_dynamic_batch.py"
    assert converter_path.is_file()
    assert not converter_path.is_relative_to(
        project_path.parent / "vie_plugin_indicator_light"
    )
