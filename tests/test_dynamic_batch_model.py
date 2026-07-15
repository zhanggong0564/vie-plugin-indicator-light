import subprocess
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from scripts.convert_rec_dynamic_batch import convert_dynamic_batch


def _create_fixed_batch_model(path: Path) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, 32, 100]
    )
    output_info = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, [1, 3, 32, 100]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["image"], ["embedding"])],
        "fixed_batch_recognition",
        [input_info],
        [output_info],
    )
    model = helper.make_model(graph)
    onnx.save(model, path)
    return model


def test_convert_dynamic_batch_updates_batch_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "fixed.onnx"
    destination = tmp_path / "dynamic.onnx"
    original = _create_fixed_batch_model(source)

    convert_dynamic_batch(source, destination)

    converted = onnx.load(destination)
    assert converted.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert converted.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert [node.SerializeToString() for node in converted.graph.node] == [
        node.SerializeToString() for node in original.graph.node
    ]
    onnx.checker.check_model(converted)


def test_convert_dynamic_batch_rejects_empty_symbol(tmp_path: Path) -> None:
    source = tmp_path / "fixed.onnx"
    _create_fixed_batch_model(source)

    with pytest.raises(ValueError, match="batch symbol cannot be empty"):
        convert_dynamic_batch(source, tmp_path / "dynamic.onnx", "  ")


def test_convert_dynamic_batch_rejects_same_path(tmp_path: Path) -> None:
    source = tmp_path / "fixed.onnx"
    _create_fixed_batch_model(source)

    with pytest.raises(ValueError, match="source and destination must differ"):
        convert_dynamic_batch(source, source)


@pytest.mark.parametrize("value_kind", ["input", "output"])
def test_convert_dynamic_batch_rejects_multiple_values(
    tmp_path: Path, value_kind: str
) -> None:
    source = tmp_path / "fixed.onnx"
    destination = tmp_path / "dynamic.onnx"
    model = _create_fixed_batch_model(source)
    extra = helper.make_tensor_value_info(
        f"extra_{value_kind}", TensorProto.FLOAT, [1, 3, 32, 100]
    )
    getattr(model.graph, value_kind).append(extra)
    onnx.save(model, source)

    with pytest.raises(
        ValueError, match="recognition model must have one input and one output"
    ):
        convert_dynamic_batch(source, destination)

    assert not destination.exists()


@pytest.mark.parametrize("value_kind", ["input", "output"])
def test_convert_dynamic_batch_rejects_value_without_dimensions(
    tmp_path: Path, value_kind: str
) -> None:
    source = tmp_path / "fixed.onnx"
    destination = tmp_path / "dynamic.onnx"
    model = _create_fixed_batch_model(source)
    value_info = getattr(model.graph, value_kind)[0]
    value_info.type.tensor_type.shape.ClearField("dim")
    onnx.save(model, source)

    with pytest.raises(ValueError, match=f"tensor {value_info.name!r} has no dimensions"):
        convert_dynamic_batch(source, destination)

    assert not destination.exists()


def test_cli_creates_requested_output(tmp_path: Path) -> None:
    source = tmp_path / "fixed.onnx"
    destination = tmp_path / "output" / "dynamic.onnx"
    _create_fixed_batch_model(source)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_rec_dynamic_batch.py",
            str(source),
            str(destination),
            "--symbol",
            "n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert destination.is_file()
    assert result.stdout.strip() == str(destination)
    converted = onnx.load(destination)
    assert converted.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "n"
    assert converted.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "n"


def test_cli_rejects_empty_symbol_without_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "fixed.onnx"
    destination = tmp_path / "output" / "dynamic.onnx"
    _create_fixed_batch_model(source)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_rec_dynamic_batch.py",
            str(source),
            str(destination),
            "--symbol",
            "  ",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not destination.exists()
