"""Convert a fixed-batch recognition ONNX model to dynamic batch."""

import argparse
from pathlib import Path

import onnx


def _set_dynamic_first_dimension(value_info, symbol: str) -> None:
    dimensions = value_info.type.tensor_type.shape.dim
    if not dimensions:
        raise ValueError(f"tensor {value_info.name!r} has no dimensions")
    dimensions[0].ClearField("dim_value")
    dimensions[0].dim_param = symbol


def convert_dynamic_batch(
    source: Path, destination: Path, symbol: str = "batch"
) -> None:
    """Save a model copy whose input and output batch dimensions are dynamic."""
    if not symbol.strip():
        raise ValueError("batch symbol cannot be empty")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")

    model = onnx.load(source)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("recognition model must have one input and one output")

    _set_dynamic_first_dimension(model.graph.input[0], symbol)
    _set_dynamic_first_dimension(model.graph.output[0], symbol)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a recognition ONNX model to dynamic batch."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--symbol", default="batch")
    args = parser.parse_args()

    convert_dynamic_batch(args.source, args.destination, args.symbol)
    print(args.destination)


if __name__ == "__main__":
    main()
