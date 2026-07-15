from unittest.mock import Mock

import numpy as np
import pytest

from services.base.inference_runner import TensorInfo
from schemas.exceptions import ModelInferenceError
from vie_plugin_indicator_light.indicator_light_det import IndicatorLightRecognition


class FakeRunner:
    def __init__(
        self,
        input_infos=(TensorInfo("input", ("batch", 3, 224, 224), "tensor(float)"),),
        output_infos=(TensorInfo("embedding", ("batch", 128), "tensor(float)"),),
        outputs=None,
    ) -> None:
        self.input_infos = input_infos
        self.output_infos = output_infos
        self.providers = ("CPUExecutionProvider",)
        self.run = Mock(
            return_value=(
                outputs
                if outputs is not None
                else [np.zeros((2, 128), dtype=np.float32)]
            )
        )


def test_infer_batch_empty_returns_typed_embedding_matrix_without_running() -> None:
    runner = FakeRunner()
    recognizer = IndicatorLightRecognition("unused.onnx", runner=runner)

    result = recognizer.infer_batch([])

    assert result.shape == (0, 128)
    assert result.dtype == np.float32
    runner.run.assert_not_called()


def test_infer_batch_preprocesses_all_rois_in_one_runner_call() -> None:
    runner = FakeRunner()
    recognizer = IndicatorLightRecognition("unused.onnx", runner=runner)
    roi_a = np.zeros((32, 48, 3), dtype=np.uint8)
    roi_b = np.full((64, 24, 3), 255, dtype=np.uint8)

    result = recognizer.infer_batch([roi_a, roi_b])

    runner.run.assert_called_once()
    assert runner.run.call_args.args[0]["input"].shape == (2, 3, 224, 224)
    assert result.shape == (2, 128)


@pytest.mark.parametrize(
    ("input_infos", "output_infos", "message"),
    [
        (
            (TensorInfo("input", (1, 3, 224, 224), "tensor(float)"),),
            (TensorInfo("embedding", ("batch", 128), "tensor(float)"),),
            "dynamic batch",
        ),
        (
            (TensorInfo("input", ("batch", 3, 224, 224), "tensor(float)"),),
            (TensorInfo("embedding", (1, 128), "tensor(float)"),),
            "dynamic batch",
        ),
        (
            (TensorInfo("input", ("batch", 3, 224, 224), "tensor(float)"),),
            (),
            "one output",
        ),
        (
            (TensorInfo("input", ("batch", 3, 224, 224), "tensor(float)"),),
            (
                TensorInfo("embedding", ("batch", 128), "tensor(float)"),
                TensorInfo("extra", ("batch", 128), "tensor(float)"),
            ),
            "one output",
        ),
    ],
)
def test_initialization_rejects_incompatible_model_metadata(
    input_infos, output_infos, message
) -> None:
    runner = FakeRunner(input_infos=input_infos, output_infos=output_infos)

    with pytest.raises(ValueError, match=message):
        IndicatorLightRecognition("unused.onnx", runner=runner)


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([], "exactly one output"),
        (
            [
                np.zeros((2, 128), dtype=np.float32),
                np.zeros((2, 128), dtype=np.float32),
            ],
            "exactly one output",
        ),
        ([np.zeros((128,), dtype=np.float32)], "rank 2"),
        ([np.zeros((2, 1, 128), dtype=np.float32)], "rank 2"),
        ([np.zeros((1, 128), dtype=np.float32)], "shape"),
        ([np.zeros((2, 0), dtype=np.float32)], "shape"),
        ([np.full((2, 128), np.nan, dtype=np.float32)], "finite"),
        ([np.full((2, 128), np.inf, dtype=np.float32)], "finite"),
    ],
)
def test_infer_batch_rejects_outputs_that_violate_embedding_contract(
    outputs, message
) -> None:
    runner = FakeRunner(outputs=outputs)
    recognizer = IndicatorLightRecognition("unused.onnx", runner=runner)
    rois = [
        np.zeros((32, 48, 3), dtype=np.uint8),
        np.zeros((64, 24, 3), dtype=np.uint8),
    ]

    with pytest.raises(ModelInferenceError, match=message):
        recognizer.infer_batch(rois)


@pytest.mark.parametrize(
    ("runner_output", "message"),
    [
        (None, "exactly one output"),
        (42, "exactly one output"),
        ([np.zeros((2, 128), dtype=np.int32)], "floating-point"),
    ],
)
def test_infer_batch_rejects_non_sequence_and_non_float_outputs(
    runner_output, message
) -> None:
    runner = FakeRunner()
    runner.run.return_value = runner_output
    recognizer = IndicatorLightRecognition("unused.onnx", runner=runner)
    rois = [
        np.zeros((32, 48, 3), dtype=np.uint8),
        np.zeros((64, 24, 3), dtype=np.uint8),
    ]

    with pytest.raises(ModelInferenceError, match=message):
        recognizer.infer_batch(rois)
