from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from services.base.inference_runner import TensorInfo
from schemas.exceptions import ModelInferenceError
from vie_plugin_indicator_light.indicator_light_det import (
    IndicatorLightDetRec,
    IndicatorLightRecognition,
)


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


def test_det_rec_batches_sorted_expanded_rois_and_aligns_results() -> None:
    detector = IndicatorLightDetRec.__new__(IndicatorLightDetRec)
    detector.det = Mock()
    detector.det.infer.return_value = SimpleNamespace(
        boxes=[[60.0, 10.0, 80.0, 30.0], [10.0, 20.0, 30.0, 40.0]],
        scores=[0.9, 0.8],
    )
    detector.rec = Mock()
    detector.rec.infer_batch.return_value = np.array(
        [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
    )
    image = np.zeros((45, 85, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(85, dtype=np.uint8)
    image[:, :, 1] = np.arange(45, dtype=np.uint8)[:, None]

    result = detector.infer(image)

    detector.rec.infer_batch.assert_called_once()
    rois = detector.rec.infer_batch.call_args.args[0]
    assert len(rois) == 2
    assert rois[0].shape == (35, 40, 3)
    assert rois[0][0, 0, :2].tolist() == [0, 10]
    assert rois[0][-1, -1, :2].tolist() == [39, 44]
    assert rois[1].shape == (40, 35, 3)
    assert rois[1][0, 0, :2].tolist() == [50, 0]
    assert rois[1][-1, -1, :2].tolist() == [84, 39]
    assert result.boxes == [[10.0, 20.0, 30.0, 40.0], [60.0, 10.0, 80.0, 30.0]]
    assert result.scores == [0.8, 0.9]
    assert result.embeddings == [[1.0, 2.0], [3.0, 4.0]]


def test_det_rec_empty_detection_skips_recognition() -> None:
    detector = IndicatorLightDetRec.__new__(IndicatorLightDetRec)
    detector.det = Mock()
    detector.det.infer.return_value = SimpleNamespace(boxes=[], scores=[])
    detector.rec = Mock()

    result = detector.infer(np.zeros((20, 30, 3), dtype=np.uint8))

    detector.rec.infer_batch.assert_not_called()
    assert result.boxes == []
    assert result.scores == []
    assert result.embeddings == []


def test_det_rec_rejects_empty_roi_before_batch_recognition() -> None:
    detector = IndicatorLightDetRec.__new__(IndicatorLightDetRec)
    detector.det = Mock()
    detector.det.infer.return_value = SimpleNamespace(
        boxes=[[100.0, 100.0, 110.0, 110.0]], scores=[0.7]
    )
    detector.rec = Mock()

    with pytest.raises(ModelInferenceError, match="ROI"):
        detector.infer(np.zeros((20, 30, 3), dtype=np.uint8))

    detector.rec.infer_batch.assert_not_called()


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

    with pytest.raises(ModelInferenceError, match=message):
        IndicatorLightRecognition("unused.onnx", runner=runner)


@pytest.mark.parametrize("fixed_side", ["input", "output"])
def test_fixed_batch_error_explains_rec_v3_upgrade(fixed_side: str) -> None:
    input_batch = 1 if fixed_side == "input" else "batch"
    output_batch = 1 if fixed_side == "output" else "batch"
    runner = FakeRunner(
        input_infos=(
            TensorInfo("input", (input_batch, 3, 224, 224), "tensor(float)"),
        ),
        output_infos=(
            TensorInfo("embedding", (output_batch, 128), "tensor(float)"),
        ),
    )

    with pytest.raises(ModelInferenceError) as exc_info:
        IndicatorLightRecognition("rec_v2.onnx", runner=runner)

    assert "固定 batch 不支持" in exc_info.value.error_msg
    assert "rec_v3.onnx" in exc_info.value.error_msg
    assert "转换" in exc_info.value.error_msg


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
