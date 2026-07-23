import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "run.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("indicator_light_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_comparison_image_handles_mismatched_counts_and_sizes(monkeypatch):
    example = _load_example_module()
    registered = np.zeros((80, 120, 3), dtype=np.uint8)
    current = np.zeros((40, 100, 3), dtype=np.uint8)
    registered_before = registered.copy()
    current_before = current.copy()
    labels = []
    rectangle_colors = []
    text_colors = []
    original_put_text = cv2.putText
    original_rectangle = cv2.rectangle

    def capture_put_text(image, text, *args, **kwargs):
        labels.append(text)
        text_colors.append((text, args[3]))
        return original_put_text(image, text, *args, **kwargs)

    def capture_rectangle(image, pt1, pt2, color, *args, **kwargs):
        rectangle_colors.append(color)
        return original_rectangle(image, pt1, pt2, color, *args, **kwargs)

    monkeypatch.setattr(example.cv2, "putText", capture_put_text)
    monkeypatch.setattr(example.cv2, "rectangle", capture_rectangle)

    comparison = example.build_comparison_image(
        registered,
        [[10, 10, 30, 30], [50, 10, 70, 30]],
        [0.91, 0.82],
        current,
        [[20, 10, 40, 30]],
        [0.73],
        max_height=60,
    )

    assert comparison.shape == (100, 160, 3)
    assert "Registered | ROI: 2" in labels
    assert "Current | ROI: 1" in labels
    assert "ROI count mismatch: 2 != 1" in labels
    assert "1 0.91" in labels
    assert "2 0.82" in labels
    assert "1 0.73" in labels
    assert rectangle_colors == [
        example.REGISTERED_COLOR,
        example.REGISTERED_COLOR,
        example.CURRENT_COLOR,
    ]
    assert (
        "ROI count mismatch: 2 != 1",
        example.MISMATCH_COLOR,
    ) in text_colors
    assert np.array_equal(registered, registered_before)
    assert np.array_equal(current, current_before)


def test_build_comparison_image_accepts_empty_detections():
    example = _load_example_module()

    comparison = example.build_comparison_image(
        np.zeros((30, 40, 3), dtype=np.uint8),
        [],
        [],
        np.zeros((20, 50, 3), dtype=np.uint8),
        [],
        [],
    )

    assert comparison.shape == (80, 77, 3)


def test_main_saves_comparison_when_business_result_has_no_coordinates(
    monkeypatch,
):
    example = _load_example_module()
    current = np.zeros((40, 100, 3), dtype=np.uint8)
    registered = np.zeros((80, 120, 3), dtype=np.uint8)
    raw_current = SimpleNamespace(
        boxes=[[20, 10, 40, 30]],
        scores=[0.73],
    )
    raw_registered = SimpleNamespace(
        boxes=[[10, 10, 30, 30], [50, 10, 70, 30]],
        scores=[0.91, 0.82],
    )
    detector = Mock()
    detector.detect.return_value.to_dict.return_value = {
        "detailList": [
            {
                "status": "false",
                "coordinate": [],
            }
        ],
        "status": "false",
        "error_msg": (
            "Number of ROIs does not match the registered standard image 2!=1."
        ),
        "message": "失败",
    }
    detector.detector.infer.side_effect = [raw_registered, raw_current]
    monkeypatch.setattr(
        example.scenario_registry,
        "create",
        Mock(return_value=detector),
    )
    monkeypatch.setattr(example.cv2, "imread", Mock(side_effect=[current, registered]))
    imwrite = Mock(return_value=True)
    monkeypatch.setattr(example.cv2, "imwrite", imwrite)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "current.jpg", "registered.jpg"],
    )

    example.main()

    infer_calls = detector.detector.infer.call_args_list
    assert infer_calls[0].args[0] is registered
    assert infer_calls[1].args[0] is current
    saved_path, saved_image = imwrite.call_args.args
    assert saved_path == "indicator_light_result.jpg"
    assert saved_image.shape == (100, 160, 3)
    detector.close.assert_called_once_with()
