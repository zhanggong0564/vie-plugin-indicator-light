import cv2
import numpy as np

from schemas.data_base import IndicatorLightEmbedding
from vie_plugin_indicator_light.layout_matching import (
    match_layout_fast,
    match_layout_orb,
)


def _result(boxes, embeddings=None, shape=(200, 300)):
    embeddings = embeddings or [[1.0, 0.0] for _ in boxes]
    return IndicatorLightEmbedding(
        embeddings=embeddings,
        boxes=boxes,
        scores=[0.9] * len(boxes),
        image_shape=shape,
    )


def test_fast_match_ignores_extra_detection_and_preserves_registered_order():
    registered = _result(
        [[20, 20, 40, 40], [120, 25, 140, 45], [70, 120, 90, 140]]
    )
    current = _result(
        [
            [90, 130, 110, 150],
            [250, 180, 270, 198],
            [30, 30, 50, 50],
            [150, 35, 170, 55],
        ]
    )

    matched = match_layout_fast(registered, current)

    assert matched is not None
    assert matched.pairs == ((0, 2), (1, 3), (2, 0))
    assert matched.extra_current_indices == (1,)
    assert matched.method == "fast"


def test_fast_match_rejects_missing_registered_position():
    registered = _result(
        [[20, 20, 40, 40], [120, 25, 140, 45], [70, 120, 90, 140]]
    )
    current = _result([[30, 30, 50, 50], [150, 35, 170, 55]])

    assert match_layout_fast(registered, current) is None


def test_orb_match_recovers_180_degree_rotation():
    rng = np.random.default_rng(7)
    registered_image = rng.integers(0, 256, (360, 480, 3), dtype=np.uint8)
    cv2.putText(
        registered_image,
        "INDICATOR PANEL A",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    current_image = cv2.rotate(registered_image, cv2.ROTATE_180)
    registered_boxes = [
        [30, 30, 60, 60],
        [200, 50, 230, 80],
        [100, 260, 130, 290],
    ]
    current_boxes = [
        [480 - x2, 360 - y2, 480 - x1, 360 - y1]
        for x1, y1, x2, y2 in registered_boxes
    ]
    registered = _result(registered_boxes, shape=(360, 480))
    current = _result(current_boxes, shape=(360, 480))

    assert match_layout_fast(registered, current) is None
    matched = match_layout_orb(
        registered,
        current,
        registered_image,
        current_image,
    )

    assert matched is not None
    assert matched.pairs == ((0, 0), (1, 1), (2, 2))
    assert matched.method == "orb"
