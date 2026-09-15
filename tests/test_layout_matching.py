import cv2
import numpy as np
import pytest

from schemas.data_base import IndicatorLightEmbedding
from vie_plugin_indicator_light.layout_matching import (
    _box_corners,
    _complete_single_region_match,
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


@pytest.fixture
def expanded_key_layout():
    boxes = [[20, 20, 40, 40], [100, 20, 120, 40], [180, 20, 200, 40],
             [20, 100, 40, 120], [100, 100, 120, 120], [180, 100, 200, 120]]
    return _box_corners(boxes)


def test_region_fallback_recovers_one_expanded_box_without_reordering(expanded_key_layout):
    reference = expanded_key_layout
    current = reference.copy()
    current[5] = _box_corners([[178, 98, 202, 160]])[0]
    current = current[[5, 3, 0, 4, 2, 1]]
    pairs = ((0, 2), (1, 5), (2, 4), (3, 1), (4, 3))
    assert _complete_single_region_match(reference, current, pairs) == (*pairs, (5, 0))


@pytest.mark.parametrize("box", [
    [220, 100, 240, 160],  # displaced: does not contain the registered control
    [150, 80, 250, 190],  # oversized
    [99, 99, 201, 121],  # overlaps an adjacent registered control
    [199, 119, 219, 139],  # negligible overlap
])
def test_region_fallback_rejects_wrong_regions(expanded_key_layout, box):
    reference = expanded_key_layout
    current = reference.copy()
    current[5] = _box_corners([box])[0]
    assert _complete_single_region_match(
        reference, current, tuple((i, i) for i in range(5))
    ) is None


def test_region_fallback_rejects_ambiguous_and_missing_candidates(expanded_key_layout):
    reference = expanded_key_layout
    pairs = tuple((i, i) for i in range(5))
    assert _complete_single_region_match(reference, reference[:5], pairs) is None
    ambiguous = np.concatenate([reference, reference[5:]])
    assert _complete_single_region_match(reference, ambiguous, pairs) is None
    assert _complete_single_region_match(reference, reference, pairs[:4]) is None
    assert _complete_single_region_match(reference[:4], reference[:4], pairs[:3]) is None


def test_orb_region_fallback_keeps_all_positions_in_real_matching():
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
    boxes = [[30, 30, 50, 50], [130, 30, 150, 50], [230, 30, 250, 50],
             [30, 90, 50, 110], [130, 90, 150, 110], [230, 90, 250, 110]]
    current_boxes = [*boxes[:5], [230, 90, 250, 178]]
    registered = _result(boxes, shape=(400, 400))
    current = _result(current_boxes, shape=(400, 400))
    matched = match_layout_orb(registered, current, image, image)
    assert matched is not None
    assert matched.pairs == tuple((i, i) for i in range(6))
    assert matched.method == "orb_region"
    # Removing a control still fails; no placeholder or skipped comparison is added.
    assert match_layout_orb(registered, _result(current_boxes[:5], shape=(400, 400)), image, image) is None


@pytest.mark.parametrize("scale,offset", [(1.0, (0, 0)), (1.2, (10, 8)), (0.8, (20, 30))])
def test_existing_fast_correspondence_survives_scale_translation_and_permutation(scale, offset):
    boxes = np.array([[20, 20, 40, 40], [100, 25, 120, 45],
                      [190, 30, 210, 50], [50, 120, 70, 140], [160, 130, 180, 150]])
    shifted = boxes * scale + np.tile(offset, 2)
    permutation = [3, 0, 4, 1, 2]
    current = _result(shifted[permutation].tolist())
    match = match_layout_fast(_result(boxes.tolist()), current)
    assert match is not None
    assert match.method == "fast"
    assert match.pairs == tuple((i, permutation.index(i)) for i in range(5))
