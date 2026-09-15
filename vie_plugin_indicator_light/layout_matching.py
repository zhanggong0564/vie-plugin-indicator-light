"""Match current indicator detections to registered lamp positions."""

from dataclasses import dataclass
from itertools import combinations
from math import atan2, degrees

import cv2
import numpy as np


@dataclass(frozen=True)
class LayoutMatch:
    pairs: tuple[tuple[int, int], ...]
    extra_current_indices: tuple[int, ...]
    method: str
    mean_distance: float


def _centers(boxes, image_shape) -> np.ndarray:
    height, width = image_shape
    array = np.asarray(boxes, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 4 or height <= 0 or width <= 0:
        raise ValueError("layout boxes and image_shape are invalid")
    return np.column_stack(
        (
            (array[:, 0] + array[:, 2]) / (2 * width),
            (array[:, 1] + array[:, 3]) / (2 * height),
        )
    )


def _distance_gate(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.08
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[distances == 0] = np.inf
    spacing = float(np.median(np.min(distances, axis=1)))
    return max(0.025, min(0.15, spacing * 0.4))


def _greedy_pairs(
    registered: np.ndarray,
    current: np.ndarray,
    gate: float,
) -> tuple[tuple[tuple[int, int], ...], float]:
    candidates = sorted(
        (
            (float(np.linalg.norm(reg_point - cur_point)), reg_index, cur_index)
            for reg_index, reg_point in enumerate(registered)
            for cur_index, cur_point in enumerate(current)
        ),
        key=lambda item: item[0],
    )
    used_registered = set()
    used_current = set()
    pairs = []
    distances = []
    for distance, reg_index, cur_index in candidates:
        if distance > gate:
            break
        if reg_index in used_registered or cur_index in used_current:
            continue
        used_registered.add(reg_index)
        used_current.add(cur_index)
        pairs.append((reg_index, cur_index))
        distances.append(distance)
    pairs.sort()
    mean_distance = float(np.mean(distances)) if distances else float("inf")
    return tuple(pairs), mean_distance


def _similarity_transform(current_a, current_b, registered_a, registered_b):
    current_vector = current_b - current_a
    registered_vector = registered_b - registered_a
    current_length = float(np.linalg.norm(current_vector))
    registered_length = float(np.linalg.norm(registered_vector))
    if current_length < 1e-6 or registered_length < 1e-6:
        return None
    angle = atan2(registered_vector[1], registered_vector[0]) - atan2(
        current_vector[1], current_vector[0]
    )
    scale = registered_length / current_length
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = scale * np.array(((cosine, -sine), (sine, cosine)))
    offset = registered_a - matrix @ current_a
    return matrix, offset, degrees(angle)


def match_layout_fast(registered_result, current_result) -> LayoutMatch | None:
    """Match layouts with a near-zero-rotation similarity transform."""
    if not registered_result.boxes or not current_result.boxes:
        return None
    if registered_result.image_shape is None or current_result.image_shape is None:
        return None
    registered = _centers(registered_result.boxes, registered_result.image_shape)
    current = _centers(current_result.boxes, current_result.image_shape)
    if len(current) < len(registered):
        return None
    gate = _distance_gate(registered)

    if len(registered) == 1:
        pairs, mean_distance = _greedy_pairs(registered, current, gate)
        if len(pairs) != 1:
            return None
        used = {current_index for _, current_index in pairs}
        return LayoutMatch(
            pairs=pairs,
            extra_current_indices=tuple(index for index in range(len(current)) if index not in used),
            method="fast",
            mean_distance=mean_distance,
        )

    solutions = {}
    for reg_a, reg_b in combinations(range(len(registered)), 2):
        for cur_a, cur_b in combinations(range(len(current)), 2):
            for first, second in ((cur_a, cur_b), (cur_b, cur_a)):
                transform = _similarity_transform(
                    current[first], current[second], registered[reg_a], registered[reg_b]
                )
                if transform is None:
                    continue
                matrix, offset, angle = transform
                normalized_angle = (angle + 180) % 360 - 180
                if abs(normalized_angle) > 15:
                    continue
                transformed = current @ matrix.T + offset
                pairs, mean_distance = _greedy_pairs(registered, transformed, gate)
                if len(pairs) != len(registered):
                    continue
                previous = solutions.get(pairs)
                if previous is None or mean_distance < previous:
                    solutions[pairs] = mean_distance

    if not solutions:
        return None
    ranked = sorted(solutions.items(), key=lambda item: item[1])
    best_pairs, best_distance = ranked[0]
    if len(ranked) > 1 and ranked[1][1] - best_distance < gate * 0.05:
        return None
    used = {current_index for _, current_index in best_pairs}
    return LayoutMatch(
        pairs=best_pairs,
        extra_current_indices=tuple(index for index in range(len(current)) if index not in used),
        method="fast",
        mean_distance=best_distance,
    )


def _resize_gray(image: np.ndarray, max_side: int = 1600):
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), scale



def _complete_single_region_match(registered_boxes, projected_boxes, pairs):
    """Recover one expanded ROI only when the other positions anchor registration.

    Preserve existing center matches; reject missing, ambiguous and oversized ROIs.
    Coordinates for both sets of polygons must be in registered-image pixels.
    """
    count = len(registered_boxes)
    if count < 5 or len(pairs) != count - 1:
        return None
    used_registered = {i for i, _ in pairs}
    used_current = {j for _, j in pairs}
    missing = next(i for i in range(count) if i not in used_registered)
    reference = np.asarray(registered_boxes[missing], dtype=np.float32)
    reference_area = cv2.contourArea(reference)
    if reference_area <= 0:
        return None
    center = tuple(float(v) for v in reference.mean(axis=0))
    candidates = []
    for index, polygon in enumerate(projected_boxes):
        if index in used_current:
            continue
        polygon = np.asarray(polygon, dtype=np.float32)
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            continue
        area = cv2.contourArea(polygon)
        if not 1.0 <= area / reference_area <= 5.0:
            continue
        if cv2.pointPolygonTest(polygon, center, False) < 0:
            continue
        overlap, _ = cv2.intersectConvexConvex(reference, polygon)
        if overlap / min(area, reference_area) < 0.65:
            continue
        # An expanded box must not cover another registered control center.
        if any(
            cv2.pointPolygonTest(
                polygon, tuple(float(v) for v in np.mean(box, axis=0)), False
            ) >= 0
            for i, box in enumerate(registered_boxes) if i != missing
        ):
            continue
        candidates.append(index)
    if len(candidates) != 1:
        return None
    return tuple(sorted((*pairs, (missing, candidates[0]))))


def _box_corners(boxes):
    array = np.asarray(boxes, dtype=np.float32)
    return array[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)


def match_layout_orb(
    registered_result,
    current_result,
    registered_image: np.ndarray,
    current_image: np.ndarray,
) -> LayoutMatch | None:
    """Align images with ORB/RANSAC, then match projected lamp centers."""
    if not registered_result.boxes or not current_result.boxes:
        return None
    registered_gray, registered_scale = _resize_gray(registered_image)
    current_gray, current_scale = _resize_gray(current_image)
    orb = cv2.ORB_create(nfeatures=2500)
    registered_keypoints, registered_descriptors = orb.detectAndCompute(registered_gray, None)
    current_keypoints, current_descriptors = orb.detectAndCompute(current_gray, None)
    if registered_descriptors is None or current_descriptors is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        current_descriptors, registered_descriptors, k=2
    )
    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 12:
        return None
    current_points = np.float32(
        [current_keypoints[item.queryIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    registered_points = np.float32(
        [registered_keypoints[item.trainIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(
        current_points, registered_points, cv2.RANSAC, 5.0
    )
    if homography is None or inliers is None:
        return None
    inlier_count = int(inliers.sum())
    if inlier_count < 8 or inlier_count / len(good) < 0.3:
        return None

    registered_centers = _centers(
        registered_result.boxes, registered_result.image_shape
    )
    current_boxes = np.asarray(current_result.boxes, dtype=np.float32)
    current_centers = np.column_stack(
        (
            (current_boxes[:, 0] + current_boxes[:, 2]) * 0.5 * current_scale,
            (current_boxes[:, 1] + current_boxes[:, 3]) * 0.5 * current_scale,
        )
    ).astype(np.float32)
    projected = cv2.perspectiveTransform(current_centers.reshape(-1, 1, 2), homography)
    projected = projected.reshape(-1, 2)
    projected[:, 0] /= registered_gray.shape[1]
    projected[:, 1] /= registered_gray.shape[0]
    gate = _distance_gate(registered_centers)
    pairs, mean_distance = _greedy_pairs(registered_centers, projected, gate)
    method = "orb"
    if len(pairs) != len(registered_centers):
        reference_corners = _box_corners(registered_result.boxes) * registered_scale
        current_corners = _box_corners(current_result.boxes) * current_scale
        projected_corners = cv2.perspectiveTransform(
            current_corners.reshape(-1, 1, 2), homography
        ).reshape(-1, 4, 2)
        pairs = _complete_single_region_match(reference_corners, projected_corners, pairs)
        if pairs is None:
            return None
        method = "orb_region"
    used = {current_index for _, current_index in pairs}
    return LayoutMatch(
        pairs=pairs,
        extra_current_indices=tuple(
            index for index in range(len(projected)) if index not in used
        ),
        method=method,
        mean_distance=mean_distance,
    )
