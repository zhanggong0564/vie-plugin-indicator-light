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
    if len(pairs) != len(registered_centers):
        return None
    used = {current_index for _, current_index in pairs}
    return LayoutMatch(
        pairs=pairs,
        extra_current_indices=tuple(
            index for index in range(len(projected)) if index not in used
        ),
        method="orb",
        mean_distance=mean_distance,
    )
