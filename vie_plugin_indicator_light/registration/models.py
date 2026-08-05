from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def normalize_embeddings(
    embeddings: Sequence[Iterable[float]],
) -> tuple[tuple[float, ...], ...]:
    arrays = [np.asarray(vector, dtype=np.float64) for vector in embeddings]
    if not arrays:
        raise ValueError("embeddings cannot be empty")

    ranks = {array.ndim for array in arrays}
    if len(ranks) != 1 or not ranks <= {1, 2}:
        raise ValueError("embeddings must use a consistent vector representation")
    if ranks == {2}:
        if any(array.shape[0] != 1 for array in arrays):
            raise ValueError("legacy embeddings must contain singleton wrappers")
        arrays = [array[0] for array in arrays]

    normalized = tuple(tuple(float(value) for value in vector) for vector in arrays)
    dimensions = {len(vector) for vector in normalized}
    if dimensions == {0}:
        raise ValueError("embeddings cannot contain zero-dimensional vectors")
    if len(dimensions) != 1:
        raise ValueError("embeddings have inconsistent dimensions")
    if any(not math.isfinite(value) for vector in normalized for value in vector):
        raise ValueError("embeddings must contain only finite values")
    if any(not any(value != 0.0 for value in vector) for vector in normalized):
        raise ValueError("embeddings cannot contain all-zero vectors")

    return normalized


def normalize_boxes(
    boxes: Sequence[Sequence[float]],
    image_shape: Sequence[int],
) -> tuple[tuple[float, float, float, float], ...]:
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain height and width")
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0:
        raise ValueError("image_shape values must be positive")

    normalized = []
    for box in boxes:
        if len(box) < 4:
            raise ValueError("boxes must contain xyxy coordinates")
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        values = (x1 / width, y1 / height, x2 / width, y2 / height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("boxes must contain only finite values")
        if not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
            raise ValueError("normalized boxes must be valid xyxy coordinates")
        normalized.append(values)
    return tuple(normalized)


@dataclass(frozen=True)
class RegistrationDescriptor:
    registration_id: str
    material_no: str
    product_name: str
    version: int
    model_file: str
    create_time: str | None
    update_time: str | None
    register_mode: bool | None

    @property
    def source_fingerprint(self) -> str:
        source = {
            "create_time": self.create_time,
            "id": self.registration_id,
            "model_file": self.model_file,
            "product_name": self.product_name,
            "update_time": self.update_time,
            "version": self.version,
        }
        canonical = json.dumps(
            source,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedEmbeddingGeneration:
    registration_id: str
    source_fingerprint: str
    pipeline_fingerprint: str
    generation_id: str
    embeddings: tuple[tuple[float, ...], ...]
    boxes: tuple[tuple[float, float, float, float], ...]
    image_shape: tuple[int, int]
    layout_version: int = 1
    material_no: str = ""
    version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "embeddings", normalize_embeddings(self.embeddings))
        if self.layout_version != 1:
            raise ValueError("unsupported registration layout version")
        height, width = self.image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image_shape values must be positive")
        normalized_boxes = normalize_boxes(
            [
                (x1 * width, y1 * height, x2 * width, y2 * height)
                for x1, y1, x2, y2 in self.boxes
            ],
            self.image_shape,
        )
        if len(normalized_boxes) != len(self.embeddings):
            raise ValueError("boxes and embeddings must have the same length")
        object.__setattr__(self, "boxes", normalized_boxes)

    @property
    def vector_dimension(self) -> int:
        return len(self.embeddings[0])

    @classmethod
    def create(
        cls,
        registration_id: str,
        source_fingerprint: str,
        pipeline_fingerprint: str,
        embeddings: Sequence[Iterable[float]],
        boxes: Sequence[Sequence[float]],
        image_shape: Sequence[int],
        material_no: str = "",
        version: int = 0,
    ) -> CachedEmbeddingGeneration:
        return cls(
            registration_id=registration_id,
            source_fingerprint=source_fingerprint,
            pipeline_fingerprint=pipeline_fingerprint,
            generation_id=uuid.uuid4().hex,
            embeddings=embeddings,
            boxes=normalize_boxes(boxes, image_shape),
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            material_no=material_no,
            version=version,
        )

    def to_inference_result(self):
        from schemas.data_base import IndicatorLightEmbedding

        height, width = self.image_shape
        return IndicatorLightEmbedding(
            embeddings=[list(vector) for vector in self.embeddings],
            boxes=[
                [x1 * width, y1 * height, x2 * width, y2 * height]
                for x1, y1, x2, y2 in self.boxes
            ],
            image_shape=self.image_shape,
        )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_fingerprint(det_model_path: str | Path, rec_model_path: str | Path) -> str:
    model_hashes = {
        "detector": _file_sha256(det_model_path),
        "recognizer": _file_sha256(rec_model_path),
    }
    canonical = json.dumps(model_hashes, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
