from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


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
    material_no: str = ""
    version: int = 0

    def __post_init__(self) -> None:
        normalized = tuple(tuple(float(value) for value in vector) for vector in self.embeddings)
        if not normalized:
            raise ValueError("embeddings cannot be empty")

        dimensions = {len(vector) for vector in normalized}
        if dimensions == {0}:
            raise ValueError("embeddings cannot contain zero-dimensional vectors")
        if len(dimensions) != 1:
            raise ValueError("embeddings have inconsistent dimensions")
        if any(not math.isfinite(value) for vector in normalized for value in vector):
            raise ValueError("embeddings must contain only finite values")
        if any(not any(value != 0.0 for value in vector) for vector in normalized):
            raise ValueError("embeddings cannot contain all-zero vectors")

        object.__setattr__(self, "embeddings", normalized)

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
        material_no: str = "",
        version: int = 0,
    ) -> CachedEmbeddingGeneration:
        return cls(
            registration_id=registration_id,
            source_fingerprint=source_fingerprint,
            pipeline_fingerprint=pipeline_fingerprint,
            generation_id=uuid.uuid4().hex,
            embeddings=tuple(tuple(vector) for vector in embeddings),
            material_no=material_no,
            version=version,
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
