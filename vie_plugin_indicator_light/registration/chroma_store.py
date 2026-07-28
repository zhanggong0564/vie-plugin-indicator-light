from __future__ import annotations

import hashlib
import ipaddress
import re
import time
from math import isclose
from pathlib import Path
from typing import Any, Mapping, Sequence

import chromadb

from .chroma_generation import (
    MAX_INT64,
    MIN_INT64,
    complete_generations,
    finite_number,
    nonnegative_integer,
    parse_generation,
    positive_integer,
    signed_integer,
)
from .models import CachedEmbeddingGeneration


_COLLECTION_BASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]$")


def namespaced_collection_name(base: str, pipeline_fingerprint: str) -> str:
    if not isinstance(base, str) or not 3 <= len(base) <= 487:
        raise ValueError("Chroma collection base name must be 3-487 characters")
    if not _COLLECTION_BASE_RE.fullmatch(base):
        raise ValueError("Chroma collection base name contains invalid characters")
    if ".." in base:
        raise ValueError("Chroma collection base name cannot contain consecutive periods")
    try:
        ipaddress.ip_address(base)
    except ValueError:
        pass
    else:
        raise ValueError("Chroma collection base name cannot be an IP address")
    if not isinstance(pipeline_fingerprint, str) or not pipeline_fingerprint:
        raise ValueError("pipeline fingerprint cannot be empty")
    if re.fullmatch(r"[0-9a-fA-F]+", pipeline_fingerprint):
        digest = pipeline_fingerprint.lower()
    else:
        digest = hashlib.sha256(pipeline_fingerprint.encode("utf-8")).hexdigest()
    return f"{base}-{digest[:24]}"


class ChromaRegistrationStore:
    def __init__(
        self,
        path: str | Path,
        collection_name: str,
        pipeline_fingerprint: str | None = None,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        if pipeline_fingerprint is not None:
            # Keep pipeline collections side by side for rollback and rolling deploys.
            collection_name = namespaced_collection_name(
                collection_name, pipeline_fingerprint
            )
        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )

    def get(
        self,
        registration_id: str,
        source_fingerprint: str,
        pipeline_fingerprint: str,
    ) -> CachedEmbeddingGeneration | None:
        result = self._collection.get(
            where={
                "$and": [
                    {"registration_id": registration_id},
                    {"source_fingerprint": source_fingerprint},
                    {"pipeline_fingerprint": pipeline_fingerprint},
                ]
            },
            include=["embeddings", "metadatas"],
        )
        generations = self._complete_generations(result)
        if not generations:
            return None
        return max(
            generations,
            key=lambda item: (item[0], item[1].generation_id),
        )[1]

    def replace(self, generation: CachedEmbeddingGeneration) -> None:
        created_at = time.time_ns()
        roi_count = len(generation.embeddings)
        ids = [
            self._record_id(generation.registration_id, generation.generation_id, index)
            for index in range(roi_count)
        ]
        metadatas = [
            {
                "registration_id": generation.registration_id,
                "source_fingerprint": generation.source_fingerprint,
                "pipeline_fingerprint": generation.pipeline_fingerprint,
                "generation_id": generation.generation_id,
                "roi_index": index,
                "roi_count": roi_count,
                "vector_dimension": generation.vector_dimension,
                "created_at": created_at,
                "material_no": generation.material_no,
                "version": generation.version,
            }
            for index in range(roi_count)
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=[list(vector) for vector in generation.embeddings],
            metadatas=metadatas,
        )

        result = self._collection.get(
            ids=ids,
            include=["embeddings", "metadatas"],
        )
        validated = {
            item.generation_id: item
            for _, item in self._complete_generations(result)
        }.get(generation.generation_id)
        if validated is None or not self._same_generation(validated, generation):
            raise RuntimeError("new embedding generation is not complete")

        registration_records = self._collection.get(
            where={"registration_id": generation.registration_id},
            include=["metadatas"],
        )
        current_ids = set(ids)
        obsolete_ids = [
            record_id
            for record_id in registration_records["ids"]
            if record_id not in current_ids
        ]
        if obsolete_ids:
            self._collection.delete(ids=obsolete_ids)

    @staticmethod
    def _record_id(
        registration_id: str,
        generation_id: str,
        roi_index: int,
    ) -> str:
        return f"{registration_id}:{generation_id}:{roi_index}"

    @staticmethod
    def _same_generation(
        actual: CachedEmbeddingGeneration,
        expected: CachedEmbeddingGeneration,
    ) -> bool:
        if (
            actual.registration_id != expected.registration_id
            or actual.source_fingerprint != expected.source_fingerprint
            or actual.pipeline_fingerprint != expected.pipeline_fingerprint
            or actual.generation_id != expected.generation_id
            or actual.material_no != expected.material_no
            or actual.version != expected.version
            or len(actual.embeddings) != len(expected.embeddings)
        ):
            return False
        return all(
            len(actual_vector) == len(expected_vector)
            and all(
                isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-7)
                for actual_value, expected_value in zip(
                    actual_vector, expected_vector
                )
            )
            for actual_vector, expected_vector in zip(
                actual.embeddings, expected.embeddings
            )
        )

    @classmethod
    def _complete_generations(
        cls,
        result: Mapping[str, Any],
    ) -> list[tuple[int | float, CachedEmbeddingGeneration]]:
        return complete_generations(result)

    @staticmethod
    def _parse_generation(
        generation_id: str,
        records: list[tuple[str, Mapping[str, Any], Sequence[float]]],
    ) -> tuple[int | float, CachedEmbeddingGeneration] | None:
        return parse_generation(generation_id, records)

    @staticmethod
    def _positive_integer(value: Any) -> int | None:
        return positive_integer(value)

    @staticmethod
    def _nonnegative_integer(value: Any) -> int | None:
        return nonnegative_integer(value)

    @staticmethod
    def _signed_integer(value: Any) -> int | None:
        return signed_integer(value)

    @staticmethod
    def _finite_number(value: Any) -> int | float | None:
        return finite_number(value)
