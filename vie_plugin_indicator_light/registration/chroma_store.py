from __future__ import annotations

import time
import hashlib
import ipaddress
import re
from collections import defaultdict
from math import isclose
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import chromadb

from .models import CachedEmbeddingGeneration


MIN_INT64 = -(2**63)
MAX_INT64 = 2**63 - 1
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
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            return []

        grouped: dict[str, list[tuple[str, Mapping[str, Any], Sequence[float]]]]
        grouped = defaultdict(list)
        for record_id, metadata, embedding in zip(ids, metadatas, embeddings):
            if not isinstance(metadata, Mapping):
                continue
            generation_id = metadata.get("generation_id")
            if not isinstance(generation_id, str) or not generation_id:
                continue
            grouped[generation_id].append((record_id, metadata, embedding))

        complete: list[tuple[int | float, CachedEmbeddingGeneration]] = []
        for generation_id, records in grouped.items():
            parsed = cls._parse_generation(generation_id, records)
            if parsed is not None:
                complete.append(parsed)
        return complete

    @staticmethod
    def _parse_generation(
        generation_id: str,
        records: list[tuple[str, Mapping[str, Any], Sequence[float]]],
    ) -> tuple[int | float, CachedEmbeddingGeneration] | None:
        first = records[0][1]
        required = (
            "registration_id",
            "source_fingerprint",
            "pipeline_fingerprint",
            "roi_count",
            "vector_dimension",
            "created_at",
            "material_no",
            "version",
        )
        if any(key not in first for key in required):
            return None

        try:
            roi_count = ChromaRegistrationStore._positive_integer(
                first["roi_count"]
            )
            vector_dimension = ChromaRegistrationStore._positive_integer(
                first["vector_dimension"]
            )
            created_at = ChromaRegistrationStore._finite_number(
                first["created_at"]
            )
            version = ChromaRegistrationStore._signed_integer(
                first["version"]
            )
            if (
                roi_count is None
                or vector_dimension is None
                or created_at is None
                or version is None
                or not isinstance(first["material_no"], str)
            ):
                return None

            expected = {
                key: first[key]
                for key in required
            }
            if any(
                any(metadata.get(key) != value for key, value in expected.items())
                for _, metadata, _ in records
            ):
                return None

            indexes = [
                ChromaRegistrationStore._nonnegative_integer(
                    metadata["roi_index"]
                )
                for _, metadata, _ in records
            ]
            if any(index is None for index in indexes):
                return None
            if sorted(indexes) != list(range(roi_count)):
                return None

            ordered = sorted(
                zip(indexes, records),
                key=lambda item: item[0],
            )
            embeddings = tuple(
                tuple(float(value) for value in record[2])
                for _, record in ordered
            )
            if any(len(vector) != vector_dimension for vector in embeddings):
                return None

            generation = CachedEmbeddingGeneration(
                registration_id=str(first["registration_id"]),
                source_fingerprint=str(first["source_fingerprint"]),
                pipeline_fingerprint=str(first["pipeline_fingerprint"]),
                generation_id=generation_id,
                embeddings=embeddings,
                material_no=first["material_no"],
                version=version,
            )
        except (KeyError, OverflowError, TypeError, ValueError):
            return None
        return created_at, generation

    @staticmethod
    def _positive_integer(value: Any) -> int | None:
        parsed = ChromaRegistrationStore._nonnegative_integer(value)
        if parsed is None or parsed == 0:
            return None
        return parsed

    @staticmethod
    def _nonnegative_integer(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        if isinstance(value, Integral):
            integer = int(value)
            if integer < 0 or integer > MAX_INT64:
                return None
            return integer
        numeric = float(value)
        if (
            not isfinite(numeric)
            or not numeric.is_integer()
            or numeric < 0
            or numeric > MAX_INT64
        ):
            return None
        return int(numeric)

    @staticmethod
    def _signed_integer(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        if isinstance(value, Integral):
            integer = int(value)
            if integer < MIN_INT64 or integer > MAX_INT64:
                return None
            return integer
        numeric = float(value)
        if (
            not isfinite(numeric)
            or not numeric.is_integer()
            or numeric < MIN_INT64
            or numeric > MAX_INT64
        ):
            return None
        return int(numeric)

    @staticmethod
    def _finite_number(value: Any) -> int | float | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        if isinstance(value, Integral):
            integer = int(value)
            if integer < MIN_INT64 or integer > MAX_INT64:
                return None
            return integer
        if not isfinite(float(value)):
            return None
        return value
