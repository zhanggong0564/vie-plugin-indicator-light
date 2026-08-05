"""Parse and validate Chroma embedding-generation records."""

from collections import defaultdict
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from .models import CachedEmbeddingGeneration


MIN_INT64 = -(2**63)
MAX_INT64 = 2**63 - 1
_REQUIRED_METADATA = (
    "registration_id",
    "source_fingerprint",
    "pipeline_fingerprint",
    "roi_count",
    "vector_dimension",
    "created_at",
    "material_no",
    "version",
    "image_height",
    "image_width",
    "layout_version",
)

GenerationRecord = tuple[str, Mapping[str, Any], Sequence[float]]
ParsedGeneration = tuple[int | float, CachedEmbeddingGeneration]


def complete_generations(
    result: Mapping[str, Any],
) -> list[ParsedGeneration]:
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    if embeddings is None:
        return []

    grouped: dict[str, list[GenerationRecord]] = defaultdict(list)
    for record_id, metadata, embedding in zip(ids, metadatas, embeddings):
        if not isinstance(metadata, Mapping):
            continue
        generation_id = metadata.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            continue
        grouped[generation_id].append((record_id, metadata, embedding))

    complete = []
    for generation_id, records in grouped.items():
        parsed = parse_generation(generation_id, records)
        if parsed is not None:
            complete.append(parsed)
    return complete


def parse_generation(
    generation_id: str,
    records: list[GenerationRecord],
) -> ParsedGeneration | None:
    first = records[0][1]
    if any(key not in first for key in _REQUIRED_METADATA):
        return None

    try:
        roi_count = positive_integer(first["roi_count"])
        vector_dimension = positive_integer(first["vector_dimension"])
        created_at = finite_number(first["created_at"])
        version = signed_integer(first["version"])
        image_height = positive_integer(first["image_height"])
        image_width = positive_integer(first["image_width"])
        layout_version = positive_integer(first["layout_version"])
        if (
            roi_count is None
            or vector_dimension is None
            or created_at is None
            or version is None
            or image_height is None
            or image_width is None
            or layout_version != 1
            or not isinstance(first["material_no"], str)
        ):
            return None

        expected = {key: first[key] for key in _REQUIRED_METADATA}
        if any(
            any(metadata.get(key) != value for key, value in expected.items())
            for _, metadata, _ in records
        ):
            return None

        indexes = [
            nonnegative_integer(metadata["roi_index"])
            for _, metadata, _ in records
        ]
        if any(index is None for index in indexes):
            return None
        if sorted(indexes) != list(range(roi_count)):
            return None

        ordered = sorted(zip(indexes, records), key=lambda item: item[0])
        embeddings = tuple(
            tuple(float(value) for value in record[2])
            for _, record in ordered
        )
        if any(len(vector) != vector_dimension for vector in embeddings):
            return None
        boxes = tuple(
            tuple(float(metadata[key]) for key in ("box_x1", "box_y1", "box_x2", "box_y2"))
            for _, (_, metadata, _) in ordered
        )

        generation = CachedEmbeddingGeneration(
            registration_id=str(first["registration_id"]),
            source_fingerprint=str(first["source_fingerprint"]),
            pipeline_fingerprint=str(first["pipeline_fingerprint"]),
            generation_id=generation_id,
            embeddings=embeddings,
            boxes=boxes,
            image_shape=(image_height, image_width),
            layout_version=layout_version,
            material_no=first["material_no"],
            version=version,
        )
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    return created_at, generation


def positive_integer(value: Any) -> int | None:
    parsed = nonnegative_integer(value)
    return parsed if parsed not in (None, 0) else None


def nonnegative_integer(value: Any) -> int | None:
    return _bounded_integer(value, minimum=0, maximum=MAX_INT64)


def signed_integer(value: Any) -> int | None:
    return _bounded_integer(value, minimum=MIN_INT64, maximum=MAX_INT64)


def _bounded_integer(
    value: Any,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    if isinstance(value, Integral):
        integer = int(value)
        return integer if minimum <= integer <= maximum else None

    numeric = float(value)
    if (
        not isfinite(numeric)
        or not numeric.is_integer()
        or not minimum <= numeric <= maximum
    ):
        return None
    return int(numeric)


def finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    if isinstance(value, Integral):
        integer = int(value)
        if MIN_INT64 <= integer <= MAX_INT64:
            return integer
        return None
    return value if isfinite(float(value)) else None
