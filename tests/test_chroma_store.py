from __future__ import annotations

import math
from typing import Any

import pytest

pytest.importorskip("chromadb")

from vie_plugin_indicator_light.registration.chroma_store import (
    ChromaRegistrationStore,
    MAX_INT64,
    MIN_INT64,
    namespaced_collection_name,
)
from vie_plugin_indicator_light.registration.models import (
    CachedEmbeddingGeneration,
)


COLLECTION = "test_registered_embeddings"


def generation(
    generation_id: str = "g1",
    embeddings: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0),
        (0.0, 1.0),
    ),
) -> CachedEmbeddingGeneration:
    return CachedEmbeddingGeneration(
        registration_id="model-1",
        source_fingerprint="source-1",
        pipeline_fingerprint="pipeline-1",
        generation_id=generation_id,
        embeddings=embeddings,
        material_no="A0SW1821",
        version=1,
    )


def metadata(
    *,
    generation_id: str,
    roi_index: int,
    roi_count: int = 2,
    vector_dimension: int = 2,
    created_at: int = 1,
) -> dict[str, str | int]:
    return {
        "registration_id": "model-1",
        "source_fingerprint": "source-1",
        "pipeline_fingerprint": "pipeline-1",
        "generation_id": generation_id,
        "roi_index": roi_index,
        "roi_count": roi_count,
        "vector_dimension": vector_dimension,
        "created_at": created_at,
        "material_no": "A0SW1821",
        "version": 1,
    }


def test_round_trip_restores_roi_order(tmp_path):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    expected = generation(
        embeddings=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    )

    store.replace(expected)

    assert store.get("model-1", "source-1", "pipeline-1") == expected


def test_pipeline_namespaces_support_different_dimensions(tmp_path):
    first = ChromaRegistrationStore(tmp_path, COLLECTION, "a" * 64)
    second = ChromaRegistrationStore(tmp_path, COLLECTION, "b" * 64)
    first_generation = generation(embeddings=((1.0, 0.0),))
    second_generation = CachedEmbeddingGeneration(
        registration_id="model-1",
        source_fingerprint="source-1",
        pipeline_fingerprint="pipeline-2",
        generation_id="g2",
        embeddings=((1.0, 0.0, 0.0),),
        material_no="A0SW1821",
        version=1,
    )

    first.replace(first_generation)
    second.replace(second_generation)

    assert first.collection_name != second.collection_name
    assert {
        collection.name for collection in first._client.list_collections()
    } == {first.collection_name, second.collection_name}
    assert first.get("model-1", "source-1", "pipeline-1") == first_generation
    assert second.get("model-1", "source-1", "pipeline-2") == second_generation


def test_collection_name_is_deterministic_and_accepts_uppercase_base():
    assert namespaced_collection_name("registered_embeddings", "pipeline-v1") == (
        namespaced_collection_name("registered_embeddings", "pipeline-v1")
    )
    assert namespaced_collection_name("registered_embeddings", "ABCDEF").endswith(
        "-abcdef"
    )
    assert namespaced_collection_name("Registered_Embeddings", "ABCDEF").startswith(
        "Registered_Embeddings-"
    )


@pytest.mark.parametrize(
    "base",
    [
        "ab",
        "a" * 488,
        "-invalid",
        "invalid-",
        "bad name",
        "bad/name",
        "bad..name",
        "127.0.0.1",
        "2001:db8::1",
    ],
)
def test_collection_name_rejects_invalid_base(base):
    with pytest.raises(ValueError):
        namespaced_collection_name(base, "a" * 64)


@pytest.mark.parametrize(
    "base",
    [
        "abc",
        "a" * 487,
        "A.B_c-9",
        "999.999.999.999",
    ],
)
def test_collection_name_accepts_valid_boundaries(base):
    name = namespaced_collection_name(base, "a" * 64)

    assert name.startswith(f"{base}-")
    assert len(name) <= 512


def test_negative_version_round_trip(tmp_path):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    expected = CachedEmbeddingGeneration(
        registration_id="model-negative-version",
        source_fingerprint="source-1",
        pipeline_fingerprint="pipeline-1",
        generation_id="g-negative",
        embeddings=((1.0, 0.0),),
        material_no="A0SW1821",
        version=-1,
    )

    store.replace(expected)

    assert (
        store.get(
            "model-negative-version",
            "source-1",
            "pipeline-1",
        )
        == expected
    )


def test_inconsistent_diagnostic_metadata_is_ignored(tmp_path):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    records = [
        metadata(generation_id="bad", roi_index=0),
        metadata(generation_id="bad", roi_index=1),
    ]
    records[1]["material_no"] = "OTHER"
    store._collection.upsert(
        ids=["model-1:bad:0", "model-1:bad:1"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=records,
    )

    assert store.get("model-1", "source-1", "pipeline-1") is None


def test_round_trip_survives_client_reopen(tmp_path):
    first = ChromaRegistrationStore(tmp_path, COLLECTION)
    first.replace(generation())

    reopened = ChromaRegistrationStore(tmp_path, COLLECTION)

    assert reopened.get("model-1", "source-1", "pipeline-1") == generation()


def test_replacement_removes_old_generation(tmp_path):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    store.replace(generation("old"))
    replacement = generation("new", ((0.5, 0.5),))

    store.replace(replacement)

    assert store.get("model-1", "source-1", "pipeline-1") == replacement
    records = store._collection.get(
        where={"registration_id": "model-1"},
        include=["metadatas"],
    )
    assert {item["generation_id"] for item in records["metadatas"]} == {"new"}


def test_same_generation_replacement_removes_stale_roi_records(tmp_path):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    store.replace(
        generation(
            "same",
            ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)),
        )
    )
    replacement = generation("same", ((0.25, 0.75),))

    store.replace(replacement)

    assert store.get("model-1", "source-1", "pipeline-1") == replacement
    records = store._collection.get(
        where={"registration_id": "model-1"},
        include=["metadatas"],
    )
    assert records["ids"] == ["model-1:same:0"]


def test_partial_newer_generation_is_ignored(tmp_path):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    store.replace(generation("complete"))
    store._collection.upsert(
        ids=["model-1:partial:0"],
        embeddings=[[0.5, 0.5]],
        metadatas=[
            metadata(
                generation_id="partial",
                roi_index=0,
                roi_count=2,
                created_at=999999999999999999,
            )
        ],
    )

    loaded = store.get("model-1", "source-1", "pipeline-1")

    assert loaded is not None
    assert loaded.generation_id == "complete"


@pytest.mark.parametrize(
    ("records", "embeddings"),
    [
        (
            [
                metadata(generation_id="bad", roi_index=0),
                metadata(generation_id="bad", roi_index=1, vector_dimension=3),
            ],
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            [
                metadata(generation_id="bad", roi_index=0),
                metadata(generation_id="bad", roi_index=1, roi_count=3),
            ],
            [[1.0, 0.0], [0.0, 1.0]],
        ),
        (
            [
                metadata(generation_id="bad", roi_index=0),
                metadata(generation_id="bad", roi_index=1),
            ],
            [[1.0, 0.0], [0.0, 0.0]],
        ),
    ],
)
def test_malformed_generation_is_ignored(tmp_path, records, embeddings):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    store._collection.upsert(
        ids=[f"model-1:bad:{index}" for index in range(len(records))],
        embeddings=embeddings,
        metadatas=records,
    )

    assert store.get("model-1", "source-1", "pipeline-1") is None


def test_newest_complete_generation_is_returned_deterministically(tmp_path):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    for generation_id, created_at, vector in (
        ("older", 10, [1.0, 0.0]),
        ("newer", 20, [0.0, 1.0]),
    ):
        store._collection.upsert(
            ids=[f"model-1:{generation_id}:0"],
            embeddings=[vector],
            metadatas=[
                metadata(
                    generation_id=generation_id,
                    roi_index=0,
                    roi_count=1,
                    created_at=created_at,
                )
            ],
        )

    loaded = store.get("model-1", "source-1", "pipeline-1")

    assert loaded is not None
    assert loaded.generation_id == "newer"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("roi_count", True),
        ("roi_count", 0),
        ("roi_count", -1),
        ("roi_count", 1.9),
        ("roi_count", math.nan),
        ("roi_count", 10**400),
        ("vector_dimension", False),
        ("vector_dimension", 0),
        ("vector_dimension", -1),
        ("vector_dimension", 2.1),
        ("vector_dimension", math.inf),
        ("vector_dimension", 10**400),
        ("roi_index", True),
        ("roi_index", -1),
        ("roi_index", 0.5),
        ("roi_index", math.nan),
        ("roi_index", 10**400),
        ("created_at", True),
        ("created_at", math.nan),
        ("created_at", math.inf),
        ("created_at", -math.inf),
        ("created_at", 10**400),
        ("version", True),
        ("version", 1.5),
        ("version", math.nan),
        ("version", math.inf),
        ("version", MIN_INT64 - 1),
        ("version", MAX_INT64 + 1),
    ],
)
def test_strictly_rejects_malformed_numeric_metadata(
    field, invalid_value
):
    item = metadata(
        generation_id="bad",
        roi_index=0,
        roi_count=1,
    )
    item[field] = invalid_value
    parsed = ChromaRegistrationStore._parse_generation(
        "bad",
        [("model-1:bad:0", item, [1.0, 0.0])],
    )

    assert parsed is None


def test_equal_timestamps_use_generation_id_as_deterministic_tiebreaker(
    tmp_path,
):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    for generation_id, vector in (
        ("generation-a", [1.0, 0.0]),
        ("generation-b", [0.0, 1.0]),
    ):
        store._collection.upsert(
            ids=[f"model-1:{generation_id}:0"],
            embeddings=[vector],
            metadatas=[
                metadata(
                    generation_id=generation_id,
                    roi_index=0,
                    roi_count=1,
                    created_at=10,
                )
            ],
        )

    loaded = store.get("model-1", "source-1", "pipeline-1")

    assert loaded is not None
    assert loaded.generation_id == "generation-b"


def test_partial_replacement_preserves_previous_complete_generation(
    tmp_path, monkeypatch
):
    store = ChromaRegistrationStore(tmp_path, COLLECTION)
    old = generation("old")
    store.replace(old)
    real_upsert = store._collection.upsert

    def partial_upsert(**kwargs: Any) -> None:
        real_upsert(
            ids=kwargs["ids"][:1],
            embeddings=kwargs["embeddings"][:1],
            metadatas=kwargs["metadatas"][:1],
        )

    monkeypatch.setattr(store._collection, "upsert", partial_upsert)

    with pytest.raises(RuntimeError, match="complete"):
        store.replace(generation("new"))

    assert store.get("model-1", "source-1", "pipeline-1") == old
