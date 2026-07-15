from dataclasses import FrozenInstanceError

import pytest

from vie_plugin_indicator_light.registration.models import (
    CachedEmbeddingGeneration,
    RegistrationDescriptor,
    normalize_embeddings,
    pipeline_fingerprint,
)


def _descriptor(**overrides):
    values = {
        "registration_id": "registration-1",
        "material_no": "A0SW1821",
        "product_name": "A0SW1821",
        "version": 1,
        "model_file": "http://example.test/registered.jpg",
        "create_time": "2026-07-01T08:39:56",
        "update_time": "2026-07-01T08:39:56",
        "register_mode": False,
    }
    values.update(overrides)
    return RegistrationDescriptor(**values)


def test_source_fingerprint_is_stable_and_ignores_request_only_fields():
    first = _descriptor()
    second = _descriptor(material_no="OTHER", register_mode=True)

    assert first.source_fingerprint == second.source_fingerprint
    assert len(first.source_fingerprint) == 64

    with pytest.raises(FrozenInstanceError):
        first.version = 2


@pytest.mark.parametrize("register_mode", [None, False, True])
def test_descriptor_preserves_optional_register_mode(register_mode):
    assert _descriptor(register_mode=register_mode).register_mode is register_mode


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registration_id", "registration-2"),
        ("product_name", "different-product"),
        ("version", 2),
        ("model_file", "http://example.test/replacement.jpg"),
        ("create_time", "2026-07-02T08:39:56"),
        ("update_time", "2026-07-02T08:39:56"),
    ],
)
def test_source_fingerprint_changes_with_canonical_source_fields(field, value):
    assert _descriptor().source_fingerprint != _descriptor(**{field: value}).source_fingerprint


def test_generation_create_normalizes_vectors_and_generates_identity():
    generation = CachedEmbeddingGeneration.create(
        registration_id="registration-1",
        source_fingerprint="source-fingerprint",
        pipeline_fingerprint="pipeline-fingerprint",
        embeddings=[[1, 2], [3.5, 4]],
        material_no="A0SW1821",
        version=1,
    )

    assert generation.embeddings == ((1.0, 2.0), (3.5, 4.0))
    assert generation.vector_dimension == 2
    assert generation.material_no == "A0SW1821"
    assert generation.version == 1
    assert len(generation.generation_id) == 32
    assert generation.generation_id != CachedEmbeddingGeneration.create(
        registration_id="registration-1",
        source_fingerprint="source-fingerprint",
        pipeline_fingerprint="pipeline-fingerprint",
        embeddings=[[1, 2]],
    ).generation_id

    with pytest.raises(FrozenInstanceError):
        generation.generation_id = "replacement"


def test_normalize_embeddings_accepts_legacy_singleton_wrappers():
    assert normalize_embeddings([[[1, 2]], [[3, 4]]]) == (
        (1.0, 2.0),
        (3.0, 4.0),
    )


def test_generation_create_accepts_legacy_singleton_wrappers():
    generation = CachedEmbeddingGeneration.create(
        registration_id="registration-1",
        source_fingerprint="source-fingerprint",
        pipeline_fingerprint="pipeline-fingerprint",
        embeddings=[[[1, 2]]],
    )

    assert generation.embeddings == ((1.0, 2.0),)


@pytest.mark.parametrize(
    "embeddings",
    [
        [[1, 2], [[3, 4]]],
        [[[1, 2], [3, 4]]],
        [1],
        [[]],
        [[1, 2], [3]],
        [[0, 0]],
        [[1, float("nan")]],
        [[1, float("inf")]],
    ],
)
def test_normalize_embeddings_rejects_invalid_shapes_and_values(embeddings):
    with pytest.raises((TypeError, ValueError)):
        normalize_embeddings(embeddings)


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        ([], "empty"),
        ([[]], "zero-dimensional"),
        ([[1, 2], [3]], "inconsistent"),
        ([[0, 0]], "all-zero"),
        ([[1, 0], [0, 0]], "all-zero"),
    ],
)
def test_generation_rejects_invalid_vectors(embeddings, message):
    with pytest.raises(ValueError, match=message):
        CachedEmbeddingGeneration.create(
            registration_id="registration-1",
            source_fingerprint="source-fingerprint",
            pipeline_fingerprint="pipeline-fingerprint",
            embeddings=embeddings,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_generation_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="finite"):
        CachedEmbeddingGeneration.create(
            registration_id="registration-1",
            source_fingerprint="source-fingerprint",
            pipeline_fingerprint="pipeline-fingerprint",
            embeddings=[[1, value]],
        )


def test_pipeline_fingerprint_changes_when_either_model_changes(tmp_path):
    det_model = tmp_path / "det.onnx"
    rec_model = tmp_path / "rec.onnx"
    det_model.write_bytes(b"det-v1")
    rec_model.write_bytes(b"rec-v1")

    original = pipeline_fingerprint(det_model, rec_model)
    assert original == pipeline_fingerprint(det_model, rec_model)
    assert len(original) == 64

    det_model.write_bytes(b"det-v2")
    changed_det = pipeline_fingerprint(det_model, rec_model)
    assert changed_det != original

    det_model.write_bytes(b"det-v1")
    rec_model.write_bytes(b"rec-v2")
    assert pipeline_fingerprint(det_model, rec_model) != original
