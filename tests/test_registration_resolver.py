from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vie_plugin_indicator_light.registration import (
    CachedEmbeddingGeneration,
    RegistrationDescriptor,
    RegistrationResolver,
)
from vie_plugin_indicator_light.registration import resolver as resolver_module


def _descriptor(**overrides) -> RegistrationDescriptor:
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


def _generation(descriptor: RegistrationDescriptor, pipeline: str = "pipeline"):
    return CachedEmbeddingGeneration.create(
        descriptor.registration_id,
        descriptor.source_fingerprint,
        pipeline,
        [[1.0, 0.0], [0.0, 1.0]],
        material_no=descriptor.material_no,
        version=descriptor.version,
    )


def _resolver(store, downloader=None, infer=None, **kwargs) -> RegistrationResolver:
    return RegistrationResolver(
        store=store,
        downloader=downloader or Mock(return_value=object()),
        infer=infer or Mock(return_value=SimpleNamespace(embeddings=[[1.0, 0.0]])),
        pipeline_fingerprint=kwargs.pop("pipeline_fingerprint", "pipeline"),
        download_options=kwargs.pop("download_options", {"timeout": (1.0, 2.0)}),
        **kwargs,
    )


def test_cache_hit_skips_download_inference_and_write():
    descriptor = _descriptor()
    generation = _generation(descriptor)
    store = Mock()
    store.get.return_value = generation
    downloader = Mock()
    infer = Mock()

    result = _resolver(store, downloader, infer).resolve(descriptor)

    assert result == generation.embeddings
    store.get.assert_called_once_with(
        descriptor.registration_id,
        descriptor.source_fingerprint,
        "pipeline",
    )
    downloader.assert_not_called()
    infer.assert_not_called()
    store.replace.assert_not_called()


@pytest.mark.parametrize("mismatch", ["source", "pipeline"])
def test_fingerprint_mismatch_is_treated_as_cache_miss(mismatch):
    descriptor = _descriptor()
    generation = CachedEmbeddingGeneration.create(
        descriptor.registration_id,
        "different" if mismatch == "source" else descriptor.source_fingerprint,
        "different" if mismatch == "pipeline" else "pipeline",
        [[1.0, 0.0]],
    )
    store = Mock()
    store.get.side_effect = [generation, None]
    downloader = Mock(return_value="image")
    infer = Mock(return_value=SimpleNamespace(embeddings=[[0.0, 1.0]]))

    result = _resolver(store, downloader, infer).resolve(descriptor)

    assert result == ((0.0, 1.0),)
    downloader.assert_called_once_with(
        descriptor.model_file,
        timeout=(1.0, 2.0),
    )
    infer.assert_called_once_with("image")
    store.replace.assert_called_once()


def test_cache_miss_downloads_infers_validates_and_writes_once():
    descriptor = _descriptor()
    store = Mock()
    store.get.return_value = None
    downloader = Mock(return_value="image")
    infer = Mock(return_value=SimpleNamespace(embeddings=[[1, 0], [0, 2]]))

    result = _resolver(store, downloader, infer).resolve(descriptor)

    assert result == ((1.0, 0.0), (0.0, 2.0))
    assert store.get.call_count == 2
    downloader.assert_called_once_with(
        descriptor.model_file,
        timeout=(1.0, 2.0),
    )
    infer.assert_called_once_with("image")
    generation = store.replace.call_args.args[0]
    assert generation.registration_id == descriptor.registration_id
    assert generation.source_fingerprint == descriptor.source_fingerprint
    assert generation.pipeline_fingerprint == "pipeline"
    assert generation.material_no == descriptor.material_no
    assert generation.version == descriptor.version


@pytest.mark.parametrize("field", ["material_no", "version"])
def test_diagnostic_metadata_mismatch_is_treated_as_cache_miss(field):
    descriptor = _descriptor()
    values = {
        "material_no": descriptor.material_no,
        "version": descriptor.version,
    }
    values[field] = "OTHER" if field == "material_no" else 2
    stale = CachedEmbeddingGeneration.create(
        descriptor.registration_id,
        descriptor.source_fingerprint,
        "pipeline",
        [[1.0, 0.0]],
        **values,
    )
    store = Mock()
    store.get.side_effect = [stale, None]

    _resolver(store).resolve(descriptor)

    assert store.replace.call_args.args[0].material_no == descriptor.material_no


def test_write_failure_returns_fresh_embeddings():
    descriptor = _descriptor()
    store = Mock()
    store.get.return_value = None
    store.replace.side_effect = RuntimeError("write unavailable")

    result = _resolver(store).resolve(descriptor)

    assert result == ((1.0, 0.0),)


@pytest.mark.parametrize("stage", ["read", "write"])
def test_store_failure_logs_do_not_include_exception_message(stage, monkeypatch):
    secret = "[0.123456, 0.654321]-sensitive-vector"
    descriptor = _descriptor()
    store = Mock()
    store.get.return_value = None
    if stage == "read":
        store.get.side_effect = RuntimeError(secret)
    else:
        store.replace.side_effect = RuntimeError(secret)
    warning = Mock()
    monkeypatch.setattr(resolver_module.vision_logger, "warning", warning)

    _resolver(store).resolve(descriptor)

    assert warning.call_count >= 1
    rendered_calls = " ".join(
        repr(value)
        for call in warning.call_args_list
        for value in (call.args, call.kwargs)
    )
    assert secret not in rendered_calls
    assert descriptor.registration_id in rendered_calls
    assert f"stage={stage}" in rendered_calls
    assert "RuntimeError" in rendered_calls


def test_read_failure_falls_back_to_registration():
    descriptor = _descriptor()
    store = Mock()
    store.get.side_effect = RuntimeError("read unavailable")

    result = _resolver(store).resolve(descriptor)

    assert result == ((1.0, 0.0),)
    assert store.get.call_count == 2
    store.replace.assert_called_once()


@pytest.mark.parametrize(
    "inference_result",
    [SimpleNamespace(embeddings=None), SimpleNamespace()],
)
def test_missing_embeddings_propagates_validation_failure(inference_result):
    store = Mock()
    store.get.return_value = None
    resolver = _resolver(store, infer=Mock(return_value=inference_result))

    with pytest.raises(ValueError, match="inference result must expose non-null embeddings"):
        resolver.resolve(_descriptor())

    store.replace.assert_not_called()


def test_download_and_inference_failures_propagate():
    descriptor = _descriptor()
    store = Mock()
    store.get.return_value = None

    with pytest.raises(RuntimeError, match="download"):
        _resolver(
            store,
            downloader=Mock(side_effect=RuntimeError("download failed")),
        ).resolve(descriptor)

    with pytest.raises(RuntimeError, match="inference"):
        _resolver(
            store,
            infer=Mock(side_effect=RuntimeError("inference failed")),
        ).resolve(descriptor)


@pytest.mark.parametrize("failure_stage", ["download", "inference", "validation"])
def test_lock_registry_is_released_after_registration_failure(failure_stage):
    store = Mock()
    store.get.return_value = None
    downloader = Mock(return_value="image")
    infer = Mock(return_value=SimpleNamespace(embeddings=[[1.0, 0.0]]))
    if failure_stage == "download":
        downloader.side_effect = RuntimeError("download failed")
    elif failure_stage == "inference":
        infer.side_effect = RuntimeError("inference failed")
    else:
        infer.return_value = SimpleNamespace(embeddings=[[0.0, 0.0]])
    resolver = _resolver(store, downloader=downloader, infer=infer)

    expected_error = ValueError if failure_stage == "validation" else RuntimeError
    with pytest.raises(expected_error):
        resolver.resolve(_descriptor())

    assert resolver.lock_registry_size == 0


def test_concurrent_same_id_misses_register_only_once():
    descriptor = _descriptor()
    stored = {}
    store_lock = threading.Lock()

    class Store:
        def get(self, registration_id, source_fingerprint, pipeline_fingerprint):
            with store_lock:
                return stored.get(registration_id)

        def replace(self, generation):
            with store_lock:
                stored[generation.registration_id] = generation

    download_started = threading.Event()
    release_download = threading.Event()

    def download(url, **kwargs):
        download_started.set()
        assert release_download.wait(timeout=1)
        return "image"

    downloader = Mock(side_effect=download)
    infer = Mock(return_value=SimpleNamespace(embeddings=[[1.0, 0.0]]))
    resolver = _resolver(Store(), downloader, infer)

    with ThreadPoolExecutor(max_workers=8) as executor:
        first = executor.submit(resolver.resolve, descriptor)
        assert download_started.wait(timeout=1)
        remaining = [
            executor.submit(resolver.resolve, descriptor)
            for _ in range(15)
        ]
        release_download.set()
        results = [first.result(), *(future.result() for future in remaining)]

    assert results == [((1.0, 0.0),)] * 16
    assert downloader.call_count == 1
    assert infer.call_count == 1
    assert resolver.lock_registry_size == 0


def test_two_resolvers_share_same_id_registration_lock():
    descriptor = _descriptor()
    stored = {}
    store_lock = threading.Lock()
    get_count = 0
    second_resolver_checked_cache = threading.Event()
    download_started = threading.Event()
    release_download = threading.Event()

    class Store:
        def get(self, registration_id, source_fingerprint, pipeline_fingerprint):
            nonlocal get_count
            with store_lock:
                get_count += 1
                if get_count >= 3:
                    second_resolver_checked_cache.set()
                return stored.get(registration_id)

        def replace(self, generation):
            with store_lock:
                stored[generation.registration_id] = generation

    def download(url, **kwargs):
        download_started.set()
        assert release_download.wait(timeout=1)
        return "image"

    store = Store()
    downloader = Mock(side_effect=download)
    infer = Mock(return_value=SimpleNamespace(embeddings=[[1.0, 0.0]]))
    first_resolver = _resolver(store, downloader, infer)
    second_resolver = _resolver(store, downloader, infer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_resolver.resolve, descriptor)
        assert download_started.wait(timeout=1)
        second = executor.submit(second_resolver.resolve, descriptor)
        assert second_resolver_checked_cache.wait(timeout=1)
        release_download.set()
        results = [first.result(), second.result()]

    assert results == [((1.0, 0.0),), ((1.0, 0.0),)]
    assert downloader.call_count == 1
    assert infer.call_count == 1
    assert first_resolver.lock_registry_size == 0
    assert second_resolver.lock_registry_size == 0


def test_lock_registry_reclaims_entry_when_lock_acquire_raises():
    class InterruptedLock:
        def acquire(self):
            raise KeyboardInterrupt

        def release(self):
            pytest.fail("an unacquired lock must not be released")

    registry = resolver_module._RegistrationLockRegistry()
    entry = resolver_module._LockEntry(lock=InterruptedLock())
    registry._entries["registration-1"] = entry

    with pytest.raises(KeyboardInterrupt):
        with registry.acquire("registration-1"):
            pytest.fail("context body must not run")

    assert registry.size == 0


def test_different_registration_ids_do_not_serialize():
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    barrier = threading.Barrier(2)
    store = Mock()
    store.get.return_value = None

    def download(url, **kwargs):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=1)
        with active_lock:
            active -= 1
        return "image"

    resolver = _resolver(store, Mock(side_effect=download))
    descriptors = [
        _descriptor(registration_id="registration-1"),
        _descriptor(registration_id="registration-2"),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(resolver.resolve, descriptors))

    assert results == [((1.0, 0.0),), ((1.0, 0.0),)]
    assert max_active == 2
    assert resolver.lock_registry_size == 0


def test_lock_registry_does_not_retain_sequential_registration_ids():
    store = Mock()
    store.get.return_value = None
    resolver = _resolver(store)

    for index in range(100):
        resolver.resolve(_descriptor(registration_id=f"registration-{index}"))

    assert resolver.lock_registry_size == 0
