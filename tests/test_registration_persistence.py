from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("chromadb")

from vie_plugin_indicator_light.registration.chroma_store import (
    ChromaRegistrationStore,
)
from vie_plugin_indicator_light.registration.models import RegistrationDescriptor
from vie_plugin_indicator_light.registration.resolver import RegistrationResolver


def _descriptor() -> RegistrationDescriptor:
    return RegistrationDescriptor(
        registration_id="registration-1",
        material_no="A0SW1821",
        product_name="A0SW1821",
        version=1,
        model_file="http://example.test/reference.jpg",
        create_time="2026-07-01T08:39:56",
        update_time="2026-07-01T08:39:56",
        register_mode=False,
    )


def _resolver(store, downloader, infer, fingerprint):
    return RegistrationResolver(
        store=store,
        downloader=downloader,
        infer=infer,
        pipeline_fingerprint=fingerprint,
        download_options={"timeout": (1.0, 2.0)},
    )


def test_registration_persists_across_resolver_and_client_reopen(tmp_path):
    fingerprint = "a" * 64
    descriptor = _descriptor()
    downloader = Mock(return_value="registered-image")
    infer = Mock(
        return_value=SimpleNamespace(
            embeddings=[[1.0, 0.0], [0.0, 1.0]]
        )
    )
    first_store = ChromaRegistrationStore(
        tmp_path, "registered_embeddings", fingerprint
    )
    first_resolver = _resolver(first_store, downloader, infer, fingerprint)

    first = first_resolver.resolve(descriptor)
    second = first_resolver.resolve(descriptor)

    assert first == second == ((1.0, 0.0), (0.0, 1.0))
    assert downloader.call_count == 1
    assert infer.call_count == 1

    reopened_store = ChromaRegistrationStore(
        tmp_path, "registered_embeddings", fingerprint
    )
    reopened_downloader = Mock()
    reopened_infer = Mock()
    reopened_resolver = _resolver(
        reopened_store, reopened_downloader, reopened_infer, fingerprint
    )

    third = reopened_resolver.resolve(descriptor)
    cached = reopened_store.get(
        descriptor.registration_id,
        descriptor.source_fingerprint,
        fingerprint,
    )

    assert third == first
    reopened_downloader.assert_not_called()
    reopened_infer.assert_not_called()
    assert cached is not None
    assert cached.material_no == descriptor.material_no
    assert cached.version == descriptor.version
