from __future__ import annotations

from typing import Protocol

from .models import CachedEmbeddingGeneration


class RegistrationVectorStore(Protocol):
    def get(
        self,
        registration_id: str,
        source_fingerprint: str,
        pipeline_fingerprint: str,
    ) -> CachedEmbeddingGeneration | None:
        ...

    def replace(self, generation: CachedEmbeddingGeneration) -> None:
        ...


class NullRegistrationStore:
    """No-op store used when persistent registration caching is unavailable."""

    def get(
        self,
        registration_id: str,
        source_fingerprint: str,
        pipeline_fingerprint: str,
    ) -> None:
        return None

    def replace(self, generation: CachedEmbeddingGeneration) -> None:
        return None
