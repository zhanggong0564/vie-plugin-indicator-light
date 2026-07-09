from .models import (
    CachedEmbeddingGeneration,
    RegistrationDescriptor,
    pipeline_fingerprint,
)


def __getattr__(name: str):
    """Keep legacy explicit imports lazy while the package import stays lightweight."""
    if name == "RegistrationResolver":
        from .resolver import RegistrationResolver

        return RegistrationResolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CachedEmbeddingGeneration",
    "RegistrationDescriptor",
    "pipeline_fingerprint",
]
