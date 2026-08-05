from .models import (
    CachedEmbeddingGeneration,
    RegistrationDescriptor,
    pipeline_fingerprint,
)


def __getattr__(name: str):
    """Keep legacy explicit imports lazy while the package import stays lightweight."""
    if name in ("RegistrationResolver", "ResolvedRegistration"):
        from .resolver import RegistrationResolver, ResolvedRegistration

        return {
            "RegistrationResolver": RegistrationResolver,
            "ResolvedRegistration": ResolvedRegistration,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CachedEmbeddingGeneration",
    "RegistrationDescriptor",
    "ResolvedRegistration",
    "pipeline_fingerprint",
]
