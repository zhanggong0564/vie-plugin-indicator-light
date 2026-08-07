from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Protocol

from utils import vision_logger

from .models import CachedEmbeddingGeneration, RegistrationDescriptor
from .store import RegistrationVectorStore


@dataclass
class _LockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class _RegistrationLockRegistry:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, _LockEntry] = {}

    @contextmanager
    def acquire(self, registration_id: str) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(registration_id)
            if entry is None:
                entry = _LockEntry()
                self._entries[registration_id] = entry
            entry.users += 1

        acquired = False
        try:
            entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(registration_id, None)

    @property
    def size(self) -> int:
        with self._guard:
            return len(self._entries)


_PROCESS_REGISTRATION_LOCKS = _RegistrationLockRegistry()


class InferenceResult(Protocol):
    embeddings: Any


class RegistrationImageDownloader(Protocol):
    def __call__(self, url: str, **options: Any) -> Any: ...


class RegistrationInfer(Protocol):
    def __call__(self, image: Any) -> InferenceResult: ...


@dataclass(frozen=True)
class ResolvedRegistration:
    generation: CachedEmbeddingGeneration
    image: Any | None = None


class RegistrationResolver:
    def __init__(
        self,
        store: RegistrationVectorStore,
        downloader: RegistrationImageDownloader,
        infer: RegistrationInfer,
        pipeline_fingerprint: str,
        download_options: Mapping[str, Any],
        lock_registry: _RegistrationLockRegistry | None = None,
        image_callback: Callable[[RegistrationDescriptor, Any], None] | None = None,
        image_required: Callable[[RegistrationDescriptor], bool] | None = None,
    ) -> None:
        self._store = store
        self._downloader = downloader
        self._infer = infer
        self._pipeline_fingerprint = pipeline_fingerprint
        self._download_options = dict(download_options)
        self._locks = lock_registry or _PROCESS_REGISTRATION_LOCKS
        self._image_callback = image_callback
        self._image_required = image_required

    def resolve(
        self,
        descriptor: RegistrationDescriptor,
    ) -> ResolvedRegistration:
        force_refresh = descriptor.register_mode is True
        if not force_refresh:
            cached = self._safe_get(descriptor)
            if cached is not None:
                vision_logger.info(
                    "注册向量缓存命中: registration_id={} source={}",
                    descriptor.registration_id,
                    descriptor.source_fingerprint[:12],
                )
                self._archive_cached_registration_if_needed(descriptor)
                return ResolvedRegistration(cached)

        with self._locks.acquire(descriptor.registration_id):
            if not force_refresh:
                cached = self._safe_get(descriptor)
                if cached is not None:
                    vision_logger.info(
                        "注册向量缓存等待后命中: registration_id={} source={}",
                        descriptor.registration_id,
                        descriptor.source_fingerprint[:12],
                    )
                    self._archive_cached_registration_if_needed(descriptor)
                    return ResolvedRegistration(cached)

            vision_logger.info(
                "注册向量刷新: registration_id={} reason={} source={}",
                descriptor.registration_id,
                "register_true" if force_refresh else "cache_miss_or_source_changed",
                descriptor.source_fingerprint[:12],
            )
            image = self._downloader(
                descriptor.model_file,
                **self._download_options,
            )
            if self._image_callback is not None:
                try:
                    self._image_callback(descriptor, image)
                except Exception as exc:
                    vision_logger.warning(
                        "注册图归档失败: registration_id={}, error_type={}",
                        descriptor.registration_id,
                        type(exc).__name__,
                    )
            inference_result = self._infer(image)
            embeddings = getattr(inference_result, "embeddings", None)
            boxes = getattr(inference_result, "boxes", None)
            image_shape = getattr(inference_result, "image_shape", None)
            if embeddings is None or boxes is None or image_shape is None:
                raise ValueError(
                    "inference result must expose embeddings, boxes, and image_shape"
                )
            generation = CachedEmbeddingGeneration.create(
                registration_id=descriptor.registration_id,
                source_fingerprint=descriptor.source_fingerprint,
                pipeline_fingerprint=self._pipeline_fingerprint,
                embeddings=embeddings,
                boxes=boxes,
                image_shape=image_shape,
                material_no=descriptor.material_no,
                version=descriptor.version,
            )
            self._safe_replace(generation)
            return ResolvedRegistration(generation, image)

    def download_image(self, descriptor: RegistrationDescriptor) -> Any:
        image = self._downloader(descriptor.model_file, **self._download_options)
        if self._image_callback is not None:
            try:
                self._image_callback(descriptor, image)
            except Exception as exc:
                vision_logger.warning(
                    "注册图归档失败: registration_id={}, error_type={}",
                    descriptor.registration_id,
                    type(exc).__name__,
                )
        return image

    def _archive_cached_registration_if_needed(
        self,
        descriptor: RegistrationDescriptor,
    ) -> None:
        if self._image_required is None or not self._image_required(descriptor):
            return
        try:
            image = self._downloader(descriptor.model_file, **self._download_options)
            if self._image_callback is not None:
                self._image_callback(descriptor, image)
        except Exception as exc:
            vision_logger.warning(
                "缓存注册图归档失败: registration_id={}, error_type={}",
                descriptor.registration_id,
                type(exc).__name__,
            )

    def _safe_get(
        self,
        descriptor: RegistrationDescriptor,
    ) -> CachedEmbeddingGeneration | None:
        try:
            generation = self._store.get(
                descriptor.registration_id,
                descriptor.source_fingerprint,
                self._pipeline_fingerprint,
            )
        except Exception as exc:
            vision_logger.warning(
                "注册向量缓存读取失败: registration_id={}, stage=read, error_type={}",
                descriptor.registration_id,
                type(exc).__name__,
            )
            return None

        if generation is None:
            return None
        if (
            generation.registration_id != descriptor.registration_id
            or generation.source_fingerprint != descriptor.source_fingerprint
            or generation.pipeline_fingerprint != self._pipeline_fingerprint
            or generation.material_no != descriptor.material_no
            or generation.version != descriptor.version
        ):
            return None
        return generation

    def _safe_replace(self, generation: CachedEmbeddingGeneration) -> None:
        try:
            self._store.replace(generation)
        except Exception as exc:
            vision_logger.warning(
                "注册向量缓存写入失败: registration_id={}, stage=write, error_type={}",
                generation.registration_id,
                type(exc).__name__,
            )

    @property
    def lock_registry_size(self) -> int:
        return self._locks.size
