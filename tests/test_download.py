import cv2
import numpy as np
import pytest

from schemas.exceptions import InvalidImageError
from vie_plugin_indicator_light.download import download_image


class _FakeResponse:
    def __init__(self, chunks, status_error=None):
        self._chunks = chunks
        self._status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def iter_content(self, chunk_size):
        assert chunk_size == 64 * 1024
        yield from self._chunks


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.timeout = None
        self.allow_redirects = None

    def get(self, url, **kwargs):
        self.timeout = kwargs["timeout"]
        self.allow_redirects = kwargs["allow_redirects"]
        assert kwargs["stream"] is True
        return self.response


def _public_resolver(host, port):
    assert host == "example.com"
    assert port == 443
    return ["93.184.216.34"]


def test_rejects_non_http_scheme():
    with pytest.raises(InvalidImageError, match="URL"):
        download_image("file:///etc/passwd")


def test_rejects_invalid_port():
    with pytest.raises(InvalidImageError, match="URL"):
        download_image("https://example.com:not-a-port/image.jpg")


def test_rejects_private_resolved_address():
    with pytest.raises(InvalidImageError, match="内网"):
        download_image(
            "http://example.com/image.jpg",
            resolver=lambda host, port: ["127.0.0.1"],
        )


def test_streams_with_timeout_and_size_limit():
    session = _FakeSession(_FakeResponse([b"a" * 6, b"b" * 6]))
    with pytest.raises(InvalidImageError, match="过大"):
        download_image(
            "https://example.com/image.jpg",
            max_bytes=10,
            timeout=(3.0, 10.0),
            session=session,
            resolver=_public_resolver,
        )
    assert session.timeout == (3.0, 10.0)
    assert session.allow_redirects is False


def test_decodes_valid_image():
    source = np.full((4, 5, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", source)
    assert ok
    session = _FakeSession(_FakeResponse([encoded.tobytes()]))

    image = download_image(
        "https://example.com/image.jpg",
        max_bytes=1024 * 1024,
        session=session,
        resolver=_public_resolver,
    )

    assert image.shape == source.shape


def test_rejects_invalid_image_bytes():
    session = _FakeSession(_FakeResponse([b"not-an-image"]))
    with pytest.raises(InvalidImageError, match="解码失败"):
        download_image(
            "https://example.com/image.jpg",
            max_bytes=1024,
            session=session,
            resolver=_public_resolver,
        )


def test_rejects_host_not_in_allowed_hosts():
    with pytest.raises(InvalidImageError, match="不在允许列表"):
        download_image(
            "https://evil.example/image.jpg",
            allowed_hosts=("example.com",),
            resolver=lambda host, port: ["93.184.216.34"],
        )


def test_allows_host_in_allowed_hosts_and_decodes_image():
    source = np.full((4, 5, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", source)
    assert ok
    session = _FakeSession(_FakeResponse([encoded.tobytes()]))

    image = download_image(
        "https://example.com/image.jpg",
        allowed_hosts=("example.com",),
        max_bytes=1024 * 1024,
        session=session,
        resolver=_public_resolver,
    )

    assert image.shape == source.shape
