"""指示灯注册参考图的受控下载与解码。"""

import ipaddress
import socket
from typing import Callable, Iterable, Optional
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests

from schemas.exceptions import InvalidImageError


def resolve_host_addresses(host: str, port: int) -> list[str]:
    """解析主机的全部 IPv4/IPv6 地址，供 SSRF 校验。"""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise InvalidImageError("注册参考图域名解析失败") from exc
    return list({info[4][0] for info in infos})


def download_image(
    url: str,
    *,
    max_bytes: int = 20 * 1024 * 1024,
    timeout: tuple[float, float] = (3.0, 10.0),
    allowed_hosts: Optional[Iterable[str]] = None,
    session=requests,
    resolver: Callable[[str, int], Iterable[str]] = resolve_host_addresses,
) -> np.ndarray:
    """从受控 HTTP(S) 地址下载图片。

    DNS 主机名必须显式列入 ``allowed_hosts``；该配置表示调用方信任该
    名称及其解析结果。IP 字面量直接校验，非公网地址同样必须显式允许。
    """
    try:
        parsed = urlsplit(url)
        hostname_value = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname_value:
            raise InvalidImageError("注册参考图 URL 非法")
        hostname = hostname_value.lower()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise InvalidImageError("注册参考图 URL 非法") from exc
    allowed = {host.strip().lower() for host in (allowed_hosts or []) if host and host.strip()}

    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if not literal_ip.is_global and hostname not in allowed:
            raise InvalidImageError("注册参考图地址不允许访问内网")
    else:
        if hostname not in allowed:
            raise InvalidImageError("注册参考图域名不在允许列表")
        addresses = list(resolver(hostname, port))
        if not addresses:
            raise InvalidImageError("注册参考图域名未解析到有效地址")
        try:
            for address in addresses:
                ipaddress.ip_address(address)
        except ValueError as exc:
            raise InvalidImageError("注册参考图域名解析结果非法") from exc

    try:
        with session.get(
            url,
            stream=True,
            timeout=timeout,
            allow_redirects=False,
        ) as response:
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_content(64 * 1024):
                if not chunk:
                    continue
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise InvalidImageError("注册参考图过大")
    except InvalidImageError:
        raise
    except requests.RequestException as exc:
        raise InvalidImageError(f"注册参考图下载失败: {exc}") from exc

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("注册参考图解码失败")
    return image
