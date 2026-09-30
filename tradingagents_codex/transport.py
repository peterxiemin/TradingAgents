"""Explicit, narrow transport configuration for isolated worker processes.

Nothing is inherited merely by importing this module. A caller must opt in to
``read_transport_env`` and pass its result to the worker. Only credential-free
loopback proxies and existing CA paths are supported; TLS verification is never
disabled. Values are deliberately omitted from validation errors.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

_PROXY_KEYS = frozenset({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"})
_CA_FILE_KEYS = frozenset({"SSL_CERT_FILE", "CODEX_CA_CERTIFICATE", "REQUESTS_CA_BUNDLE"})
_CA_DIR_KEYS = frozenset({"SSL_CERT_DIR"})
_TRANSPORT_KEYS = _PROXY_KEYS | _CA_FILE_KEYS | _CA_DIR_KEYS
_PROXY_SCHEMES = frozenset({"http", "https", "socks5", "socks5h"})


def _validate_proxy(value: str) -> str:
    error = "Transport proxy must be a credential-free loopback URL"
    # urlsplit normalizes some control characters. Reject them before parsing,
    # as well as characters that different HTTP clients interpret differently.
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(error)
    if any(char in value for char in ("\\", "?", "#", "@", "%")):
        raise ValueError(error)
    try:
        url = urlsplit(value)
        hostname = url.hostname
        port = url.port
        if (
            url.scheme not in _PROXY_SCHEMES
            or not hostname
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
            or url.netloc.endswith(":")
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError(error)
        if hostname != "localhost" and not ipaddress.ip_address(hostname).is_loopback:
            raise ValueError(error)
        authority = f"[{hostname}]" if ":" in hostname else hostname
        if port is not None:
            authority += f":{port}"
        if url.netloc.lower() != authority:
            raise ValueError(error)
    except ValueError:
        raise ValueError(error) from None
    return value


def _validate_ca_path(value: str, *, directory: bool) -> str:
    error = "Transport CA path must be an existing readable absolute file or directory"
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(error)
    try:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(error)
        path = path.resolve(strict=True)
        valid_type = path.is_dir() if directory else path.is_file()
        required_access = os.R_OK | os.X_OK if directory else os.R_OK
        if not valid_type or not os.access(path, required_access):
            raise ValueError(error)
    except (OSError, RuntimeError, ValueError):
        raise ValueError(error) from None
    return str(path)


def validate_transport_env(mapping: Mapping[str, str] | None) -> dict[str, str]:
    """Copy and validate explicitly supplied settings, without network access.

    Filesystem metadata is checked for CA paths, but their contents are not read.
    Proxy names must identify localhost or a literal loopback address; no DNS
    resolution, network requests, authentication, or global settings are changed.
    """
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping) or any(key not in _TRANSPORT_KEYS for key in mapping):
        raise ValueError("Unsupported transport environment setting")
    result = {}
    for key, value in mapping.items():
        if not isinstance(value, str) or not value:
            raise ValueError("Transport settings require nonempty string values")
        if key in _PROXY_KEYS:
            result[key] = _validate_proxy(value)
        else:
            result[key] = _validate_ca_path(value, directory=key in _CA_DIR_KEYS)
    return result


def read_transport_env() -> dict[str, str]:
    """Opt in to reading only the documented proxy/CA environment allowlist."""
    return validate_transport_env(
        {key: os.environ[key] for key in _TRANSPORT_KEYS if key in os.environ}
    )
