"""Explicit transport opt-in must not become general environment inheritance."""

from __future__ import annotations

import os

import pytest

from tradingagents_codex.transport import (
    _TRANSPORT_KEYS,
    read_transport_env,
    validate_transport_env,
)


@pytest.mark.parametrize(
    "proxy",
    [
        "http://localhost:8080",
        "https://LOCALHOST:8443/",
        "http://127.0.0.1:1234",
        "http://127.42.0.8:1234",
        "http://[::1]:1234",
        "socks5://localhost:1080",
        "socks5h://[0:0:0:0:0:0:0:1]:1080",
        "http://localhost",
    ],
)
@pytest.mark.parametrize("key", ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"])
def test_only_credential_free_loopback_proxies_are_accepted(proxy, key):
    assert validate_transport_env({key: proxy}) == {key: proxy}


@pytest.mark.parametrize(
    "proxy",
    [
        "http://secret-user:secret-password@localhost:8080",
        "http://secret-user@localhost:8080",
        "http://@localhost:8080",
        "http://localhost:8080?password=secret",
        "http://localhost:8080#secret",
        "http://localhost:8080/secret",
        "http://localhost:8080?",
        "http://localhost:8080#",
        "http://example.com:8080",
        "http://10.0.0.1:8080",
        "http://0.0.0.0:8080",
        "http://[::]:8080",
        "http://localhost.evil:8080",
        "http://localhost.:8080",
        "http://127.0.0.1.evil:8080",
        "http://2130706433:8080",
        "http://127.1:8080",
        "file:///etc/passwd",
        "ftp://localhost:8080",
        "localhost:8080",
        "//localhost:8080",
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost:secret",
        "http://localhost:",
        "http://localhost:08080",
        "http://[::1]evil:8080",
        "http://[::1]evil",
        "http://[::1",
        "http://[::1%25zone]:8080",
        "http://local%68ost:8080",
        "http://localhost\\@example.com",
        " http://localhost:8080",
        "http://local\nhost:8080",
        "http://localhost:8080\x00",
        "http://localhost:8080\t",
        "http://localhost:8080\x7f",
        "",
    ],
)
def test_malformed_remote_or_credential_bearing_proxy_fails_closed(proxy):
    with pytest.raises(ValueError) as error:
        validate_transport_env({"HTTPS_PROXY": proxy})
    assert proxy not in str(error.value) or not proxy
    assert "secret" not in str(error.value)


def test_ca_files_and_directory_require_explicit_existing_paths(tmp_path):
    cert = tmp_path / "certificate.pem"
    cert.write_text("fixture contents are not read by transport validation")
    directory = tmp_path / "certificates"
    directory.mkdir()
    mapping = {
        "SSL_CERT_FILE": str(cert),
        "CODEX_CA_CERTIFICATE": str(cert),
        "REQUESTS_CA_BUNDLE": str(cert),
        "SSL_CERT_DIR": str(directory),
    }
    assert validate_transport_env(mapping) == mapping


@pytest.mark.parametrize(
    "value",
    [
        "",
        "relative.pem",
        "~/certificate.pem",
        "/not-present/secret.pem",
        "/not-present/secret\x00.pem",
    ],
)
def test_invalid_ca_paths_fail_without_exposing_values(value):
    with pytest.raises(ValueError) as error:
        validate_transport_env({"SSL_CERT_FILE": value})
    assert "secret" not in str(error.value)


def test_ca_type_and_readability_checked(tmp_path, monkeypatch):
    cert = tmp_path / "certificate.pem"
    cert.write_text("fixture")
    for mapping in ({"SSL_CERT_FILE": str(tmp_path)}, {"SSL_CERT_DIR": str(cert)}):
        with pytest.raises(ValueError):
            validate_transport_env(mapping)
    monkeypatch.setattr(os, "access", lambda *_: False)
    with pytest.raises(ValueError):
        validate_transport_env({"SSL_CERT_FILE": str(cert)})


@pytest.mark.parametrize(
    "key",
    [
        "OPENAI_API_KEY",
        "CODEX_HOME",
        "HOME",
        "PATH",
        "NO_PROXY",
        "https_proxy",
        "SSL_VERIFY",
        "PYTHONHTTPSVERIFY",
        "CURL_SSL_VERIFYHOST",
        "LD_PRELOAD",
        "secret-key",
    ],
)
def test_unknown_environment_or_tls_disable_settings_rejected(key):
    with pytest.raises(ValueError) as error:
        validate_transport_env({key: "secret-value"})
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("value", [None, 0, False, [], {}])
def test_non_string_values_rejected(value):
    with pytest.raises(ValueError):
        validate_transport_env({"HTTPS_PROXY": value})


def test_read_environment_is_explicit_and_allowlisted(monkeypatch):
    for key in _TRANSPORT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-must-not-forward")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("PYTHONHTTPSVERIFY", "0")
    assert validate_transport_env(None) == {}
    assert validate_transport_env({}) == {}
    assert read_transport_env() == {"HTTPS_PROXY": "http://127.0.0.1:8080"}


def test_validation_copies_mapping():
    original = {"HTTP_PROXY": "http://localhost:8080"}
    validated = validate_transport_env(original)
    original["HTTP_PROXY"] = "http://secret:secret@localhost"
    assert validated == {"HTTP_PROXY": "http://localhost:8080"}
