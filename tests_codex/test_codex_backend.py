"""Offline inference tests using the installed SDK's real protocol models."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from openai_codex.client import CodexClient as RealCodexClient
from openai_codex.generated.v2_all import (
    ConfigReadResponse,
    ItemCompletedNotification,
    ItemStartedNotification,
    ThreadStartParams,
    ThreadStartResponse,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
    TurnStartParams,
    TurnStartResponse,
)
from openai_codex.models import InitializeResponse, Notification

from tradingagents_codex import codex_backend as adapter

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def _effective_config():
    result = {}
    for key, value in adapter._CONFIG_VALUES.items():
        current = result
        *parents, leaf = key.split(".")
        for part in parents:
            current = current.setdefault(part, {})
        current[leaf] = value
    return result


class FakeClient:
    """Fake transport, real typed SDK request/response boundaries."""

    def __init__(self, config, approval_handler, owner):
        inspect.signature(RealCodexClient).bind(config, approval_handler=approval_handler)
        self.config = config
        self.approval_handler = approval_handler
        self.owner = owner
        self.cwd = Path(config.cwd)
        self.thread_id = f"thread-{len(owner.clients)}"
        self.turn_id = f"turn-{len(owner.clients)}"
        self.events = []
        self.started = False
        self.closed = threading.Event()
        self.waiting = threading.Event()
        self.interrupted = []
        self.unregistered = []
        self.thread_params = None
        self.turn_params = None
        owner.clients.append(self)

    def start(self):
        assert self.cwd.is_dir()
        assert list(self.cwd.iterdir()) == []
        assert self.cwd.stat().st_mode & 0o077 == 0
        self.started = True

    def close(self):
        self.closed.set()

    def initialize(self):
        if self.owner.fail_initialize:
            raise RuntimeError("initialization failed")
        return InitializeResponse(userAgent=f"test/{adapter.SUPPORTED_SDK_VERSION} (linux)")

    def request(self, method, params, *, response_model):
        inspect.signature(RealCodexClient.request).bind(
            self,
            method,
            params,
            response_model=response_model,
        )
        assert method == "config/read"
        assert params == {"cwd": str(self.cwd), "includeLayers": False}
        assert response_model is ConfigReadResponse
        config = _effective_config()
        if self.owner.mutate_config:
            self.owner.mutate_config(config)
        return response_model.model_validate({"config": config, "origins": {}})

    def thread_start(self, params):
        inspect.signature(RealCodexClient.thread_start).bind(self, params)
        assert isinstance(params, ThreadStartParams)
        self.thread_params = params
        thread = {
            "id": self.thread_id,
            "sessionId": self.thread_id,
            "cwd": str(self.cwd),
            "ephemeral": True,
            "path": None,
            "cliVersion": adapter.SUPPORTED_SDK_VERSION,
            "createdAt": 0,
            "updatedAt": 0,
            "modelProvider": "openai",
            "preview": "",
            "source": "appServer",
            "status": {"type": "idle"},
            "turns": [],
        }
        response = {
            "thread": thread,
            "cwd": str(self.cwd),
            "model": "fake-model",
            "modelProvider": "openai",
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "sandbox": {"type": "readOnly", "networkAccess": False},
            "instructionSources": [],
        }
        if self.owner.mutate_thread:
            self.owner.mutate_thread(response)
        return ThreadStartResponse.model_validate(response)

    def turn_start(self, thread_id, input_items, params=None):
        inspect.signature(RealCodexClient.turn_start).bind(
            self,
            thread_id,
            input_items,
            params=params,
        )
        assert thread_id == self.thread_id
        assert isinstance(params, TurnStartParams)
        assert input_items == [item.model_dump(mode="json", by_alias=True) for item in params.input]
        self.turn_params = params
        if self.owner.block_turn_start:
            self.waiting.set()
            assert self.closed.wait(5), "test transport was not closed during turn start"
            raise RuntimeError("transport closed during turn start")
        for usage in self.owner.usage_updates:
            self.events.append(
                Notification(
                    "thread/tokenUsage/updated",
                    ThreadTokenUsageUpdatedNotification.model_validate(
                        {
                            "threadId": self.thread_id,
                            "turnId": self.turn_id,
                            "tokenUsage": {"total": usage, "last": dict.fromkeys(usage, 0)},
                        }
                    ),
                )
            )
        message = {
            "type": "agentMessage",
            "id": "message",
            "phase": "final_answer",
            "text": self.owner.response,
        }
        item = self.owner.tool_item or message
        self.events.append(
            Notification(
                "item/started",
                ItemStartedNotification.model_validate(
                    {
                        "threadId": self.thread_id,
                        "turnId": self.turn_id,
                        "startedAtMs": 0,
                        "item": item,
                    }
                ),
            )
        )
        self.events.append(
            Notification(
                "item/completed",
                ItemCompletedNotification.model_validate(
                    {
                        "threadId": self.thread_id,
                        "turnId": self.turn_id,
                        "completedAtMs": 1,
                        "item": item,
                    }
                ),
            )
        )
        self.events.append(
            Notification(
                "turn/completed",
                TurnCompletedNotification.model_validate(
                    {
                        "threadId": self.thread_id,
                        "turn": {"id": self.turn_id, "items": [], "status": self.owner.status},
                    }
                ),
            )
        )
        return TurnStartResponse.model_validate(
            {
                "turn": {"id": self.turn_id, "items": [], "status": "inProgress"},
            }
        )

    def next_turn_notification(self, turn_id):
        assert turn_id == self.turn_id
        if self.owner.block_stream or (
            self.owner.block_after_usage and self.events[0].method != "thread/tokenUsage/updated"
        ):
            self.waiting.set()
            assert self.closed.wait(5), "test transport was not closed during stream"
            raise RuntimeError("transport closed")
        return self.events.pop(0)

    def unregister_turn_notifications(self, turn_id):
        self.unregistered.append(turn_id)

    def turn_interrupt(self, thread_id, turn_id):
        self.interrupted.append((thread_id, turn_id))
        if self.owner.block_interrupt:
            assert self.closed.wait(5), "test transport was not closed after interrupt timeout"
            raise RuntimeError("transport closed during interrupt")


class FakeFactory:
    def __init__(self):
        self.clients = []
        self.response = '{"summary":"test research"}'
        self.status = "completed"
        self.block_stream = False
        self.block_turn_start = False
        self.block_interrupt = False
        self.block_after_usage = False
        self.usage_updates = []
        self.fail_initialize = False
        self.mutate_config = None
        self.mutate_thread = None
        self.tool_item = None

    def __call__(self, config, approval_handler):
        return FakeClient(config, approval_handler, self)


@pytest.fixture
def factory(monkeypatch):
    result = FakeFactory()
    monkeypatch.setattr(adapter, "CodexClient", result)
    return result


def _backend(tmp_path):
    return adapter.CodexBackend(
        model=None,
        codex_home=tmp_path / "research-auth",
        working_dir=tmp_path / "sessions",
    )


def _generate(backend, role="market"):
    return backend.generate(role=role, prompt="Inline host data only", output_schema=SCHEMA)


def test_typed_sdk_schema_and_per_call_isolation(factory, tmp_path):
    async def exercise():
        async with _backend(tmp_path) as backend:
            results = await asyncio.gather(
                _generate(backend), _generate(backend), _generate(backend, "news")
            )
            assert results == [factory.response] * 3
            assert not backend._sessions
        with pytest.raises(RuntimeError, match="closed"):
            await _generate(backend)

    asyncio.run(exercise())
    assert len(factory.clients) == 3
    assert len({client.cwd for client in factory.clients}) == 3
    for client in factory.clients:
        assert client.closed.is_set() and not client.cwd.exists()
        assert client.unregistered == [client.turn_id]
        thread = client.thread_params.model_dump(by_alias=True, mode="json", exclude_none=True)
        turn = client.turn_params.model_dump(by_alias=True, mode="json", exclude_none=True)
        assert thread["ephemeral"] is True
        assert thread["sandbox"] == "read-only"
        assert thread["approvalPolicy"] == "never"
        assert thread["approvalsReviewer"] == "user"
        assert turn["sandboxPolicy"] == {"type": "readOnly", "networkAccess": False}
        assert turn["outputSchema"] == SCHEMA
        assert client.turn_params.output_schema is not SCHEMA
        assert turn["input"][0]["text"] == "Inline host data only"
        assert client.interrupted == []


def test_launch_clears_parent_environment_and_uses_pinned_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-must-not-inherit")
    monkeypatch.setenv("CODEX_HOME", "/synthetic-parent-codex")
    monkeypatch.setenv("HTTPS_PROXY", "synthetic-proxy-secret")
    home, cwd = tmp_path / "app-home", tmp_path / "private-cwd"
    config = adapter._launch_config(home, cwd)
    args = config.launch_args_override
    assert args[:2] == ("/usr/bin/env", "-i")
    assert f"CODEX_HOME={home}" in args and f"HOME={cwd}" in args
    assert "--strict-config" in args
    assert not any("synthetic" in arg for arg in args)
    assert config.env is None
    assert args[-3:] == ("app-server", "--listen", "stdio://")
    overrides = dict(arg.split("=", 1) for arg in args if "=" in arg)
    for feature in adapter._DISABLED_FEATURES:
        assert overrides[f"features.{feature}"] == "false"
    assert overrides["web_search"] == '"disabled"'
    assert overrides["shell_environment_policy.inherit"] == '"none"'
    assert overrides["project_doc_max_bytes"] == "0"


def test_transport_is_opt_in_and_only_validated_values_are_forwarded(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTPS_PROXY", "http://secret:secret@localhost:1234")
    cert = tmp_path / "test-ca.pem"
    cert.write_text("contents are not read for metadata validation")
    transport = {"HTTPS_PROXY": "http://localhost:8080", "SSL_CERT_FILE": str(cert)}
    backend = adapter.CodexBackend(
        None,
        tmp_path / "home",
        tmp_path / "sessions",
        transport_env=transport,
    )
    transport["HTTPS_PROXY"] = "http://secret:secret@localhost:1234"
    config = adapter._launch_config(backend.codex_home, tmp_path, backend.transport_env)
    args = config.launch_args_override
    assert args[:2] == ("/usr/bin/env", "-i")
    assert "HTTPS_PROXY=http://localhost:8080" in args
    assert f"SSL_CERT_FILE={cert}" in args
    assert not any("secret" in arg for arg in args)
    default_args = adapter._launch_config(backend.codex_home, tmp_path).launch_args_override
    assert not any(arg.startswith(("HTTPS_PROXY=", "SSL_CERT_FILE=")) for arg in default_args)
    with pytest.raises(ValueError):
        adapter._launch_config(backend.codex_home, tmp_path, transport)


def test_generate_passes_explicit_transport_settings(factory, tmp_path):
    async def exercise():
        async with adapter.CodexBackend(
            None,
            tmp_path / "home",
            tmp_path / "sessions",
            transport_env={"ALL_PROXY": "socks5://localhost:1080"},
        ) as backend:
            await _generate(backend)

    asyncio.run(exercise())
    assert "ALL_PROXY=socks5://localhost:1080" in factory.clients[0].config.launch_args_override


@pytest.mark.parametrize(
    "method",
    [
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
        "item/permissions/requestApproval",
        "item/tool/call",
        "future/unknown/request",
    ],
)
def test_every_server_request_is_rejected(method):
    with pytest.raises(adapter.CodexSecurityError):
        adapter._reject_server_request(method, {"untrusted": "ignored"})


@pytest.mark.parametrize("feature", adapter._DISABLED_FEATURES)
def test_missing_or_enabled_capabilities_fail_before_thread(factory, tmp_path, feature):
    factory.mutate_config = lambda config: config["features"].update({feature: True})
    with pytest.raises(adapter.CodexSecurityError, match=feature):
        asyncio.run(_generate(_backend(tmp_path)))
    assert factory.clients[0].thread_params is None
    assert factory.clients[0].closed.is_set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.update(mcp_servers={"unexpected": {"command": "not-run"}}),
        lambda config: config.update(model_providers={"unexpected": {"name": "not-used"}}),
        lambda config: config.update(notify=["not-run"]),
        lambda config: config.update(model_instructions_file="/not-read"),
        lambda config: config["agents"].update(enabled=True),
        lambda config: config.update(web_search="live"),
        lambda config: config.update(approval_policy="on-request"),
        lambda config: config.update(sandbox_mode="workspace-write"),
    ],
)
def test_unsafe_effective_config_fails_closed(factory, tmp_path, mutation):
    factory.mutate_config = mutation
    with pytest.raises(adapter.CodexSecurityError):
        asyncio.run(_generate(_backend(tmp_path)))
    assert factory.clients[0].thread_params is None
    assert factory.clients[0].closed.is_set()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda response: response["sandbox"].update(networkAccess=True),
        lambda response: response.update(sandbox={"type": "dangerFullAccess"}),
        lambda response: response.update(approvalPolicy="on-request"),
        lambda response: response.update(approvalsReviewer="auto_review"),
        lambda response: response["thread"].update(ephemeral=False),
        lambda response: response["thread"].update(path="/unexpected/history"),
        lambda response: response.update(cwd="/unexpected"),
        lambda response: response.update(instructionSources=["/unexpected/AGENTS.md"]),
    ],
)
def test_returned_sandbox_and_isolation_verified_before_turn(factory, tmp_path, mutation):
    factory.mutate_thread = mutation
    with pytest.raises(adapter.CodexSecurityError):
        asyncio.run(_generate(_backend(tmp_path)))
    assert factory.clients[0].turn_params is None
    assert factory.clients[0].closed.is_set()


@pytest.mark.parametrize(
    "response,status",
    [
        ("not JSON", "completed"),
        ("", "completed"),
        ('{"summary":"x"}', "failed"),
        ('{"summary":"x"}', "interrupted"),
    ],
)
def test_invalid_or_unsuccessful_result_is_not_returned(factory, tmp_path, response, status):
    factory.response, factory.status = response, status
    with pytest.raises(RuntimeError):
        asyncio.run(_generate(_backend(tmp_path)))
    assert factory.clients[0].closed.is_set()
    assert len(factory.clients) == 1  # The host, not this adapter, owns retries.


def test_unexpected_tool_event_interrupts_and_closes(factory, tmp_path):
    factory.tool_item = {
        "type": "fileChange",
        "id": "unexpected",
        "status": "inProgress",
        "changes": [],
    }
    with pytest.raises(adapter.CodexSecurityError, match="tool activity"):
        asyncio.run(_generate(_backend(tmp_path)))
    client = factory.clients[0]
    assert client.interrupted == [(client.thread_id, client.turn_id)]
    assert client.closed.is_set()


async def _wait_until_blocked(factory):
    while not factory.clients:
        await asyncio.sleep(0)
    client = factory.clients[0]
    assert await asyncio.wait_for(asyncio.to_thread(client.waiting.wait, 2), timeout=3)
    return client


@pytest.mark.parametrize("during_start", [False, True])
def test_cancellation_joins_worker_and_removes_session(factory, tmp_path, during_start):
    factory.block_turn_start = during_start
    factory.block_stream = not during_start

    async def exercise():
        async with _backend(tmp_path) as backend:
            task = asyncio.create_task(_generate(backend))
            client = await _wait_until_blocked(factory)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3)
            assert client.closed.is_set()
            assert not client.cwd.exists() and not backend._sessions
            if not during_start:
                assert client.interrupted == [(client.thread_id, client.turn_id)]

    asyncio.run(exercise())


def test_failed_interrupt_still_kills_transport(factory, tmp_path, monkeypatch):
    factory.block_stream = factory.block_interrupt = True
    monkeypatch.setattr(adapter, "_INTERRUPT_TIMEOUT", 0.01)

    async def exercise():
        async with _backend(tmp_path) as backend:
            task = asyncio.create_task(_generate(backend))
            client = await _wait_until_blocked(factory)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3)
            assert client.closed.is_set() and not client.cwd.exists()

    asyncio.run(exercise())


def test_aclose_stops_an_active_generation(factory, tmp_path):
    factory.block_stream = True

    async def exercise():
        backend = _backend(tmp_path)
        task = asyncio.create_task(_generate(backend))
        client = await _wait_until_blocked(factory)
        await asyncio.wait_for(backend.aclose(), timeout=3)
        with pytest.raises(RuntimeError):
            await task
        assert client.closed.is_set() and not client.cwd.exists()
        await backend.aclose()  # Idempotent shutdown.

    asyncio.run(exercise())


def test_cancelled_session_never_starts_a_late_process(factory, tmp_path):
    backend = _backend(tmp_path)
    client = factory(adapter._launch_config(backend.codex_home, tmp_path), lambda *_: {})
    session = adapter._Session(client, tmp_path)
    session.stop.set()
    with pytest.raises(RuntimeError, match="stopped"):
        adapter._run_session(session, None, "market", "inline", SCHEMA)
    assert not client.started


def test_initialize_failure_closes_process(factory, tmp_path):
    factory.fail_initialize = True
    with pytest.raises(RuntimeError, match="initialization failed"):
        asyncio.run(_generate(_backend(tmp_path)))
    assert factory.clients[0].closed.is_set()


def _usage(input_tokens=10, cached=4, output=6, reasoning=3, total=16):
    return {
        "inputTokens": input_tokens,
        "cachedInputTokens": cached,
        "outputTokens": output,
        "reasoningOutputTokens": reasoning,
        "totalTokens": total,
    }


def test_usage_records_latest_cumulative_total_without_double_counting(factory, tmp_path):
    factory.usage_updates = [_usage(), _usage(20, 8, 12, 5, 32)]
    backend = _backend(tmp_path)
    asyncio.run(_generate(backend))
    assert backend.usage_records == [
        {
            "role": "market",
            "model": "fake-model",
            "input_tokens": 20,
            "cached_input_tokens": 8,
            "output_tokens": 12,
            "reasoning_output_tokens": 5,
            "total_tokens": 32,
        }
    ]


def test_usage_record_marks_unavailable_counters_null(factory, tmp_path):
    backend = _backend(tmp_path)
    asyncio.run(_generate(backend))
    assert backend.usage_records == [
        {
            "role": "market",
            "model": "fake-model",
            **dict.fromkeys(adapter._USAGE_COUNTERS),
        }
    ]


def test_failed_attempt_preserves_reported_usage(factory, tmp_path):
    factory.status = "failed"
    factory.usage_updates = [_usage()]
    backend = _backend(tmp_path)
    with pytest.raises(RuntimeError):
        asyncio.run(_generate(backend))
    assert len(backend.usage_records) == 1
    assert backend.usage_records[0]["total_tokens"] == 16


def test_cancelled_attempt_preserves_latest_known_usage(factory, tmp_path):
    factory.usage_updates = [_usage()]
    factory.block_after_usage = True
    backend = _backend(tmp_path)

    async def exercise():
        task = asyncio.create_task(_generate(backend))
        await _wait_until_blocked(factory)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert len(backend.usage_records) == 1
    assert backend.usage_records[0]["total_tokens"] == 16
    assert backend.usage_records[0]["model"] == "fake-model"


def test_initialization_failure_has_a_usage_record(factory, tmp_path):
    factory.fail_initialize = True
    backend = _backend(tmp_path)
    with pytest.raises(RuntimeError):
        asyncio.run(_generate(backend))
    assert backend.usage_records == [
        {
            "role": "market",
            "model": None,
            **dict.fromkeys(adapter._USAGE_COUNTERS),
        }
    ]


def test_unreviewed_sdk_version_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "version", lambda _: "0.159.3")
    with pytest.raises(adapter.CodexSecurityError, match="not been reviewed"):
        adapter._launch_config(tmp_path / "home", tmp_path / "cwd")


def test_default_home_and_overlapping_paths_rejected(tmp_path):
    for home, work in [
        (Path.home() / ".codex", tmp_path),
        (Path.home(), tmp_path),
        (tmp_path, tmp_path),
        (tmp_path, tmp_path / "child"),
    ]:
        with pytest.raises(ValueError):
            adapter.CodexBackend(None, home, work)


def test_stalled_stdin_writer_is_terminated_before_streams_close(tmp_path):
    """Exercise real SDK pipe buffering with a local non-reading fake child."""
    client = adapter.CodexClient(
        adapter.CodexConfig(
            launch_args_override=(sys.executable, "-c", "import time; time.sleep(60)"),
            cwd=str(tmp_path),
        ),
        approval_handler=adapter._reject_server_request,
    )
    client.start()
    process = client._proc
    errors = []

    def write_large_prompt():
        try:
            client.request(
                "offline-test", {"prompt": "x" * 250_000}, response_model=ConfigReadResponse
            )
        except Exception as exc:
            errors.append(exc)

    writer = threading.Thread(target=write_large_prompt)
    closer = threading.Thread(target=client.close)
    try:
        writer.start()
        deadline = time.monotonic() + 2
        while not client._lock.locked() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert client._lock.locked()  # Child has not read the large pipe write.
        closer.start()
        closer.join(timeout=3)
        assert not closer.is_alive(), "stdin.close blocked before process termination"
        writer.join(timeout=1)
        assert not writer.is_alive()
        assert process.poll() is not None
        assert errors
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        if closer.ident is not None:
            closer.join(timeout=2)
        writer.join(timeout=2)
        client.close()


@pytest.mark.smoke
@pytest.mark.skipif(os.name != "posix", reason="Isolated backend is POSIX-only")
def test_pinned_runtime_handshake_without_auth_or_inference(tmp_path):
    """Only initialize + config/read. Never start a thread, login, or model call."""
    home, cwd = tmp_path / "empty-application-home", tmp_path / "empty-role"
    home.mkdir(mode=0o700)
    cwd.mkdir(mode=0o700)
    client = adapter.CodexClient(
        adapter._launch_config(home, cwd),
        approval_handler=adapter._reject_server_request,
    )
    with client:
        metadata = client.initialize()
        assert f"/{adapter.SUPPORTED_SDK_VERSION}" in metadata.userAgent
        adapter._check_effective_config(client, cwd)
    assert not (home / "auth.json").exists()
    assert not (home / "sessions").exists()
    assert not (home / "history.jsonl").exists()
    assert json.loads(json.dumps(SCHEMA)) == SCHEMA
