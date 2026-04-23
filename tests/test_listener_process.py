"""Tests for ``shared.listener_process``.

Covers the three listener responsibilities:
  1. Binding aioquic on a real UDP port (ready event fires).
  2. Routing HEARTBEAT locally and emitting a ``peer_heartbeat`` event
     to the parent without requiring a coordinator response.
  3. Forwarding non-HEARTBEAT messages via ``inbound_message`` and
     sending the parent's ``inbound_response`` back on QUIC.
"""
from __future__ import annotations

import asyncio
import base64
import json
import socket
import time

import pytest

from shared.listener_process import (
    ListenerProcessHandle,
    spawn_listener_process,
)
from shared.node_client import (
    MAX_DATAGRAM_FRAME_SIZE,
    MAX_DATAGRAM_MESSAGE_SIZE,
    QUIP_ALPN_PROTOCOL,
    QuicMessage,
    QuicMessageType,
    NodeClient,
    generate_self_signed_cert,
)


def _free_port() -> int:
    """Ask the kernel for an unused UDP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def listener():
    """Spawn a listener bound to a free port, tear it down on exit."""
    try:
        cert, key = generate_self_signed_cert()
    except ValueError as exc:
        pytest.skip(f"cert generation unavailable: {exc}")
    port = _free_port()
    config_bundle = {
        "bind_address": "127.0.0.1",
        "port": port,
        "tls_cert_file": cert,
        "tls_key_file": key,
        "alpn_protocol": QUIP_ALPN_PROTOCOL,
        "max_datagram_frame_size": MAX_DATAGRAM_FRAME_SIZE,
        "max_datagram_message_size": MAX_DATAGRAM_MESSAGE_SIZE,
        "idle_timeout": 30.0,
    }
    handle = spawn_listener_process(config_bundle)
    # Wait for the listener_ready event (up to 10s).
    ready = _wait_event(handle, "listener_ready", timeout=10.0)
    assert ready is not None, "listener did not report ready in time"
    assert ready.get("port") == port
    yield handle, port
    handle.shutdown()
    handle.force_stop(timeout=3.0)


def _wait_event(
    handle: ListenerProcessHandle, name: str, timeout: float,
):
    """Poll the listener's pipe for a named event."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = handle.recv()
        if msg is not None and msg.get("event") == name:
            return msg
        time.sleep(0.05)
    return None


class TestListenerLifecycle:
    def test_spawn_and_bind(self, listener):
        """Listener spawns, binds its port, and stays alive."""
        handle, port = listener
        assert handle.is_alive()
        assert port > 0

    def test_shutdown_clean(self, listener):
        """Shutdown command causes the child to exit."""
        handle, _ = listener
        handle.shutdown()
        handle.process.join(timeout=5.0)
        assert not handle.is_alive()


@pytest.mark.timeout(30)
class TestHeartbeatLocalPath:
    """HEARTBEAT should be answered from the local cache and emit an
    event. The coordinator does not need to respond for the remote
    peer to get its heartbeat response."""

    @pytest.mark.asyncio
    async def test_heartbeat_emits_event_and_responds(self, listener):
        """Client sends HEARTBEAT, gets response, listener fires event.

        The coordinator pipe is never written to — if the listener
        tried to forward, this test would hang.
        """
        handle, port = listener

        # Push an empty snapshot so the listener's cache is initialized.
        handle.send_cmd({
            "cmd": "peer_snapshot", "peers": {}, "banned": [],
        })

        client = NodeClient(
            node_timeout=5.0, verify_tls=False, connect_timeout=2.0,
        )
        await client.start()
        try:
            ok = await client.connect_to_peer(f"127.0.0.1:{port}")
            assert ok, "failed to connect QUIC to listener"

            # Forge a minimal MinerInfo-like payload. The listener
            # doesn't parse it — it just forwards sender/version.
            from shared.block import MinerInfo
            info = MinerInfo(
                miner_id="m-test",
                miner_type="CPU",
                reward_address=b"\x01" * 32,
                ecdsa_public_key=b"\x02" * 32,
                wots_public_key=b"\x03" * 32,
                next_wots_public_key=b"\x04" * 32,
            )
            result = await client.send_heartbeat(
                f"127.0.0.1:{port}", "peer-test:20049", info,
            )
            assert result is True, "heartbeat did not succeed"

            event = _wait_event_async(
                handle, "peer_heartbeat", timeout=3.0,
            )
            assert event is not None
            assert event.get("peer") == "peer-test:20049"
        finally:
            await client.stop()


def _wait_event_async(
    handle: ListenerProcessHandle, name: str, timeout: float,
):
    """Async-safe wait for a pipe event."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = handle.recv()
        if msg is not None and msg.get("event") == name:
            return msg
        # Thread-sleep is fine here: this is called inside a test
        # task, not the listener's loop.
        time.sleep(0.05)
    return None


class TestForwardedInbound:
    """Non-HEARTBEAT messages are forwarded and the response cycle
    completes: parent replies via inbound_response, QUIC receives it."""

    @pytest.mark.asyncio
    async def test_forwards_non_heartbeat_and_sends_response(self, listener):
        """STATUS_REQUEST should forward; parent's reply round-trips."""
        handle, port = listener
        handle.send_cmd({
            "cmd": "peer_snapshot", "peers": {}, "banned": [],
        })

        # Background task: emulate coordinator by echoing a canned
        # STATUS_RESPONSE when it sees inbound_message.
        parent_done = asyncio.Event()

        async def parent_emulator():
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not parent_done.is_set():
                msg = handle.recv()
                if msg is None:
                    await asyncio.sleep(0.02)
                    continue
                if msg.get("event") != "inbound_message":
                    continue
                msg_id = msg.get("msg_id")
                raw = base64.b64decode(msg.get("raw_b64", ""))
                incoming = QuicMessage.from_bytes(raw)
                assert incoming.msg_type == QuicMessageType.STATUS_REQUEST
                payload = json.dumps({"ok": True, "from": "parent"}).encode()
                resp = incoming.create_response(payload)
                handle.send_cmd({
                    "cmd": "inbound_response",
                    "msg_id": msg_id,
                    "payload_b64": base64.b64encode(
                        resp.to_bytes(),
                    ).decode("ascii"),
                })
                return

        emu = asyncio.create_task(parent_emulator())
        client = NodeClient(
            node_timeout=5.0, verify_tls=False, connect_timeout=2.0,
        )
        await client.start()
        try:
            ok = await client.connect_to_peer(f"127.0.0.1:{port}")
            assert ok
            status = await client.get_peer_status(f"127.0.0.1:{port}")
            assert status is not None, "no response from listener"
            assert status.get("from") == "parent"
        finally:
            parent_done.set()
            await client.stop()
            await emu


class TestListenerReadSnapshot:
    """When a fresh read snapshot is present, STATUS/STATS/PEERS and
    TELEMETRY_* are answered from cache without any IPC round-trip
    to the parent."""

    @pytest.mark.asyncio
    async def test_status_served_from_cache(self, listener):
        """STATUS_REQUEST uses cached bytes; no inbound_message event."""
        handle, port = listener
        handle.send_cmd({
            "cmd": "peer_snapshot", "peers": {}, "banned": [],
        })
        cached_payload = json.dumps(
            {"host": "cached-host:1", "cached": True},
        ).encode("utf-8")
        handle.send_cmd({
            "cmd": "read_snapshot",
            "status": cached_payload,
            "stats": None, "peers": None, "telemetry": {},
        })
        # Give the listener a moment to apply the snapshot command
        # (it's pumped off the main event loop via run_in_executor).
        await asyncio.sleep(0.6)

        saw_forward = asyncio.Event()

        async def watch_no_forward():
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                msg = handle.recv()
                if msg is None:
                    await asyncio.sleep(0.02)
                    continue
                if msg.get("event") == "inbound_message":
                    saw_forward.set()
                    return

        watcher = asyncio.create_task(watch_no_forward())
        client = NodeClient(
            node_timeout=5.0, verify_tls=False, connect_timeout=2.0,
        )
        await client.start()
        try:
            ok = await client.connect_to_peer(f"127.0.0.1:{port}")
            assert ok
            status = await client.get_peer_status(f"127.0.0.1:{port}")
            assert status is not None
            assert status.get("cached") is True
            assert status.get("host") == "cached-host:1"
            # No forward event should have been emitted.
            await asyncio.sleep(0.2)
            assert not saw_forward.is_set(), (
                "STATUS_REQUEST should have been served locally"
            )
        finally:
            watcher.cancel()
            await client.stop()

    @pytest.mark.asyncio
    async def test_missing_field_falls_back_to_forward(self, listener):
        """If read_snapshot has None for a field, the listener forwards."""
        handle, port = listener
        handle.send_cmd({
            "cmd": "peer_snapshot", "peers": {}, "banned": [],
        })
        # Send a read snapshot with status=None so STATUS must forward.
        handle.send_cmd({
            "cmd": "read_snapshot",
            "status": None, "stats": None, "peers": None,
            "telemetry": {},
        })
        await asyncio.sleep(0.6)

        forwarded = asyncio.Event()

        async def parent_emulator():
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                msg = handle.recv()
                if msg is None:
                    await asyncio.sleep(0.02)
                    continue
                if msg.get("event") != "inbound_message":
                    continue
                msg_id = msg.get("msg_id")
                raw = base64.b64decode(msg.get("raw_b64", ""))
                incoming = QuicMessage.from_bytes(raw)
                if incoming.msg_type != QuicMessageType.STATUS_REQUEST:
                    continue
                forwarded.set()
                resp = incoming.create_response(
                    json.dumps({"from": "fallback"}).encode(),
                )
                handle.send_cmd({
                    "cmd": "inbound_response",
                    "msg_id": msg_id,
                    "payload_b64": base64.b64encode(
                        resp.to_bytes(),
                    ).decode("ascii"),
                })
                return

        emu = asyncio.create_task(parent_emulator())
        client = NodeClient(
            node_timeout=5.0, verify_tls=False, connect_timeout=2.0,
        )
        await client.start()
        try:
            ok = await client.connect_to_peer(f"127.0.0.1:{port}")
            assert ok
            status = await client.get_peer_status(f"127.0.0.1:{port}")
            assert status is not None
            assert status.get("from") == "fallback"
            assert forwarded.is_set()
        finally:
            await client.stop()
            await emu
