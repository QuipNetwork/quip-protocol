"""Tests for ZMQ IPC transport (ROUTER/DEALER)."""

import asyncio

import pytest

from shared.ipc_transport import ZMQ_AVAILABLE

# Skip all tests if pyzmq is not installed
pytestmark = pytest.mark.skipif(
    not ZMQ_AVAILABLE, reason="pyzmq not installed"
)


@pytest.fixture
async def router_dealer():
    """Create a connected ROUTER/DEALER pair, clean up after test."""
    from shared.ipc_transport import IPCRouter, IPCDealer, get_default_ipc_address

    addr = get_default_ipc_address("test")
    router = IPCRouter(addr)
    await router.start()

    dealer = IPCDealer(addr, b"test-child")
    await dealer.start()

    # Small delay for ZMQ connection handshake
    await asyncio.sleep(0.1)

    yield router, dealer

    await dealer.stop()
    await router.stop()


@pytest.mark.asyncio
async def test_round_trip(router_dealer):
    """Messages round-trip between ROUTER and DEALER."""
    router, dealer = router_dealer

    # Parent -> Child
    sent = await router.send_to(b"test-child", b"hello child")
    assert sent is True
    data = await dealer.recv()
    assert data == b"hello child"

    # Child -> Parent
    await dealer.send(b"hello parent")
    identity, data = await router.recv()
    assert identity == b"test-child"
    assert data == b"hello parent"


@pytest.mark.asyncio
async def test_send_to_unknown_identity(router_dealer):
    """Sending to unknown identity returns False."""
    router, _ = router_dealer
    result = await router.send_to(b"nonexistent", b"data")
    assert result is False


@pytest.mark.asyncio
async def test_broadcast(router_dealer):
    """Broadcast sends to all specified identities."""
    from shared.ipc_transport import IPCDealer

    router, dealer1 = router_dealer

    # Add a second dealer
    dealer2 = IPCDealer(router.bind_address, b"test-child-2")
    await dealer2.start()
    await asyncio.sleep(0.1)

    sent = await router.broadcast(
        b"broadcast msg", [b"test-child", b"test-child-2"]
    )
    assert sent == 2

    data1 = await dealer1.recv()
    data2 = await dealer2.recv()
    assert data1 == b"broadcast msg"
    assert data2 == b"broadcast msg"

    await dealer2.stop()


@pytest.mark.asyncio
async def test_dealer_recv_timeout(router_dealer):
    """recv_timeout returns None when no message available."""
    _, dealer = router_dealer
    result = await dealer.recv_timeout(timeout_ms=100)
    assert result is None


@pytest.mark.asyncio
async def test_multiple_messages(router_dealer):
    """Multiple messages are delivered in order."""
    router, dealer = router_dealer

    for i in range(5):
        await router.send_to(b"test-child", f"msg-{i}".encode())

    for i in range(5):
        data = await dealer.recv()
        assert data == f"msg-{i}".encode()


@pytest.mark.asyncio
async def test_large_message(router_dealer):
    """Large messages (1MB) transfer correctly."""
    router, dealer = router_dealer
    big_data = b"x" * (1024 * 1024)

    await router.send_to(b"test-child", big_data)
    received = await dealer.recv()
    assert len(received) == len(big_data)
    assert received == big_data
