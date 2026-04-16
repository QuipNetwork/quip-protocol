"""Tests for MinerServiceHandle (long-lived miner service)."""

import time

import pytest

from shared.miner_service import MinerServiceHandle


# CPU miner spec for testing (lightweight, no GPU required)
_CPU_SPEC = {
    "id": "test-cpu-1",
    "kind": "cpu",
    "cfg": {},
    "args": {},
}


@pytest.fixture
def service():
    """Create a CPU miner service, shut it down after test."""
    handle = MinerServiceHandle(_CPU_SPEC)
    yield handle
    if handle.is_alive():
        handle.close()


def test_service_starts(service):
    """Miner service process starts and is alive."""
    assert service.is_alive()
    assert service.miner_id == "test-cpu-1"
    assert service.miner_type == "CPU"


def test_service_status(service):
    """Status command returns miner info."""
    # Wait a moment for miner to initialize
    time.sleep(1.0)
    status = service.get_status()
    assert status is not None
    assert status["event"] == "status"
    assert status["miner_id"] == "test-cpu-1"
    assert status["mining"] is False


def test_service_close(service):
    """Service shuts down cleanly."""
    service.close()
    assert not service.is_alive()


def test_service_close_idempotent():
    """Calling close twice doesn't raise."""
    handle = MinerServiceHandle(_CPU_SPEC)
    handle.close()
    handle.close()


def test_service_cancel_when_not_mining(service):
    """Cancel command when not mining sends stopped event."""
    time.sleep(0.5)
    service.cancel()
    # Should get a "stopped" event
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            msg = service.result_queue.get(timeout=0.5)
            if isinstance(msg, dict) and msg.get("event") == "stopped":
                return  # Success
        except Exception:
            pass
    pytest.fail("Did not receive 'stopped' event")


def test_miner_type_mappings():
    """Verify miner_type property for different spec kinds."""
    cases = [
        ({"id": "x", "kind": "cpu"}, "CPU"),
        ({"id": "x", "kind": "metal"}, "GPU-MPS"),
        ({"id": "x", "kind": "qpu"}, "QPU"),
        ({"id": "x", "kind": "cuda", "args": {"device": "1"}}, "GPU-LOCAL:1"),
        ({"id": "x", "kind": "modal", "args": {"gpu_type": "a100"}}, "GPU-A100"),
    ]
    for spec, expected in cases:
        # Don't actually start the process, just check the property
        handle = object.__new__(MinerServiceHandle)
        handle.spec = spec
        assert handle.miner_type == expected, f"kind={spec['kind']}"


def test_compatibility_properties(service):
    """req and resp properties map to the underlying queues."""
    assert service.req is service.cmd_queue
    assert service.resp is service.result_queue
