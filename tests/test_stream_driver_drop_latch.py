# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""The stream driver must not latch into permanent no-wait ring claims.

QUI-867 regression guard. ``stream_driver_main`` used
``ring.claim_free(timeout=0.0 if dropped else 0.005)``, where ``dropped`` is the
CUMULATIVE drop counter, not a per-iteration flag. After the very first drop the
timeout collapsed to 0.0 for the life of the driver process, so every later
momentary ring-full dropped instantly instead of waiting the 5ms that would have
found a slot.

That matters because a dropped sample never reaches the worker and is never
evaluated -- if it held a winning solution, the win is silently lost. The ring is
32 slots and both sides run ~3/s, so a worker stall of ~10s (a slow chain submit,
cf. QUI-829) is enough to arm the latch permanently.

These tests drive ``stream_driver_main`` in-process against a fake ring that
distinguishes a waiting claim from a non-waiting one.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

import QPU.stream_driver as stream_driver

_FAKE_CTX = "tests.fakes.fake_stream:build_fake_persistent_context"


class _FakeRing:
    """Ring whose slot availability depends on whether the caller waits.

    Models the real race the latch destroys: the slot is not free *right now*,
    but a free one arrives well within the 5ms budget. A caller that waits gets
    a slot; a caller that passes timeout=0.0 does not.

    ``fail_first`` non-waiting claims return None regardless, to arm the latch.

    Once ``expect_claims`` claims have been seen the ring enqueues the driver's
    shutdown sentinel. Driving shutdown off observed progress (rather than a
    timer) keeps the test deterministic.
    """

    max_rows = 64
    max_cols = 64

    def __init__(self, fail_first: int = 1, expect_claims: int = 0, ctl_q=None):
        self.timeouts: list[float] = []
        self.writes = 0
        self._fail_first = fail_first
        self._expect_claims = expect_claims
        self._ctl_q = ctl_q
        self._calls = 0

    def claim_free(self, timeout: float = 0.0):
        self.timeouts.append(timeout)
        self._calls += 1
        if self._expect_claims and self._calls >= self._expect_claims:
            # All results seen: release the driver from its post-round idle.
            if self._ctl_q is not None:
                self._ctl_q.put(None)
                self._ctl_q = None
        if self._calls <= self._fail_first:
            return None  # arm the latch: first claim(s) find the ring full
        # A slot frees within ~5ms. Only a caller willing to wait sees it.
        return 0 if timeout > 0.0 else None

    def write(self, slot, sample, energy):
        self.writes += 1

    def release(self, slot):
        pass

    def close_unlink(self):
        """Owner-side teardown the driver calls in its finally block."""


def _run_driver(monkeypatch, ring_factory, n_results: int):
    """Run stream_driver_main in-process against a fake ring.

    Args:
        ring_factory: Called with the ctl_q so the ring can signal shutdown
            once it has observed every result.
        n_results: How many results the fake context produces.

    Returns:
        The _FakeRing that was used.
    """
    # The parent-death watchdog polls getppid in a thread; irrelevant here and
    # it would outlive the test.
    monkeypatch.setattr(
        stream_driver, "_start_parent_death_watchdog", lambda _stop: None,
    )
    monkeypatch.setattr(
        stream_driver, "setup_child_process_logging", lambda _q: None,
    )

    ctx = mp.get_context("spawn")
    desc_q = ctx.Queue(maxsize=64)
    ctl_q = ctx.Queue(maxsize=64)
    stop = ctx.Event()
    ring = ring_factory(ctl_q)
    monkeypatch.setattr(stream_driver, "SampleView", lambda **_kw: ring)

    # A switch starts the round. The shutdown sentinel is enqueued by the ring
    # once every result has been claimed -- enqueueing it up front would shut
    # the driver down before it produced anything.
    ctl_q.put(("switch", 1, b"\x01" * 32, b"\x02" * 32, 0, 8, 80.0))

    stream_driver.stream_driver_main(
        ring_args={},
        desc_q=desc_q,
        ctl_q=ctl_q,
        stop_event=stop,
        stream_factory_dotted=_FAKE_CTX,
        factory_kwargs={"num_reads": 4, "nodes": [0, 1, 2], "n": n_results},
    )
    return ring


class TestClaimFreeNeverLatches:
    """A past drop must not disable waiting on every future claim."""

    def test_claim_free_always_waits(self, monkeypatch):
        """Every claim gets a real wait budget, including after a drop.

        Pre-fix this fails: claims after the first drop are made with
        timeout=0.0.
        """
        ring = _run_driver(
            monkeypatch,
            lambda q: _FakeRing(fail_first=1, expect_claims=5, ctl_q=q),
            n_results=5,
        )

        assert ring.timeouts, "driver never attempted a ring claim"
        assert all(t > 0.0 for t in ring.timeouts), (
            f"claim_free was called with a zero timeout: {ring.timeouts}. "
            "A cumulative-drop latch disabled waiting."
        )

    def test_results_after_a_drop_are_still_placed(self, monkeypatch):
        """The real cost: post-drop results must not be discarded.

        Every result after the armed drop could be placed within 5ms, so the
        driver must place them. Pre-fix all of them are dropped instead.
        """
        ring = _run_driver(
            monkeypatch,
            lambda q: _FakeRing(fail_first=1, expect_claims=5, ctl_q=q),
            n_results=5,
        )

        # 5 produced, 1 genuinely unplaceable => the other 4 must be written.
        assert ring.writes == 4, (
            f"only {ring.writes}/4 post-drop results were placed; the rest "
            "were silently dropped (each could hold a winning solution)"
        )

class TestNoDropsPath:
    """A healthy ring must be unaffected by the fix."""

    def test_all_results_placed_when_ring_never_full(self, monkeypatch):
        ring = _run_driver(
            monkeypatch,
            lambda q: _FakeRing(fail_first=0, expect_claims=4, ctl_q=q),
            n_results=4,
        )

        assert ring.writes == 4
        assert all(t > 0.0 for t in ring.timeouts)
