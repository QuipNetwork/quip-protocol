# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""throttle_if_busy backs off only when the scheduler reports a busy GPU."""
from __future__ import annotations

from GPU.gpu_scheduler import throttle_if_busy


class _FakeScheduler:
    def __init__(self, busy: bool):
        self._busy = busy

    def should_throttle(self) -> bool:
        return self._busy


def _recorder():
    calls = []
    return calls, (lambda s: calls.append(s))


def test_none_scheduler_never_sleeps():
    calls, sleep = _recorder()
    throttle_if_busy(None, sleep_fn=sleep)
    assert calls == []


def test_not_busy_does_not_sleep():
    calls, sleep = _recorder()
    throttle_if_busy(_FakeScheduler(busy=False), sleep_fn=sleep)
    assert calls == []


def test_busy_sleeps_once_with_configured_duration():
    calls, sleep = _recorder()
    throttle_if_busy(_FakeScheduler(busy=True), sleep_fn=sleep, sleep_s=0.25)
    assert calls == [0.25]
