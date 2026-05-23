"""Validator URL rotation with all-down exponential backoff.

Used by both `substrate.pool.ValidatorPool` (RPC + child-process) and
the telemetry sibling process (its own simple SubstrateClient). Keeps
the rotation/backoff logic in one place so the two consumers don't
drift apart.

Contract:
    * `current()` returns the URL to try right now.
    * `advance_after_failure(failed_url)` records a failure and returns
      the next URL in the rotation. When every URL has failed in the
      current cycle, raises `AllUrlsDown` carrying the backoff duration
      the caller should wait before retrying.
    * `reset_after_backoff()` is called by the caller after sleeping
      the backoff; clears the bad set so a new cycle can begin.
    * `confirm_success(url)` is called after any successful RPC; resets
      both the bad set and the backoff schedule.
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


class AllUrlsDown(Exception):
    """Every URL has failed in the current rotation cycle.

    The `backoff_s` attribute carries the duration the caller should
    sleep before invoking `reset_after_backoff()` and trying again.
    """

    def __init__(self, backoff_s: float) -> None:
        super().__init__(f"all validator URLs are down; back off {backoff_s:.2f}s")
        self.backoff_s = backoff_s


class SubstrateUrlFailover:
    """Validator URL rotation with all-down exponential backoff.

    Args:
        urls: Non-empty list of validator URLs to rotate through.
        initial_backoff_s: First backoff duration after all URLs are
            down in one cycle. Default 1.0s.
        max_backoff_s: Cap on the exponential backoff. Default 60.0s.
    """

    def __init__(
        self,
        urls: Sequence[str],
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 60.0,
    ) -> None:
        if not urls:
            raise ValueError("SubstrateUrlFailover requires at least one URL")
        self._urls = list(urls)
        self._idx = 0
        self._bad: set[str] = set()
        self._initial_backoff_s = float(initial_backoff_s)
        self._max_backoff_s = float(max_backoff_s)
        self._next_backoff_s = self._initial_backoff_s

    def current(self) -> str:
        """Return the URL currently in use."""
        return self._urls[self._idx]

    def advance_after_failure(self, failed_url: str) -> str:
        """Mark `failed_url` as bad and return the next URL.

        If every URL is marked bad, raises `AllUrlsDown` with the
        backoff duration the caller should wait.

        `failed_url` not matching `current()` is logged-and-ignored
        (still advances from current to next) to avoid skipping URLs
        on caller races.
        """
        if failed_url != self._urls[self._idx]:
            logger.warning(
                "advance_after_failure called with %s but current URL is %s; "
                "advancing from current anyway",
                failed_url, self._urls[self._idx],
            )
        self._bad.add(self._urls[self._idx])
        self._idx = (self._idx + 1) % len(self._urls)

        if self._bad.issuperset(self._urls):
            backoff = self._next_backoff_s
            self._next_backoff_s = min(self._next_backoff_s * 2.0, self._max_backoff_s)
            raise AllUrlsDown(backoff)

        return self._urls[self._idx]

    def reset_after_backoff(self) -> None:
        """Clear the bad set after the caller has slept the backoff."""
        self._bad.clear()

    def confirm_success(self) -> None:
        """Note a successful RPC. Reset bad set and backoff schedule."""
        self._bad.clear()
        self._next_backoff_s = self._initial_backoff_s
