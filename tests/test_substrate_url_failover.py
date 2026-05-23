"""Tests for substrate.url_failover.SubstrateUrlFailover.

The pool and telemetry both need URL rotation with backoff when every
validator URL is unreachable. This helper holds that logic in one
place; both consumers depend on it without depending on each other.
"""
from __future__ import annotations

import math

import pytest

from substrate.url_failover import (
    AllUrlsDown,
    SubstrateUrlFailover,
)


def test_returns_first_url_initially():
    """Initial `current()` returns the first URL in the list."""
    fo = SubstrateUrlFailover(["http://a:9944", "http://b:9944"])
    assert fo.current() == "http://a:9944"


def test_advance_after_failure_rotates_to_next():
    """After a failure, the next URL is returned."""
    fo = SubstrateUrlFailover(["http://a", "http://b", "http://c"])
    assert fo.advance_after_failure("http://a") == "http://b"
    assert fo.current() == "http://b"
    assert fo.advance_after_failure("http://b") == "http://c"
    assert fo.current() == "http://c"


def test_advance_past_last_url_raises_all_urls_down():
    """When every URL has failed in one cycle, AllUrlsDown signals backoff time."""
    fo = SubstrateUrlFailover(
        ["http://a", "http://b"],
        initial_backoff_s=2.0,
        max_backoff_s=60.0,
    )
    fo.advance_after_failure("http://a")  # → b
    with pytest.raises(AllUrlsDown) as exc_info:
        fo.advance_after_failure("http://b")
    assert exc_info.value.backoff_s == pytest.approx(2.0)


def test_backoff_grows_exponentially_across_cycles():
    """Each completed all-down cycle doubles the backoff up to the cap."""
    fo = SubstrateUrlFailover(
        ["http://a"],
        initial_backoff_s=1.0,
        max_backoff_s=8.0,
    )
    backoffs = []
    for _ in range(5):
        try:
            fo.advance_after_failure("http://a")
        except AllUrlsDown as exc:
            backoffs.append(exc.backoff_s)
            fo.reset_after_backoff()
    # 1, 2, 4, 8, 8 (cap)
    assert backoffs == [
        pytest.approx(1.0),
        pytest.approx(2.0),
        pytest.approx(4.0),
        pytest.approx(8.0),
        pytest.approx(8.0),
    ]


def test_confirm_success_resets_backoff_and_bad_set():
    """A successful use of a URL clears the bad set and resets the backoff."""
    fo = SubstrateUrlFailover(["http://a", "http://b"], initial_backoff_s=1.0)
    fo.advance_after_failure("http://a")
    try:
        fo.advance_after_failure("http://b")
    except AllUrlsDown:
        fo.reset_after_backoff()
    fo.confirm_success(fo.current())
    # Should be able to fail-rotate from start again
    next_url = fo.advance_after_failure(fo.current())
    assert next_url in ("http://a", "http://b")
    # And backoff is back to initial
    with pytest.raises(AllUrlsDown) as exc_info:
        fo.advance_after_failure(next_url)
    assert exc_info.value.backoff_s == pytest.approx(1.0)


def test_empty_url_list_rejected():
    """Constructing with no URLs is a usage error."""
    with pytest.raises(ValueError):
        SubstrateUrlFailover([])


def test_advance_after_failure_with_wrong_url_is_a_noop_on_state():
    """If the caller passes a URL that isn't current, log + treat as if current failed.

    This avoids races between concurrent callers; we don't want to skip URLs
    just because callers raced.
    """
    fo = SubstrateUrlFailover(["http://a", "http://b"])
    # Caller incorrectly says http://b failed even though current is http://a.
    # We still advance from current to the next URL.
    result = fo.advance_after_failure("http://b")  # not the current one
    assert result == "http://b"  # we've advanced from a→b
