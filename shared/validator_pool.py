"""Role-indexed validator connection pool for the quip-miner.

`substrate-interface`'s subscribe call holds the websocket in
receive-mode for the entire subscription lifetime, deadlocking any
concurrent RPC on the same connection. The historical fix has been
"every controller owns two `SubstrateClient` instances" — one for
subscribe, one for everything else. That works but propagates the
URL rotation across N independent state machines and makes adding a
new role (e.g. telemetry-side reads, a Prometheus scraper) a
multi-file refactor.

`ValidatorPool` makes the constraint explicit: the pool owns one URL
rotation pointer; callers ask for slots by role name (`rpc`,
`subscribe.pow`, `subscribe.mempool`, …) and the pool lazy-constructs
a dedicated `SubstrateClient` per role.

Failover coordination is forward-only and idempotent under races:
when a slot's `_run()` detects a websocket drop, it calls
`pool.advance_rotation(from_url=...)`. The pool checks whether
`from_url` matches its current pointer — if yes, advance; if no
(another slot raced first), return the current pointer unchanged.
This means a single dead validator triggers exactly one rotation
event regardless of how many slots noticed.
"""
from __future__ import annotations

import asyncio
from typing import Mapping, Optional, Sequence

from shared.logging_config import get_logger
from shared.substrate_client import SubstrateClient


logger = get_logger("validator_pool")


class ValidatorPool:
    """Connection holder indexed by role.

    Lifetime:
      - `__init__` does NOT open any connections; it just records the
        URL list.
      - `get(role)` lazy-constructs the slot on first call, connects
        it to `current_url`, and caches it for subsequent calls.
      - `close()` closes every constructed slot. Idempotent. Test
        slots (injected via `test_slots=`) are not closed — they are
        caller-owned.

    Threading:
      - All async methods are safe under concurrent callers thanks to
        `_rotation_lock`.
      - Individual `SubstrateClient` instances retain their own
        `_call_lock` for SCALE-decoder safety; the pool does not
        serialize calls across slots.
    """

    def __init__(
        self,
        urls: Sequence[str],
        *,
        test_slots: Optional[Mapping[str, object]] = None,
    ) -> None:
        url_list = list(urls)
        if not url_list:
            raise ValueError(
                "ValidatorPool: `urls=` must contain at least one validator URL"
            )
        self._urls: tuple[str, ...] = tuple(url_list)
        self._current_index: int = 0
        # Pre-constructed mock clients keyed by role. When a `get(role)`
        # call hits a name in this map, the mapped object is returned
        # directly without any real-network construction.
        self._test_slots: dict[str, object] = dict(test_slots or {})
        # Owned (real) slots that the pool constructed itself. These
        # are the ones `close()` tears down.
        self._owned_slots: dict[str, SubstrateClient] = {}
        # Async lock guarding rotation and slot-construction critical
        # sections. Lazy-binds in connect()-style methods so the lock
        # captures the running event loop.
        self._rotation_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def urls(self) -> tuple[str, ...]:
        return self._urls

    @property
    def current_url(self) -> str:
        return self._urls[self._current_index]

    # ------------------------------------------------------------------
    # Slot access
    # ------------------------------------------------------------------

    async def get(self, role: str) -> object:
        """Return the slot client for `role`, lazy-constructing on first
        access. Subsequent calls for the same role return the same
        instance.

        Returns the injected object directly when `role` is in
        `test_slots`; otherwise constructs a real `SubstrateClient`
        and connects it to the pool's current URL.
        """
        if role in self._test_slots:
            return self._test_slots[role]
        lock = self._ensure_lock()
        async with lock:
            existing = self._owned_slots.get(role)
            if existing is not None:
                return existing
            client = SubstrateClient(urls=self._urls)
            # Pin to the pool's current URL on construction. Today
            # `SubstrateClient.connect()` walks all URLs and lands on
            # the first reachable — that's fine: if the pool's current
            # URL is reachable, it lands there; if not, the client's
            # own walk picks the next reachable one and we resync the
            # pool pointer below.
            await client.connect()
            if client.current_url != self.current_url:
                # The pool was anchored to a dead URL; sync forward.
                try:
                    self._current_index = self._urls.index(client.current_url)
                    logger.info(
                        "validator pool: anchor URL was unreachable; "
                        "synced pool pointer to %s",
                        client.current_url,
                    )
                except ValueError:
                    # client.current_url isn't in our list — should be
                    # impossible since we passed self._urls. Defensive.
                    pass
            self._owned_slots[role] = client
            return client

    # ------------------------------------------------------------------
    # Rotation coordination
    # ------------------------------------------------------------------

    async def advance_rotation(self, from_url: str) -> str:
        """Move the rotation pointer forward if `from_url` matches the
        pool's current pointer; otherwise return the current URL
        unchanged.

        This is the contract slots' failover handlers use: "I'm on
        `from_url` and it just died — what do I move to?" The check
        prevents two slots that both noticed the same death from
        double-advancing the pointer.
        """
        lock = self._ensure_lock()
        async with lock:
            if from_url != self.current_url:
                # Either the pool already advanced past this URL
                # (another slot noticed the death first) or the slot
                # reported a URL that isn't ours (shouldn't happen).
                # Either way, return current — let the slot reconnect
                # to wherever we're pointing now.
                return self.current_url
            old_url = self.current_url
            self._current_index = (self._current_index + 1) % len(self._urls)
            new_url = self.current_url
            logger.warning(
                "validator pool: advancing rotation from %s to %s",
                old_url,
                new_url,
            )
            return new_url

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close every pool-owned slot. Idempotent.

        Test slots (injected via `test_slots=`) are caller-owned and
        are NOT closed here — the test that constructed them is
        responsible for their lifetime.
        """
        slots = self._owned_slots
        self._owned_slots = {}
        for role, client in slots.items():
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001 — cleanup, log + continue
                logger.warning(
                    "validator pool: close() error on slot %r: %s: %s",
                    role,
                    type(exc).__name__,
                    exc,
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_lock(self) -> asyncio.Lock:
        """Lazy-instantiate the rotation lock on first async use so it
        binds to the currently running event loop. `asyncio.Lock`
        captures the loop at __init__ time, and pool construction
        typically happens before the loop is set up."""
        if self._rotation_lock is None:
            self._rotation_lock = asyncio.Lock()
        return self._rotation_lock


__all__ = ["ValidatorPool"]
