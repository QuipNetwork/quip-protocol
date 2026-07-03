"""Unit tests for `substrate.mempool_producer` (T5 extraction).

Covers the handle-free discovery pipeline extracted from
`MempoolMinerController` plus the two NEW producer guards:

  - deadline: drop orders already expired or with fewer than
    `deadline_margin_blocks` blocks to effective expiry (pallet
    `lifecycle.rs` semantics — `deadline_blocks` is relative to
    `created_at`, tightened by `first_solution_at + block_wait`);
  - min-reward: drop orders paying below `min_reward` (default 0).

All tests use a fake pool_client — no chain required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from shared.quantum_proof_of_work import (
    DEFAULT_ALLOWED_H,
    DEFAULT_ALLOWED_J,
    DEFAULT_ALLOWED_SPIN,
)
from shared.topology_hash import topology_hash
from substrate.mempool_producer import (
    MempoolJobProducer,
    deadline_blocks_remaining,
    job_matches_sampler,
)
from substrate.mempool_types import (
    IsingParams,
    JobMode,
    JobOrder,
    MinerType,
    OrderStatus,
    OrderTiming,
    ResultDelivery,
    RewardResolution,
)


NODES = (0, 1, 2)
EDGES = ((0, 1), (1, 2))
ACCOUNT = b"\xAA" * 32


def _hash(nodes, edges) -> bytes:
    return topology_hash(
        nodes, edges, DEFAULT_ALLOWED_H, DEFAULT_ALLOWED_J, DEFAULT_ALLOWED_SPIN
    )


def _order(
    *,
    nodes=NODES,
    edges=EDGES,
    mode: JobMode | None = None,
    status: OrderStatus = OrderStatus.OPENED,
    reward: int = 1_000_000_000_000,
    created_at: int = 1,
    deadline_blocks: int = 100,
    block_wait: int = 10,
    first_solution_at: int | None = None,
) -> JobOrder:
    return JobOrder(
        spec_id=b"\x42" * 32,
        proposer=b"\x11" * 32,
        ising_params=IsingParams(
            nodes=tuple(nodes),
            edges=tuple(edges),
            h_values=tuple(0 for _ in nodes),
            j_values=tuple(0 for _ in edges),
        ),
        reward=reward,
        mode=mode if mode is not None else JobMode.open(),
        resolution=RewardResolution.single_best(),
        timing=OrderTiming(deadline_blocks=deadline_blocks, block_wait=block_wait),
        delivery=ResultDelivery.on_chain_only(),
        status=status,
        created_at=created_at,
        first_solution_at=first_solution_at,
        solution_count=0,
    )


class FakePoolClient:
    """Duck-typed PoolClient: canned events + orders, call recording."""

    def __init__(self, orders=None, events=None, *, events_exc=None, query_exc=None):
        self.orders = orders or {}
        self.events = events or []
        self.events_exc = events_exc
        self.query_exc = query_exc
        self.query_calls: list[int] = []

    async def get_events_at(self, block_hash):
        if self.events_exc is not None:
            raise self.events_exc
        return self.events

    async def query_job_order(self, order_id):
        self.query_calls.append(order_id)
        if self.query_exc is not None:
            raise self.query_exc
        return self.orders.get(order_id)


def _job_proposed(order_id: int) -> dict:
    return {
        "module_id": "QuantumComputeMempool",
        "event_id": "JobProposed",
        "attributes": {"order_id": order_id},
    }


def _order_expired(order_id: int) -> dict:
    return {
        "module_id": "QuantumComputeMempool",
        "event_id": "OrderExpired",
        "attributes": {"order_id": order_id},
    }


def _signer(account: bytes = ACCOUNT):
    return SimpleNamespace(account_id_bytes=lambda: account)


def _producer(pool_client, **kwargs) -> MempoolJobProducer:
    defaults = dict(
        pool_client=pool_client,
        signer=_signer(),
        sampler_topology_hash=_hash(NODES, EDGES),
        allowed_h_values=DEFAULT_ALLOWED_H,
        allowed_j_values=DEFAULT_ALLOWED_J,
        allowed_spin_values=DEFAULT_ALLOWED_SPIN,
        solver_type=MinerType.CPU,
    )
    defaults.update(kwargs)
    return MempoolJobProducer(**defaults)


def _ctx(block_number: int = 10, block_hash_byte: int = 0x10):
    return SimpleNamespace(
        block_hash=bytes([block_hash_byte]) * 32,
        block_number=block_number,
    )


async def _drive(producer: MempoolJobProducer, block_number: int = 10) -> None:
    await producer.on_new_block(_ctx(block_number=block_number))


def _drain(producer: MempoolJobProducer) -> list[int]:
    out = []
    while not producer.accepted.empty():
        out.append(producer.accepted.get_nowait())
    return out


# ----------------------------------------------------------------------
# deadline_blocks_remaining — pallet lifecycle.rs parity
# ----------------------------------------------------------------------


def test_deadline_remaining_relative_to_created_at():
    order = _order(created_at=5, deadline_blocks=100)
    assert deadline_blocks_remaining(order, current_block=5) == 100
    assert deadline_blocks_remaining(order, current_block=105) == 0  # now >= expiry
    assert deadline_blocks_remaining(order, current_block=200) == -95


def test_deadline_remaining_tightened_by_first_solution():
    # hard deadline at 1+100=101, but first solution at 10 + block_wait 10 = 20.
    order = _order(
        created_at=1, deadline_blocks=100, block_wait=10, first_solution_at=10
    )
    assert deadline_blocks_remaining(order, current_block=15) == 5


def test_deadline_remaining_first_solution_never_extends_hard_deadline():
    # first_solution_at + block_wait (120) past the hard deadline (101):
    # the pallet takes the min, so the hard deadline still binds.
    order = _order(
        created_at=1, deadline_blocks=100, block_wait=100, first_solution_at=20
    )
    assert deadline_blocks_remaining(order, current_block=99) == 2


# ----------------------------------------------------------------------
# Acceptance and filtering through the event pipeline
# ----------------------------------------------------------------------


async def test_accepts_matching_open_order():
    client = FakePoolClient(orders={5: _order()}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p)
    assert _drain(p) == [5]
    assert p.stats.jobs_accepted == 1
    assert p.stats.heads_observed == 1
    assert p.stats.events_seen == 1


async def test_drops_topology_mismatch():
    other = _order(nodes=(0, 1, 2, 3), edges=((0, 1),))
    client = FakePoolClient(orders={5: other}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p)
    assert _drain(p) == []
    assert p.stats.jobs_filtered == 1


async def test_drops_bid_ineligible():
    order = _order(
        mode=JobMode.bid(miners=(b"\xBB" * 32,), miner_types=(MinerType.GPU,))
    )
    client = FakePoolClient(orders={5: order}, events=[_job_proposed(5)])
    p = _producer(client)  # CPU solver, account \xAA
    await _drive(p)
    assert _drain(p) == []
    assert p.stats.jobs_filtered == 1


async def test_accepts_bid_matching_account():
    order = _order(mode=JobMode.bid(miners=(ACCOUNT,)))
    client = FakePoolClient(orders={5: order}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p)
    assert _drain(p) == [5]


async def test_drops_non_opened_order():
    client = FakePoolClient(
        orders={5: _order(status=OrderStatus.EXPIRED)}, events=[_job_proposed(5)]
    )
    p = _producer(client)
    await _drive(p)
    assert _drain(p) == []


# ----------------------------------------------------------------------
# NEW guard: deadline
# ----------------------------------------------------------------------


async def test_drops_expired_order():
    # expiry at 1+100=101; block 200 is long past.
    client = FakePoolClient(orders={5: _order()}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p, block_number=200)
    assert _drain(p) == []
    assert p.stats.jobs_deadline_dropped == 1


async def test_drops_order_below_deadline_margin():
    # expiry at 101; at block 100 only 1 block remains (< default margin 2).
    client = FakePoolClient(orders={5: _order()}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p, block_number=100)
    assert _drain(p) == []
    assert p.stats.jobs_deadline_dropped == 1


async def test_accepts_order_at_exact_deadline_margin():
    # expiry at 101; at block 99 exactly 2 blocks remain (== margin).
    client = FakePoolClient(orders={5: _order()}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p, block_number=99)
    assert _drain(p) == [5]


async def test_first_solution_tightens_producer_deadline():
    # hard deadline far out, but first solution at 8 + block_wait 3 = 11:
    # at block 10 only 1 block remains — drop.
    order = _order(deadline_blocks=1000, block_wait=3, first_solution_at=8)
    client = FakePoolClient(orders={5: order}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p, block_number=10)
    assert _drain(p) == []
    assert p.stats.jobs_deadline_dropped == 1


# ----------------------------------------------------------------------
# NEW guard: min-reward
# ----------------------------------------------------------------------


async def test_drops_below_min_reward():
    client = FakePoolClient(orders={5: _order(reward=9)}, events=[_job_proposed(5)])
    p = _producer(client, min_reward=10)
    await _drive(p)
    assert _drain(p) == []
    assert p.stats.jobs_reward_dropped == 1


async def test_accepts_reward_equal_to_min_reward():
    client = FakePoolClient(orders={5: _order(reward=10)}, events=[_job_proposed(5)])
    p = _producer(client, min_reward=10)
    await _drive(p)
    assert _drain(p) == [5]


async def test_default_min_reward_zero_accepts_all():
    client = FakePoolClient(orders={5: _order(reward=0)}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p)
    assert _drain(p) == [5]


def test_negative_min_reward_rejected():
    with pytest.raises(ValueError):
        _producer(FakePoolClient(), min_reward=-1)


# ----------------------------------------------------------------------
# Dedup, expiry routing, robustness
# ----------------------------------------------------------------------


async def test_dedup_repeated_job_proposed():
    client = FakePoolClient(orders={5: _order()}, events=[_job_proposed(5)])
    p = _producer(client)
    await _drive(p, block_number=10)
    await _drive(p, block_number=11)  # same JobProposed again next block
    assert _drain(p) == [5]
    assert client.query_calls == [5], "second event must short-circuit on dedup"


async def test_order_expired_routes_to_callback():
    expired: list[int] = []
    client = FakePoolClient(events=[_order_expired(7)])
    p = _producer(client, on_order_expired=expired.append)
    await _drive(p)
    assert expired == [7]


async def test_none_ctx_is_noop():
    client = FakePoolClient(events=[_job_proposed(5)], orders={5: _order()})
    p = _producer(client)
    await p.on_new_block(None)
    assert _drain(p) == []
    assert p.stats.heads_observed == 0


async def test_ignores_non_mempool_events():
    client = FakePoolClient(
        events=[{"module_id": "System", "event_id": "ExtrinsicSuccess", "attributes": {}}]
    )
    p = _producer(client)
    await _drive(p)
    assert p.stats.events_seen == 0
    assert p.stats.heads_observed == 1


async def test_parked_producer_skips_event_poll():
    """MEMPOOL_DISABLE parks the producer, not just the feed loop.

    A parked producer must stop issuing per-block get_events_at RPCs on
    the shared loop and stop growing `accepted` — nothing consumes it
    after the stack parks.
    """
    client = FakePoolClient(orders={5: _order()}, events=[_job_proposed(5)])
    p = _producer(client)
    p.park()
    await _drive(p)
    assert _drain(p) == []
    assert p.stats.heads_observed == 0
    assert client.query_calls == []


async def test_get_events_failure_is_swallowed():
    client = FakePoolClient(events_exc=RuntimeError("rpc down"))
    p = _producer(client)
    await _drive(p)  # must not raise
    assert _drain(p) == []


async def test_query_failure_leaves_order_reconsiderable():
    client = FakePoolClient(events=[_job_proposed(5)], query_exc=RuntimeError("rpc down"))
    p = _producer(client)
    await _drive(p)  # must not raise
    client.query_exc = None
    client.orders = {5: _order()}
    await _drive(p, block_number=11)
    assert _drain(p) == [5], "a failed query must not poison the dedup set"


# ----------------------------------------------------------------------
# job_matches_sampler — shared pure filter
# ----------------------------------------------------------------------


def test_job_matches_sampler_type_bid():
    order = _order(mode=JobMode.bid(miner_types=(MinerType.GPU, MinerType.QPU_DWAVE)))
    kwargs = dict(
        sampler_topology_hash=_hash(NODES, EDGES),
        allowed_h_values=DEFAULT_ALLOWED_H,
        allowed_j_values=DEFAULT_ALLOWED_J,
        allowed_spin_values=DEFAULT_ALLOWED_SPIN,
        account=b"\xCC" * 32,
    )
    assert job_matches_sampler(order, solver_type=MinerType.GPU, **kwargs) is True
    assert job_matches_sampler(order, solver_type=MinerType.CPU, **kwargs) is False
