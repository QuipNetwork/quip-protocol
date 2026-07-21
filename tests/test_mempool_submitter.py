"""Unit tests for `substrate.mempool_submitter` (T5 extraction).

Covers the submit/claim mechanics extracted from `MempoolMinerController`
plus the T5 behavior change: receipt classification returns an outcome
enum and the previously-fatal error class maps to `MEMPOOL_DISABLE`
instead of raising (the T7 scheduler parks the producer on that signal;
raising out of the shared stack would kill pow too).

All tests use fake build/pool clients — no chain required.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from substrate.mempool_submitter import (
    CLAIM_STALE_ERRORS,
    SOLUTION_FATAL_ERRORS,
    SOLUTION_STALE_ERRORS,
    ClaimOutcome,
    MempoolSubmitter,
    SubmitOutcome,
    classify_claim_receipt,
    classify_submit_receipt,
)


# ----------------------------------------------------------------------
# Receipt classifiers — outcome enums
# ----------------------------------------------------------------------


def test_classify_submit_none_is_ok():
    assert classify_submit_receipt(None) is SubmitOutcome.OK


@pytest.mark.parametrize("name", SOLUTION_STALE_ERRORS)
def test_classify_submit_stale_names(name):
    outcome = classify_submit_receipt(f"Module(error={name}, index=42)")
    assert outcome is SubmitOutcome.STALE


@pytest.mark.parametrize("name", SOLUTION_FATAL_ERRORS)
def test_classify_submit_fatal_names_map_to_disable(name):
    outcome = classify_submit_receipt(f"Module(error={name}, index=42)")
    assert outcome is SubmitOutcome.MEMPOOL_DISABLE


def test_classify_submit_unknown_maps_to_disable():
    assert (
        classify_submit_receipt("Module(error=NeverSeenError)")
        is SubmitOutcome.MEMPOOL_DISABLE
    )


def test_classify_claim_none_is_ok():
    assert classify_claim_receipt(None) is ClaimOutcome.OK


def test_classify_claim_not_expired_retries():
    assert (
        classify_claim_receipt("Module(error=OrderNotExpired)") is ClaimOutcome.RETRY
    )


@pytest.mark.parametrize(
    "name", [n for n in CLAIM_STALE_ERRORS if n != "OrderNotExpired"]
)
def test_classify_claim_terminal_stale_gives_up(name):
    assert classify_claim_receipt(f"Module(error={name})") is ClaimOutcome.GIVE_UP


def test_classify_claim_unknown_is_failed():
    assert classify_claim_receipt("Module(error=NeverSeenError)") is ClaimOutcome.FAILED


# ----------------------------------------------------------------------
# Submitter fixtures
# ----------------------------------------------------------------------


def _receipt(error=None):
    return SimpleNamespace(error=error, extrinsic_hash="0xfeed")


def _submitter(*, submit_receipt=None, submit_exc=None, **kwargs) -> MempoolSubmitter:
    build_client = MagicMock()
    build_client.build_signed_extrinsic = AsyncMock(return_value="0xdeadbeef")
    pool_client = MagicMock()
    if submit_exc is not None:
        pool_client.submit_signed_extrinsic = AsyncMock(side_effect=submit_exc)
    else:
        pool_client.submit_signed_extrinsic = AsyncMock(
            return_value=submit_receipt if submit_receipt is not None else _receipt()
        )
    defaults = dict(
        build_client=build_client,
        pool_client=pool_client,
        signer=MagicMock(),
    )
    defaults.update(kwargs)
    return MempoolSubmitter(**defaults)


def _result(solutions=None):
    return MagicMock(solutions=solutions if solutions is not None else [[1, -1]])


# ----------------------------------------------------------------------
# submit_solution
# ----------------------------------------------------------------------


async def test_submit_ok_payload_shape_and_bookkeeping():
    s = _submitter()
    report = await s.submit_solution(42, _result([[1, -1], [-1, 1]]))
    assert report.outcome is SubmitOutcome.OK
    assert 42 in s.submitted_orders
    assert s.stats.solutions_submitted == 1

    call = s.build_client.build_signed_extrinsic.await_args
    module, call_name, params, signer = call.args
    assert module == "QuantumComputeMempool"
    assert call_name == "submit_solution"
    assert params["order_id"] == 42
    # Both BoundedVec layers are 1-field composites: each inner solution
    # AND the outer list need 1-tuple wrapping.
    assert params["solutions"] == ([([1, -1],), ([-1, 1],)],)
    assert signer is s.signer
    s.pool_client.submit_signed_extrinsic.assert_awaited_once_with(
        "0xdeadbeef", wait_for="inblock"
    )


async def test_submit_clips_to_20_solutions():
    s = _submitter()
    report = await s.submit_solution(42, _result([[1, -1]] * 25))
    assert report.outcome is SubmitOutcome.OK
    params = s.build_client.build_signed_extrinsic.await_args.args[2]
    assert len(params["solutions"][0]) == 20


async def test_submit_stale_receipt_drops_without_raise():
    s = _submitter(submit_receipt=_receipt("Module(error=OrderNotOpen)"))
    report = await s.submit_solution(42, _result())
    assert report.outcome is SubmitOutcome.STALE
    assert 42 not in s.submitted_orders
    assert s.stats.solution_stale_drops == 1
    assert s.stats.solutions_submitted == 0


@pytest.mark.parametrize("name", SOLUTION_FATAL_ERRORS)
async def test_submit_fatal_receipt_returns_disable_without_raise(name):
    s = _submitter(submit_receipt=_receipt(f"Module(error={name})"))
    report = await s.submit_solution(42, _result())
    assert report.outcome is SubmitOutcome.MEMPOOL_DISABLE
    assert name in report.error
    assert s.stats.solution_errors == 1
    assert 42 not in s.submitted_orders


async def test_submit_rpc_failure_returns_failed_without_raise():
    s = _submitter(submit_exc=ConnectionError("rpc down"))
    report = await s.submit_solution(42, _result())
    assert report.outcome is SubmitOutcome.FAILED
    assert s.stats.solution_errors == 1


async def test_submit_and_claim_carry_the_mempool_tip():
    """Mempool extrinsics tip so they outrank same-account pow traffic.

    The signer account continuously pools tip-0 pow extrinsics (proofs,
    markers, anticipatory fires). A tip-0 submit_solution has lower
    txpool priority and can neither replace a pooled sibling at its
    nonce nor win the recompose race — starved indefinitely on an
    actively-winning node (found live by T9). The tip mirrors the
    scheduler's priority inversion at the txpool layer; its cost is
    noise next to the order reward."""
    s = _submitter()
    await s.submit_solution(42, _result())
    assert s.build_client.build_signed_extrinsic.await_args.kwargs.get(
        "tip"
    ) == s.tip_plancks
    assert s.tip_plancks > 0

    s.claimable.add(7)
    await s.claim_expired_orders()
    assert s.build_client.build_signed_extrinsic.await_args.kwargs.get(
        "tip"
    ) == s.tip_plancks


async def test_submit_retries_txpool_nonce_race_with_fresh_compose():
    """A txpool rejection (1010/1014 nonce race with a sibling pow
    extrinsic from the same account) must be retried with a fresh
    compose — each build_signed_extrinsic reads the next nonce anew.
    Found live by T9: the pow proof + participation remark race the
    mempool submit on a shared signer, and a one-shot FAILED meant the
    SolverNotRegistered receipt (and its MEMPOOL_DISABLE park) was
    never reached."""
    from substrateinterface.exceptions import SubstrateRequestException

    s = _submitter()
    s.txpool_retry_backoff_s = 0.0
    s.pool_client.submit_signed_extrinsic = AsyncMock(
        side_effect=[
            SubstrateRequestException(
                {"code": 1014, "message": "Priority is too low"}
            ),
            _receipt(),
        ]
    )
    report = await s.submit_solution(42, _result())
    assert report.outcome is SubmitOutcome.OK
    assert s.stats.solutions_submitted == 1
    assert s.build_client.build_signed_extrinsic.await_count == 2  # recomposed
    assert s.stats.solution_errors == 0


async def test_submit_retries_usurped_watch_rejection():
    """Our own tipped sibling (or any higher-priority replacement) can
    usurp a pooled mempool extrinsic; the watch reports it as a typed
    ExtrinsicRejected. Retry with a fresh compose, same as the txpool
    nonce race."""
    from substrate.client import ExtrinsicRejected

    s = _submitter()
    s.txpool_retry_backoff_s = 0.0
    s.pool_client.submit_signed_extrinsic = AsyncMock(
        side_effect=[
            ExtrinsicRejected("extrinsic rejected: usurped"),
            _receipt(),
        ]
    )
    report = await s.submit_solution(42, _result())
    assert report.outcome is SubmitOutcome.OK
    assert s.build_client.build_signed_extrinsic.await_count == 2


async def test_submit_txpool_retries_are_bounded():
    from substrateinterface.exceptions import SubstrateRequestException

    s = _submitter(
        submit_exc=SubstrateRequestException(
            {"code": 1010, "message": "Invalid Transaction",
             "data": "Transaction is outdated"}
        )
    )
    s.txpool_retry_backoff_s = 0.0
    report = await s.submit_solution(42, _result())
    assert report.outcome is SubmitOutcome.FAILED
    assert s.stats.solution_errors == 1
    attempts = s.pool_client.submit_signed_extrinsic.await_count
    assert attempts == 1 + s.TXPOOL_RETRIES


async def test_submit_ok_invokes_callback():
    calls = []

    async def on_submitted(order_id, result):
        calls.append(order_id)

    s = _submitter(on_solution_submitted=on_submitted)
    await s.submit_solution(42, _result())
    assert calls == [42]


async def test_submit_ok_callback_exception_is_swallowed():
    async def on_submitted(order_id, result):
        raise RuntimeError("callback boom")

    s = _submitter(on_solution_submitted=on_submitted)
    report = await s.submit_solution(42, _result())
    assert report.outcome is SubmitOutcome.OK
    assert s.stats.solutions_submitted == 1


# ----------------------------------------------------------------------
# note_order_expired — claimable routing
# ----------------------------------------------------------------------


def test_note_order_expired_only_for_submitted_orders():
    s = _submitter()
    s.submitted_orders.add(7)
    s.note_order_expired(7)
    s.note_order_expired(8)  # never submitted — ignored
    assert s.claimable == {7}


# ----------------------------------------------------------------------
# claim_expired_orders
# ----------------------------------------------------------------------


async def test_claim_ok_removes_and_counts():
    claimed = []

    async def on_claimed(order_id, amount):
        claimed.append((order_id, amount))

    s = _submitter(on_reward_claimed=on_claimed)
    s.claimable.add(7)
    await s.claim_expired_orders()
    assert s.claimable == set()
    assert s.stats.rewards_claimed == 1
    assert claimed == [(7, 0)]


async def test_claim_not_expired_retries_next_tick():
    s = _submitter(submit_receipt=_receipt("Module(error=OrderNotExpired)"))
    s.claimable.add(7)
    await s.claim_expired_orders()
    assert 7 in s.claimable
    assert s.stats.rewards_claimed == 0
    assert s.stats.claim_errors == 0


async def test_claim_not_winner_gives_up():
    s = _submitter(submit_receipt=_receipt("Module(error=NotWinner)"))
    s.claimable.add(7)
    await s.claim_expired_orders()
    assert 7 not in s.claimable
    assert s.stats.claim_errors == 0


async def test_claim_unknown_error_counts_and_gives_up():
    s = _submitter(submit_receipt=_receipt("Module(error=NeverSeenError)"))
    s.claimable.add(7)
    await s.claim_expired_orders()
    assert 7 not in s.claimable
    assert s.stats.claim_errors == 1


async def test_claim_rpc_failure_keeps_order_for_retry():
    s = _submitter(submit_exc=ConnectionError("rpc down"))
    s.claimable.add(7)
    await s.claim_expired_orders()  # must not raise
    assert 7 in s.claimable
    assert s.stats.rewards_claimed == 0


# ----------------------------------------------------------------------
# Proactive claim on lazy expiry (found live by T9)
# ----------------------------------------------------------------------
#
# The pallet expires orders LAZILY: `expire_order_if_needed` runs only
# inside submit_solution / claim_reward / reclaim_order — there is no
# on_initialize sweep, so OrderExpired never fires on a quiet mempool
# and an event-driven claim loop waits forever. The claim loop must also
# scan `submitted_orders` for orders past their COMPUTED expiry
# (claim_reward performs the lazy expiry inline and pays).


def _order(status, *, created_at=100, deadline_blocks=50,
           first_solution_at=None, block_wait=5):
    from substrate.mempool_types import OrderStatus as _OS

    return SimpleNamespace(
        status=_OS[status] if isinstance(status, str) else status,
        created_at=created_at,
        first_solution_at=first_solution_at,
        timing=SimpleNamespace(
            deadline_blocks=deadline_blocks, block_wait=block_wait,
        ),
    )


async def test_claim_scans_submitted_orders_for_lazy_expiry():
    """An order we won whose status already reads EXPIRED is claimed even
    though no OrderExpired event was ever routed."""
    s = _submitter()
    s.submitted_orders.add(13)
    s.pool_client.query_job_order = AsyncMock(return_value=_order("EXPIRED"))
    s.pool_client.get_block_number = AsyncMock(return_value=200)
    await s.claim_expired_orders()
    assert s.stats.rewards_claimed == 1
    assert 13 not in s.submitted_orders
    assert 13 not in s.claimable


async def test_claim_scan_claims_opened_order_past_computed_expiry():
    """Lazy expiry: the order still READS Opened (nobody poked it), but
    first_solution_at + block_wait is behind the current block — claiming
    triggers the pallet's inline expiry and pays."""
    s = _submitter()
    s.submitted_orders.add(13)
    s.pool_client.query_job_order = AsyncMock(
        return_value=_order(
            "OPENED", created_at=100, deadline_blocks=50,
            first_solution_at=110, block_wait=5,
        )
    )
    s.pool_client.get_block_number = AsyncMock(return_value=116)  # >= 115
    await s.claim_expired_orders()
    assert s.stats.rewards_claimed == 1
    assert 13 not in s.submitted_orders


async def test_claim_scan_leaves_unexpired_opened_order_alone():
    s = _submitter()
    s.submitted_orders.add(13)
    s.pool_client.query_job_order = AsyncMock(
        return_value=_order(
            "OPENED", created_at=100, deadline_blocks=50,
            first_solution_at=110, block_wait=5,
        )
    )
    s.pool_client.get_block_number = AsyncMock(return_value=112)  # < 115
    await s.claim_expired_orders()
    assert s.stats.rewards_claimed == 0
    assert 13 in s.submitted_orders  # kept for the next tick
    s.pool_client.submit_signed_extrinsic.assert_not_awaited()


async def test_claim_scan_query_failure_keeps_order_for_next_tick():
    s = _submitter()
    s.submitted_orders.add(13)
    s.pool_client.query_job_order = AsyncMock(
        side_effect=ConnectionError("rpc down")
    )
    s.pool_client.get_block_number = AsyncMock(return_value=200)
    await s.claim_expired_orders()  # must not raise
    assert 13 in s.submitted_orders
    assert s.stats.rewards_claimed == 0


# ----------------------------------------------------------------------
# run_claim_loop
# ----------------------------------------------------------------------


async def test_run_claim_loop_claims_until_shutdown():
    s = _submitter(claim_poll_interval=0.01)
    s.claimable.add(7)
    shutdown = asyncio.Event()
    task = asyncio.create_task(s.run_claim_loop(shutdown))
    for _ in range(200):
        if s.stats.rewards_claimed:
            break
        await asyncio.sleep(0.01)
    shutdown.set()
    await asyncio.wait_for(task, timeout=5)
    assert s.stats.rewards_claimed == 1
