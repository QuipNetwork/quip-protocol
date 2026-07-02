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
