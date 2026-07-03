"""Unit tests for `substrate.solver_registration` — Guard D+ branch matrix.

MEMPOOL_PRIORITY_PLAN.md §8: query-first (matching registration costs no
extrinsic), vendor-resolved MinerType, SolverAlreadyRegistered race
tolerance, never-deregister, and non-fatal RPC/chain failures. Also covers
the refactored `register-solver` CLI outcome → exit-code mapping (0/4/3).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

import quip_cli
from substrate.mempool_types import MempoolSolverInfo, MinerType
from substrate.solver_registration import SolverGuardOutcome, ensure_solver_registered
from substrate.types import ExtrinsicReceipt


ACCOUNT = b"\x11" * 32
RACE_ERROR = "quantumComputeMempool.SolverAlreadyRegistered"


def _signer() -> MagicMock:
    signer = MagicMock()
    signer.account_id_bytes.return_value = ACCOUNT
    signer.ss58_address.return_value = "5FakeSolverAddress"
    return signer


def _solver_info(solver_type: MinerType) -> MempoolSolverInfo:
    return MempoolSolverInfo(
        account=ACCOUNT,
        solver_type=solver_type,
        registered_at=1,
        solutions_submitted=0,
        rewards_earned=0,
    )


def _receipt(error: str | None = None) -> ExtrinsicReceipt:
    return ExtrinsicReceipt(extrinsic_hash="0xabc", block_hash="0xdef", error=error)


def _client(query_solver=None, register_solver=None) -> MagicMock:
    client = MagicMock()
    client.query_solver = query_solver or AsyncMock(return_value=None)
    client.register_solver = register_solver or AsyncMock(return_value=_receipt())
    client.deregister_solver = AsyncMock()
    return client


# ----------------------------------------------------------------------
# ensure_solver_registered branch matrix
# ----------------------------------------------------------------------


async def test_already_registered_matching_type_skips_extrinsic():
    client = _client(
        query_solver=AsyncMock(return_value=_solver_info(MinerType.QPU_DWAVE))
    )
    outcome = await ensure_solver_registered(client, _signer(), "qpu")
    assert outcome is SolverGuardOutcome.ALREADY_REGISTERED
    client.register_solver.assert_not_awaited()
    client.deregister_solver.assert_not_awaited()


async def test_unregistered_registers_vendor_resolved_type():
    signer = _signer()
    client = _client()
    outcome = await ensure_solver_registered(client, signer, "qpu_ibm")
    assert outcome is SolverGuardOutcome.REGISTERED
    # The vendor-resolved kind must reach the chain — QpuIbm, never the
    # backend-group default QpuDwave.
    client.register_solver.assert_awaited_once_with(signer, MinerType.QPU_IBM)


async def test_pre_existing_mismatch_is_type_mismatch_without_deregister():
    client = _client(query_solver=AsyncMock(return_value=_solver_info(MinerType.CPU)))
    outcome = await ensure_solver_registered(client, _signer(), "gpu")
    assert outcome is SolverGuardOutcome.TYPE_MISMATCH
    client.register_solver.assert_not_awaited()
    client.deregister_solver.assert_not_awaited()


async def test_registration_race_requery_matching_is_already_registered():
    # Sibling child on the same account registered between our query and
    # our extrinsic — the loser re-queries and accepts the matching type.
    client = _client(
        query_solver=AsyncMock(side_effect=[None, _solver_info(MinerType.CPU)]),
        register_solver=AsyncMock(return_value=_receipt(RACE_ERROR)),
    )
    outcome = await ensure_solver_registered(client, _signer(), "cpu")
    assert outcome is SolverGuardOutcome.ALREADY_REGISTERED
    assert client.query_solver.await_count == 2
    client.deregister_solver.assert_not_awaited()


async def test_registration_race_requery_mismatch_is_type_mismatch_no_deregister():
    client = _client(
        query_solver=AsyncMock(side_effect=[None, _solver_info(MinerType.GPU)]),
        register_solver=AsyncMock(return_value=_receipt(RACE_ERROR)),
    )
    outcome = await ensure_solver_registered(client, _signer(), "cpu")
    assert outcome is SolverGuardOutcome.TYPE_MISMATCH
    client.deregister_solver.assert_not_awaited()


async def test_query_rpc_failure_returns_failed_without_raising():
    client = _client(query_solver=AsyncMock(side_effect=ConnectionError("ws down")))
    outcome = await ensure_solver_registered(client, _signer(), "cpu")
    assert outcome is SolverGuardOutcome.FAILED
    client.register_solver.assert_not_awaited()


async def test_register_submit_exception_returns_failed():
    client = _client(register_solver=AsyncMock(side_effect=TimeoutError("no receipt")))
    outcome = await ensure_solver_registered(client, _signer(), "cpu")
    assert outcome is SolverGuardOutcome.FAILED


async def test_register_non_race_receipt_error_returns_failed():
    client = _client(
        register_solver=AsyncMock(return_value=_receipt("Token.FundsUnavailable"))
    )
    outcome = await ensure_solver_registered(client, _signer(), "cpu")
    assert outcome is SolverGuardOutcome.FAILED
    client.deregister_solver.assert_not_awaited()


async def test_race_requery_rpc_failure_returns_failed():
    client = _client(
        query_solver=AsyncMock(side_effect=[None, ConnectionError("ws down")]),
        register_solver=AsyncMock(return_value=_receipt(RACE_ERROR)),
    )
    outcome = await ensure_solver_registered(client, _signer(), "cpu")
    assert outcome is SolverGuardOutcome.FAILED


async def test_race_requery_none_returns_failed():
    # SolverAlreadyRegistered followed by an empty re-query means the chain
    # view is inconsistent (a concurrent deregister) — FAILED, not a
    # confirmed type mismatch.
    client = _client(
        query_solver=AsyncMock(side_effect=[None, None]),
        register_solver=AsyncMock(return_value=_receipt(RACE_ERROR)),
    )
    outcome = await ensure_solver_registered(client, _signer(), "cpu")
    assert outcome is SolverGuardOutcome.FAILED


# ----------------------------------------------------------------------
# register-solver CLI: outcome → exit-code mapping
# ----------------------------------------------------------------------


def _invoke_register_solver(monkeypatch, outcome: SolverGuardOutcome):
    keystore = MagicMock()
    keystore.signer = _signer()
    monkeypatch.setattr(quip_cli, "_load_keystore_or_fail", lambda path: keystore)
    client = MagicMock()
    client.close = AsyncMock()
    monkeypatch.setattr(quip_cli, "_connect_or_fail", AsyncMock(return_value=client))
    guard = AsyncMock(return_value=outcome)
    monkeypatch.setattr(quip_cli, "ensure_solver_registered", guard)
    result = CliRunner().invoke(
        quip_cli.quip_miner_register_solver,
        ["--validator", "ws://localhost:9944", "--miner-type", "qpu_ibm"],
    )
    return result, guard, client


@pytest.mark.parametrize(
    "outcome,expected_code",
    [
        (SolverGuardOutcome.ALREADY_REGISTERED, 0),
        (SolverGuardOutcome.REGISTERED, 0),
        (SolverGuardOutcome.TYPE_MISMATCH, 4),
        (SolverGuardOutcome.FAILED, 3),
    ],
)
def test_register_solver_cli_exit_codes(monkeypatch, outcome, expected_code):
    result, guard, client = _invoke_register_solver(monkeypatch, outcome)
    assert result.exit_code == expected_code, result.output
    guard.assert_awaited_once()
    # The CLI hands the guard the vendor-resolved kind string verbatim.
    assert guard.await_args.args[2] == "qpu_ibm"
    client.close.assert_awaited_once()
