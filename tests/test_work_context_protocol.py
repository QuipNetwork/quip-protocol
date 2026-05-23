"""Verify both context types satisfy the WorkContext Protocol structurally."""
from __future__ import annotations

from shared.work_context import WorkContext
from shared.mempool_types import MempoolJobContext
from substrate.types import SubstrateMiningContext


def test_substrate_mining_context_satisfies_protocol():
    """SubstrateMiningContext must satisfy WorkContext without explicit inheritance."""
    # Construct a minimal valid instance — only fields needed for the Protocol check.
    ctx = SubstrateMiningContext(
        last_proof_block_hash=b"\x00" * 32,
        topology_hash=b"\x00" * 32,
        nodes=[0, 1],
        edges=[(0, 1)],
        difficulty=None,  # Protocol doesn't dictate type, only presence
        miner_account_bytes=b"\x00" * 32,
        allowed_h_values=None,
        allowed_j_values=None,
        allowed_spin_values=None,
    )
    # runtime_checkable Protocol: isinstance() works.
    assert isinstance(ctx, WorkContext)


def test_mempool_job_context_satisfies_protocol():
    """MempoolJobContext must satisfy WorkContext without explicit inheritance."""
    ctx = MempoolJobContext(
        order_id=1,
        nodes=(0, 1),
        edges=((0, 1),),
        h_values=(0, 0),
        j_values=(0,),
        min_energy_milli=None,
        min_diversity_milli=None,
        min_solutions=None,
    )
    assert isinstance(ctx, WorkContext)


def test_arbitrary_object_does_not_satisfy_protocol():
    """A bare dict or unrelated class must not be considered a WorkContext."""
    assert not isinstance({"nodes": [0, 1]}, WorkContext)
    assert not isinstance(object(), WorkContext)
