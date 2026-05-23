"""Verify context types implement their own resolve_ising and requirements()."""
from __future__ import annotations

from shared.mempool_types import MempoolJobContext
from substrate.types import SubstrateMiningContext


def test_substrate_context_has_resolve_ising_method():
    """SubstrateMiningContext must implement resolve_ising as a method."""
    ctx = SubstrateMiningContext(
        last_proof_block_hash=b"\x00" * 32,
        topology_hash=b"\x00" * 32,
        nodes=[0, 1],
        edges=[(0, 1)],
        difficulty=None,
        miner_account_bytes=b"\x00" * 32,
        allowed_h_values=None,
        allowed_j_values=None,
        allowed_spin_values=None,
    )
    assert callable(getattr(ctx, "resolve_ising", None))
    assert callable(getattr(ctx, "requirements", None))


def test_mempool_context_has_resolve_ising_method():
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
    assert callable(getattr(ctx, "resolve_ising", None))
    assert callable(getattr(ctx, "requirements", None))
