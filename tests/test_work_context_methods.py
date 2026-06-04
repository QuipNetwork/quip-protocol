"""Verify context types implement their own resolve_ising and requirements()."""
from __future__ import annotations

from substrate.mempool_types import MempoolJobContext
from substrate.types import SubstrateMiningContext


def _pow_context(*, decay_schedule=None) -> SubstrateMiningContext:
    """Minimal valid PoW context; decay_schedule defaults to absent (None)."""
    return SubstrateMiningContext(
        last_proof_block_hash=b"\x00" * 32,
        topology_hash=b"\x00" * 32,
        nodes=[0, 1],
        edges=[(0, 1)],
        difficulty=None,
        miner_account_bytes=b"\x00" * 32,
        allowed_h_values=None,
        allowed_j_values=None,
        allowed_spin_values=None,
        block_hash=b"\x00" * 32,
        block_number=0,
        decay_schedule=decay_schedule,
    )


def _mempool_context() -> MempoolJobContext:
    return MempoolJobContext(
        order_id=1,
        nodes=(0, 1),
        edges=((0, 1),),
        h_values=(0, 0),
        j_values=(0,),
        min_energy_milli=None,
        min_diversity_milli=None,
        min_solutions=None,
    )


def test_pow_context_uses_decay_ratchet():
    """PoW work always takes the decay-ratchet loop."""
    assert _pow_context(decay_schedule=[100, 200, 300]).uses_decay_ratchet() is True


def test_pow_context_uses_ratchet_even_without_decay_schedule():
    """A decay-less PoW context still takes the ratchet path, not the mempool
    path — the discriminator is the work source, not whether decay is active.
    (The ratchet loop falls back to strict energy ranking when the schedule is
    None; routing it to the mempool path would be a regression.)"""
    assert _pow_context(decay_schedule=None).uses_decay_ratchet() is True


def test_mempool_context_does_not_use_decay_ratchet():
    """Mempool jobs use strict-energy evaluation, not the ratchet."""
    assert _mempool_context().uses_decay_ratchet() is False


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


def test_work_context_module_has_no_substrate_imports():
    """Architectural regression guard: shared/work_context.py must not import substrate.

    The whole point of the WorkContext Protocol refactor is to eliminate
    the backwards `shared/` → `substrate/` dependency. If anyone adds
    `from substrate.X import …` back to shared/work_context.py — at module
    scope OR inside a function — this test catches it.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).parent.parent
    src = (repo_root / "shared" / "work_context.py").read_text()
    # Strip docstrings and comments before scanning so docstring references don't trip the test.
    # A line containing an EVEN number of `"""` opens and closes within itself
    # (single-line docstring) — net state unchanged. An ODD count toggles the
    # in_docstring state for the lines that follow.
    code_lines = []
    in_triple = False
    for line in src.splitlines():
        stripped = line.lstrip()
        triple_count = stripped.count('"""')
        if triple_count > 0:
            # Single-line `"""foo"""` keeps in_triple state stable; multi-line
            # opening or closing toggles it. Either way the line itself is part
            # of a docstring — skip it.
            if triple_count % 2 == 1:
                in_triple = not in_triple
            continue
        if in_triple:
            continue
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    assert "from substrate" not in code, (
        "shared/work_context.py contains `from substrate ...` — "
        "this reintroduces the backwards shared→substrate dependency"
    )
    assert "import substrate" not in code, (
        "shared/work_context.py contains `import substrate` — "
        "this reintroduces the backwards shared→substrate dependency"
    )
