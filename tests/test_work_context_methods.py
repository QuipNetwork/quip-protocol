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
    code_lines = []
    in_triple = False
    for line in src.splitlines():
        stripped = line.lstrip()
        if '"""' in stripped:
            # Toggle for triple-quote start/end (handles both opening and closing).
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
