# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Layering guard: ``shared/`` must not import UP into the chain/backends.

``shared/`` is meant to be the project's foundation layer — value types,
crypto, ising math, the shared-memory ring, and the ``WorkContext`` Protocol.
A foundation must be a *sink*: backends (``CPU``/``GPU``/``QPU``) and the
``substrate`` chain layer may import ``shared``, but ``shared`` must not import
*them*. Every such edge is a layering inversion (see ARCHITECTURE_REVIEW.md).

This test pins the inversions that exist **today** in ``_ALLOWED`` and asserts
the set of module-level offenders is *exactly* that allow-list — so no new
inversion can be added, and each one removed by the layering roadmap must be
struck from ``_ALLOWED`` in the same commit. When ``_ALLOWED`` is empty,
``shared/`` is a true foundation and this guard becomes a permanent ratchet.

"Module-level" means *executes at import time*: imports inside module-scope
``try:``/``if:`` blocks count, but imports deferred inside a ``def``/``async
def`` body do **not** (that lazy-import pattern is the sanctioned escape hatch,
e.g. ``base_miner._ensure_driver`` importing ``QPU.stream_driver`` only when it
spawns the stream driver).
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = REPO_ROOT / "shared"

# Top-level packages that sit ABOVE the shared foundation. A module-level
# absolute import of any of these from within shared/ is a layering inversion.
_UPWARD_PACKAGES = frozenset({"substrate", "CPU", "GPU", "QPU"})

# The inversions that exist today, as (filename, imported-dotted-path). The
# layering roadmap removes these one move at a time; strike each entry here in
# the same commit that eliminates the corresponding import. Target: empty set.
_ALLOWED = frozenset(
    {
        ("miner_bootstrap.py", "substrate.client"),
        ("miner_bootstrap.py", "substrate.types"),
        ("telemetry_process.py", "substrate.client"),
        ("telemetry_process.py", "substrate.url_failover"),
    }
)


def _is_upward(dotted: str) -> bool:
    """True if a dotted module path's top segment is an upward package."""
    return dotted.split(".", 1)[0] in _UPWARD_PACKAGES


class _ModuleLevelImportCollector(ast.NodeVisitor):
    """Collect upward imports that run at import time, skipping function bodies.

    Descending into ``FunctionDef``/``AsyncFunctionDef`` is suppressed so that
    deferred in-function imports (the sanctioned lazy-coupling escape hatch) are
    not counted as foundation-level inversions.
    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.offenders: set[tuple[str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        # Do not descend: in-function imports are deferred, not import-time.
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if _is_upward(alias.name):
                self.offenders.add((self.filename, alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        # level > 0 is a relative (intra-shared) import — never an inversion.
        if node.level == 0 and node.module and _is_upward(node.module):
            self.offenders.add((self.filename, node.module))


def _collect_inversions() -> set[tuple[str, str]]:
    """Walk shared/ and return every module-level upward import edge."""
    offenders: set[tuple[str, str]] = set()
    for path in sorted(SHARED_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        collector = _ModuleLevelImportCollector(path.name)
        collector.visit(tree)
        offenders |= collector.offenders
    return offenders


def test_shared_has_no_unapproved_upward_imports() -> None:
    """shared/ must not import substrate/CPU/GPU/QPU beyond the allow-list."""
    found = _collect_inversions()

    new_inversions = found - _ALLOWED
    assert not new_inversions, (
        "New layering inversion(s) added to shared/ (foundation must not import "
        "UP into substrate/CPU/GPU/QPU). Move the orchestration code out of "
        "shared/ instead of adding the import:\n  "
        + "\n  ".join(f"{f} -> import {m}" for f, m in sorted(new_inversions))
    )

    stale_allowances = _ALLOWED - found
    assert not stale_allowances, (
        "These inversions are listed in _ALLOWED but no longer exist — the "
        "ratchet only tightens, so strike them from _ALLOWED:\n  "
        + "\n  ".join(f"{f} -> import {m}" for f, m in sorted(stale_allowances))
    )
