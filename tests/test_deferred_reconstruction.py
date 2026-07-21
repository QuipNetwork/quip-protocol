# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for consumer-side deferred-reconstruction machinery (Task 1 / Step 3).

All tests are deterministic (no real D-Wave, no multiprocessing queues beyond
what is needed).  The oracle: with ``defect_info=None`` (every current code
path) behaviour is identical to before this change.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from shared.base_miner import BaseMiner, _MiningLoopState
from shared.miner_types import BlockRequirements, MiningResult


# ---------------------------------------------------------------------------
# Minimal no-op attempt log / solution store (writes nowhere)
# ---------------------------------------------------------------------------


class _NullLogger:
    """AttemptLogger/SolutionStore stand-in that discards everything."""

    def record(self, **_kwargs: Any) -> None:
        pass

    def metadata_update(self, **_kwargs: Any) -> None:
        pass

    def flush(self, *_a: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NODES = [0, 1, 2]
_EDGES: list = []


def _make_requirements(energy: float = -15000.0) -> BlockRequirements:
    return BlockRequirements(
        difficulty_energy=energy,
        min_diversity=0.0,
        min_solutions=1,
        timeout_to_difficulty_adjustment_decay=99999,
    )


def _make_loop_state(nodes=None, energy: float = -15000.0) -> _MiningLoopState:
    nodes = list(nodes or _NODES)
    edges: list = []
    null = _NullLogger()
    return _MiningLoopState(
        requirements=_make_requirements(energy),
        nodes=nodes,
        edges=edges,
        prev_timestamp=0,
        start_time=0.0,
        solution_number_for_log=1,
        dispatch_id_for_log=1,
        attempt_log=null,  # type: ignore[arg-type]
        solution_store=null,  # type: ignore[arg-type]
        live_threshold_var=None,
        top_k_cap=5,
        top_k=[],
        previewed_wintime=(10**9, 10**9),
    )


def _make_sampleset(n_rows: int, n_cols: int, energy_val: float = -16000.0):
    """Return a minimal sampleset-like namespace."""
    sample = np.ones((n_rows, n_cols), dtype=np.int8)
    energy = np.full(n_rows, energy_val, dtype=np.float64)
    return SimpleNamespace(
        record=SimpleNamespace(sample=sample, energy=energy),
        info={},
    )


# ---------------------------------------------------------------------------
# Minimal concrete miner for unit-testing ratchet hooks
# ---------------------------------------------------------------------------


class _UnitMiner(BaseMiner):
    """Concrete BaseMiner subclass for direct _run_substrate_ratchet tests.

    ``evaluate_sampleset`` always returns ``None`` (never a winner) unless
    overridden by a subclass.  ``_adapt_mining_params`` raises so any
    accidental setup path is caught immediately.
    """

    def __init__(self) -> None:
        super().__init__("unit-miner", sampler=object(), miner_type="test")
        self.time_manager = None

    def _adapt_mining_params(self, requirements, nodes, edges) -> dict:
        raise AssertionError("should not reach _adapt_mining_params in unit tests")

    def evaluate_sampleset(self, *_args: Any, **_kwargs: Any) -> Optional[MiningResult]:
        return None


# ---------------------------------------------------------------------------
# Step 3 — _finalize_sample base is identity
# ---------------------------------------------------------------------------


def test_base_finalize_sample_is_identity() -> None:
    """BaseMiner._finalize_sample returns the sampleset unchanged."""
    miner = _UnitMiner()
    ss = _make_sampleset(4, 3)
    defect = object()  # arbitrary non-None object
    result = miner._finalize_sample(ss, defect, [0, 1, 2])
    assert result is ss, "_finalize_sample base must return the sampleset unchanged"


# ---------------------------------------------------------------------------
# defect_info=None + full-width sampleset → evaluates as before
# ---------------------------------------------------------------------------


class _EvalCountMiner(_UnitMiner):
    """Records every evaluate_sampleset call."""

    def __init__(self) -> None:
        super().__init__()
        self.eval_calls: list = []

    def evaluate_sampleset(  # type: ignore[override]
        self,
        sampleset: Any,
        requirements: Any,
        nodes: Any,
        edges: Any,
        nonce: Any,
        salt: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[MiningResult]:
        self.eval_calls.append(sampleset)
        return None


def test_null_defect_info_full_width_evaluates_normally() -> None:
    """With defect_info=None and a full-width sampleset, evaluate is called.

    This is the regression guard: the defect_info machinery must be fully
    transparent when defect_info is None (all current paths).
    """
    miner = _EvalCountMiner()
    state = _make_loop_state()
    # Full-width: cols == len(state.nodes) == 3
    ss = _make_sampleset(n_rows=4, n_cols=len(state.nodes), energy_val=-16000.0)

    miner._run_substrate_ratchet(
        state, ss, b"\x00" * 32, b"\x01" * 32, 0.0,
        preview_cb=None,
        attempt_log_kwargs={},
        defect_info=None,
    )

    assert len(miner.eval_calls) == 1, (
        f"expected exactly one evaluate_sampleset call, got {len(miner.eval_calls)}"
    )


# ---------------------------------------------------------------------------
# Reduced-width + defect_info → _finalize_sample called (then evaluate)
# ---------------------------------------------------------------------------


class _FinalizeSpy(_UnitMiner):
    """Records _finalize_sample invocations; returns a full-width sampleset."""

    def __init__(self, nodes: list) -> None:
        super().__init__()
        self._nodes = nodes
        self.finalize_calls: list = []
        self.eval_calls: list = []

    def _finalize_sample(self, sampleset: Any, defect_info: Any, nodes: Any) -> Any:
        self.finalize_calls.append((sampleset, defect_info, nodes))
        # Return a full-width sampleset so evaluate can proceed.
        full = _make_sampleset(
            n_rows=sampleset.record.sample.shape[0],
            n_cols=len(self._nodes),
            energy_val=float(sampleset.record.energy[0]),
        )
        return full

    def evaluate_sampleset(  # type: ignore[override]
        self,
        sampleset: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[MiningResult]:
        self.eval_calls.append(sampleset)
        return None


def test_reduced_width_with_defect_info_calls_finalize_then_evaluate() -> None:
    """A reduced-width sampleset that passes energy pre-check + has defect_info
    triggers _finalize_sample and then evaluate_sampleset.
    """
    nodes = [0, 1, 2]
    miner = _FinalizeSpy(nodes)
    state = _make_loop_state(nodes=nodes)

    # Reduced: only 2 columns instead of len(nodes)=3
    reduced_cols = len(nodes) - 1
    ss = _make_sampleset(n_rows=4, n_cols=reduced_cols, energy_val=-16000.0)

    # Fake defect_info: needs only .energy_offset for the ratchet
    fake_defect = SimpleNamespace(energy_offset=0.0)

    miner._run_substrate_ratchet(
        state, ss, b"\x00" * 32, b"\x01" * 32, 0.0,
        preview_cb=None,
        attempt_log_kwargs={},
        defect_info=fake_defect,
    )

    assert len(miner.finalize_calls) == 1, (
        "_finalize_sample must be called once for a reduced-width survivor"
    )
    called_ss, called_defect, called_nodes = miner.finalize_calls[0]
    assert called_ss is ss
    assert called_defect is fake_defect
    assert called_nodes == nodes, "_finalize_sample must receive the topology nodes"
    assert len(miner.eval_calls) == 1, (
        "evaluate_sampleset must be called after _finalize_sample"
    )


# ---------------------------------------------------------------------------
# Reduced-width WITHOUT defect_info → skipped (improves_stash forced False)
# ---------------------------------------------------------------------------


def test_reduced_width_without_defect_info_skips_evaluation() -> None:
    """A reduced-width sampleset with defect_info=None is skipped entirely.

    This preserves the old under-reconstructed-sample guard behaviour:
    evaluate_sampleset must NOT be called on a narrow sample without defect.
    """
    nodes = [0, 1, 2]
    miner = _FinalizeSpy(nodes)
    state = _make_loop_state(nodes=nodes)

    # Reduced: only 2 columns instead of len(nodes)=3
    reduced_cols = len(nodes) - 1
    ss = _make_sampleset(n_rows=4, n_cols=reduced_cols, energy_val=-16000.0)

    miner._run_substrate_ratchet(
        state, ss, b"\x00" * 32, b"\x01" * 32, 0.0,
        preview_cb=None,
        attempt_log_kwargs={},
        defect_info=None,
    )

    assert miner.finalize_calls == [], (
        "_finalize_sample must NOT be called when defect_info is None"
    )
    assert miner.eval_calls == [], (
        "evaluate_sampleset must NOT be called for an under-reconstructed sample"
        " without defect_info"
    )


# ===========================================================================
# _run_mempool_eval — width/finalize handling (Task 3 / Step 4)
# ===========================================================================


def test_mempool_reduced_width_with_defect_info_calls_finalize_then_evaluate() -> None:
    """Reduced-width QPU mempool sample + defect_info → finalize then evaluate."""
    nodes = [0, 1, 2]
    miner = _FinalizeSpy(nodes)
    state = _make_loop_state(nodes=nodes)

    # Reduced: only 2 columns instead of len(nodes)=3
    reduced_cols = len(nodes) - 1
    ss = _make_sampleset(n_rows=4, n_cols=reduced_cols, energy_val=-16000.0)

    fake_defect = SimpleNamespace(energy_offset=0.0)

    miner._run_mempool_eval(
        state, ss, b"\x00" * 32, b"\x01" * 32, 0.0,
        attempt_log_kwargs={},
        defect_info=fake_defect,
    )

    assert len(miner.finalize_calls) == 1, (
        "_finalize_sample must be called once for a reduced-width QPU mempool sample"
    )
    called_ss, called_defect, called_nodes = miner.finalize_calls[0]
    assert called_ss is ss
    assert called_defect is fake_defect
    assert called_nodes == nodes, "_finalize_sample must receive the topology nodes"
    assert len(miner.eval_calls) == 1, (
        "evaluate_sampleset must be called after _finalize_sample"
    )


def test_mempool_reduced_width_without_defect_info_returns_none() -> None:
    """Reduced-width mempool sample with defect_info=None → returns None; eval not called."""
    nodes = [0, 1, 2]
    miner = _FinalizeSpy(nodes)
    state = _make_loop_state(nodes=nodes)

    # Reduced: only 2 columns instead of len(nodes)=3
    reduced_cols = len(nodes) - 1
    ss = _make_sampleset(n_rows=4, n_cols=reduced_cols, energy_val=-16000.0)

    result = miner._run_mempool_eval(
        state, ss, b"\x00" * 32, b"\x01" * 32, 0.0,
        attempt_log_kwargs={},
        defect_info=None,
    )

    assert result is None, (
        "_run_mempool_eval must return None for under-reconstructed sample without defect_info"
    )
    assert miner.finalize_calls == [], (
        "_finalize_sample must NOT be called when defect_info is None"
    )
    assert miner.eval_calls == [], (
        "evaluate_sampleset must NOT be called for an under-reconstructed sample"
        " without defect_info"
    )


def test_mempool_full_width_no_defect_info_evaluates_normally() -> None:
    """Full-width mempool sample + defect_info=None → evaluate called, no finalize.

    Regression guard: Metal/CUDA mempool samples must be unaffected.
    """
    nodes = [0, 1, 2]
    miner = _FinalizeSpy(nodes)
    state = _make_loop_state(nodes=nodes)

    # Full-width: cols == len(nodes)
    ss = _make_sampleset(n_rows=4, n_cols=len(nodes), energy_val=-16000.0)

    miner._run_mempool_eval(
        state, ss, b"\x00" * 32, b"\x01" * 32, 0.0,
        attempt_log_kwargs={},
        defect_info=None,
    )

    assert miner.finalize_calls == [], (
        "_finalize_sample must NOT be called for full-width samples"
    )
    assert len(miner.eval_calls) == 1, (
        "evaluate_sampleset must be called exactly once for full-width mempool samples"
    )
