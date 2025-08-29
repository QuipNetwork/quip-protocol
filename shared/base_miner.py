"""Abstract base miner for long-running worker architecture.

Provides a thin wrapper around the existing shared.miner.Miner implementation
by constructing the appropriate sampler in subclasses and delegating mining.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import multiprocessing

from shared.miner import Miner as ImplMiner, MiningResult


class BaseMiner:
    """Abstract base class for concrete miners.

    Subclasses must set:
      - self.miner_type (str)
      - self.sampler (object with sample_ising)
    and construct the delegate implementation (self._impl).
    """

    def __init__(
        self,
        miner_id: str,
        difficulty_energy: float,
        min_diversity: float,
        min_solutions: int,
    ) -> None:
        if type(self) is BaseMiner:
            raise TypeError("BaseMiner is abstract; instantiate a concrete subclass")
        self.miner_id = miner_id
        self.miner_type: str = "UNKNOWN"
        self.sampler = None
        self._impl: Optional[ImplMiner] = None
        self.difficulty_energy = difficulty_energy
        self.min_diversity = min_diversity
        self.min_solutions = min_solutions

    def _init_impl(self) -> None:
        """Create the delegate ImplMiner once sampler and miner_type are set."""
        if self.sampler is None:
            raise RuntimeError("Sampler not initialized in miner subclass")
        if not self.miner_type:
            raise RuntimeError("miner_type not set in miner subclass")
        if self._impl is None:
            self._impl = ImplMiner(
                miner_id=self.miner_id,
                miner_type=self.miner_type,
                sampler=self.sampler,
                difficulty_energy=self.difficulty_energy,
                min_diversity=self.min_diversity,
                min_solutions=self.min_solutions,
            )

    # API expected by the worker loop
    def mine_block(
        self,
        block_header: str,
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
    ) -> Optional[MiningResult]:
        """Delegate to the shared.miner.Miner implementation."""
        self._init_impl()
        return self._impl.mine_block(block_header, result_queue, stop_event)  # type: ignore[union-attr]

    def get_stats(self) -> Dict[str, Any]:
        """Return machine-readable stats for this miner."""
        self._init_impl()
        # timing_stats exists on ImplMiner; add basic identity info
        stats = dict(self._impl.timing_stats)  # type: ignore[union-attr]
        stats.update({
            "miner_id": self.miner_id,
            "miner_type": self.miner_type,
        })
        return stats

    def stop_mining(self) -> None:
        """No-op: cancellation is handled by stop_event provided to mine_block."""
        return None

    def shutdown(self) -> None:
        """Close underlying sampler if it supports close()."""
        try:
            if hasattr(self.sampler, "close"):
                self.sampler.close()
        except Exception:
            pass

