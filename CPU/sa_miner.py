"""CPU miner using SimulatedAnnealingStructuredSampler."""
from __future__ import annotations

from shared.base_miner import BaseMiner
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler


class SimulatedAnnealingMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        super().__init__(miner_id, cfg["difficulty_energy"], cfg["min_diversity"], cfg["min_solutions"])
        self.miner_type = "CPU"
        self.sampler = SimulatedAnnealingStructuredSampler()

