"""QPU miner using D-Wave sampler or mock via create_dwave_sampler."""
from __future__ import annotations

from shared.base_miner import BaseMiner
from QPU.dwave_sampler import create_dwave_sampler


class DWaveMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        super().__init__(miner_id, cfg["difficulty_energy"], cfg["min_diversity"], cfg["min_solutions"])
        self.miner_type = "QPU"
        self.sampler = create_dwave_sampler()

