"""GPU miner using Metal/MPS via GPUSampler('mps')."""
from __future__ import annotations

from shared.base_miner import BaseMiner
from GPU.sampler import LocalGPUSampler as GPUSampler  # temporary alias until rename


class MetalMiner(BaseMiner):
    def __init__(self, miner_id: str, **cfg):
        super().__init__(miner_id, cfg["difficulty_energy"], cfg["min_diversity"], cfg["min_solutions"])
        self.miner_type = "GPU-MPS"
        self.sampler = GPUSampler("mps")

