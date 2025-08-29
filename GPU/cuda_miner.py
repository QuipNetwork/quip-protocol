"""GPU miner using CUDA via GPUSampler(device)."""
from __future__ import annotations

from shared.base_miner import BaseMiner
from GPU.sampler import LocalGPUSampler as GPUSampler  # temporary alias until rename


class CudaMiner(BaseMiner):
    def __init__(self, miner_id: str, device: str = "0", **cfg):
        super().__init__(miner_id, cfg["difficulty_energy"], cfg["min_diversity"], cfg["min_solutions"])
        self.miner_type = f"GPU-LOCAL:{device}"
        self.sampler = GPUSampler(str(device))

