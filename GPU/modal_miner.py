"""GPU miner using Modal via ModalSampler(gpu_type)."""
from __future__ import annotations

from shared.base_miner import BaseMiner
from GPU.modal_sampler import ModalSampler


class ModalMiner(BaseMiner):
    def __init__(self, miner_id: str, gpu_type: str = "t4", **cfg):
        super().__init__(miner_id, cfg["difficulty_energy"], cfg["min_diversity"], cfg["min_solutions"])
        self.miner_type = f"GPU-{gpu_type.upper()}"
        self.sampler = ModalSampler(gpu_type)

