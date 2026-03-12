# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Gibbs miner using chromatic parallel block Gibbs sampling."""
from __future__ import annotations

import random
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)
from GPU.cuda_gibbs_sa import CudaGibbsSampler
from GPU.gpu_scheduler import KernelScheduler

try:
    import cupy as cp
except ImportError:
    cp = None


class CudaGibbsMiner(BaseMiner):
    """CUDA GPU miner using chromatic block Gibbs sampling.

    Uses CudaGibbsSampler with KernelScheduler for SM budgeting.

    Config (via **cfg):
        gpu_utilization: 1-100 (default 100).
        yielding: True = NVML-adaptive, yield to other GPU
            users. False (default) = static SM budget.
    """

    ADAPT_MIN_SWEEPS = 256
    ADAPT_MAX_SWEEPS = 2048
    ADAPT_MIN_READS = 64
    ADAPT_MAX_READS = 256
    ADAPT_EXTRA_PARAMS = {'num_sweeps_per_beta': 1}

    def __init__(
        self,
        miner_id: str,
        device: str = "0",
        topology=None,
        update_mode: str = "gibbs",
        **cfg,
    ):
        if cp is None:
            raise ImportError("cupy not available")

        cp.cuda.Device(int(device)).use()

        self.device = device
        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {self.gpu_utilization}"
            )

        dev_id = int(device)
        dev_obj = cp.cuda.Device(dev_id)
        device_sms = dev_obj.attributes[
            'MultiProcessorCount'
        ]
        dev_props = cp.cuda.runtime.getDeviceProperties(
            dev_id,
        )
        dev_name = dev_props.get('name', 'unknown')
        if isinstance(dev_name, bytes):
            dev_name = dev_name.decode()

        yielding = bool(cfg.get('yielding', False))

        self.sms_per_nonce = cfg.get('sms_per_nonce', 4)

        self._scheduler = KernelScheduler(
            device_id=dev_id,
            device_sms=device_sms,
            gpu_utilization_pct=self.gpu_utilization,
            yielding=yielding,
        )

        # Static ceiling for buffer allocation
        sm_ceiling = self._scheduler.get_sm_budget()
        max_nonces = max(
            1, 2 * (sm_ceiling // self.sms_per_nonce),
        )

        sampler = CudaGibbsSampler(
            topology=topology,
            update_mode=update_mode,
            max_sms=sm_ceiling,
        )
        sampler.prepare(
            num_reads=self.ADAPT_MAX_READS,
            num_sweeps=self.ADAPT_MAX_SWEEPS,
            num_sweeps_per_beta=1,
            max_nonces=max_nonces,
        )
        super().__init__(
            miner_id, sampler, miner_type="GPU-CUDA-Gibbs",
        )

        signal.signal(signal.SIGTERM, self._cleanup_handler)

        self.logger.info(
            "GPU %s: %s | utilization=%d%% | "
            "SMs=%d/%d | yielding=%s",
            device, dev_name,
            self.gpu_utilization,
            sm_ceiling, device_sms,
            yielding,
        )

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA resource cleanup."""
        if hasattr(self, 'logger'):
            self.logger.info(
                f"CUDA Gibbs miner {self.miner_id} received "
                f"SIGTERM, cleaning up..."
            )

        try:
            self.sampler.close()
            self._scheduler.stop()
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(
                    f"Error during CUDA Gibbs cleanup: {e}"
                )

        sys.exit(0)

    def _post_mine_cleanup(self) -> None:
        """Release CUDA streams and stop NVML monitor."""
        self.sampler.close()
        self._scheduler.stop()

    def _pre_mine_setup(self, *args, **kwargs) -> bool:
        """Set CUDA device context before mining."""
        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(
                f"Failed to set device context: {e}"
            )
            return False
        return True

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        return self.adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
        )

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int = 1,
        **kwargs,
    ) -> dimod.SampleSet:
        results = self.sampler.sample_ising(
            [h], [J],
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
        )
        return results[0]

    def _generate_nonce_batch(
        self,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        num_nonces: int,
    ) -> Tuple[
        List[Dict], List[Dict], List[int], List[bytes]
    ]:
        """Generate h/J/nonce/salt for a batch on CPU."""
        h_list: List[Dict] = []
        J_list: List[Dict] = []
        nonces: List[int] = []
        salts: List[bytes] = []
        for _ in range(num_nonces):
            salt = random.randbytes(32)
            nonce = ising_nonce_from_block(
                prev_hash, miner_id, cur_index, salt,
            )
            h, J = generate_ising_model_from_nonce(
                nonce, nodes, edges,
            )
            h_list.append(h)
            J_list.append(J)
            nonces.append(nonce)
            salts.append(salt)
        return h_list, J_list, nonces, salts

    def _sample_batch(
        self,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        *,
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int = 1,
        **kwargs,
    ) -> Optional[
        List[Tuple[int, bytes, dimod.SampleSet]]
    ]:
        """3-stage pipeline: GPU compute | CPU generate | H2D.

        CPU nonce generation overlaps with GPU kernel via
        launch/harvest split.
        """
        if self._scheduler.should_throttle():
            time.sleep(0.5)

        sm_budget = self._scheduler.get_sm_budget()
        num_nonces = max(
            1, 2 * (sm_budget // self.sms_per_nonce),
        )

        sampler = self.sampler
        mn_params = dict(
            reads_per_nonce=num_reads,
            num_sweeps=num_sweeps,
            sms_per_nonce=self.sms_per_nonce,
        )

        # Step 1: launch kernel (async — GPU starts computing)
        if sampler._mn_preloaded:
            nonces = self._preload_nonces
            salts = self._preload_salts
            sampler.launch_multi_nonce(
                [], [], **mn_params,
            )
        else:
            h_list, J_list, nonces, salts = (
                self._generate_nonce_batch(
                    prev_hash, miner_id, cur_index,
                    nodes, edges, num_nonces,
                )
            )
            sampler.launch_multi_nonce(
                h_list, J_list, **mn_params,
            )

        # Step 2: generate NEXT batch on CPU (overlaps GPU!)
        next_h, next_J, next_nonces, next_salts = (
            self._generate_nonce_batch(
                prev_hash, miner_id, cur_index,
                nodes, edges, num_nonces,
            )
        )

        # Step 3: preload next batch (async H2D — overlaps kernel!)
        sampler.preload_multi_nonce(
            next_h, next_J, **mn_params,
        )
        self._preload_nonces = next_nonces
        self._preload_salts = next_salts

        # Step 4: harvest results (sync GPU)
        results = sampler.harvest_multi_nonce()

        return [
            (nonces[i], salts[i], results[i])
            for i in range(num_nonces)
        ]
