# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unified CUDA miner — SA and Gibbs via streaming API.

Supports both simulated annealing (SA) and chromatic block
Gibbs sampling via the ``update_mode`` parameter. All slot
management is handled by sample_ising_streaming()
through the GPUMiner base class.
"""
from __future__ import annotations

from GPU.gpu_miner import GPUMiner
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None


class CudaMiner(GPUMiner):
    """Unified CUDA GPU miner for SA and Gibbs kernels.

    Thin wrapper that creates the appropriate sampler and
    delegates all pipeline logic to GPUMiner + streaming API.

    Args:
        update_mode: "sa" for simulated annealing (1 SM/model),
            "gibbs" or "metropolis" for chromatic block Gibbs
            (sms_per_nonce SMs/model, default 4).
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
        update_mode: str = "sa",
        **cfg,
    ):
        if cp is None:
            raise ImportError("cupy not available")

        self._is_gibbs = update_mode.lower() in (
            "gibbs", "metropolis",
        )
        self._update_mode = update_mode.lower()

        gpu_util = cfg.pop('gpu_utilization', 100)
        yielding = cfg.pop('yielding', False)
        self.sms_per_nonce = cfg.pop('sms_per_nonce', 4)

        dev_id = int(device)
        GPUMiner._init_cuda_device(
            self, dev_id, gpu_util, yielding,
        )

        topology_obj = (
            topology
            if topology is not None
            else DEFAULT_TOPOLOGY
        )

        device_sms = cp.cuda.Device(
            dev_id,
        ).attributes['MultiProcessorCount']
        sm_ceiling = max(
            1, int(device_sms * gpu_util / 100),
        )

        if self._is_gibbs:
            from GPU.cuda_gibbs_sa import CudaGibbsSampler
            self._sampler = CudaGibbsSampler(
                topology=topology_obj,
                update_mode=self._update_mode,
                max_sms=sm_ceiling,
            )
            miner_type = "GPU-CUDA-Gibbs"
        else:
            from GPU.cuda_sa import CudaSASampler
            self._sampler = CudaSASampler(
                topology=topology_obj,
                max_sms=sm_ceiling,
            )
            miner_type = "GPU-CUDA"

        super().__init__(
            miner_id,
            self._sampler,
            device=device,
            gpu_utilization=gpu_util,
            yielding=yielding,
            miner_type=miner_type,
        )

        if self._is_gibbs:
            self.logger.info(
                "CUDA Gibbs miner on device %s "
                "(mode=%s, utilization=%d%%, "
                "sms_per_nonce=%d)",
                device, self._update_mode, gpu_util,
                self.sms_per_nonce,
            )
        else:
            self.logger.info(
                "CUDA SA miner on device %s "
                "(self-feeding, utilization=%d%%)",
                device, gpu_util,
            )
