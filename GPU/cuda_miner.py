# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unified CUDA miner — SA and Gibbs via streaming API.

Supports both simulated annealing (SA) and chromatic block
Gibbs sampling via the ``update_mode`` parameter. All slot
management is handled by sample_ising_streaming()
through the GPUMiner base class.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from GPU.gpu_miner import GPUMiner
from shared.miner_types import BlockRequirements
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

    # Gibbs needs ~2x sweeps to match SA at the same reads.
    # Measured via sweep_reads_grid on Advantage2 topology.
    GIBBS_SWEEP_MULTIPLIER = 2

    # Sampling runs in a stream-driver process (GPU/cuda_stream.py).
    STREAM_FACTORY_DOTTED = "GPU.cuda_stream:build_persistent_context"

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        """Compute adaptive params, doubling sweeps for Gibbs."""
        params = self.adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
            allowed_h=getattr(current_requirements, "allowed_h_values", None),
        )
        if self._is_gibbs:
            params['num_sweeps'] = min(
                params['num_sweeps'] * self.GIBBS_SWEEP_MULTIPLIER,
                self.ADAPT_MAX_SWEEPS * self.GIBBS_SWEEP_MULTIPLIER,
            )
        return params

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

        gpu_util = cfg.pop(
            'utilization', cfg.pop('gpu_utilization', 100),
        )
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
        self.topology = topology_obj
        self._yielding = yielding

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
                sms_per_nonce=self.sms_per_nonce,
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

    def _stream_factory_kwargs(
        self, sample_ctx: Dict[str, Any], nodes: List[int],
    ) -> Dict[str, Any]:
        """Return kwargs forwarded to GPU.cuda_stream:build_persistent_context.

        Called by BaseMiner._ensure_driver when spawning the stream driver.
        """
        return {
            "miner_id": self.miner_id,
            "nodes": nodes,
            "edges": sample_ctx["edges"],
            "feeder_buffer_size": self.FEEDER_BUFFER_SIZE,
            "num_reads": sample_ctx["num_reads"],
            "num_sweeps": sample_ctx["num_sweeps"],
            "topology": getattr(self, "topology", None),
            "device": str(getattr(self, "device", "0")),
            "update_mode": getattr(self, "_update_mode", "sa"),
            "sms_per_nonce": getattr(self, "sms_per_nonce", 4),
            "utilization": getattr(self, "gpu_utilization", 100),
            "yielding": getattr(self, "_yielding", False),
        }
