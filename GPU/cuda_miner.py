# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA SA miner — thin kernel adapter over GPUMiner.

Uses CudaKernelRealSA persistent kernel with 1 SM per model.
Each SM has a ring-buffer slot: the host enqueues models,
the kernel grabs them, processes SA sweeps, writes output.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import dimod
import numpy as np

from shared.beta_schedule import _default_ising_beta_range
from shared.ising_model import IsingModel
from GPU.gpu_miner import GPUMiner
from GPU.cuda_kernel import CudaKernelRealSA
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None


class _SASampler:
    """Minimal sampler stub for BaseMiner topology access."""

    def __init__(self, nodes, edges, properties):
        self.nodes = nodes
        self.edges = edges
        self.nodelist = nodes
        self.edgelist = edges
        self.properties = properties
        self.sampler_type = "cuda-persistent"

    def sample_ising(self, h, J, **kw):
        raise NotImplementedError


class CudaMiner(GPUMiner):
    """CUDA GPU miner using persistent SA kernel.

    1 SM per model. Inherits pipeline, feeder, scheduler,
    and SIGTERM cleanup from GPUMiner.
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
        **cfg,
    ):
        if cp is None:
            raise ImportError("cupy not available")

        gpu_util = cfg.pop('gpu_utilization', 100)
        yielding = cfg.pop('yielding', False)

        topology_obj = (
            topology
            if topology is not None
            else DEFAULT_TOPOLOGY
        )
        nodes = list(topology_obj.graph.nodes)
        edges = list(topology_obj.graph.edges)

        sampler = _SASampler(
            nodes, edges, topology_obj.properties,
        )

        # GPUMiner.__init__ sets MPS + CUDA device context
        super().__init__(
            miner_id,
            sampler,
            device=device,
            gpu_utilization=gpu_util,
            yielding=yielding,
            miner_type="GPU-CUDA",
        )

        # Kernel must be created after CUDA context
        self._kernel = CudaKernelRealSA(
            max_N=5000, verbose=False,
        )

        self.logger.info(
            "CUDA SA miner on device %s "
            "(persistent, utilization=%d%%)",
            device, gpu_util,
        )

    # -- Kernel adapter protocol --

    def _kernel_sms_per_model(self) -> int:
        return 1

    def _kernel_enqueue(
        self,
        model: IsingModel,
        job_id: int,
        num_reads: int,
        num_sweeps: int,
        **params,
    ) -> None:
        beta_range = _default_ising_beta_range(
            model.h, model.J,
        )
        self._kernel.enqueue_job(
            job_id=job_id,
            h=model.h,
            J=model.J,
            num_reads=num_reads,
            num_betas=num_sweeps,
            num_sweeps_per_beta=params.get(
                'num_sweeps_per_beta', 1,
            ),
            beta_range=beta_range,
        )

    def _kernel_signal_ready(self) -> None:
        self._kernel.signal_batch_ready()

    def _kernel_try_dequeue(
        self,
    ) -> Optional[Tuple[int, Any]]:
        result = self._kernel.try_dequeue_result()
        if result is None:
            return None
        return (result.get('job_id', 0), result)

    def _kernel_harvest(
        self, raw_result: Any,
    ) -> dimod.SampleSet:
        samples = self._kernel.get_samples(raw_result)
        energies = self._kernel.get_energies(raw_result)
        return dimod.SampleSet.from_samples(
            samples.astype(np.int8),
            vartype='SPIN',
            energy=energies,
        )

    def _kernel_stop(self) -> None:
        self._kernel.stop_immediate()
