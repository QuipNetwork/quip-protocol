# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unified GPU miner base class for CUDA SA and Gibbs kernels.

Owns the shared pipeline infrastructure: RandomIsingFeeder for
background model generation, KernelScheduler for SM budget,
SIGTERM cleanup, sparse topology filtering, and the streaming
mining loop via sample_ising_streaming().

Subclasses create the appropriate sampler and pass it here.
"""
from __future__ import annotations

import os
import signal
import threading
from typing import (
    Iterator, List, Optional, Tuple,
)

import dimod
from shared.base_miner import BaseMiner
from shared.miner_types import BlockRequirements
from shared.ising_feeder import RandomIsingFeeder
from GPU.gpu_scheduler import (
    KernelScheduler,
    configure_mps_thread_limit,
)

try:
    import cupy as cp
except ImportError:
    cp = None


class GPUMiner(BaseMiner):
    """Shared pipeline base for CUDA GPU miners.

    Provides RandomIsingFeeder, KernelScheduler, SIGTERM cleanup,
    the streaming mining loop, sparse topology filtering, and
    adaptive parameter calculation.

    Subclasses create a sampler (CudaSASampler or
    CudaGibbsSampler) and pass it to __init__.
    """

    # Keep the feeder roughly 2x the per-iteration nonce batch. Old code
    # sized it as ``num_k * 2`` from the live SM budget; 16 covers the
    # common case (4–8 in-flight kernels on a typical GPU) without
    # over-spawning Python workers.
    FEEDER_BUFFER_SIZE = 16

    def __init__(
        self,
        miner_id: str,
        sampler,
        *,
        device: str = "0",
        gpu_utilization: int = 100,
        yielding: bool = False,
        miner_type: str = "GPU-CUDA",
    ):
        if cp is None:
            raise ImportError("cupy not available")

        dev_id = int(device)

        # MPS + device context (idempotent if subclass
        # already called _init_cuda_device)
        if not getattr(self, '_cuda_initialized', False):
            self._init_cuda_device(
                dev_id, gpu_utilization, yielding,
            )

        super().__init__(
            miner_id, sampler, miner_type=miner_type,
        )

        self.device = device

        if not 0 < gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {gpu_utilization}"
            )
        self.gpu_utilization = gpu_utilization

        device_sms = cp.cuda.Device(
            int(device),
        ).attributes['MultiProcessorCount']
        self._device_sms = device_sms

        self._scheduler = KernelScheduler(
            device_id=int(device),
            device_sms=device_sms,
            gpu_utilization_pct=gpu_utilization,
            yielding=yielding,
        )

        # Pipeline state (reset per mine_block call)
        self._feeder: Optional[RandomIsingFeeder] = None
        self._stream: Optional[Iterator] = None

        if threading.current_thread() is threading.main_thread():
            signal.signal(
                signal.SIGTERM, self._cleanup_handler,
            )

    def _init_cuda_device(
        self,
        dev_id: int,
        gpu_utilization: int,
        yielding: bool,
    ) -> None:
        """Set MPS thread limit and activate CUDA device.

        Must be called before any CUDA API call. Safe to call
        multiple times — subsequent calls are no-ops.

        Subclasses that need CUDA before super().__init__()
        should call this explicitly in their __init__.
        """
        if getattr(self, '_cuda_initialized', False):
            return
        self._mps_enforced = configure_mps_thread_limit(
            gpu_utilization_pct=gpu_utilization,
            device_id=dev_id,
            yielding=yielding,
        )
        cp.cuda.Device(dev_id).use()
        self._cuda_initialized = True

    # ----------------------------------------------------------
    # BaseMiner hooks
    # ----------------------------------------------------------

    def _pre_mine_setup(self, *args, **kwargs) -> bool:
        """Activate the CUDA device and size adaptive batching.

        The Ising feeder is now built by ``BaseMiner.mine_work_item``
        via ``context.make_feeder(...)`` immediately before the loop;
        this hook handles the GPU-specific setup that needs a live
        CUDA context (SM budget sizing, device activation).
        """
        # Extract block context from BaseMiner's positional
        # args — no CUDA needed for this.
        prev_block = args[0] if len(args) > 0 else None
        node_info = args[1] if len(args) > 1 else None
        if prev_block is None or node_info is None:
            self.logger.error(
                "Missing prev_block or node_info",
            )
            return False

        # get_sm_budget() uses cached _device_sms — no
        # CUDA call needed.
        budget = self._scheduler.get_sm_budget()
        num_k = max(
            1, budget // self.sampler._sms_per_nonce,
        )

        # Adaptive nonce tracking for yielding mode
        self._max_nonces = num_k
        self._active_nonces = num_k

        try:
            cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(
                f"Failed to set device context: {e}",
            )
            return False

        self._stream = None

        return True

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        """Compute adaptive params from difficulty."""
        return self.adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
        )

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """No-op: unpack_packed_results already returns dense-indexed samples."""
        return sampleset

    def _post_mine_cleanup(self) -> None:
        """Stop stream, sync GPU, kill feeder, free buffers.

        Ordering:
        1. Close stream — signals kernel to exit
        2. Sync GPU compute stream — kernel must stop
           before feeder data it reads is freed
        3. Stop feeder — kills worker processes
        4. Close sampler — frees GPU buffers
        """
        if self._stream is not None:
            self._stream.close()
            self._stream = None

        # Sync GPU before killing feeder — kernel must be
        # stopped before feeder data is freed.
        if (
            hasattr(self, 'sampler')
            and self.sampler is not None
            and getattr(self.sampler, '_sf_prepared', False)
        ):
            self.sampler._sf_stream_compute.synchronize()

        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None

        if hasattr(self, 'sampler') and self.sampler is not None:
            self.sampler.close()

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM: stop feeder, signal kernel, exit."""
        # Stop the RandomIsingFeeder first — its ProcessPoolExecutor
        # workers will block atexit if not shut down.
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None

        if hasattr(self, '_scheduler'):
            self._scheduler.stop()

        if hasattr(self, 'sampler') and self.sampler._sf_prepared:
            self.sampler.signal_exit(wait=False)

        self.logger.info(
            f"{self.miner_type} miner {self.miner_id} "
            f"received SIGTERM, cleaning up...",
        )

        # Hard exit — skip atexit handlers that might block
        # on GPU sync or orphaned thread pools.
        os._exit(0)
