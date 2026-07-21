# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unified GPU miner base class for CUDA SA and Gibbs kernels.

Owns the per-backend setup harvested by the CUDA stream-driver factory:
the sampler and KernelScheduler (SM budget), plus SIGTERM cleanup, sparse
topology filtering, and adaptive parameter calculation. The streaming
pipeline itself runs in the driver process (GPU/cuda_stream.py).

Subclasses create the appropriate sampler and pass it here.
"""
from __future__ import annotations

import os
import signal
import threading

import dimod
from shared.base_miner import BaseMiner
from GPU.gpu_scheduler import (
    KernelScheduler,
    configure_mps_thread_limit,
)

try:
    import cupy as cp
except ImportError:
    cp = None


class GPUMiner(BaseMiner):
    """Shared base for CUDA GPU miners.

    Provides the sampler + KernelScheduler the stream-driver factory
    harvests, SIGTERM cleanup, sparse topology filtering, and adaptive
    parameter calculation. The streaming loop runs in the driver process.

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

        # KernelScheduler is harvested by the CUDA stream-driver factory
        # (GPU/cuda_stream.py:build_persistent_context) to size num_kernels and
        # honor should_throttle in the producer process. The worker keeps no
        # feeder/stream — the driver owns the streaming pipeline.
        self._scheduler = KernelScheduler(
            device_id=int(device),
            device_sms=device_sms,
            gpu_utilization_pct=gpu_utilization,
            yielding=yielding,
        )

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
        """Validate the work context and activate the CUDA device.

        The sampler + feeder + scheduler live in the stream-driver process
        (GPU/cuda_stream.py); this hook only validates ``prev_block`` /
        ``node_info`` early — the controller treats ``False`` as "skip this
        attempt" — and re-activates the device context.
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

        try:
            cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(
                f"Failed to set device context: {e}",
            )
            return False

        return True

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """No-op: unpack_packed_results already returns dense-indexed samples."""
        return sampleset

    def _post_mine_cleanup(self) -> None:
        """Sync the GPU compute stream and free sampler buffers.

        The feeder + streaming pipeline live in the driver process; the worker
        only syncs any in-flight kernel and closes its sampler.
        """
        if (
            hasattr(self, 'sampler')
            and self.sampler is not None
            and getattr(self.sampler, '_sf_prepared', False)
        ):
            self.sampler._sf_stream_compute.synchronize()

        if hasattr(self, 'sampler') and self.sampler is not None:
            self.sampler.close()

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM: stop scheduler, signal kernel, exit."""
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
