# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unified GPU miner base class for CUDA SA and Gibbs kernels.

Owns the shared pipeline infrastructure: IsingFeeder for
background model generation, KernelScheduler for SM budget,
SIGTERM cleanup, sparse topology filtering, and the streaming
mining loop via sample_ising_streaming().

Subclasses create the appropriate sampler and pass it here.
"""
from __future__ import annotations

import signal
import sys
import time
from typing import (
    Dict, Iterator, List, Optional, Tuple,
)

import dimod
import numpy as np

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.ising_feeder import IsingFeeder
from GPU.gpu_scheduler import (
    KernelScheduler,
    configure_mps_thread_limit,
)

try:
    import cupy as cp
except ImportError:
    cp = None


# ----------------------------------------------------------
# Pipeline constants
# ----------------------------------------------------------

_PIPELINE_STALL_TIMEOUT = 30.0


class GPUMiner(BaseMiner):
    """Shared pipeline base for CUDA GPU miners.

    Provides IsingFeeder, KernelScheduler, SIGTERM cleanup,
    the streaming mining loop, sparse topology filtering, and
    adaptive parameter calculation.

    Subclasses create a sampler (CudaSASampler or
    CudaGibbsSampler) and pass it to __init__.
    """

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

        # Sparse topology node indices for filtering
        self._node_indices = np.array(
            sampler.nodes, dtype=np.int32,
        )

        # Pipeline state (reset per mine_block call)
        self._feeder: Optional[IsingFeeder] = None
        self._stream: Optional[Iterator] = None

        signal.signal(signal.SIGTERM, self._cleanup_handler)

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
        """Set CUDA device and create IsingFeeder."""
        try:
            cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(
                f"Failed to set device context: {e}",
            )
            return False

        # Extract block context from BaseMiner's positional args
        prev_block = args[0] if len(args) > 0 else None
        node_info = args[1] if len(args) > 1 else None
        if prev_block is None or node_info is None:
            self.logger.error(
                "Missing prev_block or node_info",
            )
            return False

        cur_index = prev_block.header.index + 1
        budget = self._scheduler.get_sm_budget()
        num_k = max(
            1, budget // self.sampler._sms_per_nonce,
        )

        self._feeder = IsingFeeder(
            prev_hash=prev_block.hash,
            miner_id=node_info.miner_id,
            cur_index=cur_index,
            nodes=self.sampler.nodes,
            edges=self.sampler.edges,
            buffer_size=num_k * 2,
        )

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
        **kwargs,
    ) -> Optional[
        List[Tuple[int, bytes, dimod.SampleSet]]
    ]:
        """Stream one result from the GPU pipeline.

        Lazily creates the streaming iterator on first call.
        Returns one (nonce, salt, sampleset) per call.
        """
        if self._scheduler.should_throttle():
            time.sleep(0.5)

        if self._stream is None:
            extra = {
                k: v for k, v in kwargs.items()
                if k not in ('num_reads', 'num_sweeps')
            }
            num_k = max(
                1,
                self._scheduler.get_sm_budget()
                // self.sampler._sms_per_nonce,
            )
            self._stream = (
                self.sampler.sample_ising_streaming(
                    self._feeder,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                    num_kernels=num_k,
                    poll_timeout=_PIPELINE_STALL_TIMEOUT,
                    **extra,
                )
            )

        try:
            model, ss = next(self._stream)
        except TimeoutError:
            self.logger.warning(
                "Pipeline stall: no completions after "
                f"{_PIPELINE_STALL_TIMEOUT}s",
            )
            return None
        except StopIteration:
            return None

        ss = self._filter_sparse_topology(ss)
        return [(model.nonce, model.salt, ss)]

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        **kwargs,
    ) -> dimod.SampleSet:
        """Single-nonce fallback (synchronous)."""
        extra = {
            k: v for k, v in kwargs.items()
            if k not in ('num_reads', 'num_sweeps')
        }
        results = self.sampler.sample_ising(
            [h], [J],
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            **extra,
        )
        return results[0]

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples for sparse topology."""
        return self._filter_sparse_topology(sampleset)

    def _filter_sparse_topology(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Extract only active topology nodes from kernel output.

        Kernel returns N=max_node+1 but validation expects
        only active node count.
        """
        samples = sampleset.record.sample
        filtered = samples[:, self._node_indices].astype(
            np.int8,
        )
        return dimod.SampleSet.from_samples(
            filtered,
            vartype='SPIN',
            energy=sampleset.record.energy,
            info=sampleset.info,
        )

    def _post_mine_cleanup(self) -> None:
        """Stop stream, feeder, and sync sampler."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None
        self.sampler.close()

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA cleanup."""
        if hasattr(self, '_scheduler'):
            self._scheduler.stop()

        if self._stream is not None:
            self._stream.close()
            self._stream = None

        self.logger.info(
            f"{self.miner_type} miner {self.miner_id} "
            f"received SIGTERM, cleaning up...",
        )
        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.Stream.null.synchronize()
                mem = cp.get_default_memory_pool()
                mem.free_all_blocks()
                pin = cp.get_default_pinned_memory_pool()
                pin.free_all_blocks()
        except Exception as e:
            self.logger.error(f"CUDA cleanup error: {e}")
        sys.exit(0)
