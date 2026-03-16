# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unified GPU miner base class for CUDA SA and Gibbs kernels.

Owns the shared pipeline infrastructure: IsingFeeder for
background model generation, KernelScheduler for SM budget,
SIGTERM cleanup, sparse topology filtering, and the
enqueue/poll/dequeue/re-enqueue mining loop.

Pipeline model (per kernel):
    3 slots: completed | active | next
    Kernel persists until no "next" slot, then exits.
    Host: dequeue completed → enqueue replacement.
    Feeder keeps num_kernels models buffered for burst.

Subclasses implement 6 abstract methods (kernel adapter
protocol) to plug in their specific kernel backend.
"""
from __future__ import annotations

import dataclasses
import signal
import sys
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import dimod
import numpy as np

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.ising_feeder import IsingFeeder
from shared.ising_model import IsingModel
from GPU.gpu_scheduler import (
    KernelScheduler,
    configure_mps_thread_limit,
)

try:
    import cupy as cp
except ImportError:
    cp = None

_POLL_INTERVAL = 0.001  # 1ms between completion polls
_PIPELINE_STALL_TIMEOUT = 30.0


@dataclasses.dataclass(slots=True)
class InFlightModel:
    """Tracks a model currently in the GPU pipeline.

    Attributes:
        job_id: Unique kernel job identifier.
        nonce: Blockchain nonce for proof-of-work.
        salt: Random salt used to derive the nonce.
        enqueue_time: Monotonic timestamp when enqueued.
    """

    job_id: int
    nonce: int
    salt: bytes
    enqueue_time: float


class GPUMiner(BaseMiner):
    """Shared pipeline base for CUDA GPU miners.

    Provides IsingFeeder, KernelScheduler, SIGTERM cleanup,
    the pipeline loop, sparse topology filtering, and
    adaptive parameter calculation.

    Subclasses must implement the kernel adapter protocol:
        _kernel_sms_per_model()
        _kernel_enqueue(model, job_id, num_reads, num_sweeps, **p)
        _kernel_signal_ready()
        _kernel_try_dequeue() -> Optional[Tuple[int, Any]]
        _kernel_harvest(raw_result) -> dimod.SampleSet
        _kernel_stop()
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
        self._in_flight: Dict[int, InFlightModel] = {}
        self._next_job_id = 0
        self._cold_start = True

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
    # Kernel adapter protocol (abstract)
    # ----------------------------------------------------------

    @abstractmethod
    def _kernel_sms_per_model(self) -> int:
        """SMs consumed per in-flight model."""

    @abstractmethod
    def _kernel_enqueue(
        self,
        model: IsingModel,
        job_id: int,
        num_reads: int,
        num_sweeps: int,
        **params,
    ) -> None:
        """Upload one model to a kernel slot."""

    @abstractmethod
    def _kernel_signal_ready(self) -> None:
        """Tell kernel it can start (or that new work exists)."""

    @abstractmethod
    def _kernel_try_dequeue(
        self,
    ) -> Optional[Tuple[int, Any]]:
        """Non-blocking poll. Returns (job_id, raw) or None."""

    @abstractmethod
    def _kernel_harvest(
        self, raw_result: Any,
    ) -> dimod.SampleSet:
        """Convert raw kernel result to dimod.SampleSet."""

    @abstractmethod
    def _kernel_stop(self) -> None:
        """Stop kernel and release GPU resources."""

    # ----------------------------------------------------------
    # Pipeline properties
    # ----------------------------------------------------------

    @property
    def _num_kernels(self) -> int:
        """Number of concurrent kernel instances."""
        budget = self._scheduler.get_sm_budget()
        return max(1, budget // self._kernel_sms_per_model())

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
        num_k = self._num_kernels

        self._feeder = IsingFeeder(
            prev_hash=prev_block.hash,
            miner_id=node_info.miner_id,
            cur_index=cur_index,
            nodes=self.sampler.nodes,
            edges=self.sampler.edges,
            buffer_size=num_k * 2,
        )

        self._in_flight.clear()
        self._next_job_id = 0
        self._cold_start = True

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
        """Pipeline: enqueue→poll→dequeue→re-enqueue.

        Cold start: enqueue num_kernels models, signal kernel.
        Steady state: poll for completions, dequeue + re-enqueue
        each, return harvested results.
        """
        if self._scheduler.should_throttle():
            time.sleep(0.5)

        extra = {
            k: v for k, v in kwargs.items()
            if k not in ('num_reads', 'num_sweeps')
        }

        # Cold start: fill active + next slots per kernel
        if self._cold_start:
            self._cold_start = False
            fill = self._num_kernels * 2
            for _ in range(fill):
                self._enqueue_one(
                    num_reads, num_sweeps, **extra,
                )
            self._kernel_signal_ready()

        # Poll for completions
        deadline = time.monotonic() + _PIPELINE_STALL_TIMEOUT
        while time.monotonic() < deadline:
            pair = self._kernel_try_dequeue()
            if pair is not None:
                job_id, raw_result = pair
                tracked = self._in_flight.pop(
                    job_id, None,
                )

                # Immediately enqueue replacement
                self._enqueue_one(
                    num_reads, num_sweeps, **extra,
                )
                self._kernel_signal_ready()

                sampleset = self._kernel_harvest(
                    raw_result,
                )

                if tracked is not None:
                    return [
                        (tracked.nonce, tracked.salt,
                         sampleset),
                    ]
                return []

            time.sleep(_POLL_INTERVAL)

        self.logger.warning(
            "Pipeline stall: no completions after "
            f"{_PIPELINE_STALL_TIMEOUT}s",
        )
        return None

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
        model = IsingModel(h=h, J=J, nonce=0, salt=b'')
        job_id = self._alloc_job_id()
        self._kernel_enqueue(
            model, job_id, num_reads, num_sweeps,
            **extra,
        )
        self._kernel_signal_ready()

        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            pair = self._kernel_try_dequeue()
            if pair is not None:
                _, raw_result = pair
                return self._kernel_harvest(raw_result)
            time.sleep(0.05)

        raise TimeoutError(
            "Kernel did not produce result within 300s",
        )

    def _enqueue_one(
        self,
        num_reads: int,
        num_sweeps: int,
        **extra,
    ) -> None:
        """Pop a model from feeder and enqueue it."""
        assert self._feeder is not None, (
            "_enqueue_one called before _pre_mine_setup"
        )
        model = self._feeder.pop()
        job_id = self._alloc_job_id()
        self._in_flight[job_id] = InFlightModel(
            job_id=job_id,
            nonce=model.nonce,
            salt=model.salt,
            enqueue_time=time.monotonic(),
        )
        self._kernel_enqueue(
            model, job_id, num_reads, num_sweeps,
            **extra,
        )

    def _alloc_job_id(self) -> int:
        """Allocate a monotonically increasing job ID."""
        jid = self._next_job_id
        self._next_job_id += 1
        return jid

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
        """Stop feeder, kernel, and scheduler."""
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None
        self._in_flight.clear()
        try:
            self._kernel_stop()
        except Exception:
            pass

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA cleanup."""
        if hasattr(self, '_scheduler'):
            self._scheduler.stop()

        try:
            self._kernel_stop()
        except Exception:
            pass

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
