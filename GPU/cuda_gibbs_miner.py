# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Gibbs miner — thin kernel adapter over GPUMiner.

Uses CudaGibbsSampler self-feeding kernel with 4 SMs per
model. Each kernel (nonce group) has 3 rotating slots:

    completed | active | next

The kernel works on the active slot, then switches to next.
If no next is available, the kernel exits. The host tracks
per-kernel slot assignments and polls only active slots for
completion, then downloads + re-enqueues immediately.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import dimod
import numpy as np

from shared.ising_model import IsingModel
from GPU.gpu_miner import GPUMiner
from GPU.cuda_gibbs_sa import CudaGibbsSampler
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None

# Slot states matching CudaGibbsSampler constants
_SLOT_COMPLETE = 3


@dataclasses.dataclass(slots=True)
class KernelState:
    """Per-kernel (nonce group) slot assignment.

    Tracks which slot holds the active job, which holds
    the next job, and which is free for upload.

    Attributes:
        active_slot: Slot the kernel is computing on.
        active_job: Job ID in the active slot.
        next_slot: Preloaded slot, kernel picks up next.
        next_job: Job ID in the next slot, or -1 if empty.
        free_slot: Slot available for new model upload.
    """

    active_slot: int
    active_job: int
    next_slot: int
    next_job: int
    free_slot: int


class CudaGibbsMiner(GPUMiner):
    """CUDA GPU miner using chromatic block Gibbs sampling.

    4 SMs per model. Inherits pipeline, feeder, scheduler,
    and SIGTERM cleanup from GPUMiner.

    Per-kernel slot lifecycle:
        Host uploads to free_slot → becomes next_slot.
        Kernel finishes active_slot → switches to next_slot.
        Host downloads completed active_slot → becomes free.
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

        self._update_mode = update_mode
        gpu_util = cfg.pop('gpu_utilization', 100)
        yielding = cfg.pop('yielding', False)
        self.sms_per_nonce = cfg.pop('sms_per_nonce', 4)

        dev_id = int(device)
        # MPS + device before any CUDA work
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

        self._gibbs = CudaGibbsSampler(
            topology=topology_obj,
            update_mode=update_mode,
            max_sms=sm_ceiling,
        )

        super().__init__(
            miner_id,
            self._gibbs,
            device=device,
            gpu_utilization=gpu_util,
            yielding=yielding,
            miner_type="GPU-CUDA-Gibbs",
        )

        # Per-kernel slot tracking (indexed by nonce_id)
        self._kernels: List[KernelState] = []
        self._kernel_launched = False
        self._num_betas = 0
        self._beta_uploaded = False
        self._enqueue_count = 0

        self.logger.info(
            "CUDA Gibbs miner on device %s "
            "(mode=%s, utilization=%d%%, "
            "sms_per_nonce=%d)",
            device, update_mode, gpu_util,
            self.sms_per_nonce,
        )

    # -- Kernel adapter protocol --

    def _kernel_sms_per_model(self) -> int:
        return self.sms_per_nonce

    def _kernel_enqueue(
        self,
        model: IsingModel,
        job_id: int,
        num_reads: int,
        num_sweeps: int,
        **params,
    ) -> None:
        """Upload model to a kernel's free slot.

        First N calls (cold start) fill slot 0 of each
        kernel as the initial active slot. Next N calls
        fill slot 1 as the next slot. Subsequent calls
        rotate into whatever slot is free.
        """
        num_k = self._num_kernels
        spb = params.get('num_sweeps_per_beta', 1)

        # Lazy init: prepare buffers on first enqueue
        if not self._gibbs._sf_prepared:
            self._gibbs.prepare(
                num_reads=self.ADAPT_MAX_READS,
                num_sweeps=self.ADAPT_MAX_SWEEPS,
                num_sweeps_per_beta=1,
                max_nonces=num_k,
            )
            self._gibbs.prepare_self_feeding(
                num_nonces=num_k,
                reads_per_nonce=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=spb,
                sms_per_nonce=self.sms_per_nonce,
            )
            # Initialize per-kernel state:
            # slots 0=active, 1=next, 2=free
            self._kernels = [
                KernelState(
                    active_slot=0, active_job=-1,
                    next_slot=1, next_job=-1,
                    free_slot=2,
                )
                for _ in range(num_k)
            ]

        if not self._beta_uploaded:
            self._num_betas, _ = (
                self._gibbs.upload_beta_schedule(
                    model.h, model.J, num_sweeps, spb,
                )
            )
            self._beta_uploaded = True

        # Cold start: fill active (slot 0), then next (slot 1)
        idx = self._enqueue_count
        if idx < num_k:
            # First round: fill active slot
            nonce_id = idx
            ks = self._kernels[nonce_id]
            self._gibbs.upload_slot(
                nonce_id, ks.active_slot,
                model.h, model.J,
            )
            ks.active_job = job_id
        elif idx < 2 * num_k:
            # Second round: fill next slot
            nonce_id = idx - num_k
            ks = self._kernels[nonce_id]
            self._gibbs.upload_slot(
                nonce_id, ks.next_slot,
                model.h, model.J,
            )
            ks.next_job = job_id
        else:
            # Steady state: find a kernel with a free slot
            placed = False
            for nonce_id, ks in enumerate(self._kernels):
                if ks.free_slot >= 0:
                    self._gibbs.upload_slot(
                        nonce_id, ks.free_slot,
                        model.h, model.J,
                    )
                    # Free becomes new next
                    ks.next_slot = ks.free_slot
                    ks.next_job = job_id
                    ks.free_slot = -1
                    placed = True
                    break
            assert placed, (
                f"No free slot for job {job_id}"
            )

        self._enqueue_count += 1

    def _kernel_signal_ready(self) -> None:
        """Launch self-feeding kernel if not running."""
        if not self._kernel_launched:
            self._gibbs.launch_self_feeding(
                num_betas=self._num_betas,
            )
            self._kernel_launched = True

    def _kernel_try_dequeue(
        self,
    ) -> Optional[Tuple[int, Any]]:
        """Poll only active slots for completion.

        Reads the ctrl array and checks slot_state for
        each kernel's active_slot. On COMPLETE: downloads
        results, rotates slots, returns (job_id, sampleset).
        """
        if not self._kernel_launched:
            return None

        ctrl = cp.asnumpy(self._gibbs._d_sf_ctrl)
        stride = self._gibbs.CTRL_STRIDE

        for nonce_id, ks in enumerate(self._kernels):
            if ks.active_job < 0:
                continue
            base = nonce_id * stride
            state = ctrl[base + ks.active_slot]

            if state != _SLOT_COMPLETE:
                continue

            # Download completed results
            ss = self._gibbs.download_slot(
                nonce_id, ks.active_slot,
            )
            job_id = ks.active_job

            # Rotate: active→free, next→active
            old_active = ks.active_slot
            ks.free_slot = old_active
            ks.active_slot = ks.next_slot
            ks.active_job = ks.next_job
            ks.next_slot = -1
            ks.next_job = -1

            return (job_id, ss)

        return None

    def _kernel_harvest(
        self, raw_result: Any,
    ) -> dimod.SampleSet:
        """Result is already a SampleSet from download_slot."""
        return raw_result

    def _kernel_stop(self) -> None:
        """Signal kernel exit, clean up resources."""
        if self._kernel_launched:
            try:
                self._gibbs.signal_exit()
            except Exception:
                pass
            self._kernel_launched = False
        self._kernels.clear()
        self._beta_uploaded = False
        self._enqueue_count = 0

    def _post_mine_cleanup(self) -> None:
        """Release kernel, feeder, and scheduler."""
        super()._post_mine_cleanup()
        self._gibbs.close()
        self._scheduler.stop()
