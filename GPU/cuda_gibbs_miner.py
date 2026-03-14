# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Gibbs miner using chromatic parallel block Gibbs sampling."""
from __future__ import annotations

import logging
import random
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)
from GPU.cuda_gibbs_sa import CudaGibbsSampler
from GPU.gpu_scheduler import (
    KernelScheduler,
    configure_mps_thread_limit,
)

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)

_PIPELINE_STALL_TIMEOUT = 30.0
_POLL_INTERVAL = 0.001  # 1ms between completion polls


def _generate_batch_worker(
    prev_hash, miner_id, cur_index,
    nodes, edges, num_nonces,
):
    """Generate Ising models in a worker process.

    Runs in ProcessPoolExecutor to avoid GIL contention
    with the main thread's GPU operations.

    Returns:
        (h_list, J_list, nonces, salts) tuple.
    """
    h_list, J_list = [], []
    nonces, salts = [], []
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


class IsingPipeline:
    """Producer-consumer pipeline for GPU kernel overlap.

    Architecture:
        Worker process: CPU-bound Ising model generation
        Generator thread: staging + async H2D to VRAM slots
        Main thread: kernel launch + sync + D2H download

    Buffer layout:
        2 VRAM slots (sampler double buffer, idx 0/1)
        1 RAM slot (batch from worker process)

    Slot lifecycle: FREE → READY → COMPUTING → FREE
    """

    def __init__(
        self,
        sampler,
        prev_hash,
        miner_id,
        cur_index,
        nodes,
        edges,
        num_nonces,
        mn_params,
    ):
        self._sampler = sampler
        self._gen_args = (
            prev_hash, miner_id, cur_index,
            nodes, edges, num_nonces,
        )
        self._mn_params = mn_params

        self._pool = ProcessPoolExecutor(max_workers=1)
        self._stop = threading.Event()
        self._slot_freed = threading.Event()
        self._preload_ready = threading.Event()

        self._preload_nonces = None
        self._preload_salts = None
        self._active_nonces = None
        self._active_salts = None

        self._gen_thread = None
        self._future = None
        self._error = None

    def start(self):
        """Cold start: launch first kernel, start pipeline."""
        h, J, nonces, salts = _generate_batch_worker(
            *self._gen_args,
        )
        self._sampler.launch_multi_nonce(
            h, J, **self._mn_params,
        )
        self._active_nonces = nonces
        self._active_salts = salts

        # Submit first async batch to worker process
        self._future = self._pool.submit(
            _generate_batch_worker, *self._gen_args,
        )

        self._gen_thread = threading.Thread(
            target=self._generator_loop, daemon=True,
        )
        self._gen_thread.start()

        # Kernel running on one slot, other is free
        self._slot_freed.set()

    def _generator_loop(self):
        """Fill VRAM slots as they become free."""
        try:
            self._generator_loop_inner()
        except Exception as e:
            if not self._stop.is_set():
                self._error = e
                logger.error("Pipeline generator error: %s", e)
                self._preload_ready.set()

    def _generator_loop_inner(self):
        """Inner loop — separated for clean error handling."""
        while not self._stop.is_set():
            # Get batch from worker process
            try:
                result = self._future.result(timeout=1.0)
            except TimeoutError:
                continue
            if self._stop.is_set():
                return
            h_list, J_list, nonces, salts = result

            # Submit next CPU generation immediately
            self._future = self._pool.submit(
                _generate_batch_worker, *self._gen_args,
            )

            # Wait for a free VRAM slot
            while not self._stop.is_set():
                if self._slot_freed.wait(timeout=0.1):
                    self._slot_freed.clear()
                    break
            if self._stop.is_set():
                return

            # Staging + async H2D to free slot
            self._sampler.preload_multi_nonce(
                h_list, J_list, **self._mn_params,
            )
            self._preload_nonces = nonces
            self._preload_salts = salts

            # Signal consumer: ready to launch
            self._preload_ready.set()

    def next_batch(self):
        """Return results from running kernel, launch next.

        Sequence:
            1. Wait for preloaded data
            2. Sync GPU (wait for running kernel)
            3. Launch NEXT kernel from preloaded slot
            4. Signal generator: slot freed
            5. Download OLD results (GPU busy with new kernel)
        """
        if not self._preload_ready.wait(
            timeout=_PIPELINE_STALL_TIMEOUT,
        ):
            raise RuntimeError(
                "Pipeline stall: preload not ready after "
                f"{_PIPELINE_STALL_TIMEOUT}s"
            )
        self._preload_ready.clear()

        if self._error:
            raise self._error

        prev_nonces = self._preload_nonces
        prev_salts = self._preload_salts
        active_nonces = self._active_nonces
        active_salts = self._active_salts

        # Sync GPU (wait for running kernel to finish)
        pending = self._sampler.harvest_sync()

        # Launch from preloaded buffer — GPU busy again
        self._sampler.launch_multi_nonce(
            [], [], **self._mn_params,
        )
        self._active_nonces = prev_nonces
        self._active_salts = prev_salts

        # Signal generator: old slot is free
        self._slot_freed.set()

        # Download old results (GPU running new kernel)
        results = self._sampler.download_results(pending)

        return [
            (active_nonces[i], active_salts[i], results[i])
            for i in range(len(results))
        ]

    def stop(self):
        """Shutdown pipeline and clean up resources."""
        self._stop.set()
        self._slot_freed.set()
        self._preload_ready.set()
        if self._gen_thread:
            self._gen_thread.join(timeout=5.0)
        self._pool.shutdown(
            wait=False, cancel_futures=True,
        )


class SelfFeedingPipeline:
    """Pipeline using self-feeding kernel with 3-slot buffers.

    Architecture:
        Worker process: CPU-bound Ising model generation
        Main thread: upload to free slots, poll completions,
            download results. Kernel stays resident.

    The kernel chains between models via rotating buffer slots.
    Host fills EMPTY/COMPLETE slots; kernel picks up READY ones
    automatically.

    When a scheduler is provided with yielding=True, the
    pipeline dynamically scales nonce groups up/down based on
    external GPU process detection via NVML.
    """

    _YIELD_CHECK_INTERVAL = 2.0  # seconds

    def __init__(
        self,
        sampler,
        prev_hash,
        miner_id,
        cur_index,
        nodes,
        edges,
        num_nonces,
        mn_params,
        scheduler=None,
    ):
        self._sampler = sampler
        self._gen_args = (
            prev_hash, miner_id, cur_index,
            nodes, edges, num_nonces,
        )
        self._mn_params = mn_params
        self._num_nonces = num_nonces

        self._pool = ProcessPoolExecutor(max_workers=1)
        self._future = None
        self._pending_results = []
        self._started = False

        # Track which (nonce_id, slot_id) maps to which
        # (nonce_value, salt) for result correlation
        self._slot_meta: Dict[
            Tuple[int, int], Tuple[int, bytes]
        ] = {}

        # Dynamic yielding state
        self._scheduler = scheduler
        self._max_nonces = num_nonces
        self._active_count = num_nonces
        self._last_yield_check = 0.0
        self._num_betas = 0

    def start(self):
        """Cold start: generate first 2 batches, launch kernel."""
        sampler = self._sampler
        num_nonces = self._num_nonces
        reads = self._mn_params['reads_per_nonce']
        sweeps = self._mn_params['num_sweeps']
        sms = self._mn_params['sms_per_nonce']

        # Prepare 3-slot buffers
        sampler.prepare_self_feeding(
            num_nonces=num_nonces,
            reads_per_nonce=reads,
            num_sweeps=sweeps,
            sms_per_nonce=sms,
        )

        # Generate first batch (blocking)
        h1, J1, nonces1, salts1 = _generate_batch_worker(
            *self._gen_args,
        )

        # Upload beta schedule (shared, only once)
        self._num_betas, _ = sampler.upload_beta_schedule(
            h1[0], J1[0], sweeps,
        )
        num_betas = self._num_betas

        # Upload batch 1 to slot 0
        for k in range(num_nonces):
            sampler.upload_slot(k, 0, h1[k], J1[k])
            self._slot_meta[(k, 0)] = (
                nonces1[k], salts1[k],
            )

        # Generate second batch (blocking for cold start)
        h2, J2, nonces2, salts2 = _generate_batch_worker(
            *self._gen_args,
        )

        # Upload batch 2 to slot 1
        for k in range(num_nonces):
            sampler.upload_slot(k, 1, h2[k], J2[k])
            self._slot_meta[(k, 1)] = (
                nonces2[k], salts2[k],
            )

        # Launch kernel (stays resident)
        sampler.launch_self_feeding(
            num_betas=num_betas,
        )

        # Start async CPU generation for next batch
        self._future = self._pool.submit(
            _generate_batch_worker, *self._gen_args,
        )

        self._started = True

    def _check_yield(self):
        """Adjust nonces based on NVML utilization."""
        if self._scheduler is None:
            return
        now = time.monotonic()
        if now - self._last_yield_check < self._YIELD_CHECK_INTERVAL:
            return
        self._last_yield_check = now

        target = self._scheduler.check_stable_target(
            self._max_nonces, self._active_count,
        )
        if target is None:
            return  # Not stable yet (hysteresis)

        if target < self._active_count:
            self._scale_down(target)
        elif target > self._active_count:
            self._scale_up(target)

    def _scale_down(self, target: int):
        """Yield nonces by signaling high-numbered groups."""
        sampler = self._sampler
        prev = self._active_count

        for nid in range(target, self._active_count):
            sampler.signal_nonce_exit(nid)
            for sid in range(3):
                self._slot_meta.pop((nid, sid), None)

        self._active_count = target
        logger.info(
            "Yielding: %d -> %d nonces", prev, target,
        )

    def _scale_up(self, target: int):
        """Reclaim nonces by restarting kernel with full grid.

        Requires a full kernel restart because exited blocks
        cannot be resumed. Overhead is ~1 model time (~650ms).
        """
        sampler = self._sampler
        prev = self._active_count

        # Stop current kernel entirely
        sampler.signal_exit()

        # Generate fresh models for all nonce slots
        h_list, J_list, nonces, salts = (
            _generate_batch_worker(*self._gen_args)
        )
        h2, J2, nonces2, salts2 = (
            _generate_batch_worker(*self._gen_args)
        )

        # Zero ctrl array for clean state
        sampler._d_sf_ctrl[:] = 0

        # Upload fresh slots for all nonces
        self._slot_meta.clear()
        for k in range(target):
            sampler.upload_slot(k, 0, h_list[k], J_list[k])
            self._slot_meta[(k, 0)] = (
                nonces[k], salts[k],
            )
            sampler.upload_slot(k, 1, h2[k], J2[k])
            self._slot_meta[(k, 1)] = (
                nonces2[k], salts2[k],
            )

        # Relaunch with full nonce count
        sampler.launch_self_feeding(
            num_betas=self._num_betas,
            active_nonce_count=target,
        )

        self._active_count = target
        logger.info(
            "Reclaiming: %d -> %d nonces "
            "(0 external processes)",
            prev, target,
        )

    def next_batch(self):
        """Poll for completions, refill slots, return results.

        Returns:
            List of (nonce, salt, SampleSet) for completed
            slots, or empty list if nothing ready yet.
        """
        assert self._started
        sampler = self._sampler

        self._check_yield()

        # Poll for completed slots
        completed = sampler.poll_completions()
        if not completed:
            # Check if kernel exited
            if not sampler.is_kernel_running():
                raise RuntimeError(
                    "Self-feeding kernel exited with "
                    "no completions"
                )
            time.sleep(_POLL_INTERVAL)
            completed = sampler.poll_completions()

        # Spin until at least one completion
        deadline = time.monotonic() + _PIPELINE_STALL_TIMEOUT
        while not completed:
            if not sampler.is_kernel_running():
                raise RuntimeError(
                    "Self-feeding kernel exited "
                    "unexpectedly"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Pipeline stall: no completions "
                    f"after {_PIPELINE_STALL_TIMEOUT}s"
                )
            time.sleep(_POLL_INTERVAL)
            completed = sampler.poll_completions()

        # Download results and refill slots
        results = []
        for nonce_id, slot_id in completed:
            # Skip completions from yielded nonces
            if nonce_id >= self._active_count:
                continue
            # Download
            ss = sampler.download_slot(nonce_id, slot_id)
            meta = self._slot_meta.pop(
                (nonce_id, slot_id), (None, None),
            )
            results.append((meta[0], meta[1], ss))

            # Refill this slot with new work
            self._refill_slot(nonce_id, slot_id)

        return results

    def _refill_slot(self, nonce_id, slot_id):
        """Upload new model to a completed slot."""
        # Get or wait for next CPU batch
        if self._future is None:
            self._future = self._pool.submit(
                _generate_batch_worker,
                *self._gen_args,
            )

        if not self._future.done():
            # Can't refill yet — leave slot empty,
            # kernel will handle it
            return

        h_list, J_list, nonces, salts = (
            self._future.result()
        )
        self._future = self._pool.submit(
            _generate_batch_worker, *self._gen_args,
        )

        # Upload this nonce's model to the slot
        if nonce_id < len(h_list):
            self._sampler.upload_slot(
                nonce_id, slot_id,
                h_list[nonce_id], J_list[nonce_id],
            )
            self._slot_meta[(nonce_id, slot_id)] = (
                nonces[nonce_id], salts[nonce_id],
            )

    def stop(self):
        """Signal kernel exit and clean up."""
        if self._started:
            try:
                self._sampler.signal_exit()
            except Exception as e:
                logger.warning(
                    "Error signaling exit: %s", e,
                )
            self._started = False
        self._pool.shutdown(
            wait=False, cancel_futures=True,
        )


class CudaGibbsMiner(BaseMiner):
    """CUDA GPU miner using chromatic block Gibbs sampling.

    Uses CudaGibbsSampler with KernelScheduler for SM budgeting.
    SelfFeedingPipeline provides zero-gap GPU utilization via
    a resident kernel with 3-slot rotating buffers per nonce.

    Config (via **cfg):
        gpu_utilization: 1-100 (default 100).
        yielding: True = NVML-adaptive, yield to other GPU
            users. False (default) = static SM budget.
        pipeline: "self-feeding" (default) or "double-buffer".
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

        self.device = device
        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {self.gpu_utilization}"
            )
        yielding = bool(cfg.get('yielding', False))

        # Set MPS thread limit before CUDA context creation
        dev_id = int(device)
        self._mps_enforced = configure_mps_thread_limit(
            gpu_utilization_pct=self.gpu_utilization,
            device_id=dev_id,
            yielding=yielding,
        )

        cp.cuda.Device(dev_id).use()

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

        self.sms_per_nonce = cfg.get('sms_per_nonce', 4)
        self._pipeline_mode = cfg.get(
            'pipeline', 'self-feeding',
        )

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

        self._pipeline = None

        signal.signal(signal.SIGTERM, self._cleanup_handler)

        mps_label = (
            "enforced" if self._mps_enforced
            else "off"
        )
        self.logger.info(
            "GPU %s: %s | utilization=%d%% | "
            "SMs=%d/%d | yielding=%s | mps=%s",
            device, dev_name,
            self.gpu_utilization,
            sm_ceiling, device_sms,
            yielding, mps_label,
        )

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA resource cleanup."""
        if hasattr(self, 'logger'):
            self.logger.info(
                f"CUDA Gibbs miner {self.miner_id} received "
                f"SIGTERM, cleaning up..."
            )

        try:
            if self._pipeline:
                self._pipeline.stop()
                self._pipeline = None
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
        """Release pipeline, CUDA streams, and NVML monitor."""
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
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
        """Pipelined GPU batch with async model generation.

        First call creates the IsingPipeline (cold start).
        Subsequent calls return results from the running
        kernel while the next kernel is already executing.
        """
        if self._scheduler.should_throttle():
            time.sleep(0.5)

        if self._pipeline is None:
            sm_budget = self._scheduler.get_sm_budget()
            num_nonces = max(
                1, sm_budget // self.sms_per_nonce,
            )
            mn_params = dict(
                reads_per_nonce=num_reads,
                num_sweeps=num_sweeps,
                sms_per_nonce=self.sms_per_nonce,
            )
            if self._pipeline_mode == 'self-feeding':
                sched = (
                    self._scheduler
                    if self._scheduler.yielding
                    else None
                )
                self._pipeline = SelfFeedingPipeline(
                    self.sampler,
                    prev_hash, miner_id, cur_index,
                    nodes, edges, num_nonces,
                    mn_params,
                    scheduler=sched,
                )
            else:
                self._pipeline = IsingPipeline(
                    self.sampler,
                    prev_hash, miner_id, cur_index,
                    nodes, edges, num_nonces,
                    mn_params,
                )
            self._pipeline.start()

        return self._pipeline.next_batch()
