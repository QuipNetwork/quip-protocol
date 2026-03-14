"""GPU miner using CUDA SA kernel with self-feeding pipeline."""
from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.ising_feeder import IsingFeeder
from shared.ising_model import IsingModel
from GPU.cuda_sa_kernel import CudaSAKernel
from GPU.gpu_scheduler import (
    KernelScheduler,
    configure_mps_thread_limit,
)
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)

_PIPELINE_STALL_TIMEOUT = 30.0
_POLL_INTERVAL = 0.001  # 1ms between completion polls
_YIELD_CHECK_INTERVAL = 2.0  # seconds between yield checks


class SaSelfFeedingPipeline:
    """Self-feeding pipeline for SA CUDA kernel.

    Architecture:
        Worker process: CPU-bound Ising model generation
        Main thread: upload → poll → download loop
        Kernel: stays resident, chains models via 3-slot
            rotating buffers (EMPTY→READY→ACTIVE→COMPLETE)

    1 block per nonce (SA uses 1 SM per nonce, unlike
    Gibbs which uses 4).
    """

    _YIELD_CHECK_INTERVAL = _YIELD_CHECK_INTERVAL

    def __init__(
        self,
        kernel,
        feeder: IsingFeeder,
        num_nonces: int,
        mn_params: dict,
        scheduler=None,
    ):
        self._kernel = kernel
        self._feeder = feeder
        self._mn_params = mn_params
        self._num_nonces = num_nonces
        self._started = False

        # Track (nonce_id, slot_id) -> IsingModel
        self._slot_meta: Dict[
            Tuple[int, int], IsingModel
        ] = {}

        # Dynamic yielding state
        self._scheduler = scheduler
        self._max_nonces = num_nonces
        self._active_count = num_nonces
        self._last_yield_check = 0.0
        self._num_betas = 0

    def start(self):
        """Cold start: generate 2 batches, launch kernel."""
        kernel = self._kernel
        num_nonces = self._num_nonces
        reads = self._mn_params['num_reads']
        num_betas = self._mn_params['num_betas']

        # Prepare 3-slot buffers
        kernel.prepare_self_feeding(
            num_nonces=num_nonces,
            num_reads=reads,
            num_betas=num_betas,
        )

        # Generate first batch (blocking)
        batch1 = [
            self._feeder.pop()
            for _ in range(num_nonces)
        ]

        # Upload beta schedule (shared, once)
        self._num_betas, _ = kernel.upload_beta_schedule(
            batch1[0].h, batch1[0].J, num_betas,
        )

        # Upload batch 1 to slot 0
        for k in range(num_nonces):
            m = batch1[k]
            kernel.upload_slot(k, 0, m.h, m.J)
            self._slot_meta[(k, 0)] = m

        # Generate second batch (blocking for cold start)
        batch2 = [
            self._feeder.pop()
            for _ in range(num_nonces)
        ]

        # Upload batch 2 to slot 1
        for k in range(num_nonces):
            m = batch2[k]
            kernel.upload_slot(k, 1, m.h, m.J)
            self._slot_meta[(k, 1)] = m

        # Launch kernel (stays resident)
        kernel.launch_self_feeding(
            num_betas=self._num_betas,
        )

        self._started = True

    def _check_yield(self):
        """Adjust nonces based on NVML utilization."""
        if self._scheduler is None:
            return
        now = time.monotonic()
        if now - self._last_yield_check < (
            self._YIELD_CHECK_INTERVAL
        ):
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
        kernel = self._kernel
        prev = self._active_count

        for nid in range(target, self._active_count):
            kernel.signal_nonce_exit(nid)
            for sid in range(3):
                self._slot_meta.pop((nid, sid), None)

        self._active_count = target
        logger.info(
            "Yielding: %d -> %d nonces", prev, target,
        )

    def _scale_up(self, target: int):
        """Reclaim nonces by restarting kernel.

        Requires a full kernel restart because exited blocks
        cannot be resumed.
        """
        kernel = self._kernel
        prev = self._active_count

        # Stop current kernel entirely
        kernel.signal_exit()

        # Generate fresh models for all nonce slots
        batch1 = [
            self._feeder.pop() for _ in range(target)
        ]
        batch2 = [
            self._feeder.pop() for _ in range(target)
        ]

        # Zero ctrl array for clean state
        kernel._d_sf_ctrl[:] = 0

        # Upload fresh slots for all nonces
        self._slot_meta.clear()
        for k in range(target):
            m1 = batch1[k]
            kernel.upload_slot(k, 0, m1.h, m1.J)
            self._slot_meta[(k, 0)] = m1
            m2 = batch2[k]
            kernel.upload_slot(k, 1, m2.h, m2.J)
            self._slot_meta[(k, 1)] = m2

        # Relaunch with full nonce count
        kernel.launch_self_feeding(
            num_betas=self._num_betas,
            active_nonce_count=target,
        )

        self._active_count = target
        logger.info(
            "Reclaiming: %d -> %d nonces", prev, target,
        )

    def next_batch(self):
        """Poll for completions, refill slots, return results.

        Returns:
            List of (nonce, salt, SampleSet) for completed
            slots.
        """
        assert self._started
        kernel = self._kernel

        self._check_yield()

        # Poll for completed slots
        completed = kernel.poll_completions()
        if not completed:
            if not kernel.is_kernel_running():
                raise RuntimeError(
                    "SA self-feeding kernel exited with "
                    "no completions"
                )
            time.sleep(_POLL_INTERVAL)
            completed = kernel.poll_completions()

        # Spin until at least one completion
        deadline = (
            time.monotonic() + _PIPELINE_STALL_TIMEOUT
        )
        while not completed:
            if not kernel.is_kernel_running():
                raise RuntimeError(
                    "SA self-feeding kernel exited "
                    "unexpectedly"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Pipeline stall: no completions "
                    f"after {_PIPELINE_STALL_TIMEOUT}s"
                )
            time.sleep(_POLL_INTERVAL)
            completed = kernel.poll_completions()

        # Download results and refill slots
        results = []
        for nonce_id, slot_id in completed:
            # Skip completions from yielded nonces
            if nonce_id >= self._active_count:
                continue
            ss = kernel.download_slot(nonce_id, slot_id)
            model = self._slot_meta.pop(
                (nonce_id, slot_id), None,
            )
            if model is not None:
                results.append(
                    (model.nonce, model.salt, ss),
                )

            # Refill this slot with new work
            self._refill_slot(nonce_id, slot_id)

        return results

    def _refill_slot(self, nonce_id, slot_id):
        """Upload new model to a completed slot."""
        model = self._feeder.try_pop()
        if model is None:
            return  # Leave slot empty; kernel handles it
        self._kernel.upload_slot(
            nonce_id, slot_id, model.h, model.J,
        )
        self._slot_meta[(nonce_id, slot_id)] = model

    def stop(self):
        """Signal kernel exit and clean up."""
        if self._started:
            try:
                self._kernel.signal_exit()
            except Exception as e:
                logger.warning(
                    "Error signaling exit: %s", e,
                )
            self._started = False


class CudaMiner(BaseMiner):
    """CUDA GPU miner using SA with self-feeding pipeline.

    Self-feeding pipeline provides zero-gap GPU utilization
    via a resident kernel with 3-slot rotating buffers per
    nonce. 1 block (SM) per nonce.

    Config (via **cfg):
        gpu_utilization: 1-100 (default 100).
        yielding: True = NVML-adaptive, yield to other GPU
            users. False (default) = static SM budget.
        pipeline: "self-feeding" (default) or "launch-harvest".
    """

    # CUDA GPU calibration ranges
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

        self.device = device
        self.gpu_utilization = cfg.get(
            'gpu_utilization', 100,
        )
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {self.gpu_utilization}"
            )
        yielding = bool(cfg.get('yielding', False))

        # Set MPS thread limit before CUDA context
        dev_id = int(device)
        self._mps_enforced = configure_mps_thread_limit(
            gpu_utilization_pct=self.gpu_utilization,
            device_id=dev_id,
            yielding=yielding,
        )

        cp.cuda.Device(dev_id).use()

        # Get topology
        topology_obj = (
            topology
            if topology is not None
            else DEFAULT_TOPOLOGY
        )
        self.nodes = list(topology_obj.graph.nodes)
        self.edges = list(topology_obj.graph.edges)
        self._node_indices = np.array(
            self.nodes, dtype=np.int32,
        )

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

        # Per-job SA kernel (also used for self-feeding)
        self.kernel = CudaSAKernel(max_N=5000)

        # Prepare multi-nonce SA kernel at max capacity
        self.kernel.prepare(
            nodes=self.nodes,
            edges=self.edges,
            num_reads=self.ADAPT_MAX_READS,
            max_num_betas=self.ADAPT_MAX_SWEEPS,
            max_nonces=sm_ceiling,
        )

        # Minimal sampler interface for BaseMiner
        class _Sampler:
            def __init__(self, nodes, edges, properties):
                self.nodes = nodes
                self.edges = edges
                self.nodelist = nodes
                self.edgelist = edges
                self.properties = properties
                self.sampler_type = "cuda-sa"

            def sample_ising(self, h, J, **kw):
                raise NotImplementedError

        super().__init__(
            miner_id,
            _Sampler(
                self.nodes, self.edges,
                topology_obj.properties,
            ),
        )

        self.miner_type = "GPU-CUDA"
        self._pipeline = None
        self._feeder = None
        self._preload_models = None

        signal.signal(signal.SIGTERM, self._cleanup_handler)

        mps_label = (
            "enforced" if self._mps_enforced else "off"
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
        self.logger.info(
            "CUDA miner %s received SIGTERM, cleaning up...",
            self.miner_id,
        )
        try:
            if self._pipeline:
                self._pipeline.stop()
                self._pipeline = None
            if self._feeder:
                self._feeder.stop()
                self._feeder = None
            self.kernel.close()
            self._scheduler.stop()
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool(
                ).free_all_blocks()
        except Exception as e:
            self.logger.error(
                "Error during CUDA cleanup: %s", e,
            )
        sys.exit(0)

    def _post_mine_cleanup(self) -> None:
        """Release pipeline, CUDA streams, NVML monitor."""
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._feeder:
            self._feeder.stop()
            self._feeder = None
        self.kernel.close()
        self._scheduler.stop()

    def _filter_samples_for_sparse_topology(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples to actual topology nodes."""
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

    def _pre_mine_setup(
        self,
        prev_block,
        node_info,
        requirements,
        prev_timestamp: int,
        stop_event: multiprocessing.synchronize.Event,
        **kwargs,
    ) -> bool:
        """Set CUDA device context before mining."""
        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(
                "Failed to set device context: %s", e,
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
        """Run SA on a single Ising problem via per-job kernel."""
        return self.kernel.sample_ising(
            h, J,
            num_reads=num_reads,
            num_betas=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
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
        num_sweeps_per_beta: int = 1,
        **kwargs,
    ) -> Optional[
        List[Tuple[int, bytes, dimod.SampleSet]]
    ]:
        """Self-feeding pipeline: zero-gap GPU utilization.

        First call creates the pipeline (cold start).
        Subsequent calls poll completions from the running
        resident kernel.
        """
        if self._scheduler.should_throttle():
            time.sleep(0.5)

        if self._feeder is None:
            sm_budget = self._scheduler.get_sm_budget()
            # SA uses 1 SM per nonce
            num_nonces = max(1, sm_budget)

            self._feeder = IsingFeeder(
                prev_hash, miner_id, cur_index,
                nodes, edges,
                buffer_size=num_nonces * 2,
            )

            if self._pipeline_mode == 'self-feeding':
                mn_params = dict(
                    num_reads=num_reads,
                    num_betas=num_sweeps,
                )
                sched = (
                    self._scheduler
                    if self._scheduler.yielding
                    else None
                )
                self._pipeline = SaSelfFeedingPipeline(
                    self.kernel,
                    self._feeder,
                    num_nonces,
                    mn_params,
                    scheduler=sched,
                )
                self._pipeline.start()
            else:
                self._lh_num_nonces = num_nonces

        if self._pipeline is not None:
            return self._pipeline.next_batch()

        return self._launch_harvest_batch(
            self._lh_num_nonces,
            num_reads, num_sweeps,
            num_sweeps_per_beta,
        )

    def _launch_harvest_batch(
        self,
        num_nonces, num_reads,
        num_sweeps, num_sweeps_per_beta,
    ) -> List[Tuple[int, bytes, dimod.SampleSet]]:
        """Fallback: launch/harvest pipeline (synchronous)."""
        kernel = self.kernel
        mn_params = dict(
            num_reads=num_reads,
            num_betas=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
        )

        if kernel._preloaded:
            models = self._preload_models
            kernel.launch_multi_nonce([], [], **mn_params)
        else:
            models = [
                self._feeder.pop()
                for _ in range(num_nonces)
            ]
            kernel.launch_multi_nonce(
                [m.h for m in models],
                [m.J for m in models],
                **mn_params,
            )

        # Feeder generates in background during GPU work
        next_models = [
            self._feeder.pop()
            for _ in range(num_nonces)
        ]

        kernel.preload_multi_nonce(
            [m.h for m in next_models],
            [m.J for m in next_models],
            **mn_params,
        )
        self._preload_models = next_models

        results = kernel.harvest_multi_nonce()
        return [
            (models[i].nonce, models[i].salt, results[i])
            for i in range(num_nonces)
        ]

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples for sparse topology.

        sample_ising (per-job) returns N=max_node+1 columns
        (raw IDs), needs filtering. Multi-nonce returns
        N=len(nodes) columns (dense CSR), already correct.
        """
        num_cols = sampleset.record.sample.shape[1]
        if num_cols == len(self.nodes):
            return sampleset
        return self._filter_samples_for_sparse_topology(
            sampleset,
        )
