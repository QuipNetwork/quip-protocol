"""GPU miner using per-job CUDA SA kernel via CudaSAKernel."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import random
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)
from GPU.cuda_sa_kernel import CudaSAKernel
from GPU.gpu_scheduler import KernelScheduler
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None


class CudaMiner(BaseMiner):
    """CUDA GPU miner using per-job kernel launches.

    Multi-nonce: prepare() + sample_multi_nonce() for batched
    dispatch. SM budget managed by KernelScheduler.

    Config (via **cfg):
        gpu_utilization: 1-100 (default 100).
        yielding: True = NVML-adaptive, yield to other GPU
            users. False (default) = static SM budget.
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
        """Initialize CUDA miner.

        Args:
            miner_id: Unique identifier for this miner.
            device: CUDA device ID (default "0").
            topology: Optional topology object.
            **cfg: Additional config (gpu_utilization, etc).
        """
        if cp is None:
            raise ImportError("cupy not available")
        cp.cuda.Device(int(device)).use()

        # Get topology
        topology_obj = (
            topology if topology is not None else DEFAULT_TOPOLOGY
        )
        self.nodes = list(topology_obj.graph.nodes)
        self.edges = list(topology_obj.graph.edges)
        self._node_indices = np.array(
            self.nodes, dtype=np.int32,
        )

        # Per-job SA kernel
        self.kernel = CudaSAKernel(max_N=5000)

        # Minimal sampler interface for BaseMiner
        class _Sampler:
            def __init__(self, nodes, edges, properties):
                self.nodes = nodes
                self.edges = edges
                self.nodelist = nodes
                self.edgelist = edges
                self.properties = properties
                self.sampler_type = "cuda-oneshot"

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
        self.device = device

        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {self.gpu_utilization}"
            )

        dev_id = int(device)
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

        yielding = bool(cfg.get('yielding', False))

        self._scheduler = KernelScheduler(
            device_id=dev_id,
            device_sms=device_sms,
            gpu_utilization_pct=self.gpu_utilization,
            yielding=yielding,
        )

        # Static ceiling for buffer allocation
        sm_ceiling = self._scheduler.get_sm_budget()

        # Prepare multi-nonce SA kernel at max capacity
        self.kernel.prepare(
            nodes=self.nodes,
            edges=self.edges,
            num_reads=self.ADAPT_MAX_READS,
            max_num_betas=self.ADAPT_MAX_SWEEPS,
            max_nonces=sm_ceiling,
        )

        signal.signal(signal.SIGTERM, self._cleanup_handler)

        self.logger.info(
            "GPU %s: %s | utilization=%d%% | "
            "SMs=%d/%d | yielding=%s",
            device, dev_name,
            self.gpu_utilization,
            sm_ceiling, device_sms,
            yielding,
        )

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA resource cleanup."""
        self.logger.info(
            f"CUDA miner {self.miner_id} received SIGTERM, "
            f"cleaning up..."
        )
        try:
            self.kernel.close()
            self._scheduler.stop()
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                self.logger.info(
                    f"CUDA device {self.device} cleanup completed"
                )
        except Exception as e:
            self.logger.error(f"Error during CUDA cleanup: {e}")
        sys.exit(0)

    def _post_mine_cleanup(self) -> None:
        """Release CUDA streams and stop NVML monitor."""
        self.kernel.close()
        self._scheduler.stop()

    def _filter_samples_for_sparse_topology(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples to actual topology nodes.

        Kernel returns N=4800 (max node ID + 1) but validation
        expects 4593 (actual node count). Extract only real nodes.
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
                f"Failed to set device context: {e}",
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
        """3-stage pipeline: GPU compute | CPU generate | H2D.

        SA uses 1 SM per nonce (1 block per nonce).
        CPU nonce generation overlaps with GPU kernel via
        launch/harvest split.
        """
        if self._scheduler.should_throttle():
            time.sleep(0.5)

        num_nonces = self._scheduler.get_sm_budget()
        kernel = self.kernel

        mn_params = dict(
            num_reads=num_reads,
            num_betas=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
        )

        # Step 1: launch kernel (async — GPU starts computing)
        if kernel._preloaded:
            nonces = self._preload_nonces
            salts = self._preload_salts
            kernel.launch_multi_nonce(
                [], [], **mn_params,
            )
        else:
            h_list, J_list, nonces, salts = (
                self._generate_nonce_batch(
                    prev_hash, miner_id, cur_index,
                    nodes, edges, num_nonces,
                )
            )
            kernel.launch_multi_nonce(
                h_list, J_list, **mn_params,
            )

        # Step 2: generate NEXT batch on CPU (overlaps GPU!)
        next_h, next_J, next_nonces, next_salts = (
            self._generate_nonce_batch(
                prev_hash, miner_id, cur_index,
                nodes, edges, num_nonces,
            )
        )

        # Step 3: preload next batch (async H2D — overlaps kernel!)
        kernel.preload_multi_nonce(
            next_h, next_J, **mn_params,
        )
        self._preload_nonces = next_nonces
        self._preload_salts = next_salts

        # Step 4: harvest results (sync GPU)
        results = kernel.harvest_multi_nonce()

        return [
            (nonces[i], salts[i], results[i])
            for i in range(num_nonces)
        ]

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples for sparse topology.

        sample_ising (per-job) returns N=max_node+1 columns
        (raw IDs), needs filtering. sample_multi_nonce returns
        N=len(nodes) columns (dense CSR), already correct.
        """
        num_cols = sampleset.record.sample.shape[1]
        if num_cols == len(self.nodes):
            return sampleset
        return self._filter_samples_for_sparse_topology(
            sampleset,
        )
