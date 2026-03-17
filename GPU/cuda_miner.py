"""GPU miner using per-job CUDA SA kernel via CudaSAKernel."""
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import signal
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import dimod

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from GPU.cuda_sa_kernel import CudaSAKernel
from GPU.gpu_utilization import GpuUtilizationMonitor
from dwave_topologies import DEFAULT_TOPOLOGY

try:
    import cupy as cp
except ImportError:
    cp = None


class CudaMiner(BaseMiner):
    """CUDA GPU miner using per-job kernel launches.

    Each mining attempt launches a fresh kernel that runs SA
    and returns synchronously. No persistent kernel, no ring
    buffer, no mapped memory.
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
        self._node_indices = np.array(self.nodes, dtype=np.int32)

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
            _Sampler(self.nodes, self.edges, topology_obj.properties),
        )

        self.miner_type = "GPU-CUDA"
        self.device = device

        self.gpu_utilization = cfg.get('gpu_utilization', 100)
        if not 0 < self.gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {self.gpu_utilization}"
            )

        device_sms = cp.cuda.Device(
            int(device)
        ).attributes['MultiProcessorCount']
        self._monitor = GpuUtilizationMonitor(
            device_id=int(device),
            max_utilization_pct=self.gpu_utilization,
            device_sms=device_sms,
        )

        signal.signal(signal.SIGTERM, self._cleanup_handler)

        self.logger.info(
            f"CUDA miner initialized on device {device} "
            f"(per-job kernel, gpu_utilization={self.gpu_utilization}%)"
        )
        self.logger.info(
            f"Topology: {len(self.nodes)} nodes, "
            f"{len(self.edges)} edges"
        )

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA resource cleanup."""
        if hasattr(self, '_monitor'):
            self._monitor.stop()

        self.logger.info(
            f"CUDA miner {self.miner_id} received SIGTERM, "
            f"cleaning up..."
        )
        try:
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

    def _filter_samples_for_sparse_topology(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples to actual topology nodes.

        Kernel returns N=4800 (max node ID + 1) but validation
        expects 4593 (actual node count). Extract only real nodes.
        """
        samples = sampleset.record.sample
        filtered = samples[:, self._node_indices].astype(np.int8)
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
            self.logger.error(f"Failed to set device context: {e}")
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
        # SA kernel uses 1 block, but throttle if external load
        # is very high to reduce contention
        if self._monitor.should_throttle():
            time.sleep(0.5)

        return self.kernel.sample_ising(
            h, J,
            num_reads=num_reads,
            num_betas=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
        )

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples for sparse topology."""
        return self._filter_samples_for_sparse_topology(sampleset)
