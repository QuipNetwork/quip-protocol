# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""
Metal Splash Sampler - Based on Gonzalez et al. 2011

Implements the Splash Sampler from "Parallel Gibbs Sampling: From Colored
Fields to Thin Junction Trees" for strongly coupled Ising models.

Key algorithm:
1. Build bounded treewidth junction trees ("Splashes") covering all variables
2. For each Splash: calibrate via belief propagation
3. Sample entire Splash jointly from calibrated distribution
4. Repeat for all Splashes in each sweep

This addresses slow mixing in strongly coupled models (like Zephyr)
by jointly sampling groups of tightly coupled variables.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import dimod
import Metal
import numpy as np

from GPU.metal_scheduler import DutyCycleController
from GPU.metal_utils import _create_buffer, build_csr_from_ising, compute_beta_schedule, unpack_metal_results


class MetalSplashSampler:
    """
    Splash sampler using Metal GPU.

    Based on Gonzalez et al. 2011 "Parallel Gibbs Sampling: From Colored
    Fields to Thin Junction Trees".

    The Splash sampler addresses slow mixing in strongly coupled models by:
    1. Building bounded treewidth junction trees ("Splashes") covering all variables
    2. Calibrating each junction tree via belief propagation
    3. Jointly sampling entire Splashes from the calibrated distribution

    This is more effective than coordinate-wise Gibbs on strongly coupled
    models like Zephyr (degree ~18-20).
    """

    def __init__(self, topology=None, max_treewidth: int = 4, max_splash_size: int = 64):
        """Initialize Metal Splash sampler.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY)
            max_treewidth: Maximum treewidth for junction trees (default: 4)
            max_splash_size: Maximum variables per Splash (default: 64)
        """
        self.logger = logging.getLogger(__name__)
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal is not supported on this device")

        self.max_treewidth = max_treewidth
        self.max_splash_size = max_splash_size

        # Set up topology
        from dwave_topologies import DEFAULT_TOPOLOGY
        topology_obj = topology if topology is not None else DEFAULT_TOPOLOGY
        topology_graph = topology_obj.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())
        self.nodelist = self.nodes
        self.properties = topology_obj.properties

        # Extract Zephyr parameters from topology (for logging)
        topo_shape = self.properties.get('topology', {}).get('shape', [9, 2])
        self.m = topo_shape[0]
        self.t = topo_shape[1]

        self.logger.debug(f"Topology: Z({self.m},{self.t}) with {len(self.nodes)} nodes, {len(self.edges)} edges")
        self.logger.debug(f"Splash params: max_treewidth={max_treewidth}, max_splash_size={max_splash_size}")

        # Load and compile Metal kernel
        kernel_path = os.path.join(os.path.dirname(__file__), "metal_splash.metal")
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()

        lib, err = self.device.newLibraryWithSource_options_error_(kernel_source, None, None)
        if err:
            raise RuntimeError(f"Failed to compile Metal kernels: {err}")
        if not lib:
            raise RuntimeError("Failed to create Metal library (no error reported)")

        # Load splash_sampler kernel
        self._kernel = lib.newFunctionWithName_("splash_sampler")
        if not self._kernel:
            function_names = [lib.functionNames()[i] for i in range(len(lib.functionNames()))]
            raise RuntimeError(f"Failed to find splash_sampler kernel. Available: {function_names}")

        self._pipeline, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel, None)
        if err or not self._pipeline:
            raise RuntimeError(f"Failed to create pipeline: {err}")

        self._command_queue = self.device.newCommandQueue()

    def sample_ising(
        self,
        h: List[Dict[int, float]],
        J: List[Dict[Tuple[int, int], float]],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[dimod.SampleSet]:
        """
        Sample from Ising model using Splash sampling.

        Args:
            h: List of linear biases [{node: bias}, ...] for each problem
            J: List of quadratic biases [{(node1, node2): coupling}, ...] for each problem
            num_reads: Number of independent sampling runs per problem
            num_sweeps: Total number of sweeps (default 1000)
            num_sweeps_per_beta: Sweeps per beta value (default 1)
            beta_range: (hot_beta, cold_beta) or None for auto
            beta_schedule_type: "linear", "geometric", or "custom"
            beta_schedule: Custom beta schedule (requires beta_schedule_type="custom")
            seed: RNG seed

        Returns:
            List of dimod.SampleSet with samples and energies for each problem
        """
        num_problems = len(h)
        if len(J) != num_problems:
            raise ValueError(f"h and J must have same length: {num_problems} vs {len(J)}")

        self.logger.debug(f"[MetalSplash] Processing {num_problems} problems, {num_reads} reads each, {num_sweeps} sweeps")

        # Build CSR representation for the problem (use float for Splash)
        csr_row_ptr, csr_col_ind, csr_J_vals, h_vals_array, node_to_idx, N = build_csr_from_ising(
            h[0], J[0], use_float=True
        )

        num_edges = len(csr_col_ind) // 2

        # Compute beta schedule
        beta_schedule, beta_range = compute_beta_schedule(
            h[0], J[0], num_sweeps, num_sweeps_per_beta, beta_range, beta_schedule_type, beta_schedule
        )

        self.logger.debug(f"[MetalSplash] Beta schedule: {len(beta_schedule)} betas from {beta_schedule[0]:.4f} to {beta_schedule[-1]:.4f}")

        # RNG seed
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Create Metal buffers
        csr_row_ptr_buf = _create_buffer(self.device, csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = _create_buffer(self.device, csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = _create_buffer(self.device, csr_J_vals, "csr_J_vals")
        h_vals_buf = _create_buffer(self.device, h_vals_array, "h_vals")
        beta_schedule_buf = _create_buffer(self.device, beta_schedule, "beta_schedule")

        # Scalar parameters
        N_bytes = np.int32(N).tobytes()
        num_edges_bytes = np.int32(num_edges).tobytes()
        max_splash_size_bytes = np.int32(self.max_splash_size).tobytes()
        max_treewidth_bytes = np.int32(self.max_treewidth).tobytes()
        num_betas_bytes = np.int32(len(beta_schedule)).tobytes()
        sweeps_per_beta_bytes = np.int32(num_sweeps_per_beta).tobytes()
        base_seed_bytes = np.uint32(seed).tobytes()
        num_samples_bytes = np.int32(num_problems * num_reads).tobytes()

        # Output buffers
        packed_size = (N + 7) // 8
        num_samples = num_problems * num_reads
        final_samples_buf = self.device.newBufferWithLength_options_(
            num_samples * packed_size, Metal.MTLResourceStorageModeShared
        )
        final_energies_buf = self.device.newBufferWithLength_options_(
            num_samples * 4, Metal.MTLResourceStorageModeShared
        )

        # Device memory for Splash construction (per-sample)
        # Splash struct: 5 ints = 20 bytes, Clique struct: 7 ints = 28 bytes
        # For N=4600, with 64 vars per Splash, need ~96 Splashes with ~256 cliques total
        visited_buf = self.device.newBufferWithLength_options_(
            num_samples * N * 4, Metal.MTLResourceStorageModeShared
        )
        splash_buf = self.device.newBufferWithLength_options_(
            num_samples * 96 * 20, Metal.MTLResourceStorageModeShared  # 96 Splashes per sample
        )
        clique_buf = self.device.newBufferWithLength_options_(
            num_samples * 256 * 28, Metal.MTLResourceStorageModeShared  # 256 Cliques per sample
        )
        splash_var_buf = self.device.newBufferWithLength_options_(
            num_samples * N * 4, Metal.MTLResourceStorageModeShared  # Enough for all variables
        )
        clique_var_buf = self.device.newBufferWithLength_options_(
            num_samples * 1024 * 4, Metal.MTLResourceStorageModeShared  # ~4 vars per clique * 256 cliques
        )
        sep_var_buf = self.device.newBufferWithLength_options_(
            num_samples * 512 * 4, Metal.MTLResourceStorageModeShared  # ~2 seps per clique * 256
        )

        # Execute kernel
        _duty_cycle: Optional[DutyCycleController] = kwargs.get(
            'duty_cycle',
        )
        _t0 = time.perf_counter()

        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)

        # Buffer bindings (must match kernel parameter order)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(h_vals_buf, 0, 3)

        encoder.setBytes_length_atIndex_(N_bytes, len(N_bytes), 4)
        encoder.setBytes_length_atIndex_(num_edges_bytes, len(num_edges_bytes), 5)
        encoder.setBytes_length_atIndex_(max_splash_size_bytes, len(max_splash_size_bytes), 6)
        encoder.setBytes_length_atIndex_(max_treewidth_bytes, len(max_treewidth_bytes), 7)

        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 8)
        encoder.setBytes_length_atIndex_(num_betas_bytes, len(num_betas_bytes), 9)
        encoder.setBytes_length_atIndex_(sweeps_per_beta_bytes, len(sweeps_per_beta_bytes), 10)
        encoder.setBytes_length_atIndex_(base_seed_bytes, len(base_seed_bytes), 11)

        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 12)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 13)

        encoder.setBytes_length_atIndex_(num_samples_bytes, len(num_samples_bytes), 14)

        # Device memory for Splash construction
        encoder.setBuffer_offset_atIndex_(visited_buf, 0, 15)
        encoder.setBuffer_offset_atIndex_(splash_buf, 0, 16)
        encoder.setBuffer_offset_atIndex_(clique_buf, 0, 17)
        encoder.setBuffer_offset_atIndex_(splash_var_buf, 0, 18)
        encoder.setBuffer_offset_atIndex_(clique_var_buf, 0, 19)
        encoder.setBuffer_offset_atIndex_(sep_var_buf, 0, 20)

        # Dispatch: one threadgroup per sample
        threads_per_group = min(256, self._pipeline.maxTotalThreadsPerThreadgroup())
        threads_per_threadgroup = Metal.MTLSize(width=threads_per_group, height=1, depth=1)
        num_threadgroups_size = Metal.MTLSize(width=num_samples, height=1, depth=1)

        self.logger.debug(f"[MetalSplash] Dispatch: {num_samples} threadgroups x {threads_per_group} threads")

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups_size, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Duty-cycle yielding after GPU dispatch
        if _duty_cycle and _duty_cycle.enabled:
            _compute_s = time.perf_counter() - _t0
            time.sleep(_duty_cycle.compute_sleep(_compute_s))

        # Check for errors
        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        samples_raw = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_samples * packed_size),
            dtype=np.int8
        ).reshape(num_samples, packed_size)

        energies_raw = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_samples * 4),
            dtype=np.int32
        )

        # Unpack samples and build SampleSets
        samplesets = []
        for prob_idx in range(num_problems):
            prob_start = prob_idx * num_reads
            prob_end = prob_start + num_reads
            prob_packed = samples_raw[prob_start:prob_end]
            prob_energies = energies_raw[prob_start:prob_end]

            # Unpack and build SampleSet
            sampleset = unpack_metal_results(
                prob_packed, prob_energies, N, num_reads, node_to_idx,
                beta_range, beta_schedule_type,
                sampler_type="splash",
                max_treewidth=self.max_treewidth,
                max_splash_size=self.max_splash_size
            )
            samplesets.append(sampleset)

        return samplesets
