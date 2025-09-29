"""Unified Metal Parallel Tempering Sampler - Modular Metal Kernels.

Single interface that takes raw h,J dictionaries and processes them using
separate Metal kernels for each preprocessing step.
"""

import time
import logging
from typing import Optional, Dict, Any
import numpy as np
import struct

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

import dimod


class UnifiedMetalSampler:
    """Unified Metal GPU sampler with single sample_ising() interface."""

    def __init__(self, logger: Optional[logging.Logger] = None, verbose: bool = False):
        """Initialize unified Metal sampler.

        Args:
            logger: Optional logger instance
            verbose: If True, enable DEBUG logging for this sampler
        """
        self.logger = logger or logging.getLogger(__name__)
        self._verbose = verbose

        # Configure logger
        if logger is None:
            self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            if verbose:
                self.logger.setLevel(logging.DEBUG)

        self.sampler_type = "unified_metal"

        if not METAL_AVAILABLE:
            raise ImportError("Metal not available - requires macOS with Apple Silicon")

        # Initialize Metal device and resources
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Metal device not available")

        self._command_queue = self._device.newCommandQueue()
        self._kernels = {}
        self._compile_kernels()

        self.logger.debug(f"[UnifiedMetal] Initialized on device: {self._device.name()}")

    def _compile_kernels(self):
        """Compile modular Metal kernels."""
        try:
            # Read Metal kernel source
            import os
            kernel_path = os.path.join(os.path.dirname(__file__), "metal_kernels_parallel.metal")

            if not os.path.exists(kernel_path):
                raise FileNotFoundError(f"Modular Metal kernels not found: {kernel_path}")

            with open(kernel_path, 'r') as f:
                kernel_source = f.read()

            # Compile Metal library
            result = self._device.newLibraryWithSource_options_error_(kernel_source, None, None)
            library = result[0]
            error = result[1]
            if library is None:
                error_msg = f"Failed to compile modular Metal kernels: {error}" if error else "Failed to compile modular Metal kernels"
                raise RuntimeError(error_msg)

            # Create kernel pipeline states for all modular kernels
            kernel_names = [
                "build_csr_from_hJ",
                "calculate_optimal_replicas",
                "create_temperature_ladder",
                "calculate_read_partitioning",
                "parallel_tempering_sampling"
            ]

            for name in kernel_names:
                function = library.newFunctionWithName_(name)
                if function is None:
                    raise RuntimeError(f"Kernel function '{name}' not found")

                result = self._device.newComputePipelineStateWithFunction_error_(function, None)
                pipeline_state = result[0]
                error = result[1]
                if pipeline_state is None:
                    error_msg = f"Failed to create pipeline state for '{name}': {error}" if error else f"Failed to create pipeline state for '{name}'"
                    raise RuntimeError(error_msg)

                self._kernels[name] = pipeline_state

            self.logger.debug(f"[UnifiedMetal] Compiled {len(kernel_names)} modular kernels successfully")

        except Exception as e:
            self.logger.error(f"[UnifiedMetal] Kernel compilation failed: {e}")
            raise

    def _create_buffer(self, data: np.ndarray, label: str = "") -> Any:
        """Create Metal buffer from numpy array."""
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        byte_data = data.tobytes()
        if not isinstance(byte_data, bytes):
            raise TypeError(f"Expected bytes-like object, got {type(byte_data)}")
        byte_length = len(byte_data)
        buffer = self._device.newBufferWithBytes_length_options_(
            byte_data, byte_length, 0  # MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(f"Failed to create Metal buffer: {label}")
        return buffer

    def _pack_problem_input(self, h: Dict[int, float], J: Dict[tuple, float]):
        """Pack h,J dictionaries into ProblemInput struct format."""
        # Pack h dictionary
        h_nodes = []
        h_values = []
        for node, bias in h.items():
            h_nodes.append(node)
            h_values.append(bias)

        # Pack J dictionary
        J_edges = []
        J_values = []
        for (i, j), coupling in J.items():
            J_edges.extend([i, j])
            J_values.append(coupling)

        # Create arrays with proper padding (must match Metal constants)
        MAX_NODES = 512
        MAX_EDGES = 4096

        # Pad to max sizes
        while len(h_nodes) < MAX_NODES:
            h_nodes.append(0)
            h_values.append(0.0)

        while len(J_edges) < MAX_EDGES * 2:
            J_edges.append(0)
        while len(J_values) < MAX_EDGES:
            J_values.append(0.0)

        # Pack into struct format matching Metal ProblemInput
        # struct ProblemInput {
        #     int num_nodes; int num_edges;
        #     int h_nodes[MAX_NODES]; float h_values[MAX_NODES];
        #     int J_edges[MAX_EDGES * 2]; float J_values[MAX_EDGES];
        # }

        problem_data = struct.pack('<2i', len(h), len(J))  # num_nodes, num_edges
        problem_data += struct.pack(f'<{MAX_NODES}i', *h_nodes[:MAX_NODES])  # h_nodes
        problem_data += struct.pack(f'<{MAX_NODES}f', *h_values[:MAX_NODES])  # h_values
        problem_data += struct.pack(f'<{MAX_EDGES * 2}i', *J_edges[:MAX_EDGES * 2])  # J_edges
        problem_data += struct.pack(f'<{MAX_EDGES}f', *J_values[:MAX_EDGES])  # J_values

        return np.frombuffer(problem_data, dtype=np.uint8)

    def _pack_unified_params(self, num_reads: int, num_sweeps: int, num_replicas: int,
                           swap_interval: int, sample_interval: int, T_min: float, T_max: float,
                           base_seed: int, cooling_factor: float, cooling_start_sweep: int):
        """Pack unified parameters into UnifiedParams struct format."""
        # struct UnifiedParams {
        #     int num_reads, num_sweeps, num_replicas, swap_interval, sample_interval;
        #     float T_min, T_max; uint base_seed; float cooling_factor; int cooling_start_sweep;
        # }

        params_data = struct.pack('<5i2fI1f1i',
            num_reads, num_sweeps, num_replicas, swap_interval, sample_interval,
            T_min, T_max, base_seed & 0xFFFFFFFF, cooling_factor, cooling_start_sweep
        )

        return np.frombuffer(params_data, dtype=np.uint8)

    def sample_ising(self, h: Dict[int, float], J: Dict[tuple, float],
                    num_reads: int = 256, num_sweeps: int = 1000,
                    num_replicas: int = None, swap_interval: int = 15,
                    T_min: float = 0.01, T_max: float = 1.0,
                    sample_interval: int = None, cooling_factor: float = 0.999,
                    cooling_start_sweep: int = None, **kwargs) -> dimod.SampleSet:
        """GPU-only pipeline: build CSR, choose replicas, make temperatures, partition reads, and sample — all on GPU."""
        start_time = time.time()

        # Determine dense node order and mapping (0..N-1)
        node_set = set(h.keys()) if h else set()
        if J:
            for (i, j) in J.keys():
                node_set.add(int(i)); node_set.add(int(j))
        if not node_set:
            raise ValueError("Empty problem: no variables in h or J")
        node_list = sorted(node_set)
        label_to_idx = {int(lbl): idx for idx, lbl in enumerate(node_list)}
        N = len(node_list)
        max_idx = N - 1

        # Defaults
        if sample_interval is None:
            sample_interval = max(1, num_sweeps // max(1, num_reads))

        self.logger.debug(
            f"[UnifiedMetal] GPU-only pipeline N={N}, reads={num_reads}, sweeps={num_sweeps}"
        )

        try:
            # === 1) Build CSR on GPU ===
            # Pack ProblemData {num_nodes, num_edges, max_node_index}
            edges = []
            jvals = []
            for (i_lbl, j_lbl), coupling in J.items():
                if i_lbl == j_lbl:
                    continue
                ii = label_to_idx.get(int(i_lbl))
                jj = label_to_idx.get(int(j_lbl))
                if ii is None or jj is None:
                    continue
                edges.append((int(ii), int(jj)))
                jvals.append(float(coupling))
            E = len(edges)

            problem_data = np.array([N, E, max_idx], dtype=np.int32)
            J_edges = np.array([e for pair in edges for e in pair], dtype=np.int32)
            J_values = np.array(jvals, dtype=np.float32)

            # Outputs (over-allocate col_ind/J_vals as 2*E for undirected expansion)
            csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
            csr_col_ind = np.zeros(max(1, 2 * E), dtype=np.int32)
            csr_J_vals = np.zeros(max(1, 2 * E), dtype=np.int8)
            nnz_out = np.zeros(1, dtype=np.int32)
            row_ptr_working = np.zeros(N, dtype=np.int32)

            # Create buffers
            problem_buf = self._create_buffer(problem_data, "problem_data")
            J_edges_buf = self._create_buffer(J_edges if E > 0 else np.zeros(2, dtype=np.int32), "J_edges")
            J_values_buf = self._create_buffer(J_values if E > 0 else np.zeros(1, dtype=np.float32), "J_values")
            csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
            csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
            csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
            nnz_out_buf = self._create_buffer(nnz_out, "nnz_out")
            row_ptr_working_buf = self._create_buffer(row_ptr_working, "row_ptr_working")

            # Dispatch build_csr_from_hJ
            cb1 = self._command_queue.commandBuffer()
            enc1 = cb1.computeCommandEncoder()
            enc1.setComputePipelineState_(self._kernels["build_csr_from_hJ"])
            enc1.setBuffer_offset_atIndex_(problem_buf, 0, 0)         # 0
            enc1.setBuffer_offset_atIndex_(J_edges_buf, 0, 1)         # 1
            enc1.setBuffer_offset_atIndex_(J_values_buf, 0, 2)        # 2
            enc1.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 3)     # 3
            enc1.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 4)     # 4
            enc1.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 5)      # 5
            enc1.setBuffer_offset_atIndex_(nnz_out_buf, 0, 6)         # 6
            enc1.setBuffer_offset_atIndex_(row_ptr_working_buf, 0, 7) # 7
            enc1.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=1, height=1, depth=1), Metal.MTLSize(width=1, height=1, depth=1))
            enc1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()

            # Read nnz
            nnz_contents = nnz_out_buf.contents()
            nnz_view = nnz_contents.as_buffer(nnz_out_buf.length())
            nnz_val = struct.unpack('<i', bytes([nnz_view[i] for i in range(4)]))[0]
            if nnz_val <= 0 and E > 0:
                raise RuntimeError("GPU CSR construction produced zero nnz")

            # === 2) Optimal replicas on GPU ===
            req_reps = 0 if (num_replicas is None) else int(num_replicas)
            density = (2.0 * float(E)) / max(1.0, float(N) * float(N - 1))
            N_buf = self._create_buffer(np.array([N], dtype=np.int32), "N")
            E_buf = self._create_buffer(np.array([E], dtype=np.int32), "E")
            density_buf = self._create_buffer(np.array([density], dtype=np.float32), "density")
            req_buf = self._create_buffer(np.array([req_reps], dtype=np.int32), "requested")
            opt_buf = self._create_buffer(np.array([0], dtype=np.int32), "optimal")

            cb2 = self._command_queue.commandBuffer()
            enc2 = cb2.computeCommandEncoder()
            enc2.setComputePipelineState_(self._kernels["calculate_optimal_replicas"])
            enc2.setBuffer_offset_atIndex_(N_buf, 0, 0)
            enc2.setBuffer_offset_atIndex_(E_buf, 0, 1)
            enc2.setBuffer_offset_atIndex_(density_buf, 0, 2)
            enc2.setBuffer_offset_atIndex_(req_buf, 0, 3)
            enc2.setBuffer_offset_atIndex_(opt_buf, 0, 4)
            enc2.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=1, height=1, depth=1), Metal.MTLSize(width=1, height=1, depth=1))
            enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()

            # Read back optimal replicas
            opt_contents = opt_buf.contents(); opt_view = opt_contents.as_buffer(opt_buf.length())
            num_reps = struct.unpack('<i', bytes([opt_view[i] for i in range(4)]))[0]
            num_reps = max(1, int(num_reps))

            # === 3) Temperature ladder on GPU ===
            temps = np.zeros(num_reps, dtype=np.float32)
            T_min_buf = self._create_buffer(np.array([float(T_min)], dtype=np.float32), "T_min")
            T_max_buf = self._create_buffer(np.array([float(T_max)], dtype=np.float32), "T_max")
            num_reps_buf = self._create_buffer(np.array([num_reps], dtype=np.int32), "num_reps")
            temps_buf = self._create_buffer(temps, "temperatures")

            cb3 = self._command_queue.commandBuffer()
            enc3 = cb3.computeCommandEncoder()
            enc3.setComputePipelineState_(self._kernels["create_temperature_ladder"])
            enc3.setBuffer_offset_atIndex_(T_min_buf, 0, 0)
            enc3.setBuffer_offset_atIndex_(T_max_buf, 0, 1)
            enc3.setBuffer_offset_atIndex_(num_reps_buf, 0, 2)
            enc3.setBuffer_offset_atIndex_(temps_buf, 0, 3)
            enc3.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=1, height=1, depth=1), Metal.MTLSize(width=1, height=1, depth=1))
            enc3.endEncoding(); cb3.commit(); cb3.waitUntilCompleted()

            # === 4) Read partitioning on GPU ===
            per_replica_base = np.zeros(num_reps, dtype=np.int32)
            per_replica_quota = np.zeros(num_reps, dtype=np.int32)
            reads_buf = self._create_buffer(np.array([int(num_reads)], dtype=np.int32), "num_reads")
            per_replica_base_buf = self._create_buffer(per_replica_base, "per_replica_base")
            per_replica_quota_buf = self._create_buffer(per_replica_quota, "per_replica_quota")

            cb4 = self._command_queue.commandBuffer()
            enc4 = cb4.computeCommandEncoder()
            enc4.setComputePipelineState_(self._kernels["calculate_read_partitioning"])
            enc4.setBuffer_offset_atIndex_(reads_buf, 0, 0)
            enc4.setBuffer_offset_atIndex_(num_reps_buf, 0, 1)
            enc4.setBuffer_offset_atIndex_(per_replica_base_buf, 0, 2)
            enc4.setBuffer_offset_atIndex_(per_replica_quota_buf, 0, 3)
            enc4.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=1, height=1, depth=1), Metal.MTLSize(width=1, height=1, depth=1))
            enc4.endEncoding(); cb4.commit(); cb4.waitUntilCompleted()

            # === 5) Sampling on GPU ===
            if sample_interval is None:
                sample_interval = max(1, num_sweeps // max(1, num_reads // max(1, num_reps)))
            sweeps_buf = self._create_buffer(np.array([int(num_sweeps)], dtype=np.int32), "num_sweeps")
            samp_int_buf = self._create_buffer(np.array([int(sample_interval)], dtype=np.int32), "sample_interval")
            base_seed_buf = self._create_buffer(np.array([int(time.time() * 1000) & 0xFFFFFFFF], dtype=np.uint32), "base_seed")

            final_samples = np.zeros(num_reads * N, dtype=np.int8)
            final_energies = np.zeros(num_reads, dtype=np.int32)
            final_samples_buf = self._create_buffer(final_samples, "final_samples")
            final_energies_buf = self._create_buffer(final_energies, "final_energies")

            thread_buffers = np.zeros(num_reps * N, dtype=np.int8)
            rng_states = np.zeros(num_reps, dtype=np.uint32)
            thread_buffers_buf = self._create_buffer(thread_buffers, "thread_buffers")
            rng_states_buf = self._create_buffer(rng_states, "rng_states")

            cb5 = self._command_queue.commandBuffer()
            enc5 = cb5.computeCommandEncoder()
            enc5.setComputePipelineState_(self._kernels["parallel_tempering_sampling"])
            # set buffers 0..12 as per signature
            enc5.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
            enc5.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
            enc5.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
            enc5.setBuffer_offset_atIndex_(N_buf, 0, 3)
            enc5.setBuffer_offset_atIndex_(temps_buf, 0, 4)
            enc5.setBuffer_offset_atIndex_(num_reps_buf, 0, 5)
            enc5.setBuffer_offset_atIndex_(sweeps_buf, 0, 6)
            enc5.setBuffer_offset_atIndex_(samp_int_buf, 0, 7)
            enc5.setBuffer_offset_atIndex_(base_seed_buf, 0, 8)
            enc5.setBuffer_offset_atIndex_(per_replica_base_buf, 0, 9)
            enc5.setBuffer_offset_atIndex_(per_replica_quota_buf, 0, 10)
            enc5.setBuffer_offset_atIndex_(final_samples_buf, 0, 11)
            enc5.setBuffer_offset_atIndex_(final_energies_buf, 0, 12)
            enc5.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 13)
            enc5.setBuffer_offset_atIndex_(rng_states_buf, 0, 14)

            max_threads = 256
            if num_reps <= max_threads:
                threads_per_group = Metal.MTLSize(width=num_reps, height=1, depth=1)
                num_groups = Metal.MTLSize(width=1, height=1, depth=1)
            else:
                threads_per_group = Metal.MTLSize(width=max_threads, height=1, depth=1)
                groups = (num_reps + max_threads - 1) // max_threads
                num_groups = Metal.MTLSize(width=groups, height=1, depth=1)

            self.logger.debug(f"[UnifiedMetal] Dispatch PT: {num_groups.width} groups × {threads_per_group.width} threads")
            enc5.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)
            enc5.endEncoding(); cb5.commit(); cb5.waitUntilCompleted()

            # Collect results
            samples, energies = self._collect_results(final_samples_buf, final_energies_buf, num_reads, N)
            runtime = time.time() - start_time
            self.logger.debug(f"[UnifiedMetal] Full GPU sampling completed: {runtime:.2f}s")
            return dimod.SampleSet.from_samples(samples, 'SPIN', energies)

        except Exception as e:
            self.logger.error(f"[UnifiedMetal] Sampling failed: {e}")
            raise

    def _collect_results(self, final_samples_buffer, final_energies_buffer, num_reads: int, N: int):
        """Collect samples and energies from Metal buffers."""
        # Read energies
        energies_contents = final_energies_buffer.contents()
        energies_buffer_length = final_energies_buffer.length()
        energies_buffer_view = energies_contents.as_buffer(energies_buffer_length)
        energies = []

        for i in range(num_reads):
            offset = i * 4  # 4 bytes per int32
            energy_bytes = bytes([energies_buffer_view[offset + j] for j in range(4)])
            energy = struct.unpack('<i', energy_bytes)[0]
            energies.append(energy)

        # Read samples
        samples_contents = final_samples_buffer.contents()
        samples_buffer_length = final_samples_buffer.length()
        samples_buffer_view = samples_contents.as_buffer(samples_buffer_length)
        samples_bytes = bytes(samples_buffer_view)
        samples_flat = np.frombuffer(samples_bytes, dtype=np.int8)

        expected_len = num_reads * N
        if samples_flat.size < expected_len:
            raise RuntimeError(f"Expected at least {expected_len} sample bytes, got {samples_flat.size}")
        samples_flat = samples_flat[:expected_len]

        samples = []
        for i in range(num_reads):
            start_idx = i * N
            end_idx = start_idx + N
            sample = samples_flat[start_idx:end_idx].tolist()
            samples.append(sample)

        # Validate execution
        for i in range(num_reads):
            if energies[i] == 0 and all(v == 0 for v in samples[i]):
                raise RuntimeError(f"Thread execution failure: Read {i} has zero energy and zeroed sample")

        return samples, energies

    def close(self):
        """Clean up Metal resources."""
        if hasattr(self, '_command_queue'):
            self._command_queue = None
        if hasattr(self, '_device'):
            self._device = None
        if hasattr(self, '_kernels'):
            self._kernels.clear()

    def __del__(self):
        try:
            self.close()
        except:
            pass


if __name__ == "__main__":
    # Test the unified sampler
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Unified Metal Sampler")
    print("=" * 50)

    try:
        sampler = UnifiedMetalSampler(verbose=True)

        # Test with simple 2-spin problem
        h = {0: 0.0, 1: 0.0}
        J = {(0, 1): -1.0}

        sampleset = sampler.sample_ising(
            h=h, J=J,
            num_reads=10,
            num_sweeps=100,
            num_replicas=4
        )

        energies = list(sampleset.record.energy)
        print(f"\nResults for 2-spin ferromagnetic:")
        print(f"Samples: {list(sampleset.samples())}")
        print(f"Energies: {energies}")
        print(f"Min energy: {min(energies)} (expected: -1)")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()