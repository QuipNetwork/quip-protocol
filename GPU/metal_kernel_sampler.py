"""3D Edwards-Anderson Metal Parallel Tempering Sampler.

Implements the structured approach from CURRENT_PLAN.md:
- 3D cubic lattice with L×L×L spins
- Random ±1 Edwards-Anderson couplings
- Optimized Metal kernels with structured data layout
- Target performance: >50,000 spin updates/sec on N=216
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import struct

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

import dimod
from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY


class MetalKernelDimodSampler:
    """Apple Metal GPU Parallel Tempering sampler for 3D Edwards-Anderson spin glasses."""

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None,
                 verbose: bool = False, default_sample_interval: Optional[int] = None):
        """Initialize Metal EA sampler.

        Args:
            device: Metal device ("mps" for Apple Metal Performance Shaders)
            logger: Optional logger instance
            verbose: If True, enable DEBUG logging for this sampler
            default_sample_interval: Optional default sample interval (>=1) to use
                for all calls unless overridden in sample_ising(). If None, a
                heuristic is used per call.
        """
        self.logger = logger or logging.getLogger(__name__)
        self._verbose = verbose
        # Configure logger level/handlers only if a logger wasn't provided
        if logger is None:
            self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            # Respect provided logger but elevate to DEBUG if verbose explicitly requested
            if verbose:
                self.logger.setLevel(logging.DEBUG)
        # Default sampling behavior
        self._default_sample_interval = default_sample_interval
        self.sampler_type = "metal_ea_3d"

        if not METAL_AVAILABLE:
            raise ImportError("Metal not available - requires macOS with Apple Silicon")

        # Initialize Metal device and resources
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Metal device not available")

        self._command_queue = self._device.newCommandQueue()
        self._kernels = {}
        self._compile_kernels()

        # Use the same topology as other samplers for compatibility
        topology_graph = DEFAULT_TOPOLOGY.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())

        self.logger.debug(f"[MetalEA] Initialized on device: {self._device.name()}")

    def _compile_kernels(self):
        """Compile Metal kernels from EA-specific Metal file."""
        try:
            # Read Metal kernel source
            import os
            kernel_path = os.path.join(os.path.dirname(__file__), "metal_kernels.metal")

            if not os.path.exists(kernel_path):
                raise FileNotFoundError(f"EA Metal kernels not found: {kernel_path}")

            with open(kernel_path, 'r') as f:
                kernel_source = f.read()

            # Compile Metal library
            library = self._device.newLibraryWithSource_options_error_(kernel_source, None, None)[0]
            if library is None:
                raise RuntimeError("Failed to compile Metal EA kernels")

            # Create kernel pipeline states
            kernel_names = [
                "worker_parallel_tempering"
            ]

            for name in kernel_names:
                function = library.newFunctionWithName_(name)
                if function is None:
                    raise RuntimeError(f"Kernel function '{name}' not found")

                pipeline_state = self._device.newComputePipelineStateWithFunction_error_(function, None)[0]
                if pipeline_state is None:
                    raise RuntimeError(f"Failed to create pipeline state for '{name}'")

                self._kernels[name] = pipeline_state

            self.logger.debug(f"[MetalEA] Compiled {len(kernel_names)} kernels successfully")

        except Exception as e:
            self.logger.error(f"[MetalEA] Kernel compilation failed: {e}")
            raise


    def _create_buffer(self, data: np.ndarray, label: str = "") -> Any:
        """Create Metal buffer from numpy array."""
        # Ensure data is contiguous and in bytes
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

    def _write_buffer_bytes(self, buffer: Any, byte_data: bytes, label: str = "") -> None:
        """Write raw bytes into an existing Metal buffer (shared storage).
        Length must not exceed buffer.length().
        """
        if not isinstance(byte_data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes/bytearray for {label} write, got {type(byte_data)}")
        buf_len = buffer.length()
        data_len = len(byte_data)
        if data_len > buf_len:
            raise ValueError(f"Write overflow for {label}: {data_len} > buffer.length {buf_len}")
        contents = buffer.contents()
        view = contents.as_buffer(buf_len)
        mv = memoryview(view)
        mv[:data_len] = bytearray(byte_data)

    def _convert_problem_to_buffers(self, h, J, L, num_replicas):
        """Convert arbitrary h,J problem to Metal buffer format."""
        N = len(h)  # Use actual problem size, not L^3

        # Build CSR adjacency from J (undirected: store both i->j and j->i)
        adjacency = [[] for _ in range(N)]
        weights = [[] for _ in range(N)]
        for (i, j), coupling in J.items():
            if i >= N or j >= N:
                continue
            # Store both directions
            adjacency[i].append(j)
            weights[i].append(int(np.sign(coupling)) if coupling != 0 else 0)
            adjacency[j].append(i)
            weights[j].append(int(np.sign(coupling)) if coupling != 0 else 0)

        # Convert to CSR arrays
        row_ptr = np.zeros(N + 1, dtype=np.int32)
        col_ind_list = []
        j_vals_list = []
        nnz = 0
        for i in range(N):
            row_ptr[i] = nnz
            # Keep original order; could sort(adjacency[i]) if needed
            cols = adjacency[i]
            vals = weights[i]
            col_ind_list.extend(cols)
            j_vals_list.extend(vals)
            nnz += len(cols)
        row_ptr[N] = nnz
        col_ind = np.array(col_ind_list, dtype=np.int32)
        j_vals = np.array(j_vals_list, dtype=np.int8)

        # Create adaptive temperature schedule for parallel tempering
        temperatures = self._create_adaptive_temperature_ladder(0.01, 1.0, num_replicas).astype(np.float32)

        # Create global data buffer (must match Metal GlobalData struct layout)
        # struct GlobalData { int N; int num_replicas; int swap_interval; int cooling_interval; float T_min, T_max; int step; };
        int_values = np.array([
            N,                   # N (actual number of spins)
            num_replicas,        # num_replicas
            15,                  # swap_interval
            500,                 # cooling_interval
        ], dtype=np.int32)

        float_values = np.array([
            0.01,                # T_min
            1.0,                 # T_max
        ], dtype=np.float32)

        step_value = np.array([0], dtype=np.int32)  # step counter

        global_data_bytes = int_values.tobytes() + float_values.tobytes() + step_value.tobytes()
        global_data = np.frombuffer(global_data_bytes, dtype=np.uint8)

        return {
            'csr_row_ptr': row_ptr,
            'csr_col_ind': col_ind,
            'csr_J_vals': j_vals,
            'temperatures': temperatures,
            'global_data': global_data
        }

    def _create_adaptive_temperature_ladder(self, T_min, T_max, num_replicas):
        """
        Phase 1: Optimized geometric temperature ladder for ~25-35% swap acceptance.
        Implements research-based optimal spacing for parallel tempering.
        """
        if num_replicas == 1:
            return np.array([T_min])

        # Phase 1.1: Geometric spacing with optimal ratios
        # Research shows optimal ratio depends on problem complexity and replica count
        base_ratio = (T_max / T_min) ** (1.0 / (num_replicas - 1))

        # Phase 1.2: Adaptive ratio adjustment for target acceptance
        if num_replicas <= 4:
            # For few replicas, use tighter spacing to ensure good communication
            adjustment_factor = 1.15
        elif num_replicas <= 8:
            # Moderate replica count - balanced spacing
            adjustment_factor = 1.05
        elif num_replicas <= 16:
            # Many replicas - slightly tighter spacing
            adjustment_factor = 0.98
        else:
            # Very many replicas - much tighter spacing for good overlap
            adjustment_factor = 0.92

        optimal_ratio = base_ratio * adjustment_factor

        # Generate geometric ladder
        temperatures = np.array([T_min * (optimal_ratio ** i) for i in range(num_replicas)])

        # Ensure we hit T_max exactly
        temperatures[-1] = T_max

        # Phase 1.3: Problem-complexity adaptive T_max (implemented here for convenience)
        # This will be used when N is available in calling context

        self.logger.debug(f"[MetalEA] Phase 1 geometric temperature ladder: {temperatures}")
        self.logger.debug(f"[MetalEA] Temperature ratios: {temperatures[1:] / temperatures[:-1]}")
        self.logger.debug(f"[MetalEA] Adjustment factor: {adjustment_factor:.3f}, optimal ratio: {optimal_ratio:.3f}")

        return temperatures

    def _calculate_optimal_replicas(self, problem_size, num_couplings, coupling_density):
        """
        Phase 3.2: Calculate optimal replica count based on problem characteristics.
        Based on research findings for parallel tempering performance.
        """
        # Base replica count from problem size (logarithmic scaling)
        base_replicas = max(8, int(np.log2(problem_size)) * 2)

        # Adjust based on coupling density
        if coupling_density > 0.1:  # Dense problems need more replicas for mixing
            density_factor = 1.5
        elif coupling_density < 0.01:  # Sparse problems can use fewer replicas
            density_factor = 0.8
        else:  # Medium density
            density_factor = 1.0

        # Adjust based on energy scale (larger problems with more couplings)
        if num_couplings > 1000:  # Large energy scale problems
            energy_factor = 1.3
        elif num_couplings < 100:  # Small energy scale problems
            energy_factor = 0.9
        else:
            energy_factor = 1.0

        # Calculate optimal replica count
        optimal_replicas = int(base_replicas * density_factor * energy_factor)

        # Clamp to reasonable bounds (8-64 replicas)
        optimal_replicas = min(max(optimal_replicas, 8), 64)

        self.logger.debug(f"[MetalEA] Phase 3.2 adaptive replicas: problem_size={problem_size}, "
                         f"coupling_density={coupling_density:.4f}, num_couplings={num_couplings}")
        self.logger.debug(f"[MetalEA] Replica calculation: base={base_replicas}, "
                         f"density_factor={density_factor:.2f}, energy_factor={energy_factor:.2f}")
        self.logger.debug(f"[MetalEA] Optimal replica count: {optimal_replicas}")

        return optimal_replicas

    def sample_ising(self, h, J, num_reads: int = 256, num_sweeps: int = 1000,
                            num_replicas: int = None, swap_interval: int = 15,
                            T_min: float = 0.01, T_max: float = 1.0,
                            sample_interval: Optional[int] = None,
                            cooling_factor: float = 0.999, cooling_start_sweep: int = None,
                            **kwargs) -> dimod.SampleSet:
        """GPU-only unified parallel tempering sampler - single kernel dispatch (DEFAULT).
        Args:
            sample_interval: If provided, collect a sample every this many sweeps (>=1).
                             If None, a heuristic based on num_reads/num_replicas is used.
        """

        start_time = time.time()
        N = len(h)
        L = int(round(N ** (1/3)))  # Approximate cube root for compatibility

        # Calculate coupling density early for adaptive algorithms
        coupling_density = len(J) / max(1, (N * (N - 1) / 2))  # Avoid division by zero

        # Phase 3.2: Adaptive replica count based on problem size and energy scale
        if num_replicas is None:
            num_replicas = self._calculate_optimal_replicas(N, len(J), coupling_density)

        # Speed optimization: Only scale replicas if we need more samples than replicas
        # Don't over-parallelize - it creates overhead instead of speedup
        if num_reads > num_replicas:
            # Scale to have at least as many replicas as reads, but cap at reasonable limit
            target_replicas = min(num_reads, 64)  # Cap at 64 to avoid overhead
            if target_replicas > num_replicas:
                original_replicas = num_replicas
                num_replicas = target_replicas
                self.logger.debug(f"[MetalEA] Speed optimization: Scaled replicas {original_replicas} → {num_replicas} to match read count")

        self.logger.debug(f"[MetalEA] Unified GPU sampling: N={N}, replicas={num_replicas}, sweeps={num_sweeps}")

        # Convert problem to CSR format
        buffer_data = self._convert_problem_to_buffers(h, J, L, num_replicas)

        # Debug: Check CSR data
        self.logger.debug(f"[MetalEA] Debug - CSR edges: {len(buffer_data['csr_col_ind'])}")
        self.logger.debug(f"[MetalEA] Debug - CSR J values: {buffer_data['csr_J_vals'][:10]}")  # First 10

        # Debug: Count edges with j > i condition
        csr_row_ptr = buffer_data['csr_row_ptr']
        csr_col_ind = buffer_data['csr_col_ind']
        csr_J_vals = buffer_data['csr_J_vals']
        edge_count = 0
        for i in range(N):
            start = csr_row_ptr[i]
            end = csr_row_ptr[i + 1]
            for p in range(start, end):
                j = csr_col_ind[p]
                if j > i:
                    edge_count += 1
        self.logger.debug(f"[MetalEA] Debug - Edges with j>i: {edge_count} (should be {len(J)})")

        # Phase 1.3: Problem-complexity adaptive temperature range
        # Scale T_max with problem complexity and coupling density (already calculated above)
        adaptive_T_max = max(T_max, 0.1 * np.sqrt(len(h) + len(J)))
        adaptive_T_min = min(T_min, 0.001)  # Lower minimum for better final convergence

        # Adjust based on coupling density
        if coupling_density > 0.1:  # Dense problems need higher T_max
            adaptive_T_max *= 1.5
        elif coupling_density < 0.01:  # Sparse problems can use lower T_max
            adaptive_T_max *= 0.8

        self.logger.debug(f"[MetalEA] Phase 1 adaptive temperature range: T_min={adaptive_T_min:.4f}, T_max={adaptive_T_max:.4f}")
        self.logger.debug(f"[MetalEA] Coupling density: {coupling_density:.4f}")

        # Create temperature ladder
        temperatures = self._create_adaptive_temperature_ladder(adaptive_T_min, adaptive_T_max, num_replicas).astype(np.float32)

        # Create sampling parameters struct with proper data types
        chosen_interval = sample_interval if sample_interval is not None else self._default_sample_interval
        if chosen_interval is None:
            sample_interval_local = max(1, num_sweeps // (max(1, num_reads // num_replicas) + 1))
        else:
            sample_interval_local = max(1, min(int(chosen_interval), int(num_sweeps)))

        # Phase 2.2: Problem-adaptive cooling parameters
        # Adjust cooling factor and start sweep based on problem characteristics
        if coupling_density > 0.1:  # Dense problems need more cooling
            adaptive_cooling_factor = 0.995
            adaptive_cooling_start_sweep = max(1, int(num_sweeps * 0.3))  # Start cooling earlier
        else:  # Sparse problems can start cooling later
            adaptive_cooling_factor = 0.999
            adaptive_cooling_start_sweep = max(1, int(num_sweeps * 0.5))  # Start cooling later

        # Override if explicitly provided
        final_cooling_factor = cooling_factor if cooling_factor != 0.999 else adaptive_cooling_factor
        if cooling_start_sweep is None:
            final_cooling_start_sweep = adaptive_cooling_start_sweep
        else:
            final_cooling_start_sweep = cooling_start_sweep

        self.logger.debug(f"[MetalEA] Phase 2 adaptive cooling: factor={final_cooling_factor:.4f}, start_sweep={final_cooling_start_sweep}")
        self.logger.debug(f"[MetalEA] Coupling density: {coupling_density:.4f} -> {'dense' if coupling_density > 0.1 else 'sparse'} problem type")

        try:
            # Create input buffers (reused across passes)
            csr_row_ptr_buffer = self._create_buffer(buffer_data['csr_row_ptr'], "csr_row_ptr")
            csr_col_ind_buffer = self._create_buffer(buffer_data['csr_col_ind'], "csr_col_ind")
            csr_J_vals_buffer = self._create_buffer(buffer_data['csr_J_vals'], "csr_J_vals")
            # Create reusable params buffer (written in-place per pass)
            # Updated for Phase 3: Added enable_replica_exchange (int) + exchange_threshold (float)
            params_size = struct.calcsize('<6i2fIfi1f1i1i1f')
            params_buffer = self._device.newBufferWithLength_options_(params_size, 0)
            if params_buffer is None:
                raise RuntimeError("Failed to create Metal params buffer")

            # Create reusable staging buffer for params
            params_staging_buffer = self._device.newBufferWithLength_options_(params_size, 0)
            if params_staging_buffer is None:
                raise RuntimeError("Failed to create Metal params staging buffer")

            temperatures_buffer = self._create_buffer(temperatures, "temperatures")

            # Communication buffers (reused across passes)
            thread_buffers = np.zeros(num_replicas * N, dtype=np.int8)
            rng_states = np.zeros(num_replicas, dtype=np.uint32)

            thread_buffers_buffer = self._create_buffer(thread_buffers, "thread_buffers")
            rng_states_buffer = self._create_buffer(rng_states, "rng_states")

            # Output buffers (sized to num_replicas; we'll slice to batch_reads per pass)
            final_samples = np.zeros(num_replicas * N, dtype=np.int8)
            final_energies = np.zeros(num_replicas, dtype=np.int32)

            final_samples_buffer = self._create_buffer(final_samples, "final_samples")
            final_energies_buffer = self._create_buffer(final_energies, "final_energies")

            all_samples = []
            all_energies = []
            produced = 0
            pass_idx = 0

            # Speed optimization: Use larger batches to reduce kernel dispatch overhead
            while produced < num_reads:
                batch_reads = min(num_replicas, num_reads - produced)

                # If we have many replicas, we can produce all reads in one or few passes
                if num_replicas >= num_reads:
                    batch_reads = num_reads - produced
                    self.logger.debug(f"[MetalEA] Speed optimization: Single-pass execution with {num_replicas} replicas for {batch_reads} reads")

                # Pack struct for this pass (num_reads=batch_reads, vary base_seed)
                base_seed = int(time.time() * 1000) ^ (pass_idx * 0x9E3779B1)

                # Phase 1: Target acceptance ratio and adaptation interval
                target_acceptance = 0.30  # Target 30% acceptance ratio for optimal mixing
                adaptation_interval = max(50, num_sweeps // 20)  # Adapt every 5% of sweeps

                # Phase 3: Intelligent replica exchange parameters
                enable_replica_exchange = 1 if num_replicas > 1 else 0  # Enable only with multiple replicas
                exchange_threshold = 0.05  # Temperature difference threshold for exchange consideration

                sampling_params_bytes = struct.pack('<6i2fIfi1f1i1i1f',
                    N,                    # int N
                    num_replicas,         # int num_replicas
                    num_sweeps,           # int num_sweeps
                    batch_reads,          # int num_reads (for this pass)
                    swap_interval,        # int swap_interval
                    sample_interval_local,      # int sample_interval
                    adaptive_T_min,       # float T_min (use adaptive values)
                    adaptive_T_max,       # float T_max (use adaptive values)
                    base_seed & 0xFFFFFFFF,  # uint base_seed
                    final_cooling_factor, # float cooling_factor (adaptive)
                    final_cooling_start_sweep,  # int cooling_start_sweep (adaptive)
                    target_acceptance,    # float target_acceptance_ratio
                    adaptation_interval,  # int adaptation_interval
                    enable_replica_exchange,  # int enable_replica_exchange
                    exchange_threshold    # float exchange_threshold
                )
                # Update reusable staging buffer with current params; blit into persistent params_buffer
                self._write_buffer_bytes(params_staging_buffer, sampling_params_bytes, "params_staging")

                # Compute per-thread base/quota to partition batch_reads across replicas
                quotas = np.zeros(num_replicas, dtype=np.int32)
                if batch_reads > 0:
                    base_q = batch_reads // num_replicas
                    rem_q = batch_reads % num_replicas
                    if base_q > 0:
                        quotas[:] = base_q
                    if rem_q > 0:
                        quotas[:rem_q] += 1
                bases = np.zeros(num_replicas, dtype=np.int32)
                if num_replicas > 1:
                    # Prefix sum of quotas for base indices
                    bases[1:] = np.cumsum(quotas[:-1], dtype=np.int32)

                per_thread_base_buffer = self._create_buffer(bases, "per_thread_base")
                per_thread_quota_buffer = self._create_buffer(quotas, "per_thread_quota")


                # Worker-only dispatch: single command buffer per pass
                self.logger.debug(f"[MetalEA] Dispatching worker-only (producer-free) pass {pass_idx+1}...")

                # Single command buffer and encoder per pass
                command_buffer = self._command_queue.commandBuffer()

                # STEP 1: Dispatch worker kernel

                # Set worker kernel buffers - matches worker_parallel_tempering signature
                # Update persistent params_buffer using a blit from staging_params_buffer
                blit = command_buffer.blitCommandEncoder()
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                    params_staging_buffer, 0, params_buffer, 0, params_size
                )
                blit.endEncoding()

                encoder = command_buffer.computeCommandEncoder()
                encoder.setComputePipelineState_(self._kernels["worker_parallel_tempering"])

                encoder.setBuffer_offset_atIndex_(csr_row_ptr_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(csr_col_ind_buffer, 0, 1)
                encoder.setBuffer_offset_atIndex_(csr_J_vals_buffer, 0, 2)
                encoder.setBuffer_offset_atIndex_(temperatures_buffer, 0, 3)
                encoder.setBuffer_offset_atIndex_(params_buffer, 0, 4)
                encoder.setBuffer_offset_atIndex_(final_samples_buffer, 0, 5)
                encoder.setBuffer_offset_atIndex_(final_energies_buffer, 0, 6)
                # Worker communication buffers
                encoder.setBuffer_offset_atIndex_(thread_buffers_buffer, 0, 7)
                encoder.setBuffer_offset_atIndex_(rng_states_buffer, 0, 8)
                # Per-thread output partitioning
                encoder.setBuffer_offset_atIndex_(per_thread_base_buffer, 0, 9)
                encoder.setBuffer_offset_atIndex_(per_thread_quota_buffer, 0, 10)

                # Speed optimization: Better GPU utilization with larger threadgroups
                max_threads_per_group = 32  # Apple Silicon optimal threadgroup size
                if num_replicas <= max_threads_per_group:
                    # Small number of replicas: use single threadgroup with multiple threads
                    threads_per_group = Metal.MTLSize(width=num_replicas, height=1, depth=1)
                    num_groups = Metal.MTLSize(width=1, height=1, depth=1)
                else:
                    # Large number of replicas: use multiple threadgroups
                    threads_per_group = Metal.MTLSize(width=max_threads_per_group, height=1, depth=1)
                    num_groups = Metal.MTLSize(width=(num_replicas + max_threads_per_group - 1) // max_threads_per_group, height=1, depth=1)

                total_threads = num_groups.width * threads_per_group.width
                self.logger.debug(f"[MetalEA] Optimized GPU dispatch: {num_groups.width} groups × {threads_per_group.width} threads = {total_threads} total threads")
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)



                # Finish encoding and submit once
                encoder.endEncoding()
                command_buffer.commit()
                command_buffer.waitUntilCompleted()

                # Read results for this pass
                samples, gpu_energies = self._collect_unified_samples(final_samples_buffer, final_energies_buffer, batch_reads, N)

                all_samples.extend(samples)
                all_energies.extend(gpu_energies)
                produced += batch_reads
                pass_idx += 1

            runtime = time.time() - start_time
            self.logger.debug(f"[MetalEA] Unified sampling completed (multi-pass): {runtime:.2f}s, reads={num_reads}")

            # Create SampleSet
            return dimod.SampleSet.from_samples(all_samples[:num_reads], 'SPIN', all_energies[:num_reads])

        except Exception as e:
            self.logger.error(f"[MetalEA] Unified sampling failed: {e}")
            raise


    def _collect_unified_samples(self, final_samples_buffer, final_energies_buffer, num_reads: int, N: int):
        """Collect samples from unified kernel output buffers."""
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

            # NOTE: Energy can legitimately be 0 for some graphs/samples; defer validation until samples are read

            if i < 4:  # Debug first 4 entries
                self.logger.debug(f"[MetalEA] Python read sample {i}: energy = {energy}")

        # Read samples
        samples_contents = final_samples_buffer.contents()
        samples_buffer_length = final_samples_buffer.length()
        samples_buffer_view = samples_contents.as_buffer(samples_buffer_length)
        samples_bytes = bytes(samples_buffer_view)
        samples_flat = np.frombuffer(samples_bytes, dtype=np.int8)

        # Use only the first effective set of samples; allow larger buffer (e.g., allocated for original num_reads)
        expected_len = num_reads * N
        if samples_flat.size < expected_len:
            raise RuntimeError(f"THREAD EXECUTION FAILURE: Expected at least {expected_len} sample bytes, got {samples_flat.size}.")
        samples_flat = samples_flat[:expected_len]

        samples = []
        for i in range(num_reads):
            start_idx = i * N
            end_idx = start_idx + N
            sample = samples_flat[start_idx:end_idx].tolist()
            samples.append(sample)

        # Validate execution: zero energy is only a failure if the sample is all zeros
        for i in range(num_reads):
            if energies[i] == 0 and all(v == 0 for v in samples[i]):
                raise RuntimeError(
                    f"THREAD EXECUTION FAILURE: Read {i} has zero energy and zeroed sample (thread likely didn't execute)"
                )

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
            pass  # Ignore cleanup errors during destruction


if __name__ == "__main__":
    # Test the 3D EA sampler
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing 3D Edwards-Anderson Metal Sampler")
    print("=" * 50)

    try:
        sampler = MetalKernelDimodSampler()

        # Test with dummy h, J (will be ignored)
        h = {i: 0.0 for i in range(216)}  # L=6, N=216
        J = {}

        sampleset = sampler.sample_ising(
            h=h, J=J,
            num_reads=100,
            num_sweeps=10000,
            num_replicas=64
        )

        energies = list(sampleset.record.energy)
        print(f"\nResults for L=6 (N=216):")
        print(f"Min energy: {min(energies):.1f}")
        print(f"Max energy: {max(energies):.1f}")
        print(f"Mean energy: {sum(energies)/len(energies):.1f}")
        print(f"Unique energies: {len(set(energies))}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
