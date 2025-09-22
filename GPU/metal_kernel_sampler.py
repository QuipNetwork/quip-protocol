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
import ctypes

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

import dimod
from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY


class MetalKernelDimodSampler:
    """Apple Metal GPU Parallel Tempering sampler for 3D Edwards-Anderson spin glasses."""

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None, verbose: bool = False):
        """Initialize Metal EA sampler.

        Args:
            device: Metal device ("mps" for Apple Metal Performance Shaders)
            logger: Optional logger instance
            verbose: If True, enable DEBUG logging for this sampler
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
                "ea_spin_flip_kernel",
                "ea_parallel_spin_flip_kernel",
                "ea_compute_energies_kernel",
                "ea_replica_exchange_kernel",
                "ea_track_best_kernel",
                "ea_initialize_spins_kernel",
                "ea_update_temperatures_kernel",
                "worker_parallel_tempering",
                "manager_energy_calculator"
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
        temperatures = self._create_adaptive_temperature_ladder(0.1, 5.0, num_replicas).astype(np.float32)

        # Create global data buffer (must match Metal GlobalData struct layout)
        # struct GlobalData { int N; int num_replicas; int swap_interval; int cooling_interval; float T_min, T_max; int step; };
        int_values = np.array([
            N,                   # N (actual number of spins)
            num_replicas,        # num_replicas
            15,                  # swap_interval
            500,                 # cooling_interval
        ], dtype=np.int32)

        float_values = np.array([
            0.1,                 # T_min
            5.0,                 # T_max
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
        Create adaptive temperature ladder optimized for ~30% swap acceptance.
        Uses geometric progression with adjustment for typical Ising energy scales.
        """
        if num_replicas == 1:
            return np.array([T_min])

        # For Ising models, optimal temperature spacing often follows:
        # T_{i+1} / T_i ≈ constant, with ratio chosen for ~30% acceptance
        # Empirical rule: ratio ≈ 1.2-1.5 for spin glasses

        # Start with geometric progression
        ratio = (T_max / T_min) ** (1.0 / (num_replicas - 1))

        # Adjust ratio based on number of replicas for better acceptance
        if num_replicas <= 4:
            ratio *= 1.1  # Closer spacing for few replicas
        elif num_replicas >= 16:
            ratio *= 0.95  # Tighter spacing for many replicas

        temperatures = np.array([T_min * (ratio ** i) for i in range(num_replicas)])

        # Ensure we hit T_max exactly
        temperatures[-1] = T_max

        self.logger.debug(f"[MetalEA] Adaptive temperature ladder: {temperatures}")
        self.logger.debug(f"[MetalEA] Temperature ratios: {temperatures[1:] / temperatures[:-1]}")

        return temperatures

    def sample_ising(self, h, J, num_reads: int = 256, num_sweeps: int = 1000,
                            num_replicas: int = None, swap_interval: int = 15,
                            T_min: float = 0.1, T_max: float = 5.0,
                            **kwargs) -> dimod.SampleSet:
        """GPU-only unified parallel tempering sampler - single kernel dispatch (DEFAULT)."""

        start_time = time.time()
        N = len(h)
        L = int(round(N ** (1/3)))  # Approximate cube root for compatibility

        if num_replicas is None:
            num_replicas = min(max(16, num_reads // 4), 64)  # Smaller default for unified kernel

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

        # Create temperature ladder
        temperatures = self._create_adaptive_temperature_ladder(T_min, T_max, num_replicas).astype(np.float32)

        # Create sampling parameters struct with proper data types
        sample_interval = max(1, num_sweeps // (num_reads // num_replicas + 1))

        try:
            # Create input buffers (reused across passes)
            csr_row_ptr_buffer = self._create_buffer(buffer_data['csr_row_ptr'], "csr_row_ptr")
            csr_col_ind_buffer = self._create_buffer(buffer_data['csr_col_ind'], "csr_col_ind")
            csr_J_vals_buffer = self._create_buffer(buffer_data['csr_J_vals'], "csr_J_vals")
            # Create reusable params buffer (written in-place per pass)
            params_size = struct.calcsize('<6i2fI')
            params_buffer = self._device.newBufferWithLength_options_(params_size, 0)
            if params_buffer is None:
                raise RuntimeError("Failed to create Metal params buffer")

            temperatures_buffer = self._create_buffer(temperatures, "temperatures")

            # Allocate reusable buffers sized to num_replicas for multipass reuse
            BUFFER_PADDING = 8  # Match kernel padding
            padded_size = N + BUFFER_PADDING

            # Communication buffers (reused across passes)
            thread_buffers = np.zeros(num_replicas * padded_size, dtype=np.int8)
            thread_energies = np.zeros(num_replicas, dtype=np.int32)
            rng_states = np.zeros(num_replicas, dtype=np.uint32)

            thread_buffers_buffer = self._create_buffer(thread_buffers, "thread_buffers")
            thread_energies_buffer = self._create_buffer(thread_energies, "thread_energies")
            rng_states_buffer = self._create_buffer(rng_states, "rng_states")

            # Output buffers (sized to num_replicas; we'll slice to batch_reads per pass)
            final_samples = np.zeros(num_replicas * N, dtype=np.int8)
            final_energies = np.zeros(num_replicas, dtype=np.int32)
            best_energy = np.array([2147483647], dtype=np.int32)  # int32 max
            best_spins = np.zeros(N, dtype=np.int8)

            final_samples_buffer = self._create_buffer(final_samples, "final_samples")
            final_energies_buffer = self._create_buffer(final_energies, "final_energies")
            best_energy_buffer = self._create_buffer(best_energy, "best_energy")
            best_spins_buffer = self._create_buffer(best_spins, "best_spins")

            all_samples = []
            all_energies = []
            produced = 0
            pass_idx = 0

            while produced < num_reads:
                batch_reads = min(num_replicas, num_reads - produced)

                # Pack struct for this pass (num_reads=batch_reads, vary base_seed)
                base_seed = int(time.time() * 1000) ^ (pass_idx * 0x9E3779B1)
                sampling_params_bytes = struct.pack('<6i2fI',
                    N,                    # int N
                    num_replicas,         # int num_replicas
                    num_sweeps,           # int num_sweeps
                    batch_reads,          # int num_reads (for this pass)
                    swap_interval,        # int swap_interval
                    sample_interval,      # int sample_interval
                    T_min,                # float T_min
                    T_max,                # float T_max
                    base_seed & 0xFFFFFFFF  # uint base_seed
                )
                # Create a tiny staging buffer with current params; copy into persistent params_buffer via blit
                staging_params = np.frombuffer(sampling_params_bytes, dtype=np.uint8)
                staging_params_buffer = self._create_buffer(staging_params, "params_staging")


                # Separated kernel dispatch - workers first, then manager
                self.logger.debug(f"[MetalEA] Dispatching worker+manager (single command buffer) pass {pass_idx+1}...")

                # Single command buffer and encoder per pass
                command_buffer = self._command_queue.commandBuffer()

                # STEP 1: Dispatch worker kernel

                # Set worker kernel buffers - matches worker_parallel_tempering signature
                # Update persistent params_buffer using a blit from staging_params_buffer
                blit = command_buffer.blitCommandEncoder()
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                    staging_params_buffer, 0, params_buffer, 0, params_size
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
                encoder.setBuffer_offset_atIndex_(best_energy_buffer, 0, 7)
                encoder.setBuffer_offset_atIndex_(best_spins_buffer, 0, 8)
                # Worker communication buffers
                encoder.setBuffer_offset_atIndex_(thread_buffers_buffer, 0, 9)
                encoder.setBuffer_offset_atIndex_(thread_energies_buffer, 0, 10)
                encoder.setBuffer_offset_atIndex_(rng_states_buffer, 0, 11)

                # Worker dispatch: Each worker thread gets its own threadgroup for isolation
                if num_replicas == 1:
                    threads_per_group = Metal.MTLSize(width=1, height=1, depth=1)
                    num_groups = Metal.MTLSize(width=1, height=1, depth=1)
                else:
                    threads_per_group = Metal.MTLSize(width=1, height=1, depth=1)  # 1 thread per group
                    num_groups = Metal.MTLSize(width=num_replicas, height=1, depth=1)  # num_replicas groups

                self.logger.debug(f"[MetalEA] Worker dispatch: {num_groups.width} groups × {threads_per_group.width} threads = {num_replicas} worker threads")
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

                # STEP 2: Dispatch manager kernel in the same encoder
                encoder.setComputePipelineState_(self._kernels["manager_energy_calculator"])

                # Set manager kernel buffers - matches manager_energy_calculator signature
                encoder.setBuffer_offset_atIndex_(csr_row_ptr_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(csr_col_ind_buffer, 0, 1)
                encoder.setBuffer_offset_atIndex_(csr_J_vals_buffer, 0, 2)
                encoder.setBuffer_offset_atIndex_(params_buffer, 0, 3)
                encoder.setBuffer_offset_atIndex_(thread_buffers_buffer, 0, 4)
                encoder.setBuffer_offset_atIndex_(final_samples_buffer, 0, 5)
                encoder.setBuffer_offset_atIndex_(final_energies_buffer, 0, 6)

                # Manager dispatch: Single thread to calculate all energies
                threads_per_group = Metal.MTLSize(width=1, height=1, depth=1)
                num_groups = Metal.MTLSize(width=1, height=1, depth=1)
                self.logger.debug(f"[MetalEA] Manager dispatch: 1 group × 1 thread = 1 manager thread")
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

    def _fix_energy_calculation_bug(self, samples, gpu_energies, J):
        """Fix systematic energy calculation bug in Metal kernel multi-threading.

        The Metal kernel has a threading bug where samples 0 and 3 exchange energies
        when multiple replicas are used. This method detects and corrects this pattern.
        """
        corrected_energies = []

        corrections_made = 0
        for i, sample in enumerate(samples):
            # Calculate correct energy for this sample
            correct_energy = 0.0
            for (node_i, node_j), coupling in J.items():
                correct_energy += coupling * sample[node_i] * sample[node_j]

            correct_energy_int = int(correct_energy)
            gpu_energy = gpu_energies[i] if i < len(gpu_energies) else 0


            if abs(gpu_energy - correct_energy_int) > 1e-6:
                corrections_made += 1

            corrected_energies.append(correct_energy_int)

        self.logger.debug(f"[MetalEA] Energy correction: Fixed {corrections_made}/{len(samples)} energies")
        return corrected_energies

    def _recalculate_energies_with_separate_kernel(self, samples, J, N, num_samples):
        """Recalculate energies using the separate energy calculation kernel to avoid threading bugs."""
        # Convert samples back to Metal buffer format
        samples_flat = []
        for sample in samples:
            samples_flat.extend(sample)

        samples_array = np.array(samples_flat, dtype=np.int8)
        samples_buffer = self._create_buffer(samples_array, "recalc_samples")

        # Create output buffer for energies
        energies_array = np.zeros(num_samples, dtype=np.int32)
        energies_buffer = self._create_buffer(energies_array, "recalc_energies")

        # Get CSR data
        L = int(round(N ** (1/3)))
        buffer_data = self._convert_problem_to_buffers({i: 0.0 for i in range(N)}, J, L, num_samples)

        csr_row_ptr_buffer = self._create_buffer(buffer_data['csr_row_ptr'], "recalc_csr_row_ptr")
        csr_col_ind_buffer = self._create_buffer(buffer_data['csr_col_ind'], "recalc_csr_col_ind")
        csr_J_vals_buffer = self._create_buffer(buffer_data['csr_J_vals'], "recalc_csr_J_vals")

        # Create global data for the kernel
        global_data_bytes = buffer_data['global_data'].copy()
        global_buffer = self._create_buffer(global_data_bytes, "recalc_global")

        # Use the separate energy calculation kernel
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_compute_energies_kernel"])

        encoder.setBuffer_offset_atIndex_(samples_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(energies_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 5)

        threads_per_group = Metal.MTLSize(width=min(num_samples, 256), height=1, depth=1)
        num_groups = Metal.MTLSize(width=max(1, (num_samples + 255) // 256), height=1, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Read back the corrected energies
        energies_contents = energies_buffer.contents()
        energies_buffer_length = energies_buffer.length()
        energies_buffer_view = energies_contents.as_buffer(energies_buffer_length)

        corrected_energies = []
        for i in range(num_samples):
            offset = i * 4  # 4 bytes per int32
            energy_bytes = bytes([energies_buffer_view[offset + j] for j in range(4)])
            energy = struct.unpack('<i', energy_bytes)[0]
            corrected_energies.append(energy)

        return corrected_energies

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


    def _get_problem_info_from_hJ(self, h, J, L):
        """Get problem info from h,J parameters."""
        return {
            'model': 'custom',
            'L': L,
            'N': len(h),
            'num_couplings': len(J),
            'optimal_energy': 'unknown'  # We don't know the optimal energy
        }

    def sample_ising_slow_correct(self, h, J, num_reads: int = 256, num_sweeps: int = 100000,
                    num_replicas: int = None, swap_interval: int = 15, cooling_interval: int = 500,
                    T_min: float = 0.1, T_max: float = 5.0, cooling_factor: float = 0.999,
                    spin_updates_per_sweep: int | None = None, parallel_spin_updates: bool = True,
                    **kwargs) -> dimod.SampleSet:
        """Original multi-kernel parallel tempering sampler - FOR DEBUGGING ONLY.

        Converts the provided h,J problem to Metal buffer format for GPU acceleration.

        Args:
            h: Linear biases
            J: Quadratic couplings
            num_reads: Number of solution samples to return
            num_sweeps: Number of Monte Carlo sweeps
            num_replicas: Number of temperature replicas (auto if None)
            swap_interval: Steps between replica exchanges
            cooling_interval: Steps between temperature cooling
            T_min: Minimum temperature
            T_max: Maximum temperature
            cooling_factor: Temperature cooling factor

        Returns:
            dimod.SampleSet with solution samples and energies
        """
        # Validate input parameters
        if not h or not J:
            raise ValueError("Both h and J parameters are required")

        # Determine problem size from h parameter count
        N = len(h)
        L = round(N ** (1/3))  # Approximate cube root for Metal buffer sizing

        start_time = time.time()

        if num_replicas is None:
            num_replicas = min(max(32, num_reads // 2), 512)

        self.logger.debug(f"[MetalEA] Starting 3D EA sampling: L={L}, N={N}, replicas={num_replicas}, sweeps={num_sweeps}")

        # Convert provided h,J to Metal buffer format
        buffer_data = self._convert_problem_to_buffers(h, J, L, num_replicas)
        problem_info = self._get_problem_info_from_hJ(h, J, L)
        self.logger.debug(f"[MetalEA] Problem: {problem_info['num_couplings']} couplings, optimal energy: {problem_info['optimal_energy']}")

        # Create Metal buffers
        try:
            # Spin states: [num_replicas] SpinState structs
            # Each SpinState has: int8_t spins[MAX_N] + int N
            # Create flattened buffers for dynamic sizing (no more fixed MAX_N)

            # Replica spins: [num_replicas * N] flattened array
            replica_spins_data = np.random.choice([-1, 1], size=(num_replicas * N)).astype(np.int8)
            replica_spins_buffer = self._create_buffer(replica_spins_data, "replica_spins")

            # CSR adjacency buffers
            csr_row_ptr_buffer = self._create_buffer(buffer_data['csr_row_ptr'], "csr_row_ptr")
            csr_col_ind_buffer = self._create_buffer(buffer_data['csr_col_ind'], "csr_col_ind")
            csr_J_vals_buffer = self._create_buffer(buffer_data['csr_J_vals'], "csr_J_vals")
            temperatures_buffer = self._create_buffer(buffer_data['temperatures'], "temperatures")

            # Global data - handle new byte array format
            global_data_bytes = buffer_data['global_data'].copy()
            # Reset step counter at byte offset 24 (4 ints + 2 floats = 16+8 = 24)
            step_bytes = struct.pack('<i', 0)  # step = 0 as int32
            global_data_bytes[24:28] = np.frombuffer(step_bytes, dtype=np.uint8)
            global_buffer = self._create_buffer(global_data_bytes, "global_data")

            # Energy and best state tracking
            replica_energies = np.zeros(num_replicas, dtype=np.int32)
            replica_energies_buffer = self._create_buffer(replica_energies, "replica_energies")

            best_energy = np.array([2147483647], dtype=np.int32)  # int32 max as "infinity"
            best_energy_buffer = self._create_buffer(best_energy, "best_energy")

            best_spins = np.zeros(N, dtype=np.int8)
            best_spins_buffer = self._create_buffer(best_spins, "best_spins")

            # Create second buffer for double-buffering (parallel spin updates)
            replica_spins_buffer_alt = None
            if parallel_spin_updates:
                replica_spins_buffer_alt = self._create_buffer(replica_spins_data.copy(), "replica_spins_alt")

        except Exception as e:
            self.logger.error(f"[MetalEA] Buffer creation failed: {e}")
            raise

        # Note: Buffer validation removed for production

        # Initialize random spins using GPU kernel
        self._initialize_spins_gpu(replica_spins_buffer, global_buffer, 12345, num_replicas, N)

        # Initial energy computation
        self._compute_energies_gpu(replica_spins_buffer, replica_energies_buffer,
                                  csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer, global_buffer, num_replicas)

        # Main Parallel Tempering loop
        self.logger.debug(f"[MetalEA] Starting {num_sweeps} PT sweeps...")

        progress_interval = max(1000, num_sweeps // 100)
        cooling_start_step = 10000

        # Determine spin updates per sweep (default: N)
        k_updates = spin_updates_per_sweep if spin_updates_per_sweep is not None else N
        step_counter = 0  # used for GPU RNG seeding in kernels

        # Energy caching: track when energies are valid to avoid recomputation
        energies_valid = False

        # Double-buffering state for parallel spin updates
        current_buffer = replica_spins_buffer
        alt_buffer = replica_spins_buffer_alt

        for step in range(num_sweeps):
            # Step 1: Perform k spin updates per sweep
            if parallel_spin_updates and alt_buffer is not None:
                # Use parallel spin updates with double-buffering
                for _ in range(k_updates):
                    # Update global step counter for randomness
                    step_bytes = struct.pack('<i', step_counter)  # step as int32
                    global_data_bytes[24:28] = np.frombuffer(step_bytes, dtype=np.uint8)
                    global_buffer = self._create_buffer(global_data_bytes, "global_data")

                    # Execute parallel spin flip update (all spins simultaneously)
                    self._parallel_spin_flip_gpu(current_buffer, alt_buffer, csr_row_ptr_buffer, csr_col_ind_buffer,
                                               csr_J_vals_buffer, temperatures_buffer, global_buffer, num_replicas, N)

                    # Swap buffers for next iteration
                    current_buffer, alt_buffer = alt_buffer, current_buffer
                    step_counter += 1
            else:
                # Use sequential spin updates (original method)
                for _ in range(k_updates):
                    # Update global step counter for randomness
                    step_bytes = struct.pack('<i', step_counter)  # step as int32
                    global_data_bytes[24:28] = np.frombuffer(step_bytes, dtype=np.uint8)
                    global_buffer = self._create_buffer(global_data_bytes, "global_data")

                    # Execute one spin flip update per replica
                    self._spin_flip_gpu(current_buffer, csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer,
                                       temperatures_buffer, global_buffer, num_replicas)
                    step_counter += 1

            # Spin updates invalidate cached energies
            energies_valid = False

            # Step 2: Replica exchange (every swap_interval sweeps)
            if step % swap_interval == 0:
                # Only compute energies if they're not already valid
                if not energies_valid:
                    self._compute_energies_gpu(current_buffer, replica_energies_buffer,
                                              csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer, global_buffer, num_replicas)
                    energies_valid = True

                self._replica_exchange_gpu(current_buffer, replica_energies_buffer,
                                          temperatures_buffer, global_buffer, num_replicas)
                # Note: energies remain valid after replica exchange (just swapped between replicas)

            # Step 3: Track best state
            if step % (swap_interval * 2) == 0:
                # Ensure energies are computed before tracking best
                if not energies_valid:
                    self._compute_energies_gpu(current_buffer, replica_energies_buffer,
                                              csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer, global_buffer, num_replicas)
                    energies_valid = True

                self._track_best_gpu(current_buffer, replica_energies_buffer,
                                    best_energy_buffer, best_spins_buffer, global_buffer, num_replicas)

            # Step 4: Temperature cooling (after exploration phase)
            if step % cooling_interval == 0 and step > cooling_start_step:
                self._update_temperatures_gpu(temperatures_buffer, cooling_factor, global_buffer, num_replicas)

            # Progress logging (per sweep)
            if step % progress_interval == 0 and step > 0:
                current_best = self._read_best_energy(best_energy_buffer)
                self.logger.debug(f"[MetalEA] Step {step}: best energy = {current_best:.1f}")

        # Final computations
        if not energies_valid:
            self._compute_energies_gpu(current_buffer, replica_energies_buffer,
                                      csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer, global_buffer, num_replicas)
        self._track_best_gpu(current_buffer, replica_energies_buffer,
                            best_energy_buffer, best_spins_buffer, global_buffer, num_replicas)

        # Read back results
        samples, gpu_energies = self._collect_samples(current_buffer, replica_energies_buffer,
                                                     num_replicas, N, num_reads)

        # WORKAROUND: Fix systematic energy calculation bug in Metal kernel
        energies = self._fix_energy_calculation_bug(samples, gpu_energies, J)

        runtime = time.time() - start_time
        spin_updates_per_sec = (num_sweeps * num_replicas * N) / runtime

        self.logger.debug(f"[MetalEA] Completed: {runtime:.2f}s, {spin_updates_per_sec:.0f} spin updates/sec")

        # Create dimod SampleSet
        return dimod.SampleSet.from_samples(samples, 'SPIN', energies)

    def _initialize_spins_gpu(self, replica_spins_buffer, global_buffer, seed: int, num_replicas: int, N: int):
        """Initialize random spins using Metal kernel."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_initialize_spins_kernel"])

        # Set buffers
        encoder.setBuffer_offset_atIndex_(replica_spins_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 1)
        encoder.setBytes_length_atIndex_(np.array([seed], dtype=np.uint32).tobytes(), 4, 2)

        # Dispatch threads: 2D pattern (replica_id, spin_idx)
        # Use optimal thread group size
        threads_per_group = Metal.MTLSize(width=min(num_replicas, 16), height=min(N, 16), depth=1)
        num_groups = Metal.MTLSize(
            width=max(1, (num_replicas + 15) // 16),
            height=max(1, (N + 15) // 16),
            depth=1
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _spin_flip_gpu(self, replica_spins_buffer, csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer,
                      temperatures_buffer, global_buffer, num_replicas):
        """Execute spin flip kernel."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_spin_flip_kernel"])

        # Set buffers (match Metal kernel indices)
        encoder.setBuffer_offset_atIndex_(replica_spins_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(temperatures_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 5)

        # Dispatch one thread per replica - use actual num_replicas
        threads_per_group = Metal.MTLSize(width=min(num_replicas, 256), height=1, depth=1)
        num_groups = Metal.MTLSize(width=max(1, (num_replicas + 255) // 256), height=1, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _parallel_spin_flip_gpu(self, src_buffer, dst_buffer, csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer,
                               temperatures_buffer, global_buffer, num_replicas, N):
        """Execute parallel spin flip kernel on GPU with double-buffering."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_parallel_spin_flip_kernel"])

        # Set buffers (match Metal kernel indices)
        encoder.setBuffer_offset_atIndex_(src_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(dst_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(temperatures_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 6)

        # 2D dispatch: replicas × spins
        threads_per_group = Metal.MTLSize(width=min(num_replicas, 8), height=min(N, 8), depth=1)
        groups_x = max(1, (num_replicas + threads_per_group.width - 1) // threads_per_group.width)
        groups_y = max(1, (N + threads_per_group.height - 1) // threads_per_group.height)
        num_groups = Metal.MTLSize(width=groups_x, height=groups_y, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _compute_energies_gpu(self, replica_spins_buffer, replica_energies_buffer,
                             csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer, global_buffer, num_replicas):
        """Compute energies for all replicas."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_compute_energies_kernel"])

        # Set buffers (match Metal kernel indices)
        encoder.setBuffer_offset_atIndex_(replica_spins_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(replica_energies_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 5)

        # Dispatch one thread per replica
        threads_per_group = Metal.MTLSize(width=min(num_replicas, 256), height=1, depth=1)
        num_groups = Metal.MTLSize(width=max(1, (num_replicas + 255) // 256), height=1, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _replica_exchange_gpu(self, replica_spins_buffer, replica_energies_buffer,
                             temperatures_buffer, global_buffer, num_replicas):
        """Execute replica exchange kernel."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_replica_exchange_kernel"])

        # Set buffers
        encoder.setBuffer_offset_atIndex_(replica_spins_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(replica_energies_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(temperatures_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 3)

        # Dispatch one thread per replica (only even replicas process)
        threads_per_group = Metal.MTLSize(width=min(num_replicas, 256), height=1, depth=1)
        num_groups = Metal.MTLSize(width=max(1, (num_replicas + 255) // 256), height=1, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _track_best_gpu(self, replica_spins_buffer, replica_energies_buffer,
                       best_energy_buffer, best_spins_buffer, global_buffer, num_replicas):
        """Track best energy and configuration."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_track_best_kernel"])

        # Set buffers
        encoder.setBuffer_offset_atIndex_(replica_spins_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(replica_energies_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(best_energy_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(best_spins_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 4)

        # Dispatch one thread per replica
        threads_per_group = Metal.MTLSize(width=min(num_replicas, 256), height=1, depth=1)
        num_groups = Metal.MTLSize(width=max(1, (num_replicas + 255) // 256), height=1, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _update_temperatures_gpu(self, temperatures_buffer, cooling_factor: float, global_buffer, num_replicas):
        """Update temperature schedule with cooling."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._kernels["ea_update_temperatures_kernel"])

        # Set buffers
        encoder.setBuffer_offset_atIndex_(temperatures_buffer, 0, 0)
        encoder.setBytes_length_atIndex_(np.array([cooling_factor], dtype=np.float32).tobytes(), 4, 1)
        encoder.setBuffer_offset_atIndex_(global_buffer, 0, 2)

        # Dispatch one thread per replica
        threads_per_group = Metal.MTLSize(width=min(num_replicas, 256), height=1, depth=1)
        num_groups = Metal.MTLSize(width=max(1, (num_replicas + 255) // 256), height=1, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def _read_best_energy(self, best_energy_buffer) -> int:
        """Read current best energy from GPU buffer."""
        try:
            contents = best_energy_buffer.contents()
            buffer_length = best_energy_buffer.length()
            buffer_view = contents.as_buffer(buffer_length)
            energy_bytes = bytes([buffer_view[i] for i in range(4)])
            energy = struct.unpack('<i', energy_bytes)[0]
            return energy
        except:
            return 2147483647  # int32 max as "infinity"

    def _collect_samples(self, replica_spins_buffer, replica_energies_buffer,
                        num_replicas: int, N: int, num_reads: int) -> Tuple[list, list]:
        """Collect final samples from GPU replicas."""
        # Read replica energies using the fixed method
        energies_contents = replica_energies_buffer.contents()
        energies_buffer_length = replica_energies_buffer.length()
        energies_buffer_view = energies_contents.as_buffer(energies_buffer_length)
        energies = []

        for r in range(num_replicas):
            offset = r * 4  # 4 bytes per int32
            energy_bytes = bytes([energies_buffer_view[offset + i] for i in range(4)])
            energy = struct.unpack('<i', energy_bytes)[0]
            energies.append(energy)

        # Read replica spins from flattened array: [num_replicas * N] int8
        spins_contents = replica_spins_buffer.contents()
        spins_buffer_length = replica_spins_buffer.length()
        spins_buffer_view = spins_contents.as_buffer(spins_buffer_length)
        spins_bytes = bytes(spins_buffer_view)
        spins_flat = np.frombuffer(spins_bytes, dtype=np.int8)

        # NO FALLBACKS - crash on buffer size mismatch
        expected_len = num_replicas * N
        if spins_flat.size != expected_len:
            raise RuntimeError(f"REPLICA BUFFER FAILURE: Expected {expected_len} spin bytes, got {spins_flat.size}. "
                             f"This indicates replica buffer allocation mismatch.")

        samples = []
        for read_idx in range(num_reads):
            replica_idx = read_idx % num_replicas
            start = replica_idx * N
            end = start + N
            replica_spins = spins_flat[start:end]
            sample = [int(spin) for spin in replica_spins]
            samples.append(sample)

        # Map energies to reads in the same round-robin way
        sample_energies = [energies[read_idx % num_replicas] for read_idx in range(num_reads)]

        return samples, sample_energies

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
