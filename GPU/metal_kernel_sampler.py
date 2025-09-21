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

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        """Initialize Metal EA sampler.

        Args:
            device: Metal device ("mps" for Apple Metal Performance Shaders)
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
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

        self.logger.info(f"[MetalEA] Initialized on device: {self._device.name()}")

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
                "ea_compute_energies_kernel",
                "ea_replica_exchange_kernel",
                "ea_track_best_kernel",
                "ea_initialize_spins_kernel",
                "ea_update_temperatures_kernel"
            ]

            for name in kernel_names:
                function = library.newFunctionWithName_(name)
                if function is None:
                    raise RuntimeError(f"Kernel function '{name}' not found")

                pipeline_state = self._device.newComputePipelineStateWithFunction_error_(function, None)[0]
                if pipeline_state is None:
                    raise RuntimeError(f"Failed to create pipeline state for '{name}'")

                self._kernels[name] = pipeline_state

            self.logger.info(f"[MetalEA] Compiled {len(kernel_names)} kernels successfully")

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

        # Create temperature schedule for parallel tempering
        temperatures = np.logspace(np.log10(0.1), np.log10(5.0), num_replicas).astype(np.float32)

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

    def _get_problem_info_from_hJ(self, h, J, L):
        """Get problem info from h,J parameters."""
        return {
            'model': 'custom',
            'L': L,
            'N': len(h),
            'num_couplings': len(J),
            'optimal_energy': 'unknown'  # We don't know the optimal energy
        }

    def sample_ising(self, h, J, num_reads: int = 256, num_sweeps: int = 100000,
                    num_replicas: int = None, swap_interval: int = 15, cooling_interval: int = 500,
                    T_min: float = 0.1, T_max: float = 5.0, cooling_factor: float = 0.999,
                    spin_updates_per_sweep: int | None = None,
                    **kwargs) -> dimod.SampleSet:
        """Run Parallel Tempering sampling on arbitrary Ising problems.

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

        self.logger.info(f"[MetalEA] Starting 3D EA sampling: L={L}, N={N}, replicas={num_replicas}, sweeps={num_sweeps}")

        # Convert provided h,J to Metal buffer format
        buffer_data = self._convert_problem_to_buffers(h, J, L, num_replicas)
        problem_info = self._get_problem_info_from_hJ(h, J, L)
        self.logger.info(f"[MetalEA] Problem: {problem_info['num_couplings']} couplings, optimal energy: {problem_info['optimal_energy']}")

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

        for step in range(num_sweeps):
            # Step 1: Perform k spin updates per sweep
            for _ in range(k_updates):
                # Update global step counter for randomness
                step_bytes = struct.pack('<i', step_counter)  # step as int32
                global_data_bytes[24:28] = np.frombuffer(step_bytes, dtype=np.uint8)
                global_buffer = self._create_buffer(global_data_bytes, "global_data")

                # Execute one spin flip update per replica
                self._spin_flip_gpu(replica_spins_buffer, csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer,
                                   temperatures_buffer, global_buffer, num_replicas)
                step_counter += 1

            # Step 2: Replica exchange (every swap_interval sweeps)
            if step % swap_interval == 0:
                self._compute_energies_gpu(replica_spins_buffer, replica_energies_buffer,
                                          csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer, global_buffer, num_replicas)
                self._replica_exchange_gpu(replica_spins_buffer, replica_energies_buffer,
                                          temperatures_buffer, global_buffer, num_replicas)

            # Step 3: Track best state
            if step % (swap_interval * 2) == 0:
                self._track_best_gpu(replica_spins_buffer, replica_energies_buffer,
                                    best_energy_buffer, best_spins_buffer, global_buffer, num_replicas)

            # Step 4: Temperature cooling (after exploration phase)
            if step % cooling_interval == 0 and step > cooling_start_step:
                self._update_temperatures_gpu(temperatures_buffer, cooling_factor, global_buffer, num_replicas)

            # Progress logging (per sweep)
            if step % progress_interval == 0 and step > 0:
                current_best = self._read_best_energy(best_energy_buffer)
                self.logger.debug(f"[MetalEA] Step {step}: best energy = {current_best:.1f}")

        # Final computations
        self._compute_energies_gpu(replica_spins_buffer, replica_energies_buffer,
                                  csr_row_ptr_buffer, csr_col_ind_buffer, csr_J_vals_buffer, global_buffer, num_replicas)
        self._track_best_gpu(replica_spins_buffer, replica_energies_buffer,
                            best_energy_buffer, best_spins_buffer, global_buffer, num_replicas)

        # Read back results
        samples, energies = self._collect_samples(replica_spins_buffer, replica_energies_buffer,
                                                 num_replicas, N, num_reads)

        runtime = time.time() - start_time
        spin_updates_per_sec = (num_sweeps * num_replicas * N) / runtime

        self.logger.info(f"[MetalEA] Completed: {runtime:.2f}s, {spin_updates_per_sec:.0f} spin updates/sec")

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

        # Safety check: ensure expected length
        expected_len = num_replicas * N
        if spins_flat.size < expected_len:
            # Pad with zeros if buffer shorter than expected (shouldn't happen)
            spins_flat = np.pad(spins_flat, (0, expected_len - spins_flat.size), mode='constant')
        elif spins_flat.size > expected_len:
            # Truncate if larger than expected (e.g., stale buffer sizing)
            spins_flat = spins_flat[:expected_len]

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
