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
                "preprocess_all",
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

            # Query pipeline limits for sampling kernel
            self._pt_max_threads = 256
            self._pt_thread_exec_width = 32
            try:
                pt_pipeline = self._kernels.get("parallel_tempering_sampling")
                if pt_pipeline is not None:
                    # PyObjC may expose properties as callables; handle both forms
                    mtt = getattr(pt_pipeline, "maxTotalThreadsPerThreadgroup", None)
                    if callable(mtt):
                        mtt = mtt()
                    if isinstance(mtt, (int, float)) and int(mtt) > 0:
                        self._pt_max_threads = int(mtt)
                    tew = getattr(pt_pipeline, "threadExecutionWidth", None)
                    if callable(tew):
                        tew = tew()
                    if isinstance(tew, (int, float)) and int(tew) > 0:
                        self._pt_thread_exec_width = int(tew)
            except Exception:
                # Keep conservative defaults
                pass

            self.logger.debug(
                f"[UnifiedMetal] Compiled {len(kernel_names)} kernels; PT limits: max_threads={self._pt_max_threads}, exec_width={self._pt_thread_exec_width}"
            )

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


    def sample_ising(self, h: Dict[int, float], J: Dict[tuple, float],
                    num_reads: int = 256, num_sweeps: int = 1000,
                    num_replicas: int = None, swap_interval: int = 15,
                    T_min: float = 0.01, T_max: float = 1.0,
                    sample_interval: int = None, cooling_factor: float = 0.999,
                    cooling_start_sweep: int = None,
                    threads_per_group_cap: Optional[int] = None,
                    **kwargs) -> dimod.SampleSet:
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


        self.logger.debug(
            f"[UnifiedMetal] GPU-only pipeline N={N}, reads={num_reads}, sweeps={num_sweeps}"
        )

        try:
            # === Prepare edge arrays (CPU -> GPU) ===
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

            # Allocate CSR outputs
            csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
            csr_col_ind = np.zeros(max(1, 2 * E), dtype=np.int32)
            csr_J_vals = np.zeros(max(1, 2 * E), dtype=np.int8)
            nnz_out = np.zeros(1, dtype=np.int32)
            row_ptr_working = np.zeros(N, dtype=np.int32)

            # Capacity for scheduling arrays (cap at 256)
            req_reps = 0 if (num_replicas is None) else int(num_replicas)
            cap = min(256, max(8, req_reps if req_reps > 0 else num_reads))
            temps = np.zeros(cap, dtype=np.float32)
            per_replica_base = np.zeros(cap, dtype=np.int32)
            per_replica_quota = np.zeros(cap, dtype=np.int32)
            rng_states = np.zeros(cap, dtype=np.uint32)
            thread_buffers = np.zeros(cap * N, dtype=np.int8)

            # Create buffers
            problem_buf = self._create_buffer(problem_data, "problem_data")
            J_edges_buf = self._create_buffer(J_edges if E > 0 else np.zeros(2, dtype=np.int32), "J_edges")
            J_values_buf = self._create_buffer(J_values if E > 0 else np.zeros(1, dtype=np.float32), "J_values")
            csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
            csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
            csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
            nnz_out_buf = self._create_buffer(nnz_out, "nnz_out")
            row_ptr_working_buf = self._create_buffer(row_ptr_working, "row_ptr_working")

            req_buf = self._create_buffer(np.array([req_reps], dtype=np.int32), "requested_reps")
            reads_buf = self._create_buffer(np.array([int(num_reads)], dtype=np.int32), "num_reads")
            T_min_buf = self._create_buffer(np.array([float(T_min)], dtype=np.float32), "T_min")
            T_max_buf = self._create_buffer(np.array([float(T_max)], dtype=np.float32), "T_max")
            base_seed_val = np.array([int(time.time() * 1000) & 0xFFFFFFFF], dtype=np.uint32)
            base_seed_buf = self._create_buffer(base_seed_val, "base_seed")

            num_reps_buf = self._create_buffer(np.array([0], dtype=np.int32), "num_replicas")
            temps_buf = self._create_buffer(temps, "temperatures")
            per_replica_base_buf = self._create_buffer(per_replica_base, "per_replica_base")
            per_replica_quota_buf = self._create_buffer(per_replica_quota, "per_replica_quota")
            rng_states_buf = self._create_buffer(rng_states, "rng_states")
            thread_buffers_buf = self._create_buffer(thread_buffers, "thread_buffers")

            # === Dispatch unified preprocess (single thread) ===
            cb = self._command_queue.commandBuffer()
            enc = cb.computeCommandEncoder()
            enc.setComputePipelineState_(self._kernels["preprocess_all"])
            enc.setBuffer_offset_atIndex_(problem_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(J_edges_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(J_values_buf, 0, 2)
            enc.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 3)
            enc.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 4)
            enc.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 5)
            enc.setBuffer_offset_atIndex_(nnz_out_buf, 0, 6)
            enc.setBuffer_offset_atIndex_(row_ptr_working_buf, 0, 7)
            enc.setBuffer_offset_atIndex_(req_buf, 0, 8)
            enc.setBuffer_offset_atIndex_(reads_buf, 0, 9)
            enc.setBuffer_offset_atIndex_(T_min_buf, 0, 10)
            enc.setBuffer_offset_atIndex_(T_max_buf, 0, 11)
            enc.setBuffer_offset_atIndex_(base_seed_buf, 0, 12)
            enc.setBuffer_offset_atIndex_(num_reps_buf, 0, 13)
            enc.setBuffer_offset_atIndex_(temps_buf, 0, 14)
            enc.setBuffer_offset_atIndex_(per_replica_base_buf, 0, 15)
            enc.setBuffer_offset_atIndex_(per_replica_quota_buf, 0, 16)
            enc.setBuffer_offset_atIndex_(rng_states_buf, 0, 17)
            enc.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 18)
            # New: pass num_sweeps and sample_interval buffers
            num_sweeps_buf = self._create_buffer(np.array([int(num_sweeps)], dtype=np.int32), "num_sweeps")
            # If user provided sample_interval use it; else 0 triggers GPU compute
            samp_init = int(sample_interval) if (sample_interval is not None) else 0
            sample_interval_buf = self._create_buffer(np.array([samp_init], dtype=np.int32), "sample_interval")
            enc.setBuffer_offset_atIndex_(num_sweeps_buf, 0, 19)
            enc.setBuffer_offset_atIndex_(sample_interval_buf, 0, 20)

            enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=1, height=1, depth=1), Metal.MTLSize(width=1, height=1, depth=1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

            # Read back num_reps
            opt_contents = num_reps_buf.contents(); opt_view = opt_contents.as_buffer(num_reps_buf.length())
            num_reps = struct.unpack('<i', bytes([opt_view[i] for i in range(4)]))[0]
            num_reps = max(1, int(num_reps))

            # === Sampling dispatch (replica-parallel) ===
            # Use GPU-computed sample_interval unless user provided a positive value
            sweeps_buf = self._create_buffer(np.array([int(num_sweeps)], dtype=np.int32), "num_sweeps")
            samp_int_buf = sample_interval_buf

            final_samples = np.zeros(num_reads * N, dtype=np.int8)
            final_energies = np.zeros(num_reads, dtype=np.int32)
            final_samples_buf = self._create_buffer(final_samples, "final_samples")
            final_energies_buf = self._create_buffer(final_energies, "final_energies")

            cb5 = self._command_queue.commandBuffer()
            enc5 = cb5.computeCommandEncoder()
            enc5.setComputePipelineState_(self._kernels["parallel_tempering_sampling"])
            enc5.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
            enc5.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
            enc5.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
            enc5.setBuffer_offset_atIndex_(self._create_buffer(np.array([N], dtype=np.int32), "N"), 0, 3)
            enc5.setBuffer_offset_atIndex_(temps_buf, 0, 4)
            enc5.setBuffer_offset_atIndex_(self._create_buffer(np.array([num_reps], dtype=np.int32), "num_reps"), 0, 5)
            enc5.setBuffer_offset_atIndex_(sweeps_buf, 0, 6)
            enc5.setBuffer_offset_atIndex_(samp_int_buf, 0, 7)
            enc5.setBuffer_offset_atIndex_(base_seed_buf, 0, 8)  # unused but kept for signature stability
            enc5.setBuffer_offset_atIndex_(per_replica_base_buf, 0, 9)
            enc5.setBuffer_offset_atIndex_(per_replica_quota_buf, 0, 10)
            enc5.setBuffer_offset_atIndex_(final_samples_buf, 0, 11)
            enc5.setBuffer_offset_atIndex_(final_energies_buf, 0, 12)
            enc5.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 13)
            enc5.setBuffer_offset_atIndex_(rng_states_buf, 0, 14)

            # Determine effective per-group thread cap from pipeline, with optional user cap
            cap = getattr(self, "_pt_max_threads", 256)
            if threads_per_group_cap is not None:
                try:
                    cap = max(1, min(int(threads_per_group_cap), cap))
                except Exception:
                    pass
            if num_reps <= cap:
                threads_per_group = Metal.MTLSize(width=num_reps, height=1, depth=1)
                num_groups = Metal.MTLSize(width=1, height=1, depth=1)
            else:
                threads_per_group = Metal.MTLSize(width=cap, height=1, depth=1)
                groups = (num_reps + cap - 1) // cap
                num_groups = Metal.MTLSize(width=groups, height=1, depth=1)

            self.logger.debug(f"[UnifiedMetal] Dispatch PT: {num_groups.width} groups × {threads_per_group.width} threads")
            enc5.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)
            enc5.endEncoding(); cb5.commit(); cb5.waitUntilCompleted()

            # Collect results
            samples, energies = self._collect_results(final_samples_buf, final_energies_buf, num_reads, N)
            runtime = time.time() - start_time
            self.logger.debug(f"[UnifiedMetal] Full GPU sampling completed (2-dispatch): {runtime:.2f}s")
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