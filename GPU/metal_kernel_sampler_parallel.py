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
                "worker_parallel_updates",
                "metropolis_2d_synchronous",
                "reduce_energies_2d",
                "pt_swap_adjacent",
                "compute_swap_stats",
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
                pt_pipeline = self._kernels.get("metropolis_2d_synchronous")
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
                    num_replicas: Optional[int] = None,
                    T_min: float = 0.01, T_max: float = 1.0,
                    sample_interval: Optional[int] = None) -> dimod.SampleSet:
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
            # Allocate alternate buffer for 2D double-buffering skeleton (step 1)
            thread_buffers_alt = np.zeros(cap * N, dtype=np.int8)
            thread_buffers_alt_buf = self._create_buffer(thread_buffers_alt, "thread_buffers_alt")

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

            # 2D kernel skeleton to initialize double buffers (step 1)
            cb2d = self._command_queue.commandBuffer()
            enc2d = cb2d.computeCommandEncoder()
            enc2d.setComputePipelineState_(self._kernels["worker_parallel_updates"])
            enc2d.setBuffer_offset_atIndex_(self._create_buffer(np.array([N], dtype=np.int32), "N"), 0, 0)
            enc2d.setBuffer_offset_atIndex_(self._create_buffer(np.array([num_reps], dtype=np.int32), "num_reps"), 0, 1)
            enc2d.setBuffer_offset_atIndex_(base_seed_buf, 0, 2)
            enc2d.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 3)
            enc2d.setBuffer_offset_atIndex_(thread_buffers_alt_buf, 0, 4)
            # 2D dispatch: X over spins, Y over replicas; one group per replica
            exec_w = getattr(self, "_pt_thread_exec_width", 32)
            tpg_x = max(1, int(exec_w))
            threads_per_group = Metal.MTLSize(width=tpg_x, height=1, depth=1)
            num_groups = Metal.MTLSize(width=1, height=num_reps, depth=1)
            enc2d.dispatchThreadgroups_threadsPerThreadgroup_(num_groups, threads_per_group)
            enc2d.endEncoding(); cb2d.commit(); cb2d.waitUntilCompleted()

            # Read back device-determined sample interval (if not provided)
            samp_contents = sample_interval_buf.contents(); samp_view = samp_contents.as_buffer(sample_interval_buf.length())
            sample_int_val = struct.unpack('<i', bytes([samp_view[i] for i in range(4)]))[0]
            interval = int(sample_interval) if (sample_interval is not None and int(sample_interval) > 0) else max(1, int(sample_int_val) if sample_int_val > 0 else 1)

            # Allocate persistent debug/stat buffers once
            flip_counts_buf = self._create_buffer(np.zeros(num_reps, dtype=np.int32), "flip_counts")
            pos_counts_buf  = self._create_buffer(np.zeros(num_reps, dtype=np.int32), "pos_counts")
            neg_counts_buf  = self._create_buffer(np.zeros(num_reps, dtype=np.int32), "neg_counts")
            dbg_energy_buf  = self._create_buffer(np.zeros(num_reps, dtype=np.int32), "dbg_energy")
            attempts        = self._create_buffer(np.zeros(max(0, num_reps-1), dtype=np.int32), "swap_attempts")
            accepts         = self._create_buffer(np.zeros(max(0, num_reps-1), dtype=np.int32), "swap_accepts")
            rates_buf       = self._create_buffer(np.zeros(max(0, num_reps-1), dtype=np.float32), "swap_rates")


            exec_w = getattr(self, "_pt_thread_exec_width", 32)
            tpg_x = max(1, int(exec_w))
            threads_per_group = Metal.MTLSize(width=tpg_x, height=1, depth=1)
            tg_reps = Metal.MTLSize(width=1, height=num_reps, depth=1)

            remaining = int(num_sweeps)
            while remaining > 0:
                sweeps_this = min(interval, remaining)
                self.logger.debug(f"[UnifiedMetal] 2D update sweeps chunk={sweeps_this} (remaining {remaining - sweeps_this})")
                sweeps2d_buf = self._create_buffer(np.array([sweeps_this], dtype=np.int32), "num_sweeps_2d")

                # 2D update chunk
                cb2u = self._command_queue.commandBuffer()
                enc2u = cb2u.computeCommandEncoder()
                enc2u.setComputePipelineState_(self._kernels["metropolis_2d_synchronous"])
                enc2u.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
                enc2u.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
                enc2u.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
                enc2u.setBuffer_offset_atIndex_(self._create_buffer(np.array([N], dtype=np.int32), "N"), 0, 3)
                enc2u.setBuffer_offset_atIndex_(temps_buf, 0, 4)
                enc2u.setBuffer_offset_atIndex_(num_reps_buf, 0, 5)
                enc2u.setBuffer_offset_atIndex_(sweeps2d_buf, 0, 6)
                enc2u.setBuffer_offset_atIndex_(base_seed_buf, 0, 7)
                enc2u.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 8)
                enc2u.setBuffer_offset_atIndex_(thread_buffers_alt_buf, 0, 9)
                enc2u.setBuffer_offset_atIndex_(flip_counts_buf, 0, 10)
                enc2u.setBuffer_offset_atIndex_(pos_counts_buf, 0, 11)
                enc2u.setBuffer_offset_atIndex_(neg_counts_buf, 0, 12)
                enc2u.setBuffer_offset_atIndex_(dbg_energy_buf, 0, 13)
                enc2u.dispatchThreadgroups_threadsPerThreadgroup_(tg_reps, threads_per_group)
                enc2u.endEncoding(); cb2u.commit(); cb2u.waitUntilCompleted()

                # Reduce energies after this chunk
                cbRE = self._command_queue.commandBuffer()
                encRE = cbRE.computeCommandEncoder()
                encRE.setComputePipelineState_(self._kernels["reduce_energies_2d"])
                encRE.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
                encRE.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
                encRE.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
                encRE.setBuffer_offset_atIndex_(self._create_buffer(np.array([N], dtype=np.int32), "N"), 0, 3)
                encRE.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 4)
                encRE.setBuffer_offset_atIndex_(dbg_energy_buf, 0, 5)
                encRE.setBuffer_offset_atIndex_(num_reps_buf, 0, 6)
                encRE.dispatchThreadgroups_threadsPerThreadgroup_(tg_reps, threads_per_group)
                encRE.endEncoding(); cbRE.commit(); cbRE.waitUntilCompleted()

                # Swaps: even then odd
                def dispatch_swaps(parity:int, num_pairs:int):
                    if num_pairs <= 0:
                        return
                    cbSW = self._command_queue.commandBuffer()
                    encSW = cbSW.computeCommandEncoder()
                    encSW.setComputePipelineState_(self._kernels["pt_swap_adjacent"])
                    encSW.setBuffer_offset_atIndex_(self._create_buffer(np.array([N], dtype=np.int32), "N"), 0, 0)
                    encSW.setBuffer_offset_atIndex_(temps_buf, 0, 1)
                    encSW.setBuffer_offset_atIndex_(num_reps_buf, 0, 2)
                    encSW.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 3)
                    encSW.setBuffer_offset_atIndex_(dbg_energy_buf, 0, 4)
                    encSW.setBuffer_offset_atIndex_(base_seed_buf, 0, 5)
                    encSW.setBuffer_offset_atIndex_(self._create_buffer(np.array([parity], dtype=np.int32), "swap_parity"), 0, 6)
                    encSW.setBuffer_offset_atIndex_(attempts, 0, 7)
                    encSW.setBuffer_offset_atIndex_(accepts, 0, 8)
                    encSW.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=tpg_x, height=1, depth=1), Metal.MTLSize(width=1, height=num_pairs, depth=1))
                    encSW.endEncoding(); cbSW.commit(); cbSW.waitUntilCompleted()

                dispatch_swaps(0, num_reps // 2)
                dispatch_swaps(1, (num_reps - 1) // 2)

                remaining -= sweeps_this

            # Read back stats and energies
            def read_int32_array(buf, n):
                view = buf.contents().as_buffer(buf.length())
                out = []
                for i in range(n):
                    base = i * 4
                    out.append(int.from_bytes(bytes([view[base+j] for j in range(4)]), 'little', signed=True))
                return out
            flips = read_int32_array(flip_counts_buf, num_reps)
            posd  = read_int32_array(pos_counts_buf, num_reps)
            negd  = read_int32_array(neg_counts_buf, num_reps)
            energies2d = read_int32_array(dbg_energy_buf, num_reps)
            self.logger.debug(f"[UnifiedMetal] flips: {flips}")
            self.logger.debug(f"[UnifiedMetal] dE>0: {posd}")
            self.logger.debug(f"[UnifiedMetal] dE<0: {negd}")
            if num_reps > 1:
                att = read_int32_array(attempts, max(0, num_reps-1))
                acc = read_int32_array(accepts, max(0, num_reps-1))

                # Compute acceptance rates per pair on device and read back
                if num_reps > 1:
                    cbST = self._command_queue.commandBuffer()
                    encST = cbST.computeCommandEncoder()
                    encST.setComputePipelineState_(self._kernels["compute_swap_stats"])
                    encST.setBuffer_offset_atIndex_(num_reps_buf, 0, 0)
                    encST.setBuffer_offset_atIndex_(attempts, 0, 1)
                    encST.setBuffer_offset_atIndex_(accepts, 0, 2)
                    encST.setBuffer_offset_atIndex_(rates_buf, 0, 3)
                    encST.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=1, height=max(1, (num_reps - 1)), depth=1), Metal.MTLSize(width=1, height=1, depth=1))
                    encST.endEncoding(); cbST.commit(); cbST.waitUntilCompleted()

                    # Read back float rates
                    viewf = rates_buf.contents().as_buffer(rates_buf.length())
                    rates = []
                    for i in range(max(0, num_reps-1)):
                        base = i * 4
                        rates.append(struct.unpack('<f', bytes([viewf[base+j] for j in range(4)]))[0])
                    self.logger.debug(f"[UnifiedMetal] PT accept rates: {['{:.2f}'.format(r) for r in rates]}")

                self.logger.debug(f"[UnifiedMetal] PT attempts: {att}")
                self.logger.debug(f"[UnifiedMetal] PT accepts:  {acc}")

            # Build final samples/energies on host from current 2D replica states
            spins_view = thread_buffers_buf.contents().as_buffer(thread_buffers_buf.length())
            spins_bytes = bytes(spins_view)
            spins_flat = np.frombuffer(spins_bytes, dtype=np.int8)
            spins_flat = spins_flat[:num_reps * N]

            # Select replicas by ascending energy
            idx_order = list(sorted(range(num_reps), key=lambda i: energies2d[i]))
            if num_reads <= num_reps:
                sel_indices = idx_order[:num_reads]
            else:
                reps = (num_reads + num_reps - 1) // num_reps
                sel_indices = (idx_order * reps)[:num_reads]

            samples = []
            energies = []
            for rid in sel_indices:
                start = rid * N
                end = start + N
                samples.append(spins_flat[start:end].astype(np.int8).tolist())
                energies.append(int(energies2d[rid]))

            runtime = time.time() - start_time
            self.logger.debug(f"[UnifiedMetal] 2D-only sampling completed: {runtime:.2f}s")
            return dimod.SampleSet.from_samples(samples, 'SPIN', energies)

        except Exception as e:
            self.logger.error(f"[UnifiedMetal] Sampling failed: {e}")
            raise


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