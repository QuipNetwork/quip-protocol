"""Unified Metal Parallel Tempering Sampler - Modular Metal Kernels.

Single interface that takes raw h,J dictionaries and processes them using
separate Metal kernels for each preprocessing step.
"""

import time
import logging
from typing import Optional, Dict, Any
import numpy as np
import struct

from collections import deque

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
                "metropolis_color_phase",
                "reduce_energies_2d",
                "pt_swap_adjacent",
                "compute_swap_stats",
                "compute_graph_coloring",
                "pack_selected_replicas",
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
                    sample_interval: Optional[int] = None,
                    ladder_target_acceptance: float = 0.30,
                    adapt_every: int = 2,
                    stats_window: int = 4,
                    max_replicas: Optional[int] = None,
                    enable_color_updates: bool = False) -> dimod.SampleSet:
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

            # === Phase 2: Compute graph coloring ===
            max_colors_val = 64  # Max colors (handle complete graphs and Zephyr deg~16)
            node_colors = np.zeros(N, dtype=np.int32)
            num_colors_out = np.zeros(1, dtype=np.int32)
            use_coloring_out = np.zeros(1, dtype=np.int32)
            node_degrees = np.zeros(N, dtype=np.int32)

            n_buf_coloring = self._create_buffer(np.array([N], dtype=np.int32), "N_coloring")
            max_colors_buf = self._create_buffer(np.array([max_colors_val], dtype=np.int32), "max_colors")
            node_colors_buf = self._create_buffer(node_colors, "node_colors")
            num_colors_buf_out = self._create_buffer(num_colors_out, "num_colors_out")
            use_coloring_buf = self._create_buffer(use_coloring_out, "use_coloring_out")
            node_degrees_buf = self._create_buffer(node_degrees, "node_degrees")

            # Dispatch coloring kernel
            cb_color = self._command_queue.commandBuffer()
            enc_color = cb_color.computeCommandEncoder()
            enc_color.setComputePipelineState_(self._kernels["compute_graph_coloring"])
            enc_color.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
            enc_color.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
            enc_color.setBuffer_offset_atIndex_(n_buf_coloring, 0, 2)
            enc_color.setBuffer_offset_atIndex_(max_colors_buf, 0, 3)
            enc_color.setBuffer_offset_atIndex_(node_colors_buf, 0, 4)
            enc_color.setBuffer_offset_atIndex_(num_colors_buf_out, 0, 5)
            enc_color.setBuffer_offset_atIndex_(use_coloring_buf, 0, 6)
            enc_color.setBuffer_offset_atIndex_(node_degrees_buf, 0, 7)
            enc_color.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSize(width=1, height=1, depth=1),
                Metal.MTLSize(width=1, height=1, depth=1)
            )
            enc_color.endEncoding(); cb_color.commit(); cb_color.waitUntilCompleted()

            # Read back coloring results
            nc_view = num_colors_buf_out.contents().as_buffer(num_colors_buf_out.length())
            num_colors_val = struct.unpack('<i', bytes([nc_view[i] for i in range(4)]))[0]
            uc_view = use_coloring_buf.contents().as_buffer(use_coloring_buf.length())
            use_coloring = struct.unpack('<i', bytes([uc_view[i] for i in range(4)]))[0] == 1

            if not use_coloring or num_colors_val == 0:
                raise RuntimeError(f"[Phase2] Graph coloring failed: needed >{max_colors_val} colors or invalid graph structure")

            self.logger.debug(f"[Phase2] Graph coloring successful: {num_colors_val} colors")

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


            # Read back initial temperatures into host array for adaptive ladder
            def _read_float32_array(buf, n):
                view = buf.contents().as_buffer(buf.length())
                out = []
                for i in range(n):
                    base = i * 4
                    out.append(struct.unpack('<f', bytes([view[base+j] for j in range(4)]))[0])
                return out

            def _read_int32_array(buf, n):
                view = buf.contents().as_buffer(buf.length())
                out = []
                for i in range(n):
                    base = i * 4
                    out.append(int.from_bytes(bytes([view[base+j] for j in range(4)]), 'little', signed=True))
                return out

            temps_host = _read_float32_array(temps_buf, num_reps)

            # Adaptive ladder parameters and state
            intervals_since_adapt = 0

            adapt_every_eff = int(adapt_every) if (adapt_every is not None and int(adapt_every) > 0) else 0
            stats_window_eff = int(stats_window) if (stats_window is not None and int(stats_window) > 0) else 0
            target_acc = float(ladder_target_acceptance)
            attempts_prev = np.zeros(max(0, num_reps-1), dtype=np.int32)
            accepts_prev = np.zeros(max(0, num_reps-1), dtype=np.int32)
            window_attempts = []  # list of np.int32 arrays
            window_accepts = []   # list of np.int32 arrays

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


            # Reusable small constant buffers
            n_buf = self._create_buffer(np.array([N], dtype=np.int32), "N")
            nc_buf = self._create_buffer(np.array([num_colors_val], dtype=np.int32), "num_colors")
            parity0_buf = self._create_buffer(np.array([0], dtype=np.int32), "parity0")
            parity1_buf = self._create_buffer(np.array([1], dtype=np.int32), "parity1")
            sweeps_buf_reusable = self._create_buffer(np.array([interval], dtype=np.int32), "sweeps_reusable")
            inflight_cbs = []
            inflight_tmp_bufs = []
            last_cb = None

            remaining = int(num_sweeps)
            while remaining > 0:
                sweeps_this = min(interval, remaining)
                self.logger.debug(f"[UnifiedMetal] Color-phase update sweeps chunk={sweeps_this} (remaining {remaining - sweeps_this})")

                # Update sweeps buffer value if different from interval
                if sweeps_this != interval:
                    sweeps_view = sweeps_buf_reusable.contents().as_buffer(sweeps_buf_reusable.length())
                    sweeps_bytes = struct.pack('<i', sweeps_this)
                    for i in range(4):
                        sweeps_view[i] = sweeps_bytes[i]

                # Batch per-interval work into a single command buffer; no wait here
                cbI = self._command_queue.commandBuffer()

                # Phase 2: Color-phase Metropolis update (randomization done in kernel)
                enc2u = cbI.computeCommandEncoder()
                enc2u.setComputePipelineState_(self._kernels["metropolis_color_phase"])
                enc2u.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
                enc2u.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
                enc2u.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
                enc2u.setBuffer_offset_atIndex_(n_buf, 0, 3)
                enc2u.setBuffer_offset_atIndex_(temps_buf, 0, 4)
                enc2u.setBuffer_offset_atIndex_(num_reps_buf, 0, 5)
                enc2u.setBuffer_offset_atIndex_(node_colors_buf, 0, 6)
                enc2u.setBuffer_offset_atIndex_(nc_buf, 0, 7)
                enc2u.setBuffer_offset_atIndex_(sweeps_buf_reusable, 0, 8)
                enc2u.setBuffer_offset_atIndex_(base_seed_buf, 0, 9)
                enc2u.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 10)
                enc2u.setBuffer_offset_atIndex_(thread_buffers_alt_buf, 0, 11)
                enc2u.setBuffer_offset_atIndex_(flip_counts_buf, 0, 12)
                enc2u.setBuffer_offset_atIndex_(pos_counts_buf, 0, 13)
                enc2u.setBuffer_offset_atIndex_(neg_counts_buf, 0, 14)
                enc2u.setBuffer_offset_atIndex_(dbg_energy_buf, 0, 15)
                enc2u.dispatchThreadgroups_threadsPerThreadgroup_(tg_reps, threads_per_group)
                enc2u.endEncoding()

                # Energy reduction after this chunk
                encRE = cbI.computeCommandEncoder()
                encRE.setComputePipelineState_(self._kernels["reduce_energies_2d"])
                encRE.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
                encRE.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
                encRE.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
                encRE.setBuffer_offset_atIndex_(n_buf, 0, 3)
                encRE.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 4)
                encRE.setBuffer_offset_atIndex_(dbg_energy_buf, 0, 5)

                intervals_since_adapt += 1
                if adapt_every_eff > 0 and intervals_since_adapt >= adapt_every_eff:
                    # Sync and adapt ladder using windowed swap stats
                    if last_cb is not None:
                        last_cb.waitUntilCompleted()
                    cur_att = np.array(_read_int32_array(attempts, max(0, num_reps-1)), dtype=np.int32)
                    cur_acc = np.array(_read_int32_array(accepts, max(0, num_reps-1)), dtype=np.int32)
                    delta_att = cur_att - attempts_prev
                    delta_acc = cur_acc - accepts_prev
                    attempts_prev = cur_att
                    accepts_prev = cur_acc
                    if stats_window_eff > 0:
                        window_attempts.append(delta_att)
                        window_accepts.append(delta_acc)
                        if len(window_attempts) > stats_window_eff:
                            window_attempts.pop(0)
                            window_accepts.pop(0)
                        sum_att = np.sum(window_attempts, axis=0) if window_attempts else np.zeros_like(delta_att)
                        sum_acc = np.sum(window_accepts, axis=0) if window_accepts else np.zeros_like(delta_acc)
                        # Compute windowed acceptance rates per adjacent pair
                        rates = np.where(sum_att > 0, sum_acc / np.maximum(sum_att, 1), 0.0).astype(np.float32)
                        if np.any(sum_att > 0) and len(temps_host) == num_reps:
                            # Nudge gaps toward target acceptance while preserving total beta span
                            betas = 1.0 / np.array(temps_host, dtype=np.float64)
                            beta_min = float(np.min(betas)); beta_max = float(np.max(betas))
                            gaps = np.diff(betas)
                            if gaps.size > 0 and (beta_max > beta_min):
                                # EMA-smooth acceptance rates and clamp per-step changes
                                try:
                                    ema_rates
                                except NameError:
                                    ema_rates = None
                                ema_alpha = 0.5  # smoothing factor
                                max_step = 0.15  # clamp per-adapt fractional change of each gap
                                if ema_rates is None:
                                    ema_rates = rates.astype(np.float64)
                                else:
                                    ema_rates = ema_alpha * rates.astype(np.float64) + (1.0 - ema_alpha) * ema_rates
                                rates_eff = ema_rates

                                k = 0.25  # nudge factor
                                factors = [1.0 - k * (r - target_acc) for r in rates_eff]
                                # clamp
                                factors = [min(1.0 + max_step, max(1.0 - max_step, f)) for f in factors]
                                new_gaps = np.array([max(1e-9, g * f) for g, f in zip(gaps, factors)], dtype=np.float64)
                                scale = (beta_max - beta_min) / max(1e-12, float(np.sum(new_gaps)))
                                new_gaps *= scale
                                new_betas = [beta_min]
                                for dg in new_gaps:
                                    new_betas.append(new_betas[-1] + float(dg))
                                new_betas[-1] = beta_max
                                new_betas = np.array(new_betas, dtype=np.float64)
                                new_temps = (1.0 / np.clip(new_betas, 1e-12, None)).astype(np.float32)
                                temps_host = new_temps.tolist()
                                # Write temperatures in-place to avoid buffer churn
                                vieww = temps_buf.contents().as_buffer(temps_buf.length())
                                for i, t in enumerate(new_temps):
                                    bs = struct.pack('<f', float(t))
                                    off = i * 4
                                    for j in range(4):
                                        vieww[off + j] = bs[j]
                    intervals_since_adapt = 0

                encRE.setBuffer_offset_atIndex_(num_reps_buf, 0, 6)
                encRE.dispatchThreadgroups_threadsPerThreadgroup_(tg_reps, threads_per_group)
                encRE.endEncoding()

                # Swaps: even then odd
                int_even_pairs = (num_reps // 2)
                if int_even_pairs > 0:
                    encSW0 = cbI.computeCommandEncoder()
                    encSW0.setComputePipelineState_(self._kernels["pt_swap_adjacent"])
                    encSW0.setBuffer_offset_atIndex_(n_buf, 0, 0)
                    encSW0.setBuffer_offset_atIndex_(temps_buf, 0, 1)
                    encSW0.setBuffer_offset_atIndex_(num_reps_buf, 0, 2)
                    encSW0.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 3)
                    encSW0.setBuffer_offset_atIndex_(dbg_energy_buf, 0, 4)
                    encSW0.setBuffer_offset_atIndex_(base_seed_buf, 0, 5)
                    encSW0.setBuffer_offset_atIndex_(parity0_buf, 0, 6)
                    encSW0.setBuffer_offset_atIndex_(attempts, 0, 7)
                    encSW0.setBuffer_offset_atIndex_(accepts, 0, 8)
                    encSW0.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=tpg_x, height=1, depth=1), Metal.MTLSize(width=1, height=int_even_pairs, depth=1))
                    encSW0.endEncoding()
                int_odd_pairs = ((num_reps - 1) // 2)
                if int_odd_pairs > 0:
                    encSW1 = cbI.computeCommandEncoder()
                    encSW1.setComputePipelineState_(self._kernels["pt_swap_adjacent"])
                    encSW1.setBuffer_offset_atIndex_(n_buf, 0, 0)
                    encSW1.setBuffer_offset_atIndex_(temps_buf, 0, 1)
                    encSW1.setBuffer_offset_atIndex_(num_reps_buf, 0, 2)
                    encSW1.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 3)
                    encSW1.setBuffer_offset_atIndex_(dbg_energy_buf, 0, 4)
                    encSW1.setBuffer_offset_atIndex_(base_seed_buf, 0, 5)
                    encSW1.setBuffer_offset_atIndex_(parity1_buf, 0, 6)
                    encSW1.setBuffer_offset_atIndex_(attempts, 0, 7)
                    encSW1.setBuffer_offset_atIndex_(accepts, 0, 8)
                    encSW1.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=tpg_x, height=1, depth=1), Metal.MTLSize(width=1, height=int_odd_pairs, depth=1))
                    encSW1.endEncoding()

                cbI.commit()
                last_cb = cbI
                inflight_cbs.append(cbI)
                remaining -= sweeps_this


            # Wait for all interval command buffers to complete before reading back
            if last_cb is not None:
                last_cb.waitUntilCompleted()
            inflight_cbs.clear(); inflight_tmp_bufs.clear()

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


                # GPU packing of selected replicas to reduce host read size
                # Select replicas by ascending energy (host-side selection only)
                idx_order = list(sorted(range(num_reps), key=lambda i: energies2d[i]))
                if num_reads <= num_reps:
                    sel_indices = idx_order[:num_reads]
                else:
                    reps = (num_reads + num_reps - 1) // num_reps
                    sel_indices = (idx_order * reps)[:num_reads]

                sel_idx_buf = self._create_buffer(np.array(sel_indices, dtype=np.int32), "sel_indices")
                num_reads_buf = self._create_buffer(np.array([int(num_reads)], dtype=np.int32), "num_reads")
                packed_out = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "packed_out")

                cbPK = self._command_queue.commandBuffer()
                encPK = cbPK.computeCommandEncoder()
                encPK.setComputePipelineState_(self._kernels["pack_selected_replicas"])
                encPK.setBuffer_offset_atIndex_(self._create_buffer(np.array([N], dtype=np.int32), "N"), 0, 0)
                encPK.setBuffer_offset_atIndex_(sel_idx_buf, 0, 1)
                encPK.setBuffer_offset_atIndex_(num_reads_buf, 0, 2)
                encPK.setBuffer_offset_atIndex_(thread_buffers_buf, 0, 3)
                encPK.setBuffer_offset_atIndex_(packed_out, 0, 4)
                encPK.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(width=tpg_x, height=1, depth=1), Metal.MTLSize(width=1, height=int(num_reads), depth=1))
                encPK.endEncoding(); cbPK.commit(); cbPK.waitUntilCompleted()

                # Read back packed samples only (not the full buffer)
                pview = packed_out.contents().as_buffer(packed_out.length())
                spins_bytes = bytes(pview)
                spins_flat = np.frombuffer(spins_bytes, dtype=np.int8)[:num_reads * N]

                samples = []
                energies = []
                for k, rid in enumerate(sel_indices):
                    samples.append(spins_flat[k*N:(k+1)*N].astype(np.int8).tolist())
                    energies.append(int(energies2d[rid]))

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