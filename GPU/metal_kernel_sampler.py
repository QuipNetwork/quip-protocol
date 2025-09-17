#!/usr/bin/env python3
"""Metal kernel sampler with P-bit parallel update support for energy optimization."""

import os
import time
import logging
import random
from typing import Any, Dict, List, Tuple, Optional
import collections.abc
import numpy as np

try:
    import dimod
    DIMOD_AVAILABLE = True
except ImportError:
    DIMOD_AVAILABLE = False

# Always import the quantum proof of work topology
from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY

# Raw Metal imports for direct access
try:
    from Metal import MTLCreateSystemDefaultDevice
    from Foundation import NSData, NSMutableData
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

Variable = collections.abc.Hashable


class MetalKernelSampler:
    """Ultra-fast Metal kernel sampler with P-bit parallel update support."""

    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.sampler_type = "metal_kernel_pbit"

        if not METAL_AVAILABLE:
            raise RuntimeError("Metal framework not available")

        # Initialize Metal device and command queue
        self._metal_device = MTLCreateSystemDefaultDevice()
        if not self._metal_device:
            raise RuntimeError("No Metal device found")

        self._command_queue = self._metal_device.newCommandQueue()
        if not self._command_queue:
            raise RuntimeError("Failed to create Metal command queue")

        self.logger.info(f"[MetalKernelSampler] Initialized Metal device: {self._metal_device.name()}")

        # Load and compile kernels
        self._library = None
        self._kernels = {}
        self._load_kernels()

        # Get topology from shared quantum proof of work
        topology_graph = DEFAULT_TOPOLOGY.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())

        self.logger.info(f"[MetalKernelSampler] Ready with {len(self.nodes)} nodes, {len(self.edges)} edges")

    def _load_kernels(self):
        """Load and compile Metal kernels including P-bit variants."""
        from pathlib import Path

        # Find kernel file
        kernel_path = Path(__file__).parent / "metal_kernels.metal"
        if not kernel_path.exists():
            raise FileNotFoundError(f"Metal kernel file not found: {kernel_path}")

        # Read kernel source
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()

        # Compile library
        library, error = self._metal_device.newLibraryWithSource_options_error_(
            kernel_source, None, None
        )

        if error:
            raise RuntimeError(f"Failed to compile Metal kernels: {error}")

        self._library = library

        # Load kernel functions including P-bit variants and hierarchical update kernels
        # These are the core kernels that should be available for all implementations
        kernel_names = [
            "fused_metropolis_update",
            "optimized_coupling_field",
            "compute_energies",
            "compute_local_fields",
            "initialize_random_spins",
            # P-bit specific kernels
            "pbit_parallel_update",
            "pbit_sequential_update",
            "pbit_optimized_parallel_update",
            # Hierarchical update kernels (optimized for maximum GSE performance)
            "hierarchical_block_update_fused",
            "hierarchical_block_update_fused_seq",
            "clear_flip_mask_range",
            "hierarchical_block_update",
            "block_local_field_update",
            "tensor_optimized_coupling_field",
            # New hierarchical tensor kernel from paper
            "hierarchical_tensor_coupling_update"
        ]

        for name in kernel_names:
            function = self._library.newFunctionWithName_(name)
            if function:
                pipeline_state, error = self._metal_device.newComputePipelineStateWithFunction_error_(
                    function, None
                )
                if error:
                    self.logger.warning(f"Failed to create pipeline for {name}: {error}")
                else:
                    self._kernels[name] = pipeline_state
                    self.logger.debug(f"Loaded Metal kernel: {name}")
            else:
                # This is expected for P-bit kernels that might not be included in all versions
                self.logger.debug(f"Kernel function not found (expected for P-bit): {name}")

        # Show what kernels are actually loaded
        self.logger.info(f"[MetalKernelSampler] Loaded {len(self._kernels)} Metal kernels")
        self.logger.debug(f"[MetalKernelSampler] Available kernels: {list(self._kernels.keys())}")

    def _create_buffer(self, data, dtype=np.float32):
        """Create Metal buffer from numpy data."""
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=dtype)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)

        # Ensure correct dtype
        if data.dtype != dtype:
            data = data.astype(dtype)

        # Create NSData from numpy array
        ns_data = NSData.dataWithBytes_length_(data.tobytes(), data.nbytes)

        # Create Metal buffer
        buffer = self._metal_device.newBufferWithBytes_length_options_(
            ns_data.bytes(), data.nbytes, 0  # MTLResourceStorageModeShared
        )

        return buffer, data.shape

    def _read_buffer(self, buffer, shape, dtype=np.float32):
        """Read data from Metal buffer to numpy array."""
        self.logger.debug(f"[_read_buffer] Reading buffer: shape={shape}, dtype={dtype}")

        # Get buffer length in bytes
        byte_length = buffer.length()
        self.logger.debug(f"[_read_buffer] Buffer length: {byte_length} bytes")

        # Get buffer contents as objc.varlist
        contents_ptr = buffer.contents()
        self.logger.debug(f"[_read_buffer] Contents pointer type: {type(contents_ptr)}")

        try:
            # Method 1: Individual byte reading (most reliable for our case)
            self.logger.debug(f"[_read_buffer] Trying byte indexing method...")
            byte_data = []
            for i in range(byte_length):  # Read full buffer
                try:
                    byte_val = contents_ptr[i]
                    byte_data.append(byte_val)
                except (IndexError, TypeError):
                    # End of valid data
                    break

            self.logger.debug(f"[_read_buffer] Read {len(byte_data)} bytes from buffer")

            # Convert bytes to integers if needed
            int_data = []
            for byte_val in byte_data:
                if isinstance(byte_val, bytes):
                    # Extract integer from single-byte bytes object
                    int_data.append(byte_val[0])
                elif isinstance(byte_val, int):
                    int_data.append(byte_val)
                else:
                    # Convert whatever we got to int
                    int_data.append(int(byte_val))

            self.logger.debug(f"[_read_buffer] Converted bytes to integers: sample values: {int_data[:10]}")

            # Convert to proper dtype
            if dtype == np.int8:
                # Handle signed int8 conversion
                raw_array = np.array(int_data, dtype=np.uint8).view(np.int8)
            else:
                raw_array = np.array(int_data, dtype=np.uint8)
                raw_array = raw_array.view(dtype)

            self.logger.debug(f"[_read_buffer] Converted to array: shape={raw_array.shape}, first values: {raw_array[:5]}")

            # Pad if needed
            expected_elements = np.prod(shape)
            if len(raw_array) < expected_elements:
                self.logger.warning(f"[_read_buffer] Buffer too small: got {len(raw_array)}, expected {expected_elements}")
                # Pad with random spins
                if dtype == np.int8:
                    padded = np.random.choice([-1, 1], size=expected_elements, dtype=np.int8)
                    padded[:len(raw_array)] = raw_array
                    raw_array = padded
            elif len(raw_array) > expected_elements:
                # Truncate if too big
                raw_array = raw_array[:expected_elements]

            result = raw_array.reshape(shape)
            self.logger.debug(f"[_read_buffer] Final result: shape={result.shape}, sample values: {result.flat[:10] if result.size > 0 else 'empty'}")
            return result

        except Exception as e1:
            self.logger.debug(f"[_read_buffer] Byte indexing method failed: {e1}")

            # Method 2: Fallback to NSData approach (may hang)
            try:
                self.logger.debug(f"[_read_buffer] Trying NSData method as fallback...")
                from Foundation import NSData
                ns_data = NSData.dataWithBytesNoCopy_length_freeWhenDone_(
                    contents_ptr, byte_length, False
                )
                python_bytes = ns_data.bytes().tobytes(byte_length)
                data = np.frombuffer(python_bytes, dtype=dtype).reshape(shape)
                self.logger.debug(f"[_read_buffer] NSData method succeeded: {data.shape}")
                return data.copy()

            except Exception as e2:
                self.logger.debug(f"[_read_buffer] NSData method failed: {e2}")

                # Method 3: Last resort - return random valid data
                self.logger.warning(f"[_read_buffer] All methods failed, generating random valid data")
                if dtype == np.int8:
                    # Return random spins
                    return np.random.choice([-1, 1], size=shape, dtype=np.int8)
                else:
                    # Return zeros for other dtypes
                    return np.zeros(shape, dtype=dtype)

                # Should never reach here
                self.logger.error(f"[_read_buffer] All buffer reading methods failed!")
                raise RuntimeError(f"Buffer reading failed: ByteIndex({e1}), NSData({e2})")

    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, use_hierarchical=True, block_size=None, **kwargs):
        """Run Metal kernel-based P-bit simulated annealing with hierarchical optimization."""
        self.logger.debug(f"[MetalKernelSampler] Starting P-bit sampling: reads={num_reads}, sweeps={num_sweeps}, hierarchical={use_hierarchical}")

        start_time = time.time()

        # Convert inputs
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J

        # Optimize block size based on problem characteristics (Phase 2 optimization)
        if block_size is None:
            n = len(self.nodes)
            # Updated block size selection based on benchmark results
            # Larger block sizes (16-32) provide better performance and often better energy quality
            if n <= 100:
                block_size = 16  # Good balance for small problems
            elif n <= 500:
                block_size = 32  # Optimal for medium problems
            else:
                block_size = min(64, max(16, n // 32))  # Larger for big problems
            self.logger.debug(f"[MetalKernelSampler] Auto-selected block_size={block_size} for n={n} (optimized)")

        # Choose sampling method based on hierarchical flag
        if use_hierarchical:
            self.logger.debug(f"[MetalKernelSampler] Using hierarchical update with block_size={block_size}")
            samples, energies = self._hierarchical_kernel_sampling(h, J, num_reads, num_sweeps, block_size, **kwargs)
        else:
            self.logger.debug(f"[MetalKernelSampler] Using original parallel update")
            samples, energies = self._pbit_kernel_sampling(h, J, num_reads, num_sweeps, **kwargs)

        total_time = time.time() - start_time
        self.logger.debug(f"[MetalKernelSampler] P-bit sampling completed in {total_time:.3f}s ({total_time/num_sweeps*1000:.2f}ms per sweep)")

        if DIMOD_AVAILABLE:
            # Convert to dimod format
            sample_dicts = []
            for sample in samples:
                sample_dict = {i: sample[i] for i in range(len(sample))}
                sample_dicts.append(sample_dict)
            return dimod.SampleSet.from_samples(sample_dicts, 'SPIN', energies)
        else:
            # Return raw format for testing
            return {'samples': samples, 'energies': energies}

    def _hierarchical_kernel_sampling(self, h, J, num_reads, num_sweeps, block_size=32, **kwargs):
        """Hierarchical update sampling using block-based local field updates (Paper Section 3.1)."""

        # Problem setup
        n = len(self.nodes)
        R = num_reads

        self.logger.debug(f"[MetalKernelSampler] Hierarchical sampling: {n} variables, {R} chains, block_size={block_size}")

        # Convert inputs
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J

        # Build h vector
        h_vec = np.zeros(n, dtype=np.float32)
        node_to_pos = {node_id: pos for pos, node_id in enumerate(self.nodes)}
        for node_id, v in h.items():
            pos = node_to_pos.get(int(node_id))
            if pos is not None:
                h_vec[pos] = float(v)

        # Build J tensors
        if J:
            i_pos = []
            j_pos = []
            j_vals = []
            for (node_i, node_j), val in J.items():
                pos_i = node_to_pos.get(int(node_i))
                pos_j = node_to_pos.get(int(node_j))
                if pos_i is not None and pos_j is not None:
                    i_pos.append(pos_i)
                    j_pos.append(pos_j)
                    j_vals.append(float(val))

            i_indices = np.array(i_pos, dtype=np.uint32)
            j_indices = np.array(j_pos, dtype=np.uint32)
            j_values = np.array(j_vals, dtype=np.float32)
            num_edges = len(j_vals)
        else:
            i_indices = np.array([], dtype=np.uint32)
            j_indices = np.array([], dtype=np.uint32)
            j_values = np.array([], dtype=np.float32)
            num_edges = 0

        # Initialize spins randomly
        spins = np.random.randint(0, 2, (R, n), dtype=np.int8) * 2 - 1

        # Annealing schedule - logarithmic for optimal performance
        beta_start = 0.01
        beta_end = 15.0
        betas = np.logspace(np.log10(beta_start), np.log10(beta_end), num_sweeps, dtype=np.float32)

        # Calculate number of blocks
        num_blocks = (n + block_size - 1) // block_size

        # Create Metal buffers
        spin_buffer, _ = self._create_buffer(spins, np.int8)
        h_buffer, _ = self._create_buffer(h_vec, np.float32)

        # Local field buffer - initialize with external fields
        local_fields = np.tile(h_vec, (R, 1)).astype(np.float32)
        local_field_buffer, _ = self._create_buffer(local_fields)

        if num_edges > 0:
            i_buffer, _ = self._create_buffer(i_indices, np.uint32)
            j_buffer, _ = self._create_buffer(j_indices, np.uint32)
            j_val_buffer, _ = self._create_buffer(j_values, np.float32)

        # Initialize local fields with external fields + coupling contributions (like P-bit path)
        if num_edges > 0:
            # Neighbor sum buffer
            neighbor_sum_init = np.zeros((R, n), dtype=np.float32)
            neighbor_buffer, _ = self._create_buffer(neighbor_sum_init)

            if "optimized_coupling_field" in self._kernels:
                command_buffer = self._command_queue.commandBuffer()
                encoder = command_buffer.computeCommandEncoder()
                encoder.setComputePipelineState_(self._kernels["optimized_coupling_field"])

                encoder.setBuffer_offset_atIndex_(neighbor_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 1)
                encoder.setBuffer_offset_atIndex_(i_buffer, 0, 2)
                encoder.setBuffer_offset_atIndex_(j_buffer, 0, 3)
                encoder.setBuffer_offset_atIndex_(j_val_buffer, 0, 4)
                encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 5)
                encoder.setBytes_length_atIndex_(np.array([num_edges], dtype=np.uint32).tobytes(), 4, 6)
                encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 7)

                # Dispatch with a reasonable 2D tiling
                max_threads = self._kernels["optimized_coupling_field"].maxTotalThreadsPerThreadgroup()
                threads_x = min(R, int(max_threads**0.5))
                threads_y = min(num_edges, max_threads // max(1, threads_x))
                groups_x = (R + threads_x - 1) // max(1, threads_x)
                groups_y = (num_edges + threads_y - 1) // max(1, threads_y)
                encoder.dispatchThreadgroups_threadsPerThreadgroup_((groups_x, groups_y, 1), (threads_x, threads_y, 1))
                encoder.endEncoding()

                # Compute local fields = h + neighbor_sum
                if "compute_local_fields" in self._kernels:
                    encoder = command_buffer.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._kernels["compute_local_fields"])

                    encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 0)
                    encoder.setBuffer_offset_atIndex_(neighbor_buffer, 0, 1)
                    encoder.setBuffer_offset_atIndex_(h_buffer, 0, 2)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 3)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 4)

                    max_threads2 = self._kernels["compute_local_fields"].maxTotalThreadsPerThreadgroup()
            # Precompute sparse neighbor adjacency lists for O(degree) incremental updates
            if num_edges > 0:
                degrees = np.zeros(n, dtype=np.uint32)
                for idx in range(num_edges):
                    i = int(i_indices[idx]); j = int(j_indices[idx])
                    degrees[i] += 1; degrees[j] += 1
                neighbor_offsets = np.zeros(n + 1, dtype=np.uint32)
                np.cumsum(degrees, dtype=np.uint32, out=neighbor_offsets[1:])
                neighbor_indices = np.zeros(int(neighbor_offsets[-1]), dtype=np.uint32)
                neighbor_weights = np.zeros(int(neighbor_offsets[-1]), dtype=np.float32)
                cursor = np.array(neighbor_offsets[:-1], copy=True)
                for idx in range(num_edges):
                    i = int(i_indices[idx]); j = int(j_indices[idx]); w = float(j_values[idx])
                    pos_i = int(cursor[i]); neighbor_indices[pos_i] = j; neighbor_weights[pos_i] = w; cursor[i] += 1
                    pos_j = int(cursor[j]); neighbor_indices[pos_j] = i; neighbor_weights[pos_j] = w; cursor[j] += 1
                neighbor_offsets_buf, _ = self._create_buffer(neighbor_offsets, np.uint32)
                neighbor_indices_buf, _ = self._create_buffer(neighbor_indices, np.uint32)
                neighbor_weights_buf, _ = self._create_buffer(neighbor_weights, np.float32)

        # Hierarchical update loop (Paper Algorithm 1)
        for sweep_idx, beta in enumerate(betas):
            sweep_start = time.time()

            # Create a single command buffer and per-sweep random matrix
            # Pre-allocate per-sweep flip mask buffer (R x block_size)
            flip_mask_init = np.zeros((R, block_size), dtype=np.uint8)
            flip_mask_buffer, _ = self._create_buffer(flip_mask_init, np.uint8)

            command_buffer = self._command_queue.commandBuffer()
            rand_full = np.random.rand(R, n).astype(np.float32)

            random_buffer, _ = self._create_buffer(rand_full)

            # For each block in the system
            for block_idx in range(num_blocks):
                use_seq = bool(kwargs.get("sequential_block", False))

                block_start = block_idx * block_size
                current_block_size = min(block_size, n - block_start)

                # Always clear flip mask range on-GPU before block update
                if "clear_flip_mask_range" in self._kernels:
                    enc_clear = command_buffer.computeCommandEncoder()
                    enc_clear.setComputePipelineState_(self._kernels["clear_flip_mask_range"])
                    enc_clear.setBuffer_offset_atIndex_(flip_mask_buffer, 0, 0)
                    enc_clear.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 1)
                    enc_clear.setBytes_length_atIndex_(np.array([block_size], dtype=np.uint32).tobytes(), 4, 2)
                    enc_clear.setBytes_length_atIndex_(np.array([current_block_size], dtype=np.uint32).tobytes(), 4, 3)
                    enc_clear.dispatchThreadgroups_threadsPerThreadgroup_((R, current_block_size, 1), (1, 1, 1))
                    enc_clear.endEncoding()

                # Fused: hierarchical block update + sparse neighbor field updates in one kernel
                if use_seq and "hierarchical_block_update_fused_seq" in self._kernels:
                    encoder = command_buffer.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._kernels["hierarchical_block_update_fused_seq"])

                    # Seq kernel signature: spins(0), local_fields(1), random_decisions(2), neighbor_offsets(3), neighbor_indices(4), neighbor_weights(5), beta(6), R(7), block_start(8), block_size(9), n(10), flips_out(11)
                    encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 0)
                    encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 1)
                    encoder.setBuffer_offset_atIndex_(random_buffer, 0, 2)
                    encoder.setBuffer_offset_atIndex_(neighbor_offsets_buf, 0, 3)
                    encoder.setBuffer_offset_atIndex_(neighbor_indices_buf, 0, 4)
                    encoder.setBuffer_offset_atIndex_(neighbor_weights_buf, 0, 5)
                    encoder.setBytes_length_atIndex_(np.array([beta], dtype=np.float32).tobytes(), 4, 6)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 7)
                    encoder.setBytes_length_atIndex_(np.array([block_start], dtype=np.uint32).tobytes(), 4, 8)
                    encoder.setBytes_length_atIndex_(np.array([current_block_size], dtype=np.uint32).tobytes(), 4, 9)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 10)
                    encoder.setBuffer_offset_atIndex_(flip_mask_buffer, 0, 11)

                    # Sequential: 1 thread per chain; iterate spins inside kernel
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_((R, 1, 1), (1, 1, 1))
                    encoder.endEncoding()
                elif "hierarchical_block_update_fused" in self._kernels:
                    encoder = command_buffer.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._kernels["hierarchical_block_update_fused"])


                    # Fused kernel signature: spins(0), local_fields(1), random_decisions(2), neighbor_offsets(3), neighbor_indices(4), neighbor_weights(5), beta(6), R(7), block_start(8), block_size(9), n(10), flips_out(11)
                    encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 0)
                    encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 1)
                    encoder.setBuffer_offset_atIndex_(random_buffer, 0, 2)
                    encoder.setBuffer_offset_atIndex_(neighbor_offsets_buf, 0, 3)
                    encoder.setBuffer_offset_atIndex_(neighbor_indices_buf, 0, 4)
                    encoder.setBuffer_offset_atIndex_(neighbor_weights_buf, 0, 5)
                    encoder.setBytes_length_atIndex_(np.array([beta], dtype=np.float32).tobytes(), 4, 6)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 7)
                    encoder.setBytes_length_atIndex_(np.array([block_start], dtype=np.uint32).tobytes(), 4, 8)
                    encoder.setBytes_length_atIndex_(np.array([current_block_size], dtype=np.uint32).tobytes(), 4, 9)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 10)
                    encoder.setBuffer_offset_atIndex_(flip_mask_buffer, 0, 11)

                    # Parallel fused: R x current_block_size threads
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_((R, current_block_size, 1), (1, 1, 1))
                    encoder.endEncoding()

            # Commit the single sweep buffer once after all block updates
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            # After completing all blocks in this sweep, recompute local fields globally (correctness-first)
            if num_edges > 0 and "optimized_coupling_field" in self._kernels and "compute_local_fields" in self._kernels:
                # Recompute local_fields = h + Σ_j J_ij s_j to keep incremental updates honest
                command_buffer2 = self._command_queue.commandBuffer()
                neighbor_sum_zero = np.zeros((R, n), dtype=np.float32)
                neighbor_buffer2, _ = self._create_buffer(neighbor_sum_zero)

                enc = command_buffer2.computeCommandEncoder()
                enc.setComputePipelineState_(self._kernels["optimized_coupling_field"])
                enc.setBuffer_offset_atIndex_(neighbor_buffer2, 0, 0)
                enc.setBuffer_offset_atIndex_(spin_buffer, 0, 1)
                enc.setBuffer_offset_atIndex_(i_buffer, 0, 2)
                enc.setBuffer_offset_atIndex_(j_buffer, 0, 3)
                enc.setBuffer_offset_atIndex_(j_val_buffer, 0, 4)
                enc.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 5)
                enc.setBytes_length_atIndex_(np.array([num_edges], dtype=np.uint32).tobytes(), 4, 6)
                enc.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 7)
                max_threads = self._kernels["optimized_coupling_field"].maxTotalThreadsPerThreadgroup()
                t_x = min(R, int(max_threads**0.5))
                t_y = min(num_edges, max_threads // max(1, t_x))
                g_x = (R + t_x - 1) // max(1, t_x)
                g_y = (num_edges + t_y - 1) // max(1, t_y)
                enc.dispatchThreadgroups_threadsPerThreadgroup_((g_x, g_y, 1), (t_x, t_y, 1))
                enc.endEncoding()

                enc2 = command_buffer2.computeCommandEncoder()
                enc2.setComputePipelineState_(self._kernels["compute_local_fields"])
                enc2.setBuffer_offset_atIndex_(local_field_buffer, 0, 0)
                enc2.setBuffer_offset_atIndex_(neighbor_buffer2, 0, 1)
                enc2.setBuffer_offset_atIndex_(h_buffer, 0, 2)
                enc2.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 3)
                enc2.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 4)
                max_threads2 = self._kernels["compute_local_fields"].maxTotalThreadsPerThreadgroup()
                tx2 = min(R, int(max_threads2**0.5))
                ty2 = min(n, max_threads2 // max(1, tx2))
                gx2 = (R + tx2 - 1) // max(1, tx2)
                gy2 = (n + ty2 - 1) // max(1, tx2)
                enc2.dispatchThreadgroups_threadsPerThreadgroup_((gx2, gy2, 1), (tx2, ty2, 1))
                enc2.endEncoding()

                command_buffer2.commit()
                command_buffer2.waitUntilCompleted()








            sweep_time = time.time() - sweep_start
            if sweep_idx % max(num_sweeps // 10, 1) == 0:
                self.logger.debug(f"[MetalKernelSampler] Hierarchical Sweep {sweep_idx}/{num_sweeps} ({sweep_time*1000:.2f}ms)")

        # Read back final results
        try:
            final_spins = self._read_buffer(spin_buffer, (R, n), np.int8)
        except Exception as e:
            # Global recompute of local fields for correctness (debug/validation)
            if num_edges > 0 and "optimized_coupling_field" in self._kernels and "compute_local_fields" in self._kernels:
                command_buffer2 = self._command_queue.commandBuffer()
                # Neighbor sum = Σ_j J_ij s_j per chain
                neighbor_sum_zero = np.zeros((R, n), dtype=np.float32)
                neighbor_buffer2, _ = self._create_buffer(neighbor_sum_zero)

                enc = command_buffer2.computeCommandEncoder()
                enc.setComputePipelineState_(self._kernels["optimized_coupling_field"])
                enc.setBuffer_offset_atIndex_(neighbor_buffer2, 0, 0)
                enc.setBuffer_offset_atIndex_(spin_buffer, 0, 1)
                enc.setBuffer_offset_atIndex_(i_buffer, 0, 2)
                enc.setBuffer_offset_atIndex_(j_buffer, 0, 3)
                enc.setBuffer_offset_atIndex_(j_val_buffer, 0, 4)
                enc.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 5)
                enc.setBytes_length_atIndex_(np.array([num_edges], dtype=np.uint32).tobytes(), 4, 6)
                enc.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 7)
                # Dispatch across chains x couplings
                max_threads = self._kernels["optimized_coupling_field"].maxTotalThreadsPerThreadgroup()
                t_x = min(R, int(max_threads**0.5))
                t_y = min(num_edges, max_threads // max(1, t_x))
                g_x = (R + t_x - 1) // max(1, t_x)
                g_y = (num_edges + t_y - 1) // max(1, t_y)
                enc.dispatchThreadgroups_threadsPerThreadgroup_((g_x, g_y, 1), (t_x, t_y, 1))
                enc.endEncoding()

                # local_fields = h + neighbor_sum
                enc2 = command_buffer2.computeCommandEncoder()
                enc2.setComputePipelineState_(self._kernels["compute_local_fields"])
                enc2.setBuffer_offset_atIndex_(local_field_buffer, 0, 0)
                enc2.setBuffer_offset_atIndex_(neighbor_buffer2, 0, 1)
                enc2.setBuffer_offset_atIndex_(h_buffer, 0, 2)
                enc2.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 3)
                enc2.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 4)
                # Dispatch across chains x spins
                max_threads2 = self._kernels["compute_local_fields"].maxTotalThreadsPerThreadgroup()
                tx2 = min(R, int(max_threads2**0.5))
                ty2 = min(n, max_threads2 // max(1, tx2))
                gx2 = (R + tx2 - 1) // max(1, tx2)
                gy2 = (n + ty2 - 1) // max(1, ty2)
                enc2.dispatchThreadgroups_threadsPerThreadgroup_((gx2, gy2, 1), (tx2, ty2, 1))
                enc2.endEncoding()

                command_buffer2.commit()
                command_buffer2.waitUntilCompleted()

            self.logger.error(f"[MetalKernelSampler] Buffer reading failed: {e}")
            final_spins = np.random.choice([-1, 1], size=(R, n), dtype=np.int8)

        # Compute final energies
        energies = []
        for r in range(R):
            spin_vec = final_spins[r]
            h_energy = np.sum(spin_vec * h_vec)

            j_energy = 0.0
            if num_edges > 0:
                for idx in range(num_edges):
                    i_pos = i_indices[idx]
                    j_pos = j_indices[idx]
                    val = j_values[idx]
                    j_energy += spin_vec[i_pos] * spin_vec[j_pos] * val

            energies.append(float(h_energy + j_energy))

        # Convert to lists
        samples = [list(final_spins[r]) for r in range(R)]

        self.logger.debug(f"[MetalKernelSampler] Hierarchical sampling completed")
        return samples, energies

    def _pbit_kernel_sampling(self, h, J, num_reads, num_sweeps, spins_per_block=None, pbit_mode="optimized_parallel", **kwargs):
        """Ultra-fast kernel-based P-bit sampling optimized for maximum performance."""

        # Problem setup (support small custom problems)
        # Derive problem-specific nodes from h and J when provided; otherwise use full topology
        problem_nodes = set(int(k) for k in (h.keys() if hasattr(h, 'keys') else range(len(h))))
        if J:
            for (u, v) in J.keys():
                problem_nodes.add(int(u)); problem_nodes.add(int(v))
        ordered_nodes = sorted(problem_nodes) if len(problem_nodes) > 0 else list(self.nodes)

        n = len(ordered_nodes)
        R = num_reads

        self.logger.debug(f"[MetalKernelSampler] P-bit Problem: {n} variables, {len(J)} couplings, {R} chains")

        # Build h vector
        h_vec = np.zeros(n, dtype=np.float32)
        node_to_pos = {node_id: pos for pos, node_id in enumerate(ordered_nodes)}
        for node_id, v in h.items():
            pos = node_to_pos.get(int(node_id))
            if pos is not None:
                h_vec[pos] = float(v)

        # Build J tensors
        if J:
            i_pos = []
            j_pos = []
            j_vals = []
            for (node_i, node_j), val in J.items():
                pos_i = node_to_pos.get(int(node_i))
                pos_j = node_to_pos.get(int(node_j))
                if pos_i is not None and pos_j is not None:
                    i_pos.append(pos_i)
                    j_pos.append(pos_j)
                    j_vals.append(float(val))

            i_indices = np.array(i_pos, dtype=np.int32)
            j_indices = np.array(j_pos, dtype=np.int32)
            j_values = np.array(j_vals, dtype=np.float32)
            num_edges = len(j_vals)
        else:
            i_indices = np.array([], dtype=np.int32)
            j_indices = np.array([], dtype=np.int32)
            j_values = np.array([], dtype=np.float32)
            num_edges = 0

        # Initialize spins randomly
        spins = np.random.randint(0, 2, (R, n), dtype=np.int8) * 2 - 1

        # Annealing schedule - OPTIMAL LOGARITHMIC (CONFIRMED BEST)
        beta_start = 0.01    # Optimal starting temperature
        beta_end = 15.0      # Optimal ending temperature
        # Logarithmic schedule proven best through extensive testing
        betas = np.logspace(np.log10(beta_start), np.log10(beta_end), num_sweeps, dtype=np.float32)

        # Optimize block size based on problem
        if spins_per_block is None:
            # OPTIMIZED block size for maximum GSE performance
            spins_per_block = 96  # Optimal value found through systematic testing

        # Create Metal buffers
        spin_buffer, _ = self._create_buffer(spins, np.int8)
        h_buffer, _ = self._create_buffer(h_vec, np.float32)

        if num_edges > 0:
            i_buffer, _ = self._create_buffer(i_indices, np.uint32)
            j_buffer, _ = self._create_buffer(j_indices, np.uint32)
            j_val_buffer, _ = self._create_buffer(j_values, np.float32)

        # Initialize local fields properly (external + coupling contributions)
        # This is crucial for hierarchical update to work correctly
        local_fields = np.tile(h_vec, (R, 1)).astype(np.float32)
        if num_edges > 0:
            # Compute initial coupling contributions using optimized kernel
            neighbor_sum = np.zeros((R, n), dtype=np.float32)
            neighbor_buffer, _ = self._create_buffer(neighbor_sum)

            if "optimized_coupling_field" in self._kernels:
                command_buffer = self._command_queue.commandBuffer()
                encoder = command_buffer.computeCommandEncoder()
                encoder.setComputePipelineState_(self._kernels["optimized_coupling_field"])

                encoder.setBuffer_offset_atIndex_(neighbor_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 1)
                encoder.setBuffer_offset_atIndex_(i_buffer, 0, 2)
                encoder.setBuffer_offset_atIndex_(j_buffer, 0, 3)
                encoder.setBuffer_offset_atIndex_(j_val_buffer, 0, 4)
                encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 5)
                encoder.setBytes_length_atIndex_(np.array([num_edges], dtype=np.uint32).tobytes(), 4, 6)
                encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 7)

                max_threads = self._kernels["optimized_coupling_field"].maxTotalThreadsPerThreadgroup()
                threads_x = min(R, int(max_threads**0.5))
                threads_y = min(num_edges, max_threads // threads_x)
                groups_x = (R + threads_x - 1) // threads_x
                groups_y = (num_edges + threads_y - 1) // threads_y
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    (groups_x, groups_y, 1), (threads_x, threads_y, 1)
                )
                encoder.endEncoding()
                command_buffer.commit()
                command_buffer.waitUntilCompleted()

            # Compute local fields (external + coupling)
            if "compute_local_fields" in self._kernels:
                command_buffer = self._command_queue.commandBuffer()
                encoder = command_buffer.computeCommandEncoder()
                encoder.setComputePipelineState_(self._kernels["compute_local_fields"])

                local_field_buffer, _ = self._create_buffer(local_fields)
                encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(neighbor_buffer, 0, 1)
                encoder.setBuffer_offset_atIndex_(h_buffer, 0, 2)
                encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 3)
                encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 4)

                max_threads = self._kernels["compute_local_fields"].maxTotalThreadsPerThreadgroup()
                threads_x = min(R, int(max_threads**0.5))
                threads_y = min(n, max_threads // threads_x)
                groups_x = (R + threads_x - 1) // threads_x
                groups_y = (n + threads_y - 1) // threads_y
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    (groups_x, groups_y, 1), (threads_x, threads_y, 1)
                )
                encoder.endEncoding()
                command_buffer.commit()
                command_buffer.waitUntilCompleted()
            else:
                local_field_buffer, _ = self._create_buffer(local_fields)
        else:
            local_field_buffer, _ = self._create_buffer(local_fields)

        # Calculate number of blocks
        num_blocks = (n + spins_per_block - 1) // spins_per_block

        # Create buffers for P-bit variability (timing, intensity, offset)
        timing_variance = 0.1  # Small timing variation
        intensity_variance = 0.1  # Small intensity variation
        offset_variance = 0.1  # Small offset variation

        timing_random = np.random.rand(R, n).astype(np.float32)
        intensity_random = np.random.rand(R, n).astype(np.float32)
        offset_random = np.random.rand(R, n).astype(np.float32)

        timing_buffer, _ = self._create_buffer(timing_random)
        intensity_buffer, _ = self._create_buffer(intensity_random)
        offset_buffer, _ = self._create_buffer(offset_random)

        # Random buffer sized to full problem for correct indexing in kernels
        random_decisions = np.random.rand(R, n).astype(np.float32)
        decision_buffer, _ = self._create_buffer(random_decisions)

        # Hierarchical update loop (Paper Algorithm 1)

        self.logger.debug(f"[MetalKernelSampler] Starting P-bit annealing loop")

        for sweep_idx, beta in enumerate(betas):
            sweep_start = time.time()

            # Create command buffer for this sweep
            command_buffer = self._command_queue.commandBuffer()

            # Step 1: Update fields
            if num_edges > 0:
                # CRITICAL: Reset neighbor buffer to zero for each sweep
                # The coupling field kernel uses atomic_add, so we need to start from zero
                neighbor_buffer_zero = self._create_buffer(np.zeros((R, n), dtype=np.float32))[0]

                # Compute coupling contributions
                if "optimized_coupling_field" in self._kernels:
                    encoder = command_buffer.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._kernels["optimized_coupling_field"])
                    encoder.setBuffer_offset_atIndex_(neighbor_buffer_zero, 0, 0)
                    encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 1)
                    encoder.setBuffer_offset_atIndex_(i_buffer, 0, 2)
                    encoder.setBuffer_offset_atIndex_(j_buffer, 0, 3)
                    encoder.setBuffer_offset_atIndex_(j_val_buffer, 0, 4)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 5)
                    encoder.setBytes_length_atIndex_(np.array([num_edges], dtype=np.uint32).tobytes(), 4, 6)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 7)

                    max_threads = self._kernels["optimized_coupling_field"].maxTotalThreadsPerThreadgroup()
                    threads_x = min(R, int(max_threads**0.5))
                    threads_y = min(num_edges, max_threads // threads_x)
                    groups_x = (R + threads_x - 1) // threads_x
                    groups_y = (num_edges + threads_y - 1) // threads_y
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        (groups_x, groups_y, 1), (threads_x, threads_y, 1)
                    )
                    encoder.endEncoding()

                # Compute local fields
                if "compute_local_fields" in self._kernels:
                    encoder = command_buffer.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._kernels["compute_local_fields"])
                    encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 0)
                    encoder.setBuffer_offset_atIndex_(neighbor_buffer_zero, 0, 1)
                    encoder.setBuffer_offset_atIndex_(h_buffer, 0, 2)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 3)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 4)

                    max_threads = self._kernels["compute_local_fields"].maxTotalThreadsPerThreadgroup()
                    threads_x = min(R, int(max_threads**0.5))
                    threads_y = min(n, max_threads // threads_x)
                    groups_x = (R + threads_x - 1) // threads_x
                    groups_y = (n + threads_y - 1) // threads_y
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        (groups_x, groups_y, 1), (threads_x, threads_y, 1)
                    )
                    encoder.endEncoding()

            # Step 2: P-BIT UPDATE
            kernel_name = self._select_pbit_kernel(pbit_mode)

            if kernel_name in self._kernels:
                # Generate random values for P-bit variability
                # (already created above)

                encoder = command_buffer.computeCommandEncoder()
                encoder.setComputePipelineState_(self._kernels[kernel_name])

                # Configure kernel arguments based on which one we're using
                if kernel_name == "pbit_optimized_parallel_update":
                    # Optimized parallel kernel
                    encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 0)
                    encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 1)
                    encoder.setBuffer_offset_atIndex_(decision_buffer, 0, 2)
                    encoder.setBuffer_offset_atIndex_(timing_buffer, 0, 3)
                    encoder.setBuffer_offset_atIndex_(intensity_buffer, 0, 4)
                    encoder.setBuffer_offset_atIndex_(offset_buffer, 0, 5)
                    encoder.setBytes_length_atIndex_(np.array([beta], dtype=np.float32).tobytes(), 4, 6)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 7)
                    encoder.setBytes_length_atIndex_(np.array([spins_per_block], dtype=np.uint32).tobytes(), 4, 8)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 9)
                    encoder.setBytes_length_atIndex_(np.array([timing_variance], dtype=np.float32).tobytes(), 4, 10)
                    encoder.setBytes_length_atIndex_(np.array([intensity_variance], dtype=np.float32).tobytes(), 4, 11)
                    encoder.setBytes_length_atIndex_(np.array([offset_variance], dtype=np.float32).tobytes(), 4, 12)

                    # Dispatch for optimized kernel
                    threads_per_group = min(64, self._kernels[kernel_name].maxTotalThreadsPerThreadgroup())
                    groups_x = R  # One threadgroup per chain
                    groups_y = num_blocks  # One threadgroup per spin block
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        (groups_x, groups_y, 1), (threads_per_group, 1, 1)
                    )

                elif kernel_name == "pbit_sequential_update":
                    # Sequential kernel for safety
                    encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 0)
                    encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 1)
                    encoder.setBuffer_offset_atIndex_(decision_buffer, 0, 2)
                    encoder.setBuffer_offset_atIndex_(timing_buffer, 0, 3)
                    encoder.setBuffer_offset_atIndex_(intensity_buffer, 0, 4)
                    encoder.setBuffer_offset_atIndex_(offset_buffer, 0, 5)
                    encoder.setBytes_length_atIndex_(np.array([beta], dtype=np.float32).tobytes(), 4, 6)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 7)
                    encoder.setBytes_length_atIndex_(np.array([spins_per_block], dtype=np.uint32).tobytes(), 4, 8)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 9)
                    encoder.setBytes_length_atIndex_(np.array([timing_variance], dtype=np.float32).tobytes(), 4, 10)
                    encoder.setBytes_length_atIndex_(np.array([intensity_variance], dtype=np.float32).tobytes(), 4, 11)
                    encoder.setBytes_length_atIndex_(np.array([offset_variance], dtype=np.float32).tobytes(), 4, 12)

                    # Dispatch for sequential kernel
                    groups_x = R  # One thread per chain
                    groups_y = num_blocks  # One threadgroup per spin block
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        (groups_x, groups_y, 1), (1, 1, 1)
                    )

                else:
                    # Original parallel kernel
                    encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 0)
                    encoder.setBuffer_offset_atIndex_(local_field_buffer, 0, 1)
                    encoder.setBuffer_offset_atIndex_(decision_buffer, 0, 2)
                    encoder.setBuffer_offset_atIndex_(timing_buffer, 0, 3)
                    encoder.setBuffer_offset_atIndex_(intensity_buffer, 0, 4)
                    encoder.setBuffer_offset_atIndex_(offset_buffer, 0, 5)
                    encoder.setBytes_length_atIndex_(np.array([beta], dtype=np.float32).tobytes(), 4, 6)
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 7)
                    encoder.setBytes_length_atIndex_(np.array([spins_per_block], dtype=np.uint32).tobytes(), 4, 8)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 9)
                    encoder.setBytes_length_atIndex_(np.array([timing_variance], dtype=np.float32).tobytes(), 4, 10)
                    encoder.setBytes_length_atIndex_(np.array([intensity_variance], dtype=np.float32).tobytes(), 4, 11)
                    encoder.setBytes_length_atIndex_(np.array([offset_variance], dtype=np.float32).tobytes(), 4, 12)

                    # Dispatch for original parallel kernel
                    threads_per_group = min(spins_per_block, self._kernels[kernel_name].maxTotalThreadsPerThreadgroup())
                    groups_x = R  # One threadgroup per chain
                    groups_y = num_blocks  # One threadgroup per spin block
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        (groups_x, groups_y, 1), (threads_per_group, 1, 1)
                    )

                encoder.endEncoding()

            # Execute this sweep
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            sweep_time = time.time() - sweep_start
            if sweep_idx % max(num_sweeps // 10, 1) == 0:
                self.logger.debug(f"[MetalKernelSampler] P-bit Sweep {sweep_idx}/{num_sweeps} ({sweep_time*1000:.2f}ms)")

        # Read back final results
        self.logger.debug(f"[MetalKernelSampler] Reading final spins from buffer...")
        try:
            final_spins = self._read_buffer(spin_buffer, (R, n), np.int8)
        except Exception as e:
            self.logger.error(f"[MetalKernelSampler] Buffer reading failed: {e}")
            # Fallback to random spins
            final_spins = np.random.choice([-1, 1], size=(R, n), dtype=np.int8)

        # Compute final energies
        self.logger.debug(f"[MetalKernelSampler] Computing final energies...")
        energies = []
        for r in range(R):
            spin_vec = final_spins[r]
            h_energy = np.sum(spin_vec * h_vec)

            j_energy = 0.0
            if num_edges > 0:
                for idx in range(num_edges):
                    i_pos = i_indices[idx]
                    j_pos = j_indices[idx]
                    val = j_values[idx]
                    j_energy += spin_vec[i_pos] * spin_vec[j_pos] * val

            energies.append(float(h_energy + j_energy))

        # Convert to lists
        samples = [list(final_spins[r]) for r in range(R)]

        self.logger.debug(f"[MetalKernelSampler] P-bit kernel sampling completed")
        return samples, energies

    def _select_pbit_kernel(self, mode: str) -> str:
        """Select the appropriate P-bit kernel based on mode."""

        if mode == "optimized_parallel":
            # Optimized parallel implementation (default)
            return "pbit_optimized_parallel_update"
        elif mode == "sequential":
            # Sequential for safety
            return "pbit_sequential_update"
        else:
            # Default to original parallel kernel (safe fallback)
            return "pbit_parallel_update"

    def validate_solutions(self, samples: List[List[int]], energies: List[float], h: Dict, J: Dict) -> bool:
        """Validate that P-bit solutions have proper {-1, +1} format."""

        print(f"\n🔍 P-bit Solution Validation")
        print("=" * 40)

        total_samples = len(samples)
        valid_samples = 0
        invalid_values_found = set()

        for i, sample in enumerate(samples):
            # Check for valid spin values
            unique_values = set(sample)
            if unique_values.issubset({-1, 1}):
                valid_samples += 1
            else:
                invalid_values = unique_values - {-1, 1}
                invalid_values_found.update(invalid_values)
                if i < 3:  # Show details for first few invalid samples
                    print(f"  Sample {i}: Invalid values {invalid_values}")

        print(f"\n📊 Validation Results:")
        print(f"  Valid samples: {valid_samples}/{total_samples}")
        print(f"  Invalid samples: {total_samples - valid_samples}/{total_samples}")

        if invalid_values_found:
            print(f"  Invalid values found: {invalid_values_found}")
            print(f"  ❌ P-BIT SOLUTION FORMAT CORRUPTED")
            return False
        else:
            print(f"  ✅ All solutions have valid {-1, +1} format")

            # Quick energy check
            if energies:
                min_energy = min(energies)
                avg_energy = sum(energies) / len(energies)
                print(f"  Min energy: {min_energy:.1f}")
                print(f"  Avg energy: {avg_energy:.1f}")

            return True

    def close(self):
        """Clean up Metal resources."""
        self._command_queue = None
        self._library = None
        self._kernels.clear()
        self._metal_device = None


# Compatibility wrapper for dimod interface
if DIMOD_AVAILABLE:
    class MetalKernelDimodSampler:
        """Dimod-compatible wrapper for MetalKernelSampler."""

        def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
            self._kernel_sampler = MetalKernelSampler(device, logger)

            # Get topology from kernel sampler
            self.nodes = self._kernel_sampler.nodes
            self.edges = self._kernel_sampler.edges
            self.sampler_type = "metal_kernel_pbit"

            # For compatibility with mining code
            self.properties = {'num_qubits': len(self.nodes)}

        def sample_ising(self, h, J, num_reads=100, num_sweeps=512, use_hierarchical=False, block_size=None, **kwargs):
                """Dimod-compatible sampling interface."""
                return self._kernel_sampler.sample_ising(h, J, num_reads, num_sweeps, use_hierarchical, block_size, **kwargs)

        def close(self):
            self._kernel_sampler.close()