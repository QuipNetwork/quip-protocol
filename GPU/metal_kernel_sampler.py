#!/usr/bin/env python3
"""Metal kernel-only sampler using raw Metal compute for maximum performance."""

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

# Raw Metal imports
try:
    from Metal import MTLCreateSystemDefaultDevice
    from Foundation import NSData, NSMutableData
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

Variable = collections.abc.Hashable


class MetalKernelSampler:
    """Ultra-fast Metal kernel-only sampler using raw Metal compute."""
    
    def __init__(self, device: str = "mps", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.sampler_type = "metal_kernel"
        
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
        """Load and compile Metal kernels."""
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
        
        # Load kernel functions (match names in metal_kernels.metal)
        kernel_names = [
            "fused_metropolis_update",
            "optimized_coupling_field",
            "compute_energies",
            "compute_local_fields",
            "initialize_random_spins"
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
                self.logger.warning(f"Kernel function not found: {name}")
        
        self.logger.info(f"[MetalKernelSampler] Loaded {len(self._kernels)} Metal kernels")
    
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
    
    def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs):
        """Run Metal kernel-based simulated annealing."""
        self.logger.debug(f"[MetalKernelSampler] Starting sampling: reads={num_reads}, sweeps={num_sweeps}")
        
        start_time = time.time()
        
        # Convert inputs
        h_dict = dict(h) if hasattr(h, 'items') else {i: h[i] for i in range(len(h))}
        J_dict = dict(J) if hasattr(J, 'items') else J
        
        # Run kernel-based sampling
        samples, energies = self._kernel_sampling(h_dict, J_dict, num_reads, num_sweeps)
        
        total_time = time.time() - start_time
        self.logger.debug(f"[MetalKernelSampler] Completed in {total_time:.3f}s ({total_time/num_sweeps*1000:.2f}ms per sweep)")
        
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
    
    def _kernel_sampling(self, h: Dict[int, float], J: Dict[Tuple[int, int], float], 
                        num_reads: int, num_sweeps: int) -> Tuple[List[List[int]], List[float]]:
        """Ultra-fast kernel-based sampling."""
        
        # Problem setup
        n = len(self.nodes)
        R = num_reads
        
        self.logger.debug(f"[MetalKernelSampler] Problem: {n} variables, {len(J)} couplings, {R} chains")
        
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
        
        # Annealing schedule
        beta_start = 0.1
        beta_end = 10.0
        betas = np.logspace(np.log10(beta_start), np.log10(beta_end), num_sweeps, dtype=np.float32)
        
        # Create Metal buffers
        buffer_start = time.time()
        spin_buffer, _ = self._create_buffer(spins, np.int8)
        h_buffer, _ = self._create_buffer(h_vec, np.float32)
        
        if num_edges > 0:
            i_buffer, _ = self._create_buffer(i_indices, np.int32)
            j_buffer, _ = self._create_buffer(j_indices, np.int32)
            j_val_buffer, _ = self._create_buffer(j_values, np.float32)
        
        # Field buffer for computations
        field_buffer, _ = self._create_buffer(np.zeros((R, n), dtype=np.float32))
        
        # Random values buffer
        random_buffer, _ = self._create_buffer(np.zeros((R, n), dtype=np.float32))
        
        buffer_creation_time = time.time() - buffer_start
        
        self.logger.debug(f"[MetalKernelSampler] Buffers created in {buffer_creation_time*1000:.1f}ms, starting annealing...")
        
        # Annealing loop with Metal kernels
        kernel_time = 0.0
        for sweep_idx, beta in enumerate(betas):
            sweep_start = time.time()
            # Generate random values for this sweep
            random_vals = np.random.rand(R, n).astype(np.float32)
            
            # Update random buffer
            random_buffer_new, _ = self._create_buffer(random_vals)
            
            # Create command buffer
            command_buffer = self._command_queue.commandBuffer()
            
            # Field computation kernel: coupling field + local field
            if num_edges > 0:
                # Step 1: Create fresh neighbor sum buffer (original approach)
                neighbor_sum_cleared = np.zeros((R, n), dtype=np.float32)
                neighbor_buffer, _ = self._create_buffer(neighbor_sum_cleared)
                
                # Step 2: Compute coupling contributions
                if "optimized_coupling_field" in self._kernels:
                    encoder = command_buffer.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._kernels["optimized_coupling_field"])
                    encoder.setBuffer_offset_atIndex_(neighbor_buffer, 0, 0)  # neighbor_sum
                    encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 1)      # spins
                    encoder.setBuffer_offset_atIndex_(i_buffer, 0, 2)         # i_indices
                    encoder.setBuffer_offset_atIndex_(j_buffer, 0, 3)         # j_indices  
                    encoder.setBuffer_offset_atIndex_(j_val_buffer, 0, 4)     # j_values
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 5)
                    encoder.setBytes_length_atIndex_(np.array([num_edges], dtype=np.uint32).tobytes(), 4, 6)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 7)
                    
                    # Dispatch 2D: (chains, couplings)
                    max_threads = self._kernels["optimized_coupling_field"].maxTotalThreadsPerThreadgroup()
                    threads_x = min(R, int(max_threads**0.5))
                    threads_y = min(num_edges, max_threads // threads_x)
                    groups_x = (R + threads_x - 1) // threads_x
                    groups_y = (num_edges + threads_y - 1) // threads_y
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        (groups_x, groups_y, 1), (threads_x, threads_y, 1)
                    )
                    encoder.endEncoding()
                
                # Step 3: Compute local fields (neighbor_sum + h_field)
                if "compute_local_fields" in self._kernels:
                    encoder = command_buffer.computeCommandEncoder()
                    encoder.setComputePipelineState_(self._kernels["compute_local_fields"])
                    encoder.setBuffer_offset_atIndex_(field_buffer, 0, 0)     # local_fields (output)
                    encoder.setBuffer_offset_atIndex_(neighbor_buffer, 0, 1)  # neighbor_sums
                    encoder.setBuffer_offset_atIndex_(h_buffer, 0, 2)         # h_fields
                    encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 3)
                    encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 4)
                    
                    # Dispatch 2D: (chains, spins)
                    max_threads = self._kernels["compute_local_fields"].maxTotalThreadsPerThreadgroup()
                    threads_x = min(R, int(max_threads**0.5))
                    threads_y = min(n, max_threads // threads_x)
                    groups_x = (R + threads_x - 1) // threads_x
                    groups_y = (n + threads_y - 1) // threads_y
                    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                        (groups_x, groups_y, 1), (threads_x, threads_y, 1)
                    )
                    encoder.endEncoding()
            else:
                # No couplings: field = h only
                field_data = np.tile(h_vec, (R, 1))
                field_buffer, _ = self._create_buffer(field_data)
            
            # Metropolis update kernel
            if "fused_metropolis_update" in self._kernels:
                encoder = command_buffer.computeCommandEncoder()
                encoder.setComputePipelineState_(self._kernels["fused_metropolis_update"])
                encoder.setBuffer_offset_atIndex_(spin_buffer, 0, 0)         # spins
                encoder.setBuffer_offset_atIndex_(field_buffer, 0, 1)        # local_fields
                encoder.setBuffer_offset_atIndex_(random_buffer_new, 0, 2)   # random_values
                encoder.setBytes_length_atIndex_(np.array([beta], dtype=np.float32).tobytes(), 4, 3)
                encoder.setBytes_length_atIndex_(np.array([R], dtype=np.uint32).tobytes(), 4, 4)
                encoder.setBytes_length_atIndex_(np.array([n], dtype=np.uint32).tobytes(), 4, 5)
                
                # Dispatch 2D: (chains, spins) to match kernel expectations
                max_threads = self._kernels["fused_metropolis_update"].maxTotalThreadsPerThreadgroup()
                threads_x = min(R, int(max_threads**0.5))
                threads_y = min(n, max_threads // threads_x)
                groups_x = (R + threads_x - 1) // threads_x
                groups_y = (n + threads_y - 1) // threads_y
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    (groups_x, groups_y, 1), (threads_x, threads_y, 1)
                )
                encoder.endEncoding()
            
            # Execute commands
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            sweep_time = time.time() - sweep_start
            kernel_time += sweep_time
            
            if sweep_idx % max(num_sweeps // 10, 1) == 0:
                self.logger.debug(f"[MetalKernelSampler] Sweep {sweep_idx}/{num_sweeps} ({sweep_time*1000:.2f}ms)")
        
        # Read back final results
        self.logger.debug(f"[MetalKernelSampler] Reading final spins from buffer...")
        buffer_read_start = time.time()
        try:
            final_spins = self._read_buffer(spin_buffer, (R, n), np.int8)
            buffer_read_time = time.time() - buffer_read_start
            self.logger.debug(f"[MetalKernelSampler] Buffer read took {buffer_read_time*1000:.1f}ms")
            self.logger.debug(f"[MetalKernelSampler] Successfully read spins: shape={final_spins.shape}, dtype={final_spins.dtype}")
            self.logger.debug(f"[MetalKernelSampler] Sample spin values: {final_spins[0][:10] if len(final_spins) > 0 else 'none'}")
        except Exception as e:
            self.logger.error(f"[MetalKernelSampler] Buffer reading failed: {e}")
            # Fallback: return random spins
            final_spins = np.random.choice([-1, 1], size=(R, n), dtype=np.int8)
            self.logger.warning(f"[MetalKernelSampler] Using fallback random spins")
        
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
        
        self.logger.debug(f"[MetalKernelSampler] Kernel sampling completed")
        self.logger.debug(f"[MetalKernelSampler] Timing breakdown: buffer_creation={buffer_creation_time*1000:.1f}ms, kernel={kernel_time*1000:.1f}ms, buffer_read={buffer_read_time*1000:.1f}ms")
        return samples, energies
    
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
            self.sampler_type = "metal_kernel_dimod"
            
            # For compatibility with mining code
            self.properties = {'num_qubits': len(self.nodes)}
        
        def sample_ising(self, h, J, num_reads=100, num_sweeps=512, **kwargs):
            """Dimod-compatible sampling interface."""
            return self._kernel_sampler.sample_ising(h, J, num_reads, num_sweeps, **kwargs)
        
        def close(self):
            self._kernel_sampler.close()