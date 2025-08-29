"""GPU worker process for local PyTorch-based mining."""

import os
import math

# Optional torch import
try:
    import torch
except ImportError:
    torch = None

from shared.miner import Miner, MiningResult


def gpu_mine_block_process(miner_data, block_header: str, result_queue, stop_event):
    """GPU-specific mining process function.
    
    Args:
        miner_data: Serialized miner data (type, id, config)
        block_header: Block header to mine
        result_queue: Queue to put results
        stop_event: Event to signal stop
    """
    miner_type = miner_data['type']
    miner_id = miner_data['id']
    miner_config = miner_data.get('config', {})
    
    # Create appropriate GPU sampler
    if miner_type.startswith('GPU-LOCAL'):
        # Local GPU miner - import locally to avoid circular import
        from .sampler import LocalGPUSampler
        sampler = LocalGPUSampler(miner_type.split(':')[1])
    elif miner_type.startswith('GPU-MODAL'):
        # Modal GPU miner - import locally to avoid circular import
        from .modal_sampler import ModalSampler
        sampler = ModalSampler(miner_type.split(':')[1])
    else:
        raise ValueError(f"Unknown GPU miner type: {miner_type}")
    
    miner = Miner(
        miner_id, 
        miner_type, 
        sampler, 
        difficulty_energy=miner_config['difficulty_energy'],
        min_diversity=miner_config['min_diversity'],
        min_solutions=miner_config['min_solutions']
    )
    
    # Call the original Miner.mine_block method
    miner.mine_block(block_header, result_queue, stop_event)


def gpu_worker_main(req_q, resp_q, device_str: str):
    """GPU worker main function for processing mining requests."""
    debug = os.getenv("QUIP_DEBUG") == "1"
    
    if torch is None:
        resp_q.put({"status": "error", "message": "PyTorch not available"})
        return
    
    # Resolve device
    dev: torch.device
    if device_str.lower() == "mps" or (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        dev = torch.device("mps")
    else:
        # assume CUDA ordinal
        idx = int(device_str)
        if not torch.cuda.is_available():
            resp_q.put({"status": "error", "message": "CUDA not available"})
            return
        if idx < 0 or idx >= torch.cuda.device_count():
            resp_q.put({"status": "error", "message": f"Invalid CUDA device index {idx}"})
            return
        dev = torch.device(f"cuda:{idx}")

    if debug:
        print(f"[GPU worker pid={os.getpid()}] start device={dev}", flush=True)

    while True:
        msg = req_q.get()
        if not isinstance(msg, dict):
            continue
        if msg.get("op") == "stop":
            if debug:
                print(f"[GPU worker pid={os.getpid()}] stop", flush=True)
            break
        if msg.get("op") != "sample":
            continue
        try:
            if debug:
                print(f"[GPU worker pid={os.getpid()}] received sample", flush=True)
            h = msg["h"]
            J = msg["J"]
            num_reads = int(msg.get("num_reads", 100))
            num_sweeps = int(msg.get("num_sweeps", 512))
            # Build tensors
            n = 0
            if h:
                n = max(n, max(h.keys()) + 1)
            if J:
                n = max(n, max(max(i, j) for (i, j) in J.keys()) + 1)
            if n <= 0:
                resp_q.put({"status": "error", "message": "Invalid problem size"})
                continue
            h_vec = torch.zeros(n, device=dev, dtype=torch.float32)
            for i, v in h.items():
                h_vec[i] = float(v)
            if J:
                i_idx = torch.tensor([ij[0] for ij in J.keys()], device=dev, dtype=torch.long)
                j_idx = torch.tensor([ij[1] for ij in J.keys()], device=dev, dtype=torch.long)
                j_vals = torch.tensor([float(v) for v in J.values()], device=dev, dtype=torch.float32)
            else:
                i_idx = j_idx = j_vals = None
            # Generate random spins {-1,1}
            spins = (torch.rand((num_reads, n), device=dev) > 0.5).to(torch.int8)
            spins = spins * 2 - 1  # {0,1} -> {-1,1}

            # Simulated annealing using edge list (no dense J)
            # Match CPU SA semantics: one sweep ≈ n spin updates per read.
            # Use geometric beta schedule to mimic D-Wave SA behavior.
            betas = torch.exp(torch.linspace(math.log(0.1), math.log(10.0), steps=num_sweeps, device=dev, dtype=torch.float32))
            R = num_reads
            ar = torch.arange(R, device=dev)
            updates_per_sweep = int(os.getenv("QUIP_GPU_UPDATES_PER_SWEEP", str(n)))
            recompute_interval = int(os.getenv("QUIP_GPU_RECOMPUTE_INTERVAL", "64"))

            for beta in betas:
                # Initial local field at this temperature
                if i_idx is not None:
                    sp_f = spins.to(torch.float32)
                    neighbor_sum = torch.zeros((R, n), device=dev, dtype=torch.float32)
                    # Contributions from edges (i -> j) and (j -> i)
                    neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                    neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
                else:
                    neighbor_sum = torch.zeros((R, n), device=dev, dtype=torch.float32)
                local_field = neighbor_sum + h_vec  # broadcasts h_vec over reads

                # Perform ~n updates per sweep with periodic field recomputation
                t = 0
                while t < updates_per_sweep:
                    chunk = min(recompute_interval, updates_per_sweep - t)
                    for _ in range(chunk):
                        idx = torch.randint(0, n, (R,), device=dev)
                        s_i = spins[ar, idx].to(torch.float32)
                        lf_i = local_field[ar, idx]
                        delta_e = 2.0 * s_i * lf_i
                        accept = (delta_e < 0) | (torch.rand(R, device=dev) < torch.exp(-beta * delta_e))
                        # Flip accepted spins
                        flips = torch.where(accept, torch.tensor(-1, dtype=spins.dtype, device=dev), torch.tensor(1, dtype=spins.dtype, device=dev))
                        spins[ar, idx] = spins[ar, idx] * flips
                    # Refresh local field after a chunk of updates
                    if i_idx is not None:
                        sp_f = spins.to(torch.float32)
                        neighbor_sum = torch.zeros((R, n), device=dev, dtype=torch.float32)
                        neighbor_sum.scatter_add_(1, i_idx.unsqueeze(0).expand(R, -1), sp_f[:, j_idx] * j_vals)
                        neighbor_sum.scatter_add_(1, j_idx.unsqueeze(0).expand(R, -1), sp_f[:, i_idx] * j_vals)
                    else:
                        neighbor_sum = torch.zeros((R, n), device=dev, dtype=torch.float32)
                    local_field = neighbor_sum + h_vec
                    t += chunk

            # Compute final energies with correct Ising sign convention
            sp_f = spins.to(torch.float32)
            energies = - (sp_f * h_vec).sum(dim=1)
            if i_idx is not None:
                prod = sp_f[:, i_idx] * sp_f[:, j_idx]
                energies = energies - (prod * j_vals).sum(dim=1)
            # Move to CPU lists
            resp_q.put({
                "samples": spins.to("cpu").to(torch.int8).tolist(),
                "energies": energies.to("cpu").to(torch.float32).tolist(),
            })
            if debug:
                print(f"[GPU worker pid={os.getpid()}] responded", flush=True)
        except Exception as e:
            resp_q.put({"status": "error", "message": str(e)})