#!/usr/bin/env python3
"""Compare mining rates across different hardware at fixed difficulty."""
import argparse
import json
import logging
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict

# IMPORTANT: Set spawn method before any CUDA operations
# Fork doesn't work with CUDA - child processes can't reinitialize CUDA contexts
# This must be called before any other multiprocessing code
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging to capture miner logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True  # Force reconfiguration if already configured
)

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from dataclasses import dataclass

from shared.block import Block, BlockHeader, BlockRequirements, create_genesis_block
from shared.miner_types import MiningResult
from shared.time_utils import utc_timestamp
from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies import load_topology

import os
from typing import Any, Optional


def detect_hardware() -> Dict[str, Any]:
    """Auto-detect available hardware for mining.

    Returns:
        Dict with keys: 'cpu_count', 'cuda_devices', 'has_metal'
    """
    result = {
        'cpu_count': os.cpu_count() or 1,
        'cuda_devices': [],
        'has_metal': False
    }

    # Detect CUDA GPUs - actually validate each device works
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        for device_id in range(device_count):
            try:
                # Actually try to use the device to validate it works
                cp.cuda.Device(device_id).use()
                # Try a simple allocation to confirm driver compatibility
                _ = cp.zeros(1)
                result['cuda_devices'].append(device_id)
            except Exception as e:
                print(f"   CUDA device {device_id} not usable: {e}")
    except ImportError:
        pass  # cupy not installed
    except Exception as e:
        print(f"   CUDA detection failed: {e}")

    # Detect Metal (macOS)
    try:
        import torch
        if torch.backends.mps.is_available():
            result['has_metal'] = True
    except (ImportError, Exception):
        pass

    return result


def build_miner_specs(
    miner_type: str,
    hardware: Dict[str, Any],
    num_cpus: Optional[int] = None,
    devices: Optional[str] = None
) -> List[Dict]:
    """Build list of miner specs based on hardware and CLI args.

    Args:
        miner_type: Type of miner (cpu, cuda, metal, qpu)
        hardware: Dict from detect_hardware()
        num_cpus: Optional limit on CPU miners
        devices: Optional comma-separated CUDA device IDs

    Returns:
        List of spec dicts with 'kind', 'id', and optional 'args'
    """
    specs = []

    if miner_type == 'cpu':
        count = num_cpus if num_cpus is not None else hardware['cpu_count']
        for i in range(count):
            specs.append({'kind': 'cpu', 'id': f'rate-test-cpu-{i}'})

    elif miner_type == 'cuda':
        if devices is not None:
            device_ids = [d.strip() for d in devices.split(',')]
        else:
            device_ids = [str(d) for d in hardware['cuda_devices']]

        if not device_ids:
            # Fallback to device 0 if no CUDA detected but user requested cuda
            device_ids = ['0']

        for device in device_ids:
            specs.append({
                'kind': 'cuda',
                'id': f'rate-test-cuda-{device}',
                'args': {'device': device}
            })

    elif miner_type == 'metal':
        # Metal only supports single GPU
        specs.append({'kind': 'metal', 'id': 'rate-test-metal-0'})

    elif miner_type == 'qpu':
        # QPU is single instance
        specs.append({'kind': 'qpu', 'id': 'rate-test-qpu-0'})

    return specs


def aggregate_results(miner_results: List[Dict], total_time: float) -> Dict:
    """Aggregate results from all miners into unified statistics.

    Args:
        miner_results: List of result dicts from each miner worker
        total_time: Total elapsed wall-clock time

    Returns:
        Aggregated statistics dict
    """
    total_blocks = sum(r.get('blocks_found', 0) for r in miner_results)
    total_attempts = sum(r.get('attempts', 0) for r in miner_results)

    all_energies = []
    all_diversities = []
    all_solution_counts = []
    all_mining_times = []

    for r in miner_results:
        all_energies.extend(r.get('energies', []))
        all_diversities.extend(r.get('diversities', []))
        all_solution_counts.extend(r.get('solution_counts', []))
        all_mining_times.extend(r.get('mining_times', []))

    return {
        'num_miners': len(miner_results),
        'per_miner_stats': miner_results,
        'total_blocks_found': total_blocks,
        'total_attempts': total_attempts,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'success_rate': total_blocks / total_attempts if total_attempts > 0 else 0,
        'blocks_per_minute': total_blocks / (total_time / 60) if total_time > 0 else 0,
        'energy_stats': {
            'min': min(all_energies) if all_energies else None,
            'max': max(all_energies) if all_energies else None,
            'mean': sum(all_energies) / len(all_energies) if all_energies else None,
            'all_energies': all_energies
        },
        'diversity_stats': {
            'min': min(all_diversities) if all_diversities else None,
            'max': max(all_diversities) if all_diversities else None,
            'mean': sum(all_diversities) / len(all_diversities) if all_diversities else None
        },
        'solution_count_stats': {
            'min': min(all_solution_counts) if all_solution_counts else None,
            'max': max(all_solution_counts) if all_solution_counts else None,
            'mean': sum(all_solution_counts) / len(all_solution_counts) if all_solution_counts else None
        },
        'mining_time_stats': {
            'min': min(all_mining_times) if all_mining_times else None,
            'max': max(all_mining_times) if all_mining_times else None,
            'mean': sum(all_mining_times) / len(all_mining_times) if all_mining_times else None,
            'all_times': all_mining_times
        }
    }


def mine_worker(
    miner_spec: Dict,
    difficulty_energy: float,
    duration_minutes: float,
    min_diversity: float,
    min_solutions: int,
    topology_name: Optional[str],
    result_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event
):
    """Worker function for parallel mining.

    Args:
        miner_spec: Dict with 'kind', 'id', 'args' for miner creation
        difficulty_energy: Fixed difficulty threshold
        duration_minutes: How long to mine
        min_diversity: Minimum solution diversity
        min_solutions: Minimum solutions required
        topology_name: Topology name to load (None for default)
        result_queue: Queue to send results back
        stop_event: Shared event to signal stop
    """
    # Build miner from spec
    kind = miner_spec['kind']
    miner_id = miner_spec['id']

    # CUDA debugging for worker processes
    if kind == 'cuda':
        device = miner_spec.get('args', {}).get('device', '0')
        print(f"   [{miner_id}] CUDA worker starting (PID: {os.getpid()})")
        print(f"   [{miner_id}] Target device: {device}")

        # Check environment
        print(f"   [{miner_id}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"   [{miner_id}] LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')[:100]}...")

        # Try to import and initialize CuPy fresh in this process
        try:
            import cupy as cp
            print(f"   [{miner_id}] CuPy imported: {cp.__version__}")

            # Check CUDA runtime
            try:
                runtime_ver = cp.cuda.runtime.runtimeGetVersion()
                print(f"   [{miner_id}] CUDA runtime version: {runtime_ver}")
            except Exception as e:
                print(f"   [{miner_id}] CUDA runtime query failed: {e}")

            # Try to get device count
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                print(f"   [{miner_id}] Device count: {device_count}")
            except Exception as e:
                print(f"   [{miner_id}] Device count query failed: {e}")
                result_queue.put({'error': f'CUDA device count failed: {e}', 'miner_id': miner_id})
                return

            # Try to select the device
            try:
                cp.cuda.Device(int(device)).use()
                print(f"   [{miner_id}] Device {device} selected OK")
            except Exception as e:
                print(f"   [{miner_id}] Device {device} selection failed: {e}")
                result_queue.put({'error': f'CUDA device {device} selection failed: {e}', 'miner_id': miner_id})
                return

            # Try a simple allocation
            try:
                test_arr = cp.zeros(1)
                del test_arr
                print(f"   [{miner_id}] Memory allocation test OK")
            except Exception as e:
                print(f"   [{miner_id}] Memory allocation failed: {e}")
                result_queue.put({'error': f'CUDA memory allocation failed: {e}', 'miner_id': miner_id})
                return

        except ImportError as e:
            print(f"   [{miner_id}] CuPy import failed: {e}")
            result_queue.put({'error': f'CuPy import failed: {e}', 'miner_id': miner_id})
            return
        except Exception as e:
            print(f"   [{miner_id}] CUDA init failed: {e}")
            result_queue.put({'error': f'CUDA init failed: {e}', 'miner_id': miner_id})
            return

    # Load topology in worker process
    if topology_name:
        topology = load_topology(topology_name)
    else:
        topology = DEFAULT_TOPOLOGY

    try:
        if kind == 'cpu':
            from CPU.sa_miner import SimulatedAnnealingMiner
            miner = SimulatedAnnealingMiner(miner_id=miner_id, topology=topology)
        elif kind == 'cuda':
            from GPU.cuda_miner import CudaMiner
            device = miner_spec.get('args', {}).get('device', '0')
            miner = CudaMiner(miner_id=miner_id, device=device, topology=topology)
        elif kind == 'metal':
            from GPU.metal_miner import MetalMiner
            miner = MetalMiner(miner_id=miner_id, topology=topology)
        elif kind == 'qpu':
            from QPU.dwave_miner import DWaveMiner
            miner = DWaveMiner(miner_id=miner_id, topology=topology, qpu_timeout=0.0)
        else:
            result_queue.put({'error': f'Unknown miner kind: {kind}', 'miner_id': miner_id})
            return
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"   [{miner_id}] Miner creation failed:\n{tb}")
        result_queue.put({'error': str(e), 'miner_id': miner_id})
        return

    # Run mining
    requirements = BlockRequirements(
        difficulty_energy=difficulty_energy,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        timeout_to_difficulty_adjustment_decay=0
    )

    node_info = NodeInfo(miner_id=miner_id)
    blocks_found = []
    attempts = 0
    start_time = time.time()
    prev_block = create_genesis_block()
    prev_block.next_block_requirements = requirements

    # Helper to submit current results (can be called multiple times)
    def submit_results():
        total_time = time.time() - start_time
        result_queue.put({
            'miner_id': miner_id,
            'miner_type': miner.miner_type,
            'blocks_found': len(blocks_found),
            'attempts': attempts,
            'total_time': total_time,
            'energies': [b.energy for b in blocks_found],
            'diversities': [b.diversity for b in blocks_found],
            'solution_counts': [b.num_valid for b in blocks_found],
            'mining_times': [b.mining_time for b in blocks_found if b.mining_time]
        })

    # For CUDA miners, use drain_on_stop to collect all pending GPU results
    use_drain = kind == 'cuda'

    while not stop_event.is_set():
        attempts += 1

        # Build kwargs for mine_block (CUDA miner supports drain_on_stop)
        mine_kwargs = {
            'prev_block': prev_block,
            'node_info': node_info,
            'requirements': requirements,
            'prev_timestamp': prev_block.header.timestamp,
            'stop_event': stop_event,
        }
        if use_drain:
            mine_kwargs['drain_on_stop'] = True

        result = miner.mine_block(**mine_kwargs)

        if result:
            blocks_found.append(result)
            print(f"   [{miner_id}] Block {len(blocks_found)} found! Energy: {result.energy:.1f}")

        # Submit results after each attempt (so we don't lose them if terminated)
        # Only keep the latest result in the queue (drain old ones first won't work
        # across processes, so main process will take the last one)
        submit_results()

    # Final submission (in case stop_event was set between attempts)
    submit_results()

    # Cleanup CUDA resources to allow process to exit cleanly
    if kind == 'cuda':
        try:
            # Stop the persistent kernel
            if hasattr(miner, 'async_sampler') and hasattr(miner.async_sampler, 'stop_immediate'):
                miner.async_sampler.stop_immediate()
            # Reset CUDA device to release all resources
            import cupy as cp
            cp.cuda.Device(int(miner_spec.get('args', {}).get('device', '0'))).use()
            cp.cuda.runtime.deviceReset()
        except Exception as e:
            print(f"   [{miner_id}] CUDA cleanup warning: {e}")


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to minutes.

    Supports: 30s, 5m, 2h, 1d, 1w
    Examples:
        "30s" -> 0.5 (minutes)
        "5m" -> 5.0
        "2h" -> 120.0
        "1d" -> 1440.0
        "1w" -> 10080.0
    """
    duration_str = duration_str.strip().lower()

    if duration_str.endswith('s'):
        return int(duration_str[:-1]) / 60.0
    elif duration_str.endswith('m'):
        return float(duration_str[:-1])
    elif duration_str.endswith('h'):
        return int(duration_str[:-1]) * 60.0
    elif duration_str.endswith('d'):
        return int(duration_str[:-1]) * 1440.0
    elif duration_str.endswith('w'):
        return int(duration_str[:-1]) * 10080.0
    else:
        # Try parsing as raw minutes
        return float(duration_str)


@dataclass
class NodeInfo:
    """Simple node info for testing."""
    miner_id: str


def mine_continuous(
    miner,
    difficulty_energy: float,
    duration_minutes: float,
    min_diversity: float = 0.15,
    min_solutions: int = 5,
    log_file = None
) -> Dict:
    """Mine continuously for specified duration at fixed difficulty.

    Args:
        miner: Miner instance (CPU, CUDA, QPU, etc.)
        difficulty_energy: Fixed difficulty energy threshold
        duration_minutes: How long to mine (in minutes)
        min_diversity: Minimum solution diversity requirement
        min_solutions: Minimum number of solutions required
        log_file: Optional file object to write logs to

    Returns:
        Dictionary with mining statistics
    """
    def log(msg):
        """Print to console and optionally to log file."""
        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    log(f"\n⛏️  Starting continuous mining:")
    log(f"   Duration: {duration_minutes} minutes")
    log(f"   Difficulty: {difficulty_energy:.1f}")
    log(f"   Min diversity: {min_diversity}")
    log(f"   Min solutions: {min_solutions}")

    # Setup
    requirements = BlockRequirements(
        difficulty_energy=difficulty_energy,
        min_diversity=min_diversity,
        min_solutions=min_solutions,
        timeout_to_difficulty_adjustment_decay=0  # Disable decay
    )

    node_info = NodeInfo(miner_id=f"rate-test-{miner.miner_type}-0")
    blocks_found: List[MiningResult] = []
    attempts = 0
    start_time = time.time()
    duration_seconds = duration_minutes * 60

    # QPU time tracking
    total_qpu_time_us = 0.0  # Total QPU time in microseconds

    # Progress tracking
    last_progress_time = start_time
    progress_interval = 60  # Print progress every minute

    log(f"\n⏱️  Mining started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    log(f"   Will run until {time.strftime('%H:%M:%S', time.localtime(start_time + duration_seconds))}")

    # Create genesis block as prev_block
    prev_block = create_genesis_block()
    # Update requirements for testing
    prev_block.next_block_requirements = requirements

    # Create stop event that persists across mining attempts
    stop_event = multiprocessing.Event()

    # Start a timer thread to set stop_event after duration expires
    def timer_thread():
        time.sleep(duration_seconds)
        log(f"\n✅ Duration limit reached ({duration_minutes} min)")
        stop_event.set()

    timer = threading.Thread(target=timer_thread, daemon=True)
    timer.start()

    while True:
        # Check if stop event was set by timer
        if stop_event.is_set():
            break

        # Progress update
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            elapsed = current_time - start_time
            elapsed_min = elapsed / 60
            blocks_per_min = len(blocks_found) / elapsed_min if elapsed_min > 0 else 0
            log(f"   [{elapsed_min:.1f}/{duration_minutes:.0f} min] "
                  f"Blocks: {len(blocks_found)}, "
                  f"Attempts: {attempts}, "
                  f"Rate: {blocks_per_min:.2f} blocks/min")
            last_progress_time = current_time

        attempts += 1
        prev_timestamp = prev_block.header.timestamp

        # Mine the block (using shared stop_event)
        attempt_start = time.time()
        result = miner.mine_block(
            prev_block=prev_block,
            node_info=node_info,
            requirements=requirements,
            prev_timestamp=prev_timestamp,
            stop_event=stop_event
        )
        attempt_time = time.time() - attempt_start

        # Check if stop event was set during mining
        if stop_event.is_set():
            break

        # Track QPU time for this attempt (if available)
        attempt_qpu_time_us = 0.0
        if hasattr(miner, 'timing_stats') and 'qpu_access_time' in miner.timing_stats:
            # Get the most recent QPU access time
            if miner.timing_stats['qpu_access_time']:
                attempt_qpu_time_us = miner.timing_stats['qpu_access_time'][-1]
                total_qpu_time_us += attempt_qpu_time_us

        if result:
            blocks_found.append(result)
            qpu_time_msg = f", QPU: {attempt_qpu_time_us / 1e6:.3f}s, Total QPU: {total_qpu_time_us / 1e6:.2f}s" if attempt_qpu_time_us > 0 else ""
            log(f"   ✅ Block {len(blocks_found)} found! "
                  f"Energy: {result.energy:.1f}, "
                  f"Time: {attempt_time:.1f}s, "
                  f"Diversity: {result.diversity:.3f}, "
                  f"Solutions: {result.num_valid}"
                  f"{qpu_time_msg}")

    total_time = time.time() - start_time

    # Compute statistics
    energies = [b.energy for b in blocks_found]
    diversities = [b.diversity for b in blocks_found]
    solution_counts = [b.num_valid for b in blocks_found]
    mining_times = [b.mining_time for b in blocks_found if b.mining_time]

    stats = {
        'total_blocks_found': len(blocks_found),
        'total_attempts': attempts,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'success_rate': len(blocks_found) / attempts if attempts > 0 else 0,
        'blocks_per_minute': len(blocks_found) / (total_time / 60) if total_time > 0 else 0,
        'energy_stats': {
            'min': min(energies) if energies else None,
            'max': max(energies) if energies else None,
            'mean': sum(energies) / len(energies) if energies else None,
            'all_energies': energies
        },
        'diversity_stats': {
            'min': min(diversities) if diversities else None,
            'max': max(diversities) if diversities else None,
            'mean': sum(diversities) / len(diversities) if diversities else None
        },
        'solution_count_stats': {
            'min': min(solution_counts) if solution_counts else None,
            'max': max(solution_counts) if solution_counts else None,
            'mean': sum(solution_counts) / len(solution_counts) if solution_counts else None
        },
        'mining_time_stats': {
            'min': min(mining_times) if mining_times else None,
            'max': max(mining_times) if mining_times else None,
            'mean': sum(mining_times) / len(mining_times) if mining_times else None,
            'all_times': mining_times
        },
        'qpu_time_stats': {
            'total_qpu_time_us': total_qpu_time_us,
            'total_qpu_time_seconds': total_qpu_time_us / 1e6,
            'avg_qpu_time_per_attempt_us': total_qpu_time_us / attempts if attempts > 0 else 0
        } if total_qpu_time_us > 0 else None
    }

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare mining rates at fixed difficulty'
    )
    parser.add_argument(
        '--miner-type',
        type=str,
        choices=['cpu', 'cuda', 'metal', 'qpu'],
        required=True,
        help='Miner type to test'
    )
    parser.add_argument(
        '--difficulty-energy',
        type=float,
        required=True,
        help='Fixed difficulty energy threshold (e.g., -15450.0)'
    )
    parser.add_argument(
        '--duration',
        type=str,
        default='10m',
        help='Mining duration (default: 10m). Examples: 30s, 5m, 2h, 1d, 1w'
    )
    parser.add_argument(
        '--min-diversity',
        type=float,
        default=0.15,
        help='Minimum solution diversity (default: 0.15)'
    )
    parser.add_argument(
        '--min-solutions',
        type=int,
        default=5,
        help='Minimum number of solutions (default: 5)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device ID (for cuda miner, default: 0)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default=None,
        help='Topology name (default: DEFAULT_TOPOLOGY=Z(9,2)). Examples: "Z(9,2)", "Z(10,2)", "Advantage2_system1.8"'
    )
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=None,
        help='Number of CPU miners to spawn (default: auto-detect all cores)'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default=None,
        help='Comma-separated CUDA device IDs (e.g., "0,1,2"). Default: all detected GPUs'
    )

    args = parser.parse_args()

    # Parse duration
    try:
        duration_minutes = parse_duration(args.duration)
    except (ValueError, IndexError):
        print(f"❌ Invalid duration format: '{args.duration}'. Use formats like: 30s, 5m, 2h, 1d, 1w")
        return 1

    # Parse topology if specified
    if args.topology:
        try:
            topology = load_topology(args.topology)
            print(f"✅ Loaded topology: {topology.solver_name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Failed to load topology '{args.topology}': {e}")
            return 1
    else:
        topology = DEFAULT_TOPOLOGY

    # Detect hardware
    hardware = detect_hardware()

    print("🔬 Mining Rate Comparison Tool")
    print("=" * 50)
    print(f"Miner type: {args.miner_type.upper()}")
    print(f"Topology: {topology.solver_name} ({len(topology.nodes)} nodes, {len(topology.edges)} edges)")
    print(f"Difficulty: {args.difficulty_energy:.1f}")
    print(f"Duration: {args.duration} ({duration_minutes:.1f} minutes)")

    print(f"\nDetected hardware:")
    print(f"   CPU cores: {hardware['cpu_count']}")
    print(f"   CUDA devices: {hardware['cuda_devices'] or 'none'}")
    print(f"   Metal (MPS): {'yes' if hardware['has_metal'] else 'no'}")

    # Check if requested hardware is available
    if args.miner_type == 'cuda' and not hardware['cuda_devices']:
        print(f"\n❌ No usable CUDA devices found.")
        return 1
    elif args.miner_type == 'metal' and not hardware['has_metal']:
        print(f"\n❌ Metal (MPS) not available.")
        return 1

    # Build miner specifications
    # For --devices, use it if provided; otherwise use legacy --device for single CUDA
    devices_arg = args.devices
    if devices_arg is None and args.miner_type == 'cuda' and args.device != '0':
        # Legacy single --device mode
        devices_arg = args.device

    miner_specs = build_miner_specs(
        miner_type=args.miner_type,
        hardware=hardware,
        num_cpus=args.num_cpus,
        devices=devices_arg
    )

    if not miner_specs:
        print(f"❌ No {args.miner_type} hardware detected or specified")
        return 1

    print(f"\nUsing {len(miner_specs)} miner(s):")
    for spec in miner_specs:
        device_info = f" (device {spec['args']['device']})" if 'args' in spec and 'device' in spec['args'] else ""
        print(f"   - {spec['id']}{device_info}")

    # Always use multiprocessing - subprocess exit cleans up CUDA resources properly
    result_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    # Spawn worker processes
    processes = []
    for spec in miner_specs:
        p = multiprocessing.Process(
            target=mine_worker,
            args=(
                spec,
                args.difficulty_energy,
                duration_minutes,
                args.min_diversity,
                args.min_solutions,
                args.topology,
                result_queue,
                stop_event
            )
        )
        p.start()
        processes.append(p)

    print(f"\n⛏️  Started {len(processes)} mining worker(s)")
    print(f"⏱️  Mining started at {time.strftime('%H:%M:%S')}")
    print(f"   Will run for {duration_minutes:.1f} minutes")

    # Start timer thread
    def timer_thread():
        time.sleep(duration_minutes * 60)
        print(f"\n✅ Duration limit reached ({args.duration})")
        stop_event.set()

    timer = threading.Thread(target=timer_thread, daemon=True)
    timer.start()

    # Monitor processes and stop_event (like MinerHandle.mine_with_timeout)
    start_time = time.time()
    miner_results_by_id = {}  # Keep latest result per miner (workers submit incrementally)

    def drain_queue():
        """Drain results from queue, keeping latest per miner_id."""
        while True:
            try:
                result = result_queue.get_nowait()
                miner_id = result.get('miner_id')
                if miner_id:
                    miner_results_by_id[miner_id] = result  # Keep latest
            except Exception:
                break

    try:
        # Wait for workers to exit gracefully
        # Workers check stop_event in their loop and will exit after current iteration
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)  # Check every 100ms
            drain_queue()  # Periodically drain to get incremental results

            # If stop_event is set, give workers time to finish current iteration
            if stop_event.is_set():
                # Wait up to 180 seconds for graceful shutdown (mining iteration can take 2+ min)
                shutdown_start = time.time()
                while any(p.is_alive() for p in processes):
                    drain_queue()  # Keep draining to capture incremental results
                    if time.time() - shutdown_start > 180:
                        # Force terminate if workers are stuck
                        print("   ⚠️ Timeout waiting for workers, forcing shutdown...")
                        drain_queue()  # One more drain before terminate
                        for p in processes:
                            if p.is_alive():
                                p.terminate()
                        for p in processes:
                            p.join(timeout=2.0)
                            if p.is_alive():
                                p.kill()
                        break
                    time.sleep(0.5)  # Check every 500ms during shutdown
                break

        # Drain any results after processes exit
        drain_queue()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted, waiting for workers to finish current iteration...")
        stop_event.set()
        # Give workers up to 60 seconds to finish gracefully
        shutdown_start = time.time()
        while any(p.is_alive() for p in processes):
            drain_queue()  # Keep draining
            if time.time() - shutdown_start > 60:
                print("   Forcing shutdown...")
                drain_queue()
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                for p in processes:
                    p.join(timeout=2.0)
                    if p.is_alive():
                        p.kill()
                break
            time.sleep(0.5)
        drain_queue()

    total_time = time.time() - start_time

    # Convert dict to list for processing
    miner_results = list(miner_results_by_id.values())

    # Check for errors
    errors = [r for r in miner_results if 'error' in r]
    if errors:
        for err in errors:
            print(f"❌ {err.get('miner_id', 'unknown')}: {err['error']}")

    # Filter out error results for aggregation
    valid_results = [r for r in miner_results if 'error' not in r]

    # Aggregate results (even if empty - we still want a JSON file)
    if valid_results:
        stats = aggregate_results(valid_results, total_time)
    else:
        # Create empty stats structure
        stats = {
            'num_miners': 0,
            'per_miner_stats': [],
            'total_blocks_found': 0,
            'total_attempts': 0,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'success_rate': 0,
            'blocks_per_minute': 0,
            'energy_stats': {'min': None, 'max': None, 'mean': None, 'all_energies': []},
            'diversity_stats': {'min': None, 'max': None, 'mean': None},
            'solution_count_stats': {'min': None, 'max': None, 'mean': None},
            'mining_time_stats': {'min': None, 'max': None, 'mean': None, 'all_times': []}
        }

    # Print results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)

    if not valid_results:
        print("⚠️  No miner results collected (workers may have been terminated)")
        print(f"   Total time: {stats['total_time_minutes']:.1f} min")
        if errors:
            print(f"   Errors: {len(errors)}")
    else:
        print(f"✅ Mining completed:")
        print(f"   Miners used: {stats['num_miners']}")
        print(f"   Total time: {stats['total_time_minutes']:.1f} min")
        print(f"   Blocks found: {stats['total_blocks_found']}")
        print(f"   Total attempts: {stats['total_attempts']}")
        print(f"   Success rate: {stats['success_rate'] * 100:.1f}%")
        print(f"   Mining rate: {stats['blocks_per_minute']:.3f} blocks/min")

        # Per-miner breakdown
        print(f"\n📋 Per-miner breakdown:")
        for r in stats['per_miner_stats']:
            rate = r['blocks_found'] / (r['total_time'] / 60) if r['total_time'] > 0 else 0
            print(f"   {r['miner_id']}: {r['blocks_found']} blocks, {r['attempts']} attempts, {rate:.2f} blocks/min")

        if stats['total_blocks_found'] > 0:
            print(f"\n📈 Energy distribution:")
            print(f"   Min: {stats['energy_stats']['min']:.1f}")
            print(f"   Max: {stats['energy_stats']['max']:.1f}")
            print(f"   Mean: {stats['energy_stats']['mean']:.1f}")

            print(f"\n🌈 Diversity distribution:")
            print(f"   Min: {stats['diversity_stats']['min']:.3f}")
            print(f"   Max: {stats['diversity_stats']['max']:.3f}")
            print(f"   Mean: {stats['diversity_stats']['mean']:.3f}")

            if stats['mining_time_stats']['mean']:
                print(f"\n⏱️  Mining time per block:")
                print(f"   Min: {stats['mining_time_stats']['min']:.1f}s")
                print(f"   Max: {stats['mining_time_stats']['max']:.1f}s")
                print(f"   Mean: {stats['mining_time_stats']['mean']:.1f}s")

    # Always save results (even if empty)
    output_data = {
        'miner_type': args.miner_type,
        'num_miners': stats['num_miners'],
        'difficulty_energy': args.difficulty_energy,
        'duration_spec': args.duration,
        'duration_minutes': duration_minutes,
        'min_diversity': args.min_diversity,
        'min_solutions': args.min_solutions,
        'statistics': stats,
        'errors': [e.get('error', 'unknown') for e in errors] if errors else [],
        'timestamp': utc_timestamp()
    }

    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"mining_rate_{args.miner_type}_{args.duration}min_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")

    # Return 0 if we got any results, 1 if completely empty
    return 0 if valid_results else 1


if __name__ == "__main__":
    from typing import Tuple
    sys.exit(main())
