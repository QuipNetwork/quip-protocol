#!/usr/bin/env python3
"""Launch script for quantum blockchain network with different miner types."""

import argparse
import subprocess
import time
import sys
import signal
import os


class NetworkLauncher:
    """Manages launching and coordinating blockchain network nodes."""
    
    def __init__(self):
        self.processes = []
        
    def launch_node(self, miner_type: str, node_id: int, port: int, peer_port: int = None, **kwargs):
        """Launch a mining node."""
        cmd = [sys.executable, f"{miner_type}/{miner_type.lower()}_miner.py",
               "--id", str(node_id),
               "--port", str(port)]
        
        if peer_port:
            cmd.extend(["--peer", f"localhost:{peer_port}"])
            
        # Add additional arguments
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        print(f"Launching {miner_type} node {node_id} on port {port}...")
        
        process = subprocess.Popen(cmd)
        self.processes.append(process)
        
        # Give node time to start
        time.sleep(2)
        
        return process
    
    def stop_all(self):
        """Stop all launched processes."""
        print("\nStopping all nodes...")
        for process in self.processes:
            process.terminate()
        
        # Wait for graceful shutdown
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("All nodes stopped.")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signal."""
        print("\nReceived interrupt signal...")
        self.stop_all()
        sys.exit(0)


def scenario_mixed_network():
    """Launch a mixed network with CPU, GPU, and QPU miners."""
    launcher = NetworkLauncher()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, launcher.signal_handler)
    
    print("🚀 Launching Quantum Blockchain Network")
    print("=" * 60)
    
    # Launch bootstrap CPU node
    launcher.launch_node("CPU", node_id=1, port=8080, num_sweeps=4096)
    print("✅ Bootstrap node started")
    
    # Launch additional CPU nodes
    launcher.launch_node("CPU", node_id=2, port=8081, peer_port=8080, num_sweeps=2048)
    launcher.launch_node("CPU", node_id=3, port=8082, peer_port=8080, num_sweeps=8192)
    print("✅ CPU miners started")
    
    # Launch GPU nodes if Modal is available
    try:
        launcher.launch_node("GPU", node_id=1, port=8083, peer_port=8080, gpu_type="t4")
        launcher.launch_node("GPU", node_id=2, port=8084, peer_port=8080, gpu_type="a10g")
        print("✅ GPU miners started")
    except Exception as e:
        print(f"⚠️  GPU miners not available: {e}")
    
    # Launch QPU node if D-Wave credentials available
    if os.getenv('DWAVE_API_TOKEN'):
        launcher.launch_node("QPU", node_id=1, port=8085, peer_port=8080)
        print("✅ QPU miner started")
    else:
        print("⚠️  QPU miner not available (no DWAVE_API_TOKEN)")
    
    print("\n📊 Network Summary:")
    print(f"  Total nodes: {len(launcher.processes)}")
    print("  Node types: CPU, GPU, QPU")
    print("\nNetwork is running. Press Ctrl+C to stop all nodes.")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        launcher.stop_all()


def scenario_cpu_only():
    """Launch a CPU-only network."""
    launcher = NetworkLauncher()
    signal.signal(signal.SIGINT, launcher.signal_handler)
    
    print("🚀 Launching CPU-only Blockchain Network")
    print("=" * 60)
    
    # Launch nodes with different sweep parameters
    launcher.launch_node("CPU", node_id=1, port=8080, num_sweeps=2048)
    launcher.launch_node("CPU", node_id=2, port=8081, peer_port=8080, num_sweeps=4096)
    launcher.launch_node("CPU", node_id=3, port=8082, peer_port=8080, num_sweeps=8192)
    launcher.launch_node("CPU", node_id=4, port=8083, peer_port=8080, num_sweeps=1024)
    
    print("\n✅ CPU network started with 4 nodes")
    print("Network is running. Press Ctrl+C to stop all nodes.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        launcher.stop_all()


def scenario_gpu_competition():
    """Launch GPU competition network."""
    launcher = NetworkLauncher()
    signal.signal(signal.SIGINT, launcher.signal_handler)
    
    print("🚀 Launching GPU Competition Network")
    print("=" * 60)
    
    # Launch different GPU types
    launcher.launch_node("GPU", node_id=1, port=8080, gpu_type="t4")
    launcher.launch_node("GPU", node_id=2, port=8081, peer_port=8080, gpu_type="t4")
    launcher.launch_node("GPU", node_id=3, port=8082, peer_port=8080, gpu_type="a10g")
    launcher.launch_node("GPU", node_id=4, port=8083, peer_port=8080, gpu_type="a100")
    
    print("\n✅ GPU network started")
    print("  2x T4 GPUs")
    print("  1x A10G GPU")
    print("  1x A100 GPU")
    print("\nNetwork is running. Press Ctrl+C to stop all nodes.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        launcher.stop_all()


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Launch Quantum Blockchain Network')
    parser.add_argument('--scenario', default='mixed',
                       choices=['mixed', 'cpu', 'gpu'],
                       help='Network scenario to launch')
    
    args = parser.parse_args()
    
    if args.scenario == 'mixed':
        scenario_mixed_network()
    elif args.scenario == 'cpu':
        scenario_cpu_only()
    elif args.scenario == 'gpu':
        scenario_gpu_competition()


if __name__ == "__main__":
    main()