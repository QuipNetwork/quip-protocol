#!/usr/bin/env python3
"""
Solution Explorer

Interactive tool for exploring and validating quantum blockchain solutions.
Can ingest blocks from network nodes or JSON files for detailed analysis.
"""

import sys
import json
import argparse
import asyncio
import aiohttp
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, List, Any, Optional
from shared.energy_utils import IsingModelValidator, validate_sampler_solutions
from shared.quantum_proof_of_work import generate_ising_model_from_nonce


async def fetch_block_from_network(host: str, port: int, block_index: Optional[int] = None) -> Dict[str, Any]:
    """Fetch a block from a network node."""
    url = f"http://{host}:{port}/api/blockchain"
    if block_index is not None:
        url += f"/{block_index}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            raise Exception(f"Failed to fetch from {url}: {e}")


def load_block_from_json(file_path: str) -> Dict[str, Any]:
    """Load block data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load JSON from {file_path}: {e}")


def extract_quantum_proof_from_block(block_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quantum proof data from a block."""
    if isinstance(block_data, list):
        # If it's a blockchain array, get the latest block
        if not block_data:
            raise ValueError("Empty blockchain provided")
        block_data = block_data[-1]
    
    # Handle different block formats
    if 'quantum_proof' in block_data:
        proof = block_data['quantum_proof']
    elif 'proof' in block_data:
        proof = block_data['proof']
    else:
        raise ValueError("No quantum proof found in block data")
    
    return {
        'block_index': block_data.get('index', 0),
        'solutions': proof.get('solutions', []),
        'nonce': proof.get('nonce', 0),
        'salt': proof.get('salt', b''),
        'nodes': proof.get('nodes', proof.get('node_list', [])),
        'edges': proof.get('edges', proof.get('edge_list', [])),
        'miner_id': proof.get('miner_id', 'unknown'),
        'energy': proof.get('energy', 0.0),
        'diversity': proof.get('diversity', 0.0)
    }


def explore_solutions_interactive(h: Dict[int, float], J: Dict, nodes: List[int], solutions: List[List[int]]):
    """Interactive exploration of solutions."""
    validator = IsingModelValidator(h, J, nodes)
    
    print(f"\n🔍 Interactive Solution Explorer")
    print("=" * 50)
    print(f"Problem: {len(h)} variables, {len(J)} couplings")
    print(f"Solutions: {len(solutions)} available")
    print("\nCommands:")
    print("  'all' - Validate all solutions")
    print("  'N' - Validate solution N (0-based index)")
    print("  'stats' - Show solution statistics")
    print("  'compare N M' - Compare solutions N and M")
    print("  'exit' - Exit explorer")
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'exit':
                break
            elif cmd == 'all':
                results = validate_sampler_solutions("Block Solutions", 
                                                   {'samples': solutions, 'energies': [0]*len(solutions)}, 
                                                   h, J, nodes)
                
            elif cmd == 'stats':
                print(f"\n📊 Solution Statistics:")
                print(f"  Total solutions: {len(solutions)}")
                energies = []
                for sol in solutions:
                    result = validator.validate_solution(sol, verbose=False)
                    energies.append(result['energy'])
                
                print(f"  Energy range: {min(energies):.1f} to {max(energies):.1f}")
                print(f"  Mean energy: {np.mean(energies):.1f}")
                print(f"  Std energy: {np.std(energies):.1f}")
                
            elif cmd.startswith('compare '):
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        idx1, idx2 = int(parts[1]), int(parts[2])
                        if 0 <= idx1 < len(solutions) and 0 <= idx2 < len(solutions):
                            sol1, sol2 = solutions[idx1], solutions[idx2]
                            print(f"\n🔄 Comparing Solutions {idx1} and {idx2}:")
                            
                            # Calculate Hamming distance
                            hamming_dist = sum(1 for a, b in zip(sol1, sol2) if a != b)
                            print(f"  Hamming distance: {hamming_dist}/{len(sol1)} ({hamming_dist/len(sol1):.1%})")
                            
                            # Individual energies
                            result1 = validator.validate_solution(sol1, verbose=False)
                            result2 = validator.validate_solution(sol2, verbose=False)
                            print(f"  Energy {idx1}: {result1['energy']:.1f}")
                            print(f"  Energy {idx2}: {result2['energy']:.1f}")
                            print(f"  Energy difference: {abs(result1['energy'] - result2['energy']):.1f}")
                        else:
                            print("Invalid solution indices")
                    except ValueError:
                        print("Usage: compare N M (where N, M are solution indices)")
                else:
                    print("Usage: compare N M")
                    
            elif cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < len(solutions):
                    print(f"\n--- Solution {idx} ---")
                    result = validator.validate_solution(solutions[idx], verbose=True)
                else:
                    print(f"Invalid index. Range: 0-{len(solutions)-1}")
                    
            else:
                print("Unknown command. Type 'exit' to quit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explore and validate quantum blockchain solutions")
    
    # Input source options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--network', metavar='HOST:PORT', 
                           help='Fetch block from network node (e.g., localhost:8080)')
    input_group.add_argument('--json', metavar='FILE', 
                           help='Load block from JSON file')
    
    # Block selection options
    parser.add_argument('--block-index', type=int, 
                       help='Specific block index to analyze (default: latest)')
    
    # Analysis options
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive solution explorer')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate solutions, do not explore')
    parser.add_argument('--export-results', metavar='FILE',
                       help='Export validation results to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Load block data
        if args.network:
            host, port = args.network.split(':')
            port = int(port)
            print(f"🌐 Fetching block from {host}:{port}...")
            block_data = asyncio.run(fetch_block_from_network(host, port, args.block_index))
        else:
            print(f"📁 Loading block from {args.json}...")
            block_data = load_block_from_json(args.json)
        
        # Extract quantum proof
        proof_data = extract_quantum_proof_from_block(block_data)
        
        print(f"✅ Loaded block {proof_data['block_index']} from miner {proof_data['miner_id']}")
        print(f"   Energy: {proof_data['energy']:.1f}, Diversity: {proof_data['diversity']:.3f}")
        print(f"   Solutions: {len(proof_data['solutions'])}")
        
        # Generate Ising model parameters
        h, J = generate_ising_model_from_nonce(
            proof_data['nonce'], 
            proof_data['nodes'], 
            proof_data['edges']
        )
        
        print(f"🔧 Generated Ising model: {len(h)} variables, {len(J)} couplings")
        
        # Validate solutions
        results = validate_sampler_solutions(
            f"Block {proof_data['block_index']}", 
            {'samples': proof_data['solutions'], 'energies': [0]*len(proof_data['solutions'])},
            h, J, proof_data['nodes']
        )
        
        # Export results if requested
        if args.export_results:
            with open(args.export_results, 'w') as f:
                json.dump({
                    'block_data': proof_data,
                    'validation_results': results,
                    'ising_parameters': {
                        'h': {str(k): v for k, v in h.items()},
                        'J': {f"{k[0]},{k[1]}": v for k, v in J.items()}
                    }
                }, f, indent=2)
            print(f"💾 Results exported to {args.export_results}")
        
        # Interactive exploration
        if args.interactive and not args.validate_only:
            explore_solutions_interactive(h, J, proof_data['nodes'], proof_data['solutions'])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("🔍 Solution Explorer - Example Usage")
        print("=" * 40)
        print("Examples:")
        print("  python solution_explorer.py --network localhost:8080 --interactive")
        print("  python solution_explorer.py --json block_data.json --validate-only")
        print("  python solution_explorer.py --network localhost:8080 --block-index 5 --export-results results.json")
        print("\nUse --help for full options")
    else:
        main()