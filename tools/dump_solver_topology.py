#!/usr/bin/env python3
"""
Tool to dump D-Wave solver topologies and save them as importable Python files.

This tool connects to D-Wave solvers, extracts their topology information,
and saves them as Python files in the dwave/topologies/ directory for easy import.

Usage:
    python tools/dump_solver_topology.py --solver Advantage2-System1.6
    python tools/dump_solver_topology.py --solver Advantage_system6.4 --output-dir custom/path
    python tools/dump_solver_topology.py --list-solvers
    python tools/dump_solver_topology.py --all-available
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dwave.system import DWaveSampler
    from dwave.cloud import Client
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    DWaveSampler = None
    Client = None


def normalize_solver_name(solver_name: str) -> str:
    """
    Convert solver name to Python module name format.
    
    Examples:
        Advantage2-System1.6 -> advantage2_system1_6
        Advantage_system6.4 -> advantage_system6_4
        DW_2000Q_6 -> dw_2000q_6
    """
    # Convert to lowercase
    name = solver_name.lower()
    
    # Replace dots, dashes, and other non-alphanumeric chars with underscores
    name = re.sub(r'[^a-z0-9]', '_', name)
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name


def normalize_constant_name(solver_name: str) -> str:
    """
    Convert solver name to Python constant name format.
    
    Examples:
        Advantage2-System1.6 -> ADVANTAGE2_SYSTEM1_6
        Advantage_system6.4 -> ADVANTAGE_SYSTEM6_4
    """
    return normalize_solver_name(solver_name).upper()


def get_topology_type(properties: Dict[str, Any]) -> str:
    """Determine topology type from solver properties."""
    topology = properties.get('topology', {})
    if isinstance(topology, dict):
        return topology.get('type', 'unknown')
    return str(topology).lower() if topology else 'unknown'


def get_topology_shape(properties: Dict[str, Any]) -> str:
    """Get topology shape description from solver properties."""
    topology = properties.get('topology', {})
    if isinstance(topology, dict):
        return topology.get('shape', 'unknown')
    return 'unknown'


def extract_solver_info(solver_name: str) -> Optional[Dict[str, Any]]:
    """Extract topology information from a D-Wave solver."""
    if not DWAVE_AVAILABLE or DWaveSampler is None:
        print("❌ D-Wave Ocean SDK not available")
        return None

    try:
        print(f"🔍 Connecting to solver: {solver_name}")
        sampler = DWaveSampler(solver=solver_name)
        
        # Extract basic topology info
        nodes = list(sampler.nodelist)
        edges = list(sampler.edgelist)
        properties = dict(sampler.properties)
        
        # Get topology metadata
        topology_type = get_topology_type(properties)
        topology_shape = get_topology_shape(properties)
        
        print(f"✅ Connected to {solver_name}")
        print(f"   Topology: {topology_type} ({topology_shape})")
        print(f"   Nodes: {len(nodes)}")
        print(f"   Edges: {len(edges)}")
        
        return {
            'solver_name': solver_name,
            'nodes': nodes,
            'edges': edges,
            'properties': properties,
            'topology_type': topology_type,
            'topology_shape': topology_shape,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
        
    except Exception as e:
        print(f"❌ Failed to connect to solver {solver_name}: {e}")
        return None


def generate_topology_file_content(solver_info: Dict[str, Any]) -> str:
    """Generate Python file content for the topology."""
    solver_name = solver_info['solver_name']
    constant_name = normalize_constant_name(solver_name)
    
    # Format nodes and edges for Python (limit display for readability)
    nodes_repr = repr(solver_info['nodes'])
    edges_repr = repr(solver_info['edges'])
    
    # If lists are very long, we'll write them more efficiently
    if len(solver_info['nodes']) > 100:
        # Check if nodes are contiguous integers starting from 0 or min value
        sorted_nodes = sorted(solver_info['nodes'])
        min_node = min(sorted_nodes)
        max_node = max(sorted_nodes)
        expected_range = list(range(min_node, max_node + 1))

        if sorted_nodes == expected_range:
            nodes_repr = f"list(range({min_node}, {max_node + 1}))"
        else:
            # Nodes are not contiguous, keep the full list but truncate for readability
            if len(solver_info['nodes']) > 1000:
                # For very large lists, just show the pattern
                nodes_repr = f"# {len(solver_info['nodes'])} nodes: {solver_info['nodes'][:20]} + ... + {solver_info['nodes'][-20:]}\n"
                nodes_repr += f"    {repr(solver_info['nodes'])}"
            else:
                nodes_repr = repr(solver_info['nodes'])
    
    content = f'''"""
D-Wave {solver_name} topology definition.

Auto-generated by tools/dump_solver_topology.py on {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.

Topology Information:
- Solver: {solver_name}
- Type: {solver_info['topology_type']}
- Shape: {solver_info['topology_shape']}
- Nodes: {solver_info['num_nodes']}
- Edges: {solver_info['num_edges']}
"""

from typing import List, Tuple, Dict, Any

# Topology constant for import
{constant_name} = {{
    'solver_name': '{solver_name}',
    'topology_type': '{solver_info['topology_type']}',
    'topology_shape': '{solver_info['topology_shape']}',
    'num_nodes': {solver_info['num_nodes']},
    'num_edges': {solver_info['num_edges']},
    'nodes': {nodes_repr},
    'edges': {edges_repr},
    'properties': {repr(solver_info['properties'])},
    'generated_at': '{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}',
}}

# Convenience accessors
NODES: List[int] = {constant_name}['nodes']
EDGES: List[Tuple[int, int]] = {constant_name}['edges']
PROPERTIES: Dict[str, Any] = {constant_name}['properties']
SOLVER_NAME: str = '{solver_name}'
TOPOLOGY_TYPE: str = '{solver_info['topology_type']}'
TOPOLOGY_SHAPE: str = '{solver_info['topology_shape']}'
NUM_NODES: int = {solver_info['num_nodes']}
NUM_EDGES: int = {solver_info['num_edges']}
'''
    
    return content


def list_available_solvers() -> List[str]:
    """List all available D-Wave solvers."""
    if not DWAVE_AVAILABLE or Client is None:
        print("❌ D-Wave Ocean SDK not available")
        return []

    try:
        print("🔍 Querying available D-Wave solvers...")
        client = Client.from_config()
        solvers = client.get_solvers()
        
        solver_names = []
        print("\n📋 Available solvers:")
        for solver in solvers:
            solver_names.append(solver.name)
            properties = solver.properties
            topology_type = get_topology_type(properties)
            topology_shape = get_topology_shape(properties)
            num_qubits = properties.get('num_qubits', 'unknown')

            # Handle status safely - different solver types have different status formats
            status = 'unknown'
            try:
                if hasattr(solver, 'status'):
                    if isinstance(solver.status, dict):
                        status = solver.status.get('state', 'unknown')
                    else:
                        status = str(solver.status)
                elif hasattr(solver, 'is_online'):
                    status = 'online' if solver.is_online else 'offline'
            except Exception:
                status = 'unknown'

            print(f"   {solver.name}")
            print(f"      Type: {topology_type} ({topology_shape})")
            print(f"      Qubits: {num_qubits}")
            print(f"      Status: {status}")
            print()
        
        return solver_names
        
    except Exception as e:
        print(f"❌ Failed to list solvers: {e}")
        return []


def dump_solver_topology(solver_name: str, output_dir: str = "dwave/topologies") -> bool:
    """Dump a single solver topology to a Python file."""
    # Extract solver information
    solver_info = extract_solver_info(solver_name)
    if not solver_info:
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    module_name = normalize_solver_name(solver_name)
    filename = f"{module_name}.py"
    filepath = output_path / filename
    
    # Generate file content
    content = generate_topology_file_content(solver_info)
    
    # Write file
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Topology saved to: {filepath}")
        print(f"   Import as: from {output_dir.replace('/', '.')}.{module_name} import {normalize_constant_name(solver_name)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to write file {filepath}: {e}")
        return False


def create_init_file(output_dir: str, solver_names: List[str]) -> None:
    """Create __init__.py file for the topologies package."""
    init_path = Path(output_dir) / "__init__.py"
    
    imports = []
    all_exports = []
    
    for solver_name in solver_names:
        module_name = normalize_solver_name(solver_name)
        constant_name = normalize_constant_name(solver_name)
        imports.append(f"from .{module_name} import {constant_name}")
        all_exports.append(constant_name)
    
    content = f'''"""
D-Wave solver topologies package.

Auto-generated by tools/dump_solver_topology.py on {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.

This package contains topology definitions for various D-Wave solvers,
extracted directly from the D-Wave API.
"""

{chr(10).join(imports)}

__all__ = [
{chr(10).join(f'    "{name}",' for name in all_exports)}
]
'''
    
    try:
        with open(init_path, 'w') as f:
            f.write(content)
        print(f"✅ Package init file created: {init_path}")
    except Exception as e:
        print(f"❌ Failed to create init file: {e}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Dump D-Wave solver topologies to importable Python files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --solver Advantage2-System1.6
  %(prog)s --solver Advantage_system6.4 --output-dir custom/path
  %(prog)s --list-solvers
  %(prog)s --all-available
        """
    )
    
    parser.add_argument(
        '--solver', '-s',
        type=str,
        help='Specific solver name to dump (e.g., Advantage2-System1.6)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='dwave/topologies',
        help='Output directory for topology files (default: dwave/topologies)'
    )
    
    parser.add_argument(
        '--list-solvers', '-l',
        action='store_true',
        help='List all available D-Wave solvers and exit'
    )
    
    parser.add_argument(
        '--all-available', '-a',
        action='store_true',
        help='Dump topologies for all available solvers'
    )
    
    args = parser.parse_args()
    
    if not DWAVE_AVAILABLE:
        print("❌ D-Wave Ocean SDK not available. Install with:")
        print("   pip install dwave-ocean-sdk")
        sys.exit(1)
    
    # List solvers mode
    if args.list_solvers:
        list_available_solvers()
        return
    
    # All available solvers mode
    if args.all_available:
        solver_names = list_available_solvers()
        if not solver_names:
            print("❌ No solvers available")
            sys.exit(1)
        
        print(f"\n🚀 Dumping topologies for {len(solver_names)} solvers...")
        successful = []
        
        for solver_name in solver_names:
            print(f"\n--- Processing {solver_name} ---")
            if dump_solver_topology(solver_name, args.output_dir):
                successful.append(solver_name)
        
        if successful:
            create_init_file(args.output_dir, successful)
            print(f"\n✅ Successfully dumped {len(successful)}/{len(solver_names)} solver topologies")
        else:
            print(f"\n❌ Failed to dump any solver topologies")
        
        return
    
    # Single solver mode
    if args.solver:
        print(f"🚀 Dumping topology for solver: {args.solver}")
        if dump_solver_topology(args.solver, args.output_dir):
            create_init_file(args.output_dir, [args.solver])
            print(f"\n✅ Successfully dumped topology for {args.solver}")
        else:
            print(f"\n❌ Failed to dump topology for {args.solver}")
            sys.exit(1)
        return
    
    # No action specified
    parser.print_help()
    print("\n❌ Please specify --solver, --list-solvers, or --all-available")
    sys.exit(1)


if __name__ == "__main__":
    main()
