#!/usr/bin/env python3
"""
Tool to dump D-Wave solver topologies and save them as JSON files (optionally gzipped).

This tool connects to D-Wave solvers, extracts their topology information,
and saves them as JSON files that can be loaded with load_json_topology().

Usage:
    python tools/dump_solver_topology.py --solver Advantage2-System1.6
    python tools/dump_solver_topology.py --solver Advantage_system6.4 --output-dir custom/path
    python tools/dump_solver_topology.py --list-solvers
    python tools/dump_solver_topology.py --all-available
    python tools/dump_solver_topology.py --solver Advantage2-System1.6 --gzip
"""

import argparse
import gzip
import json
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


def generate_topology_json(solver_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate JSON data structure for the topology."""
    solver_name = solver_info['solver_name']

    # Create JSON structure matching the format expected by load_json_topology()
    topology_json = {
        'metadata': {
            'solver_name': solver_name,
            'topology_type': solver_info['topology_type'],
            'topology_shape': solver_info['topology_shape'],
            'num_nodes': solver_info['num_nodes'],
            'num_edges': solver_info['num_edges'],
            'generated_from': f"D-Wave API via dump_solver_topology.py",
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        },
        'nodes': solver_info['nodes'],
        'edges': solver_info['edges'],
        'properties': solver_info['properties'],
        'docs': {
            'description': f"D-Wave {solver_name} topology definition",
            'usage': f"from dwave_topologies.topologies.json_loader import load_json_topology\ntopology = load_json_topology('{normalize_solver_name(solver_name)}.json.gz')",
        }
    }

    return topology_json


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


def dump_solver_topology(solver_name: str, output_dir: str = "dwave_topologies/topologies", use_gzip: bool = True) -> bool:
    """Dump a single solver topology to a JSON file (optionally gzipped)."""
    # Extract solver information
    solver_info = extract_solver_info(solver_name)
    if not solver_info:
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    module_name = normalize_solver_name(solver_name)
    filename = f"{module_name}.json"
    if use_gzip:
        filename += ".gz"
    filepath = output_path / filename

    # Generate JSON content
    topology_json = generate_topology_json(solver_info)

    # Write file
    try:
        if use_gzip:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(topology_json, f, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(topology_json, f, indent=2)

        file_size = filepath.stat().st_size
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB"

        print(f"✅ Topology saved to: {filepath} ({size_str})")
        print(f"   Load with: from dwave_topologies.topologies.json_loader import load_json_topology")
        print(f"              topology = load_json_topology('{filename}')")
        return True

    except Exception as e:
        print(f"❌ Failed to write file {filepath}: {e}")
        return False


def create_readme_file(output_dir: str, solver_names: List[str], use_gzip: bool) -> None:
    """Create README.md file documenting the dumped topologies."""
    readme_path = Path(output_dir) / "README.md"

    extension = ".json.gz" if use_gzip else ".json"
    topology_list = []

    for solver_name in solver_names:
        module_name = normalize_solver_name(solver_name)
        topology_list.append(f"- `{module_name}{extension}` - {solver_name}")

    content = f'''# D-Wave Solver Topologies

Auto-generated by `tools/dump_solver_topology.py` on {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.

This directory contains topology definitions for various D-Wave solvers,
extracted directly from the D-Wave API and saved in JSON format.

## Available Topologies

{chr(10).join(topology_list)}

## Usage

Load a topology using the `load_json_topology()` function:

```python
from dwave_topologies.topologies.json_loader import load_json_topology

# Load topology
topology = load_json_topology('{normalize_solver_name(solver_names[0]) if solver_names else "topology"}{extension}')

# Access topology properties
print(f"Solver: {{topology.solver_name}}")
print(f"Nodes: {{topology.num_nodes}}")
print(f"Edges: {{topology.num_edges}}")
print(f"Type: {{topology.topology_type}}")

# Access graph and data
nodes = topology.nodes
edges = topology.edges
graph = topology.graph
```

## Updating Topologies

To update or add new topologies, run:

```bash
# Dump a specific solver
python tools/dump_solver_topology.py --solver Advantage2-System1.6 --gzip

# List available solvers
python tools/dump_solver_topology.py --list-solvers

# Dump all available solvers
python tools/dump_solver_topology.py --all-available --gzip
```
'''

    try:
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"✅ README file created: {readme_path}")
    except Exception as e:
        print(f"❌ Failed to create README file: {e}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Dump D-Wave solver topologies to JSON files (optionally gzipped)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --solver Advantage2-System1.6 --gzip
  %(prog)s --solver Advantage_system6.4 --output-dir custom/path
  %(prog)s --list-solvers
  %(prog)s --all-available --gzip
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
        default='dwave_topologies/topologies',
        help='Output directory for topology files (default: dwave_topologies/topologies)'
    )

    parser.add_argument(
        '--gzip', '-g',
        action='store_true',
        help='Compress output files with gzip (recommended, reduces file size ~10x)'
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
        print(f"   Output format: {'JSON (gzip compressed)' if args.gzip else 'JSON (uncompressed)'}")
        successful = []

        for solver_name in solver_names:
            print(f"\n--- Processing {solver_name} ---")
            if dump_solver_topology(solver_name, args.output_dir, args.gzip):
                successful.append(solver_name)

        if successful:
            create_readme_file(args.output_dir, successful, args.gzip)
            print(f"\n✅ Successfully dumped {len(successful)}/{len(solver_names)} solver topologies")
        else:
            print(f"\n❌ Failed to dump any solver topologies")

        return

    # Single solver mode
    if args.solver:
        print(f"🚀 Dumping topology for solver: {args.solver}")
        print(f"   Output format: {'JSON (gzip compressed)' if args.gzip else 'JSON (uncompressed)'}")
        if dump_solver_topology(args.solver, args.output_dir, args.gzip):
            create_readme_file(args.output_dir, [args.solver], args.gzip)
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
