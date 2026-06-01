#!/usr/bin/env python3
"""
Generate JSON topology files for all standard D-Wave topologies.

This script generates static JSON files for:
- Chimera C16 (D-Wave 2000Q)
- Pegasus P16 (D-Wave Advantage)
- Zephyr Z(12, 4) (generic Advantage2)
- Advantage2-System1.6 (real solver topology)
"""

import json
import os
import dwave_networkx as dnx


def generate_generic_topology_json(topology_type: str, params: dict, output_dir: str):
    """Generate JSON file for a generic dwave_networkx topology."""

    # Generate graph based on topology type
    if topology_type == 'chimera':
        m, n, t = params['m'], params['n'], params['t']
        graph = dnx.chimera_graph(m, n, t)
        name = f"chimera_c{m}"
        solver_name = f"Chimera_C{m}_Generic"
        shape = [m, n, t]
        desc = f"D-Wave Chimera C{m} topology"
    elif topology_type == 'pegasus':
        m = params['m']
        graph = dnx.pegasus_graph(m)
        name = f"pegasus_p{m}"
        solver_name = f"Pegasus_P{m}_Generic"
        shape = [m]
        desc = f"D-Wave Pegasus P{m} topology"
    elif topology_type == 'zephyr':
        m, t = params['m'], params['t']
        graph = dnx.zephyr_graph(m, t)
        name = f"zephyr_z{m}_t{t}"
        solver_name = f"Zephyr_Z{m}_T{t}_Generic"
        shape = [m, t]
        desc = f"D-Wave Zephyr Z({m}, {t}) topology"
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    nodes = list(graph.nodes())
    edges = list(graph.edges())
    num_nodes = len(nodes)
    num_edges = len(edges)
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

    # Calculate expected GSE
    from shared.energy_utils import expected_solution_energy
    expected_gse = expected_solution_energy(
        num_nodes=num_nodes,
        num_edges=num_edges,
        c=0.75
    )

    # Create JSON topology data
    topology_data = {
        "metadata": {
            "description": desc,
            "generated_from": f"dwave_networkx.{topology_type}_graph({', '.join(map(str, shape))})",
            "solver_name": solver_name,
            "topology_type": topology_type,
            "topology_shape": shape,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": round(avg_degree, 2),
            "expected_gse": round(expected_gse, 1),
            "notes": [
                f"Generic, QPU-agnostic {topology_type} graph structure",
                "No solver-specific defect patterns",
                "Generated from dwave_networkx for reference/testing"
            ]
        },
        "properties": {
            "topology": {
                "type": topology_type,
                "shape": shape
            },
            "num_qubits": num_nodes,
            "num_couplers": num_edges,
            "chip_id": f"Generic_{solver_name}",
            "supported_problem_types": ["qubo", "ising"]
        },
        "nodes": nodes,
        "edges": [[u, v] for u, v in edges],
        "docs": {
            "topology": f"https://support.dwavesys.com/hc/en-us/articles/What-Is-the-{topology_type.capitalize()}-Topology",
            "solver": "https://docs.dwavesys.com/docs/latest/c_solver_properties.html",
            "overview": "https://docs.ocean.dwavesys.com/en/latest/concepts/topology.html"
        }
    }

    # Write JSON file
    filename = f"{name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(topology_data, f, indent=2)

    file_size = os.path.getsize(filepath) / 1024
    print(f"✓ Generated {filename}")
    print(f"  Nodes: {num_nodes:,}, Edges: {num_edges:,}, Size: {file_size:.1f} KB")

    return filename


def generate_advantage2_system_json(output_dir: str):
    """Generate JSON for real Advantage2-System1 topology."""

    # Import the existing topology data
    from dwave_topologies.topologies.advantage2_system1 import ADVANTAGE2_SYSTEM1_TOPOLOGY

    topo = ADVANTAGE2_SYSTEM1_TOPOLOGY

    # Calculate expected GSE
    from shared.energy_utils import expected_solution_energy
    expected_gse = expected_solution_energy(
        num_nodes=topo.num_nodes,
        num_edges=topo.num_edges,
        c=0.75
    )

    # Create JSON topology data
    topology_data = {
        "metadata": {
            "description": "D-Wave Advantage2-System1.7 real solver topology",
            "generated_from": "D-Wave API - real solver with defect pattern",
            "solver_name": topo.solver_name,
            "topology_type": topo.topology_type,
            "topology_shape": [12, 4],  # Zephyr Z(12,4) base
            "num_nodes": topo.num_nodes,
            "num_edges": topo.num_edges,
            "avg_degree": round(2 * topo.num_edges / topo.num_nodes, 2),
            "expected_gse": round(expected_gse, 1),
            "notes": [
                "Real Advantage2-System1.7 topology from D-Wave API",
                "Includes solver-specific defect pattern",
                "Based on Zephyr Z(12,4) architecture with missing qubits/couplers"
            ]
        },
        "properties": topo.properties,
        "nodes": topo.nodes,
        "edges": [[u, v] for u, v in topo.edges],
        "docs": topo.docs
    }

    # Write JSON file
    filename = "advantage2_system1_7.json.gz"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(topology_data, f, indent=2)

    file_size = os.path.getsize(filepath) / 1024
    print(f"✓ Generated {filename}")
    print(f"  Nodes: {topo.num_nodes:,}, Edges: {topo.num_edges:,}, Size: {file_size:.1f} KB")

    return filename


def main():
    output_dir = "dwave_topologies/topologies"

    print("="*80)
    print("Generating JSON topology files for all standard topologies")
    print("="*80)
    print()

    # Generate generic topologies
    topologies = [
        ('chimera', {'m': 16, 'n': 16, 't': 4}),
        ('pegasus', {'m': 16}),
        ('zephyr', {'m': 12, 't': 4}),
    ]

    for topo_type, params in topologies:
        generate_generic_topology_json(topo_type, params, output_dir)
        print()

    # Generate real Advantage2 topology
    generate_advantage2_system_json(output_dir)
    print()

    print("="*80)
    print("All JSON topology files generated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
