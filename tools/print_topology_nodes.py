from __future__ import annotations

"""
Utility: Print number of nodes for D-Wave topologies and show how node lists change
when topology parameters (m and/or t) are varied.

Run:
  python tools/print_topology_nodes.py

Optional: run with -q to reduce verbosity
  python tools/print_topology_nodes.py -q
"""

import argparse
from typing import Dict, Iterable, List, Optional, Set, Tuple

import dwave_networkx as dnx

from dwave_topologies import DEFAULT_TOPOLOGY
from dwave_topologies.topologies import (
    CHIMERA_C16_TOPOLOGY,
    PEGASUS_P16_TOPOLOGY,
    ZEPHYR_Z12_TOPOLOGY,
    ZEPHYR_Z11_T4_TOPOLOGY,
    ADVANTAGE2_SYSTEM1_12_TOPOLOGY
)


def build_graph(topology: str, params: Dict[str, int]):
    t = topology.lower()
    if t == "chimera":
        return dnx.chimera_graph(int(params["m"]), int(params["n"]), int(params["t"]))
    elif t == "pegasus":
        return dnx.pegasus_graph(int(params["m"]))
    elif t == "zephyr":
        return dnx.zephyr_graph(int(params["m"]), int(params["t"]))
    else:
        raise ValueError(f"Unknown topology: {topology}")


def pretty_nodes(nodes: Iterable[int], limit: int = 10) -> str:
    ns = list(nodes)
    ns_sorted = sorted(ns)[:limit]
    more = "..." if len(ns) > limit else ""
    return f"{ns_sorted}{more}"


def report_topology(name: str, quiet: bool = False):
    # Map topology names to topology objects
    topology_map = {
        'default': DEFAULT_TOPOLOGY,  # Current default topology (matches miners)
        'c16': CHIMERA_C16_TOPOLOGY,
        'chimera': CHIMERA_C16_TOPOLOGY,
        'p16': PEGASUS_P16_TOPOLOGY,
        'pegasus': PEGASUS_P16_TOPOLOGY,
        'z12': ZEPHYR_Z12_TOPOLOGY,
        'z11t4': ZEPHYR_Z11_T4_TOPOLOGY,
        'zephyr': DEFAULT_TOPOLOGY,  # Use default topology (currently Z11T4)
        'advantage2': ADVANTAGE2_SYSTEM1_12_TOPOLOGY,
    }

    topology_obj = topology_map.get(name.lower())
    if topology_obj is None:
        print(f"Unknown topology: {name}")
        return

    base_graph = topology_obj.graph
    base_nodes = set(base_graph.nodes())
    base_edges = set(base_graph.edges())
    props = topology_obj.properties

    print(
        f"== {name.upper()} (type={props['topology']['type']}, shape={props['topology']['shape']}) =="
    )
    print(
        f"Base: num_qubits(num_nodes)={len(base_nodes)} num_couplers(num_edges)={len(base_edges)} chip_id={props.get('chip_id', 'Unknown')}"  # noqa: E501
    )
    if not quiet:
        print(f"Base node sample: {pretty_nodes(base_nodes)}")

    # Define parameter sweeps per topology
    sweeps: List[Tuple[str, Dict[str, int]]] = []
    t = topology_obj.topology_type.lower()
    topo_shape = props['topology']['shape']

    if t == "chimera":
        m0, n0, t0 = topo_shape[0], topo_shape[1], topo_shape[2]
        for m in [max(2, m0 - 4), m0, m0 + 4]:
            sweeps.append((f"m={m},n={m0},t={t0}", {"m": m, "n": n0, "t": t0}))
        for tt in [max(2, t0 - 2), t0, t0 + 2]:
            sweeps.append((f"m={m0},n={n0},t={tt}", {"m": m0, "n": n0, "t": tt}))
    elif t == "pegasus":
        m0 = topo_shape[0]
        for m in [max(4, m0 - 6), m0, m0 + 8]:
            sweeps.append((f"m={m}", {"m": m}))
    elif t == "zephyr":
        m0, t0 = topo_shape[0], topo_shape[1]
        for m in [max(4, m0 - 4), m0, m0 + 4]:
            sweeps.append((f"m={m},t={t0}", {"m": m, "t": t0}))
        # Many Zephyr installs use t=2 or t=4; try both safely
        for tt in sorted(set([2, t0, max(2, t0 + 2)])):
            sweeps.append((f"m={m0},t={tt}", {"m": m0, "t": tt}))

    # Execute sweeps
    for label, p in sweeps:
        try:
            g = build_graph(t, p)
            nodes = set(g.nodes())
            edges = set(g.edges())
            changed_nodes = nodes != base_nodes
            changed_edges = edges != base_edges
            delta_nodes = len(nodes) - len(base_nodes)
            delta_edges = len(edges) - len(base_edges)
            print(
                f"  Params [{label:>16}]: qubits(num_nodes)={len(nodes):5d} (Δ {delta_nodes:+5d}) | "
                f"couplers(num_edges)={len(edges):6d} (Δ {delta_edges:+6d}) | "
                f"nodes_changed={changed_nodes} edges_changed={changed_edges}"
            )
            if not quiet and changed_nodes:
                only_in_new = sorted(list(nodes - base_nodes))[:8]
                only_in_base = sorted(list(base_nodes - nodes))[:8]
                if only_in_new:
                    print(f"    + first-only-in-new: {only_in_new} ...")
                if only_in_base:
                    print(f"    - first-only-in-base: {only_in_base} ...")
        except Exception as e:
            print(f"  Params [{label:>16}]: ERROR building graph: {e}")

    print()


def main():
    ap = argparse.ArgumentParser(description="Print node counts for D-Wave topologies and show changes when varying parameters.")
    ap.add_argument("--quiet", "-q", action="store_true", help="Reduce verbosity (hide sample node lists)")
    ap.add_argument("--topology", "-t",
                   nargs='+',
                   default=["default"],
                   help="Topology(ies) to analyze (default: default). Options: default, chimera, pegasus, zephyr, z11t4, z12, advantage2")
    args = ap.parse_args()

    for topo in args.topology:
        report_topology(topo, quiet=args.quiet)


if __name__ == "__main__":
    main()

