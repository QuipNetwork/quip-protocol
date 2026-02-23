"""
Describes the model of quantum annealer topology.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Self, Tuple

import networkx as nx

Qubit = Tuple[int, int]  # A type of qubits, defined in (x, y)-coordinates.
Coords = Tuple[int, int]


@dataclass
class Cell:
    """
    A non-divisible unit of a quantum annealer that is bundled
    with other units using external couplers.
    """

    adjlist: Dict[Qubit, List[Qubit]] = field(default_factory=dict)

    @classmethod
    def from_adjlist(cls, *coupling: Tuple[Qubit, Qubit]) -> Self:
        """Creates a new unit from a list of qubit edges."""
        adjlist: Dict[Qubit, List[Qubit]] = dict()
        for p, q in coupling:
            adjlist.setdefault(p, list()).append(q)
        return cls(adjlist=adjlist)

    @property
    def graph(self) -> nx.Graph[Qubit]:
        """A nx.Graph representation of this unit."""
        if self._graph is not None:
            return self._graph
        self._graph = nx.Graph()
        for p, qs in self.adjlist.items():
            for q in qs:
                self._graph.add_edge(p, q)
        return self._graph


@dataclass
class Coupling:
    """
    A net of external couplers that bundles two distant qubits
    on different units.

    # Attributes
    grid : Dict[Qubit, List[Qubit]]
        Describes ways of connections to inbound and outbound pairs.
    """

    grid: Dict[Qubit, List[Qubit]] = field(default_factory=dict)

    @classmethod
    def from_adjlist(cls, *qubits: Tuple[Qubit, Qubit]) -> Self:
        """Creates a new coupler net with given pairs of qubits"""
        grid: Dict[Qubit, List[Qubit]] = {}
        for p, q in qubits:
            grid.setdefault(p, list()).append(q)
        return cls(grid=grid)


@dataclass
class Topology:
    """
    A quantum annealer topology.

    # Types
    Coords
        A type of coordinates for the planar chip topology.

    # Attributes
    cells : Dict[Coords, Cell]
        Instances of chip cells.
    coupling : Dict[Coords, Dict[Coords, Coupling]]
        Description of cell coupling.
    """

    cells: Dict[Coords, Cell] = field(default_factory=dict)
    coupling: Dict[Coords, Dict[Coords, Coupling]] = field(default_factory=dict)

    @property
    def graph(self) -> nx.Graph:
        """An undirected graph that is representing this topology."""
        if self._graph is not None:
            return self._graph
        self._graph = nx.union_all(
            [
                nx.relabel_nodes(
                    unit.graph, {node: (coord, node) for node in unit.graph}
                )
                for coord, unit in self.cells.items()
            ]
        ).add_edges_from(
            [
                ((start, p), (end, q))
                for start, ends in self.coupling.items()
                for end, coupling in ends.items()
                for p, qs in coupling.grid.items()
                for q in qs
            ]
        )
        return self._graph
