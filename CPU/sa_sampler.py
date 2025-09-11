"""Simulated Annealing sampler for CPU-based quantum blockchain mining."""

from typing import Any, Dict, List, Tuple
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY

import collections.abc

Variable = collections.abc.Hashable
class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure."""
    
    def __init__(self):
        # Use the default topology (Advantage2) from quantum_proof_of_work
        topology_graph = DEFAULT_TOPOLOGY.graph
        properties = DEFAULT_TOPOLOGY.properties

        substitute_sampler = SimulatedAnnealingSampler()
        nodelist = list(topology_graph.nodes())
        edgelist = list(topology_graph.edges())

        super().__init__(
            nodelist=nodelist,
            edgelist=edgelist,
            properties=properties,
            substitute_sampler=substitute_sampler
        )
        self.sampler_type = "mock"
        self.parameters.update(substitute_sampler.parameters)
        self.mocked_parameters.add('num_sweeps')

        # Type conversions to match protocol
        self.nodelist: List[Variable] = nodelist
        self.edgelist: List[Tuple[Variable, Variable]] = edgelist
        self.properties: Dict[str, Any] = properties

        # NOTE: these are of type List[Variable], which we can't change, but AFAICT they are always ints.
        #.      it might be the case they are floats or something strange one day.
        nodes = []
        for node in self.nodelist:
            if not isinstance(node, int):
                raise ValueError(f"Expected node index to be int, got {type(node)}")
            nodes.append(int(node))
        edges = []
        for edge in self.edgelist:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError(f"Expected edge to be tuple of length 2, got {edge}")
            if not isinstance(edge[0], int) or not isinstance(edge[1], int):
                raise ValueError(f"Expected edge indices to be int, got {type(edge[0])} and {type(edge[1])}")
            edges.append((int(edge[0]), int(edge[1])))
        self.nodes = nodes
        self.edges = edges

        
