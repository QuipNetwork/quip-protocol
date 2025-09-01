"""Simulated Annealing sampler for CPU-based quantum blockchain mining."""

import os

# Set default DWave environment variables before any DWave libraries are imported
if "DWAVE_API_KEY" not in os.environ:
    os.environ["DWAVE_API_KEY"] = "MISSING IN CONFIG"
if "DWAVE_API_TOKEN" not in os.environ:
    os.environ["DWAVE_API_TOKEN"] = "MISSING IN CONFIG"

from typing import Any, Dict, List, Tuple
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import create_topology_graph, get_topology_properties
import collections.abc

Variable = collections.abc.Hashable
class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure."""
    
    def __init__(self):
        # Use the default topology (Pegasus) from quantum_proof_of_work
        topology_graph = create_topology_graph()  # Uses DEFAULT_TOPOLOGY (Pegasus)
        properties = get_topology_properties()

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

        
