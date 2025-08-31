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
        
        # Override with actual attributes that match Sampler protocol types
        self.nodelist: List[Variable] = list(nodelist)
        self.edgelist: List[Tuple[Variable, Variable]] = list(edgelist)
        self.properties: Dict[str, Any] = dict(properties)