"""Simulated Annealing sampler for CPU-based quantum blockchain mining."""

from typing import Any, Dict, List, Tuple
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler
from dwave.system import DWaveSampler
import collections.abc

Variable = collections.abc.Hashable
class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure."""
    
    def __init__(self):
        qpu = DWaveSampler()
        substitute_sampler = SimulatedAnnealingSampler()
        nodelist = []
        for node in qpu.nodelist:
            nodelist.append(node)
        edgelist = []
        for edge in qpu.edgelist:
            edgelist.append(edge)

        super().__init__(
            nodelist=nodelist,
            edgelist=edgelist,
            properties=qpu.properties,
            substitute_sampler=substitute_sampler
        )
        self.sampler_type = "mock"
        self.parameters.update(substitute_sampler.parameters)
        self.mocked_parameters.add('num_sweeps')
        
        # Override with actual attributes that match Sampler protocol types
        self.nodelist: List[Variable] = list(nodelist)
        self.edgelist: List[Tuple[Variable, Variable]] = list(edgelist)
        self.properties: Dict[str, Any] = dict(qpu.properties)