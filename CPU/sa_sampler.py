"""Simulated Annealing sampler for CPU-based quantum blockchain mining."""

from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler


class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure."""
    
    def __init__(self):
        qpu = MockDWaveSampler()

        substitute_sampler = SimulatedAnnealingSampler()
        super().__init__(
            nodelist=qpu.nodelist,
            edgelist=qpu.edgelist,
            properties=qpu.properties,
            substitute_sampler=substitute_sampler
        )
        self.sampler_type = "mock"
        self.parameters.update(substitute_sampler.parameters)  # Do not warn when SA parameters are seen.
        self.mocked_parameters.add('num_sweeps')  # Do not warn when this SA parameter is seen.