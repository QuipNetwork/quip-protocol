"""Simulated Annealing sampler for CPU-based quantum blockchain mining."""

from typing import Any, Dict, List, Tuple
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler
from shared.node_edge_coerce import coerce_int_nodes_edges
from shared.quantum_proof_of_work import DEFAULT_TOPOLOGY
from shared.stream_context import stream_from_feeder

import collections.abc

Variable = collections.abc.Hashable
class SimulatedAnnealingStructuredSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with identical structure."""

    def __init__(self, topology=None):
        # Use provided topology or fall back to default
        topology_obj = topology if topology is not None else DEFAULT_TOPOLOGY
        topology_graph = topology_obj.graph
        properties = topology_obj.properties

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
        self.nodes, self.edges = coerce_int_nodes_edges(self.nodelist, self.edgelist)

    def sample_ising_streaming(
        self, feeder, *, num_reads, num_sweeps, **_ignored,
    ):
        """Stream samplesets from a feeder, one model at a time.

        Thin generator for the unified driver path; delegates to the shared
        ``stream_from_feeder`` helper. ``**_ignored`` absorbs any
        sampler_kwargs a generic driver may pass.
        """
        yield from stream_from_feeder(
            self, feeder, num_reads=num_reads, num_sweeps=num_sweeps,
        )

