"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from shared.quantum_proof_of_work import create_topology_graph, get_topology_properties


def create_dwave_sampler():
    """Create a D-Wave sampler with proper error handling."""
    try:
        sampler = DWaveSampler()
        print(f"QPU connected to: {sampler.properties['chip_id']}")
        return sampler
    except Exception as e:
        print(f"QPU not available: {e}")
        # Fall back to mock sampler with our default topology (Pegasus)
        topology_graph = create_topology_graph()
        properties = get_topology_properties()
        return MockDWaveSampler(
            nodelist=list(topology_graph.nodes()),
            edgelist=list(topology_graph.edges()),
            properties=properties
        )


class DWaveSamplerWrapper:
    """Wrapper class for D-Wave sampler with configuration management."""
    
    def __init__(self):
        self.sampler = create_dwave_sampler()
        self.is_qpu = not isinstance(self.sampler, MockDWaveSampler)
        self.sampler_type = "qpu" if self.is_qpu else "mock"
        
        # Type conversions to match protocol expectations (nodes should be ints for quantum_proof_of_work functions)
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
    
    def sample_ising(self, h, J, **kwargs):
        """Sample from the D-Wave QPU or mock sampler."""
        if self.is_qpu:
            # QPU-specific parameters
            kwargs.setdefault('answer_mode', 'raw')
            kwargs.setdefault('annealing_time', 20.0)
        
        return self.sampler.sample_ising(h, J, **kwargs)
    
    @property
    def nodelist(self):
        return self.sampler.nodelist
    
    @property
    def edgelist(self):
        return self.sampler.edgelist
    
    @property
    def properties(self):
        return self.sampler.properties