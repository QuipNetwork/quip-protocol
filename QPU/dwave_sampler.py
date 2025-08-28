"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler


def create_dwave_sampler():
    """Create a D-Wave sampler with proper error handling."""
    try:
        sampler = DWaveSampler()
        print(f"QPU connected to: {sampler.properties['chip_id']}")
        return sampler
    except Exception as e:
        print(f"QPU not available: {e}")
        # Fall back to mock sampler
        return MockDWaveSampler()


class DWaveSamplerWrapper:
    """Wrapper class for D-Wave sampler with configuration management."""
    
    def __init__(self):
        self.sampler = create_dwave_sampler()
        self.is_qpu = not isinstance(self.sampler, MockDWaveSampler)
    
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