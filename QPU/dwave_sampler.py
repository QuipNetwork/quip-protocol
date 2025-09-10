"""D-Wave QPU sampler wrapper and configuration for quantum blockchain mining."""

from typing import Dict, List, Tuple, Any, Union, Mapping, Sequence, cast
import collections.abc
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
import dimod

# Type definitions to match base_miner
Variable = collections.abc.Hashable


class DWaveSamplerWrapper:
    """Wrapper class for D-Wave sampler with configuration management."""

    def __init__(self):
        self.sampler = DWaveSampler()
        self.is_qpu = True
        self.sampler_type = "qpu"

        # Store nodelist and edgelist as lists to match protocol expectations
        # D-Wave samplers use int nodes/edges which are compatible with Variable (Hashable)
        self.nodelist: List[Variable] = list(self.sampler.nodelist)
        self.edgelist: List[Tuple[Variable, Variable]] = list(self.sampler.edgelist)
        self.properties: Dict[str, Any] = dict(self.sampler.properties)

        # For quantum_proof_of_work functions, nodes and edges should be int lists
        # D-Wave samplers already use ints, so we can safely cast
        self.nodes: List[int] = cast(List[int], self.nodelist)
        self.edges: List[Tuple[int, int]] = cast(List[Tuple[int, int]], self.edgelist)

    def sample_ising(
        self,
        h: Union[Mapping[Variable, float], Sequence[float]],
        J: Mapping[Tuple[Variable, Variable], float],
        **kwargs
    ) -> dimod.SampleSet:
        """Sample from the D-Wave QPU or mock sampler."""
        return self.sampler.sample_ising(h, J, **kwargs)