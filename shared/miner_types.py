import collections.abc
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Protocol, Any, Union

import dimod

# Type definitions for quantum computing
Variable = collections.abc.Hashable
Bias = float

@dataclass
class MiningResult:
    """Result of a mining operation."""
    miner_id: str
    miner_type: str
    nonce: int
    salt: bytes
    timestamp: int
    prev_timestamp: int
    solutions: List[List[int]]
    energy: float
    diversity: float
    num_valid: int
    mining_time: int
    node_list: List[int]
    edge_list: List[Tuple[int, int]]
    variable_order: Optional[List[int]] = None

@dataclass
class IsingSample:
    nonce: int
    salt: bytes
    sampleset: dimod.SampleSet

class Sampler(Protocol):
    """Protocol defining the D-Wave sampler interface."""
    nodelist: List[Variable]
    edgelist: List[Tuple[Variable, Variable]]
    properties: Dict[str, Any]
    sampler_type: str
    nodes: List[int]  # Integer nodes for quantum_proof_of_work functions
    edges: List[Tuple[int, int]]  # Integer edges for quantum_proof_of_work functions

    def sample_ising(
        self,
        h: Union[Mapping[Variable, Bias], Sequence[Bias]],
        J: Mapping[Tuple[Variable, Variable], Bias],
        **kwargs
    ) -> dimod.SampleSet:
        ...
