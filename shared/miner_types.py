import collections.abc
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Protocol, Any, Union

import dimod

# Type definitions for quantum computing
Variable = collections.abc.Hashable
Bias = float


@dataclass
class BlockRequirements:
    """Mining difficulty configuration consumed by `BaseMiner` and its hooks.

    Originally part of the legacy chain block format (`shared.block`), this
    dataclass survives the v0.1 -> v0.2 refactor because the
    protocol-neutral mining loop synthesizes one from
    `SubstrateDifficulty` at the substrate boundary
    (`base_miner._block_requirements_from_difficulty`). Keeping the same
    shape means CPU/GPU/QPU hooks (`_adapt_mining_params`,
    `evaluate_sampleset`) work unchanged in substrate mode.

    `timeout_to_difficulty_adjustment_decay` is the legacy decay window in
    seconds. Substrate mode synthesizes a large sentinel so that any
    accidental call into the legacy `compute_current_requirements` decay
    code path is a no-op — the chain controller cancels work on new heads
    instead of decaying.

    `diversity_range` / `solutions_range` are chain-level bounds; in
    substrate mode they come from `SubstrateDifficulty` indirectly via the
    snapshot. Defaults preserve the legacy chain behaviour.
    """

    difficulty_energy: float
    min_diversity: float
    min_solutions: int
    timeout_to_difficulty_adjustment_decay: int
    h_values: Optional[List[float]] = None
    diversity_range: Optional[Tuple[float, float]] = None
    solutions_range: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.h_values is None:
            self.h_values = [-1.0, 0.0, 1.0]
        if self.diversity_range is None:
            from shared.energy_utils import DEFAULT_DIVERSITY_RANGE
            self.diversity_range = DEFAULT_DIVERSITY_RANGE
        if self.solutions_range is None:
            from shared.energy_utils import DEFAULT_SOLUTIONS_RANGE
            self.solutions_range = DEFAULT_SOLUTIONS_RANGE


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
