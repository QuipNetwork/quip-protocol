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
    allowed_h_values: Optional[Any] = None
    allowed_j_values: Optional[Any] = None
    diversity_range: Optional[Tuple[float, float]] = None
    solutions_range: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.allowed_h_values is None:
            from shared.quantum_proof_of_work import DEFAULT_ALLOWED_H
            self.allowed_h_values = DEFAULT_ALLOWED_H
        if self.allowed_j_values is None:
            from shared.quantum_proof_of_work import DEFAULT_ALLOWED_J
            self.allowed_j_values = DEFAULT_ALLOWED_J
        if self.diversity_range is None:
            from shared.energy_utils import DEFAULT_DIVERSITY_RANGE
            self.diversity_range = DEFAULT_DIVERSITY_RANGE
        if self.solutions_range is None:
            from shared.energy_utils import DEFAULT_SOLUTIONS_RANGE
            self.solutions_range = DEFAULT_SOLUTIONS_RANGE


@dataclass
class MiningResult:
    """Result of a mining operation.

    ``nonce`` is the canonical 32-byte big-endian representation of the
    PoW puzzle's U256 nonce, matching the on-chain ``QuantumProof.nonce``
    field (post-MR-!20).
    """
    miner_id: str
    miner_type: str
    nonce: bytes
    salt: bytes
    timestamp: int
    prev_timestamp: int
    solutions: List[List[int]]
    energy: float
    diversity: float
    # Count of unique samples whose energy meets the target threshold —
    # in snapshot/strict mode, ``energy < requirements.difficulty_energy``
    # measured against sampler energies; in ratchet mode,
    # ``energy < live_threshold_energy`` measured against chain-recomputed
    # energies. This is the count the chain actually accepts ("≥
    # min_solutions below max_energy"). The pre-dedup read count is
    # implied by the configured num_reads; not reported here.
    num_valid: int
    mining_time: int
    node_list: List[int]
    edge_list: List[Tuple[int, int]]
    variable_order: Optional[List[int]] = None
    # Independently-recomputed energy of the WORST submitted solution
    # (least negative among the diverse-selected set), in the same float
    # units as ``energy``. The chain's ``validate_proof`` filters each
    # submitted solution with strict ``energy < max_energy_milli`` before
    # checking ``valid_solution_count >= min_solutions``, so the submission
    # passes only when every solution clears — i.e. when this floor is
    # below the chain threshold. ``energy`` (best of the set) is kept for
    # ratchet/display purposes and is unsafe to gate submission on.
    # ``None`` means the worst-case wasn't recomputed; fall back to
    # ``energy`` (legacy behavior) at the cost of possible chain rejection.
    submit_floor_energy: Optional[float] = None

    @property
    def effective_floor(self) -> float:
        """Chain-gated floor: submit_floor_energy when set, else best energy."""
        return (
            self.submit_floor_energy
            if self.submit_floor_energy is not None
            else self.energy
        )


@dataclass
class IsingSample:
    nonce: bytes
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
