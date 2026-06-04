"""Substrate-side mining data types.

Mirror of the Rust `MiningSnapshot`, `DifficultyConfig`, and `QuantumProof`
shapes defined in `pallets/quantum-pow/src/types.rs` of the `quip-protocol-rs`
repository. Kept as plain dataclasses so they round-trip cleanly through
multiprocessing IPC, JSON, and SCALE.

Float-vs-milli convention:
  - Internal Python mining works in floats (e.g. h_values in (-1.0, 0.0, 1.0),
    energy in physical units).
  - The chain encodes the same values as milli-precision i32/i64. The submitter
    is the only place that converts between the two; everything else in
    Python-land speaks floats.

Post-MR-!20 (quip-protocol-rs): topologies carry three
:class:`shared.allowed_value_spec.AllowedValueSpec` instances that fully
describe how the nonce-seeded RNG samples h, j, and spin values. The
mining snapshot returns those specs so miners and the validator agree
byte-for-byte on the Ising puzzle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from shared.allowed_value_spec import AllowedValueSpec
from shared.ising_feeder import RandomIsingFeeder

if TYPE_CHECKING:
    # Type-only — avoids the substrate ↔ shared cycle at runtime.
    from shared.miner_types import BlockRequirements


def _require_len(name: str, value: bytes, n: int = 32) -> None:
    """Raise ValueError if *value* is not exactly *n* bytes long."""
    if len(value) != n:
        raise ValueError(f"{name} must be {n} bytes, got {len(value)}")


@dataclass(frozen=True)
class PowConstants:
    """Pallet constants the miner needs for local decay computation.

    These are ``#[pallet::constant]`` items on ``pallet_quantum_pow``;
    they're baked into chain metadata at runtime build time and only
    change with a runtime upgrade. Cached for the session lifetime —
    re-reading on every head would waste an RPC per block.

    All three ``curve_c_*`` values are per-mille (e.g. ``700`` = ``0.70``)
    because pallet constants must implement ``Get<_>`` and ``f64`` does
    not implement ``Encode`` (see ``pallets/quantum-pow/src/lib.rs`` for
    the explanatory docstring on the Rust side).
    """

    epoch_length: int
    curve_c_easy_milli: int
    curve_c_knee_milli: int
    curve_c_hard_milli: int


@dataclass(frozen=True)
class SubstrateDifficulty:
    """Mirror of `pallet_quantum_pow::types::DifficultyConfig`.

    All fields are milli-precision integers on chain. Helper properties expose
    the conventional float view used by the rest of the Python miner.
    """

    min_solutions: int
    max_energy_milli: int
    min_diversity_milli: int

    @property
    def max_energy(self) -> float:
        """Energy threshold as a float (chain stores ×1000)."""
        return self.max_energy_milli / 1000.0

    @property
    def min_diversity(self) -> float:
        """Diversity floor as a float in [0, 1]."""
        return self.min_diversity_milli / 1000.0


@dataclass(frozen=True)
class SubstrateMiningContext:
    """Protocol-neutral input to one mining attempt.

    Mirrors the Rust ``MiningSnapshot`` plus the per-miner identity
    material (``miner_account_bytes``). The three ``allowed_*_values``
    specs are the on-chain source of truth for how the nonce-seeded RNG
    samples per-node h, per-edge j, and per-spin solution values — miners
    must use these exact specs to produce proofs the pallet will accept.

    ``last_proof_block_hash`` is ``block_hash(LastProofBlock)`` — the only
    "time" input ``derive_nonce`` consumes. It stays constant across the
    entire round (only changes when a new proof wins), so a proof derived
    against it remains valid for as long as the round runs, no matter how
    deep the txpool gets.
    """

    last_proof_block_hash: bytes
    topology_hash: bytes
    nodes: List[int]
    edges: List[Tuple[int, int]]
    difficulty: SubstrateDifficulty
    miner_account_bytes: bytes
    allowed_h_values: AllowedValueSpec
    allowed_j_values: AllowedValueSpec
    allowed_spin_values: AllowedValueSpec
    # Best-block hash + number the snapshot was resolved against. Used by
    # the mempool controller, which reacts to every block (each may carry
    # `JobProposed` / `OrderExpired` events). The PoW controller ignores
    # them — its work key only changes when ``last_proof_block_hash`` does.
    block_hash: bytes
    block_number: int
    # Round-constant decay schedule, built once per work key by the
    # controller and attached before dispatch so workers can rank their
    # candidate stash by win-time locally (no per-iteration RPC).
    # None means the schedule was not yet computed (default/legacy callers).
    decay_schedule: Optional[List[int]] = None  # max_energy_milli per step 0..HORIZON
    last_proof_block: int = 0                   # round anchor (block number)
    epoch_length: int = 0                       # blocks per decay step (0 = disabled)

    def __post_init__(self) -> None:
        _require_len("last_proof_block_hash", self.last_proof_block_hash)
        _require_len("topology_hash", self.topology_hash)
        if len(self.miner_account_bytes) != 32:
            raise ValueError(
                "miner_account_bytes must be the 32-byte canonical miner "
                f"identity (blake2_256(SCALE(account_id))), got "
                f"{len(self.miner_account_bytes)}"
            )
        _require_len("block_hash", self.block_hash)
        if self.block_number < 0:
            raise ValueError(
                f"block_number must be non-negative, got {self.block_number}"
            )

    def resolve_ising(
        self,
        salt: bytes,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
    ) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int]:
        """Derive (h, J, nonce) for this attempt. PoW path: nonce → ChaCha8 → (h, J)."""
        # Local import avoids pulling shared.* at module load time for substrate.types
        from shared.quantum_proof_of_work import derive_nonce, generate_ising_model_from_nonce

        nonce = derive_nonce(
            self.last_proof_block_hash,
            self.miner_account_bytes,
            salt,
        )
        h, J = generate_ising_model_from_nonce(
            nonce,
            list(nodes),
            list(edges),
            allowed_h=self.allowed_h_values,
            allowed_j=self.allowed_j_values,
        )
        return h, J, nonce

    def requirements(self) -> "BlockRequirements":
        """Build BlockRequirements from this snapshot's difficulty."""
        from shared.miner_types import BlockRequirements

        d = self.difficulty
        return BlockRequirements(
            difficulty_energy=d.max_energy,
            min_diversity=d.min_diversity,
            min_solutions=int(d.min_solutions),
            timeout_to_difficulty_adjustment_decay=2**31 - 1,
            allowed_h_values=self.allowed_h_values,
            allowed_j_values=self.allowed_j_values,
        )

    def make_feeder(
        self,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
        *,
        buffer_size: int = 8,
    ) -> RandomIsingFeeder:
        """Return a ``RandomIsingFeeder`` rooted at this snapshot's round seed.

        ``last_proof_block_hash`` and ``miner_account_bytes`` together
        identify the round + miner under PoW; the feeder rolls fresh
        salts internally and derives ``(nonce, h, J)`` from them. The
        loop owns the feeder lifecycle — it stops the feeder when the
        mining attempt ends.
        """
        return RandomIsingFeeder(
            last_proof_block_hash=self.last_proof_block_hash,
            miner_bytes=self.miner_account_bytes,
            nodes=list(nodes),
            edges=list(edges),
            buffer_size=buffer_size,
        )

    def uses_decay_ratchet(self) -> bool:
        """PoW work always takes the decay-ratchet loop (see ``WorkContext``)."""
        return True


@dataclass(frozen=True)
class WinningSolution:
    """Mirror of `pallet_quantum_pow::types::WinningSolution`.

    Persisted in `QuantumPow.WinningSolutions[block_number]` alongside the
    `BlockWinner` event. ``difficulty`` is the *active* threshold the proof
    had to clear (decay applied, pre-adjust); the next block's threshold
    is whatever ``QuantumPow.Difficulty`` storage holds after the post-win
    adjustment, which is NOT duplicated here.

    ``last_proof_block_hash`` is the last proof block hash the proof actually used
    (``block_hash`` of the previous winning block). Storing it makes
    nonce re-derivation self-contained — no chain-state lookup needed,
    so it stays correct even after the prior winning block is pruned
    beyond ``BlockHashCount``.
    """

    miner: bytes
    salt: bytes
    energy_milli: int
    reward: int
    submitted_at: int
    difficulty: SubstrateDifficulty
    last_proof_block_hash: bytes

    def __post_init__(self) -> None:
        _require_len("miner", self.miner)
        _require_len("salt", self.salt)
        _require_len("last_proof_block_hash", self.last_proof_block_hash)


@dataclass(frozen=True)
class WinningSolutionWithNonce:
    """Mirror of `pallet_quantum_pow::types::WinningSolutionWithNonce`.

    Runtime-API view returned by ``QuantumPowApi_winning_solution`` that
    augments the persisted solution with the BLAKE3-derived nonce, so
    clients don't have to run BLAKE3 themselves.
    """

    solution: WinningSolution
    nonce: bytes

    def __post_init__(self) -> None:
        _require_len("nonce", self.nonce)


@dataclass(frozen=True)
class MinerInfo:
    """Mirror of `pallet_quantum_pow::types::MinerInfo`."""

    registered_at: int
    deposit: int
    proofs_submitted: int
    proofs_won: int
    rewards_earned: int


@dataclass
class ExtrinsicReceipt:
    """Outcome of submitting a signed extrinsic.

    Kept deliberately small — full event lists are available via
    `SubstrateClient.get_events_for_extrinsic` when callers need them.
    """

    extrinsic_hash: str
    block_hash: Optional[str] = None
    is_finalized: bool = False
    error: Optional[str] = None
    events: List[dict] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.error is None
