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
from typing import List, Optional, Tuple

from shared.allowed_value_spec import AllowedValueSpec


@dataclass(frozen=True)
class SubstrateDifficulty:
    """Mirror of `pallet_quantum_pow::types::DifficultyConfig`.

    All fields are milli-precision integers on chain. Helper properties expose
    the conventional float view used by the rest of the Python miner.
    """

    min_solutions: int
    max_energy_milli: int
    min_diversity_milli: int
    min_quality_milli: int

    @property
    def max_energy(self) -> float:
        """Energy threshold as a float (chain stores ×1000)."""
        return self.max_energy_milli / 1000.0

    @property
    def min_diversity(self) -> float:
        """Diversity floor as a float in [0, 1]."""
        return self.min_diversity_milli / 1000.0

    @property
    def min_quality(self) -> float:
        """Quality floor as a float in [0, 1]."""
        return self.min_quality_milli / 1000.0


@dataclass(frozen=True)
class SubstrateMiningContext:
    """Protocol-neutral input to one mining attempt.

    Mirrors the Rust ``MiningSnapshot`` (post-MR-!20) plus the per-miner
    identity material (``miner_account_bytes``). The three
    ``allowed_*_values`` specs are the on-chain source of truth for how the
    nonce-seeded RNG samples per-node h, per-edge j, and per-spin solution
    values — miners must use these exact specs to produce proofs that the
    pallet will accept.
    """

    block_number: int
    parent_hash: bytes
    topology_hash: bytes
    nodes: List[int]
    edges: List[Tuple[int, int]]
    difficulty: SubstrateDifficulty
    miner_account_bytes: bytes
    allowed_h_values: AllowedValueSpec
    allowed_j_values: AllowedValueSpec
    allowed_spin_values: AllowedValueSpec

    def __post_init__(self) -> None:
        if len(self.parent_hash) != 32:
            raise ValueError(
                f"parent_hash must be 32 bytes, got {len(self.parent_hash)}"
            )
        if len(self.topology_hash) != 32:
            raise ValueError(
                f"topology_hash must be 32 bytes, got {len(self.topology_hash)}"
            )
        if len(self.miner_account_bytes) != 32:
            raise ValueError(
                "miner_account_bytes must be the 32-byte canonical miner "
                f"identity (blake2_256(SCALE(account_id))), got "
                f"{len(self.miner_account_bytes)}"
            )
        if not (0 <= self.block_number < 2**32):
            raise ValueError(
                f"block_number must fit in u32, got {self.block_number}"
            )


@dataclass(frozen=True)
class WinningSolution:
    """Mirror of `pallet_quantum_pow::types::WinningSolution`.

    Persisted in `QuantumPow.WinningSolutions[block_number]` alongside the
    `BlockWinner` event. ``difficulty`` is the *active* threshold the proof
    had to clear (decay applied, pre-adjust); the next block's threshold
    is whatever ``QuantumPow.Difficulty`` storage holds after the post-win
    adjustment, which is NOT duplicated here.
    """

    miner: bytes
    salt: bytes
    energy_milli: int
    reward: int
    submitted_at: int
    difficulty: SubstrateDifficulty

    def __post_init__(self) -> None:
        if len(self.miner) != 32:
            raise ValueError(f"miner must be 32 bytes, got {len(self.miner)}")
        if len(self.salt) != 32:
            raise ValueError(f"salt must be 32 bytes, got {len(self.salt)}")


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
        if len(self.nonce) != 32:
            raise ValueError(f"nonce must be 32 bytes, got {len(self.nonce)}")


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
