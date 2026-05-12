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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Protocol v0 canonical h_values. Documented in PYTHON_MINER_INTEGRATION_PLAN.md
# section "Open protocol seam: h_values". The Rust pallet currently accepts
# whatever the proof carries; this constant must match what miners submit.
CANONICAL_H_VALUES: Tuple[float, ...] = (-1.0, 0.0, 1.0)


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

    Mirrors the Rust `MiningSnapshot` plus the per-miner identity material
    (`miner_account_bytes`) and the canonical `h_values` constant. Phase 3
    extracts `BaseMiner.mine_work_item(context, stop_event)` to consume this
    directly, replacing the legacy `(prev_block, node_info, requirements)`
    triple.
    """

    block_number: int
    parent_hash: bytes
    topology_hash: bytes
    nodes: List[int]
    edges: List[Tuple[int, int]]
    difficulty: SubstrateDifficulty
    miner_account_bytes: bytes
    h_values: Tuple[float, ...] = CANONICAL_H_VALUES

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
                "miner_account_bytes must be the 32-byte SCALE AccountId32, got "
                f"{len(self.miner_account_bytes)}"
            )
        if not (0 <= self.block_number < 2**32):
            raise ValueError(
                f"block_number must fit in u32, got {self.block_number}"
            )
        if not self.h_values:
            raise ValueError("h_values must be non-empty")


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
