"""Immutable dataclass bundling Ising model data with blockchain provenance."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class IsingModel:
    """One Ising model instance with blockchain provenance.

    All fields needed to upload to GPU and later validate
    a solution on-chain. Immutable after creation.

    Attributes:
        h: Linear biases (node -> field value).
        J: Quadratic couplings ((u, v) -> coupling value).
        nonce: Blockchain nonce for proof-of-work verification.
        salt: Random salt used to derive the nonce.
    """

    h: dict[int, float]
    J: dict[tuple[int, int], float]
    nonce: int
    salt: bytes
