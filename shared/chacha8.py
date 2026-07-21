"""Pure Python ChaCha8Rng matching Rust's rand_chacha::ChaCha8Rng v0.9.0.

Produces byte-identical output to the Rust crate for cross-language
deterministic Ising model generation. Not intended for cryptographic use.

SPDX-License-Identifier: AGPL-3.0-or-later
"""
from __future__ import annotations

import struct

import numpy as np

# ChaCha constants: "expand 32-byte k" as four little-endian u32 words
_CONSTANTS = (0x61707865, 0x3320646e, 0x79622d32, 0x6b206574)

# PCG32 constants (from rand_core::SeedableRng::seed_from_u64)
_PCG_MUL = 6364136223846793005
_PCG_INC = 11634580027462260723

_U32_MASK = 0xFFFFFFFF
_U64_MASK = 0xFFFFFFFFFFFFFFFF


def _rotate_right_u32(value: int, count: int) -> int:
    """Rotate a 32-bit unsigned integer right by count bits."""
    return ((value >> count) | (value << (32 - count))) & _U32_MASK


def _pcg32(state: int) -> tuple[int, int]:
    """One step of PCG-XSH-RR. Returns (new_state, output_u32).

    Matches rand_core::SeedableRng::seed_from_u64's internal pcg32.
    State is advanced BEFORE computing output.
    """
    state = (state * _PCG_MUL + _PCG_INC) & _U64_MASK
    xorshifted = (((state >> 18) ^ state) >> 27) & _U32_MASK
    rot = (state >> 59) & 0x1F
    output = _rotate_right_u32(xorshifted, rot)
    return state, output


def _seed_from_u64(seed: int) -> bytes:
    """Expand a u64 seed into a 32-byte key using PCG32.

    Matches Rust's rand_core::SeedableRng::seed_from_u64 default impl.
    Calls pcg32 eight times to fill 32 bytes (4 bytes per call).
    """
    state = seed & _U64_MASK
    key = bytearray(32)
    for i in range(8):
        state, output = _pcg32(state)
        struct.pack_into('<I', key, i * 4, output)
    return bytes(key)


def _quarter_round(s: list[int], a: int, b: int, c: int, d: int) -> None:
    """ChaCha quarter round operating on state words in-place."""
    s[a] = (s[a] + s[b]) & _U32_MASK
    s[d] = _rotate_right_u32(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _U32_MASK
    s[b] = _rotate_right_u32(s[b] ^ s[c], 20)
    s[a] = (s[a] + s[b]) & _U32_MASK
    s[d] = _rotate_right_u32(s[d] ^ s[a], 24)
    s[c] = (s[c] + s[d]) & _U32_MASK
    s[b] = _rotate_right_u32(s[b] ^ s[c], 25)


def _chacha_block(state: list[int]) -> list[int]:
    """Run the ChaCha8 block function on a 16-word state.

    Applies 4 double-rounds (8 quarter-rounds) and adds the original
    state, producing 16 output u32 words.
    """
    working = list(state)

    for _ in range(4):  # 4 double-rounds = 8 rounds
        # Column rounds
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        # Diagonal rounds
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)

    return [(working[i] + state[i]) & _U32_MASK for i in range(16)]


class ChaCha8Rng:
    """Deterministic PRNG matching Rust's rand_chacha::ChaCha8Rng v0.9.0.

    State layout (16 x u32):
        [0-3]   Constants: "expand 32-byte k"
        [4-11]  Key: 32-byte seed as 8 little-endian u32 words
        [12-13] Counter: 64-bit little-endian, starts at 0
        [14-15] Stream: 64-bit little-endian, starts at 0
    """

    __slots__ = ('_state', '_buffer', '_index')

    def __init__(self, key: bytes, counter: int = 0, stream: int = 0) -> None:
        if len(key) != 32:
            raise ValueError(f"Key must be 32 bytes, got {len(key)}")

        key_words = struct.unpack_from('<8I', key)
        self._state = [
            _CONSTANTS[0], _CONSTANTS[1], _CONSTANTS[2], _CONSTANTS[3],
            key_words[0], key_words[1], key_words[2], key_words[3],
            key_words[4], key_words[5], key_words[6], key_words[7],
            counter & _U32_MASK, (counter >> 32) & _U32_MASK,
            stream & _U32_MASK, (stream >> 32) & _U32_MASK,
        ]
        self._buffer: list[int] = []
        self._index = 0

    @classmethod
    def seed_from_u64(cls, seed: int) -> ChaCha8Rng:
        """Create a ChaCha8Rng from a u64 seed.

        Matches rand_chacha::ChaCha8Rng::seed_from_u64, which uses
        PCG32 to expand the u64 into a 32-byte key.
        """
        if not (0 <= seed <= _U64_MASK):
            raise ValueError(
                f"seed must be a u64 (0..2^64-1), got {seed}"
            )
        key = _seed_from_u64(seed)
        return cls(key)

    @classmethod
    def from_seed(cls, key: bytes) -> ChaCha8Rng:
        """Create a ChaCha8Rng directly from a 32-byte key.

        Matches ``rand_core::SeedableRng::from_seed`` for ChaCha8Rng — no
        PCG32 expansion. Use this when seeding from a full 256-bit nonce
        (the post-MR-!20 PoW path), not a truncated u64.
        """
        return cls(key)

    def _refill_buffer(self) -> None:
        """Generate the next 16-word keystream block."""
        self._buffer = _chacha_block(self._state)
        self._index = 0
        # Increment 64-bit counter (words 12-13)
        lo = (self._state[12] + 1) & _U32_MASK
        carry = 1 if lo == 0 else 0
        self._state[12] = lo
        self._state[13] = (self._state[13] + carry) & _U32_MASK

    def next_u32(self) -> int:
        """Return the next u32 from the keystream."""
        if self._index >= len(self._buffer):
            self._refill_buffer()
        value = self._buffer[self._index]
        self._index += 1
        return value


# ── Vectorized keystream (numpy) ──────────────────────────────────────────────
# Produces the SAME u32 sequence as repeated ``ChaCha8Rng.next_u32()`` but for
# many words at once: each ChaCha8 block is independent given its counter, so all
# blocks compute in parallel as a single (n_blocks, 16) uint32 array. Used by the
# vectorized Ising-model generation to skip the ~46k-element Python sample loop
# (the feeder's throughput wall). ``test_chacha8_vectorized`` asserts it matches
# the scalar RNG word-for-word.


def _rotr_vec(x: np.ndarray, count: int) -> np.ndarray:
    """Rotate a uint32 array right by ``count`` bits (uint32 wraps)."""
    return ((x >> count) | (x << (32 - count)))


def _quarter_round_vec(s: np.ndarray, a: int, b: int, c: int, d: int) -> None:
    """ChaCha quarter round across the block axis (s is (B, 16) uint32)."""
    s[:, a] = s[:, a] + s[:, b]
    s[:, d] = _rotr_vec(s[:, d] ^ s[:, a], 16)
    s[:, c] = s[:, c] + s[:, d]
    s[:, b] = _rotr_vec(s[:, b] ^ s[:, c], 20)
    s[:, a] = s[:, a] + s[:, b]
    s[:, d] = _rotr_vec(s[:, d] ^ s[:, a], 24)
    s[:, c] = s[:, c] + s[:, d]
    s[:, b] = _rotr_vec(s[:, b] ^ s[:, c], 25)


def _chacha_block_vec(states: np.ndarray) -> np.ndarray:
    """ChaCha8 block function over a (B, 16) uint32 array of initial states."""
    w = states.copy()
    for _ in range(4):  # 4 double-rounds = 8 rounds
        _quarter_round_vec(w, 0, 4, 8, 12)
        _quarter_round_vec(w, 1, 5, 9, 13)
        _quarter_round_vec(w, 2, 6, 10, 14)
        _quarter_round_vec(w, 3, 7, 11, 15)
        _quarter_round_vec(w, 0, 5, 10, 15)
        _quarter_round_vec(w, 1, 6, 11, 12)
        _quarter_round_vec(w, 2, 7, 8, 13)
        _quarter_round_vec(w, 3, 4, 9, 14)
    return w + states  # uint32 wraps


def keystream_words(
    key: bytes, n_words: int, *, counter: int = 0, stream: int = 0,
) -> np.ndarray:
    """First ``n_words`` keystream u32 words as a uint32 array.

    Byte-identical to calling :meth:`ChaCha8Rng.next_u32` ``n_words`` times on
    ``ChaCha8Rng(key, counter, stream)``: words are emitted block by block (16
    per block), the counter incrementing per block.

    Args:
        key: 32-byte seed/key.
        n_words: Number of u32 words to produce.
        counter: Starting 64-bit block counter (0 for ``from_seed``).
        stream: 64-bit stream id (0 for ``from_seed``).

    Returns:
        ``uint32`` array of length ``n_words``.
    """
    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes, got {len(key)}")
    if n_words <= 0:
        return np.empty(0, dtype=np.uint32)
    n_blocks = (n_words + 15) // 16
    key_words = np.frombuffer(key, dtype="<u4")  # 8 little-endian u32
    counters = (np.uint64(counter) + np.arange(n_blocks, dtype=np.uint64))
    states = np.empty((n_blocks, 16), dtype=np.uint32)
    states[:, 0:4] = _CONSTANTS
    states[:, 4:12] = key_words
    states[:, 12] = (counters & np.uint64(_U32_MASK)).astype(np.uint32)
    states[:, 13] = (counters >> np.uint64(32)).astype(np.uint32)
    states[:, 14] = np.uint32(stream & _U32_MASK)
    states[:, 15] = np.uint32((stream >> 32) & _U32_MASK)
    out = _chacha_block_vec(states)
    return out.reshape(-1)[:n_words]
