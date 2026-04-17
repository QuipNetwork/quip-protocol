"""Wire codecs for chain-manifest and block-by-hash sync messages.

These codecs serialize the payloads for ``CHAIN_MANIFEST_REQUEST`` /
``CHAIN_MANIFEST_RESPONSE`` and ``BLOCK_BY_HASH_REQUEST`` /
``BLOCK_BY_HASH_RESPONSE`` QUIC message types. The encoders and decoders
are pure (no I/O or network state) so they can be unit-tested in
isolation from the QUIC transport layer.

Wire formats
------------
``CHAIN_MANIFEST_REQUEST`` payload::

    [4B limit uint32][4B num_locator uint32][num_locator * 32B hashes]

``CHAIN_MANIFEST_RESPONSE`` payload::

    [4B count uint32][count * (4B index uint32 + 32B hash)]

``BLOCK_BY_HASH_REQUEST`` payload::

    [32B hash]

``BLOCK_BY_HASH_RESPONSE`` payload::

    <serialized Block>   # as produced by Block.to_network()
    <empty>              # NOT_FOUND: server has no canonical block at that hash
"""

import struct
from typing import List, Optional, Tuple

from shared.block import Block

BLOCK_HASH_SIZE = 32
MANIFEST_ENTRY_SIZE = 4 + BLOCK_HASH_SIZE

# Cap on entries per manifest response, chosen to bound wire size at
# roughly 74 KB (2048 * 36). Matches the spirit of Bitcoin's
# MAX_HEADERS_RESULTS and stays well below QUIC stream head-of-line
# blocking concerns.
MAX_MANIFEST_ENTRIES = 2048


def encode_manifest_request(locator: List[bytes], limit: int) -> bytes:
    """Encode a chain-manifest request payload.

    Args:
        locator: Block hashes tip-first, used by the server to find the
            latest common ancestor with its own canonical chain. Each
            must be exactly ``BLOCK_HASH_SIZE`` bytes.
        limit: Maximum entries the server should return; must lie in
            ``[0, MAX_MANIFEST_ENTRIES]``.

    Returns:
        Wire payload bytes.

    Raises:
        ValueError: If ``limit`` is out of range or any locator entry
            has the wrong length.
    """
    if limit < 0 or limit > MAX_MANIFEST_ENTRIES:
        raise ValueError(
            f"limit {limit} out of range [0, {MAX_MANIFEST_ENTRIES}]"
        )
    for i, h in enumerate(locator):
        if len(h) != BLOCK_HASH_SIZE:
            raise ValueError(
                f"locator[{i}] is {len(h)} bytes, expected {BLOCK_HASH_SIZE}"
            )

    out = bytearray(struct.pack('!II', limit, len(locator)))
    for h in locator:
        out.extend(h)
    return bytes(out)


def decode_manifest_request(payload: bytes) -> Tuple[List[bytes], int]:
    """Decode a chain-manifest request payload.

    Returns:
        Tuple ``(locator, limit)`` where ``locator`` is a list of
        32-byte hashes in the order sent by the client.

    Raises:
        ValueError: If the payload is malformed or truncated, or the
            declared locator length exceeds ``MAX_MANIFEST_ENTRIES``.
    """
    if len(payload) < 8:
        raise ValueError(
            f"manifest request payload too short: {len(payload)}"
        )
    limit, num = struct.unpack('!II', payload[:8])
    if num > MAX_MANIFEST_ENTRIES:
        raise ValueError(
            f"locator length {num} exceeds {MAX_MANIFEST_ENTRIES}"
        )
    expected = 8 + num * BLOCK_HASH_SIZE
    if len(payload) < expected:
        raise ValueError(
            f"manifest request payload truncated: {len(payload)} < {expected}"
        )

    locator: List[bytes] = []
    offset = 8
    for _ in range(num):
        locator.append(bytes(payload[offset:offset + BLOCK_HASH_SIZE]))
        offset += BLOCK_HASH_SIZE
    return locator, limit


def encode_manifest_response(entries: List[Tuple[int, bytes]]) -> bytes:
    """Encode a chain-manifest response payload.

    Args:
        entries: ``(block_index, block_hash)`` pairs in strictly
            ascending index order. Each hash must be
            ``BLOCK_HASH_SIZE`` bytes. At most ``MAX_MANIFEST_ENTRIES``.

    Returns:
        Wire payload bytes.

    Raises:
        ValueError: If the entry list is too long, an index is out of
            ``uint32`` range, a hash has the wrong size, or indices are
            not strictly ascending.
    """
    if len(entries) > MAX_MANIFEST_ENTRIES:
        raise ValueError(
            f"too many entries: {len(entries)} > {MAX_MANIFEST_ENTRIES}"
        )
    previous_idx = -1
    for i, (idx, h) in enumerate(entries):
        if len(h) != BLOCK_HASH_SIZE:
            raise ValueError(
                f"entries[{i}] hash is {len(h)} bytes, "
                f"expected {BLOCK_HASH_SIZE}"
            )
        if idx < 0 or idx > 0xFFFFFFFF:
            raise ValueError(
                f"entries[{i}] index {idx} out of uint32 range"
            )
        if idx <= previous_idx:
            raise ValueError(
                f"entries[{i}] index {idx} not greater than previous "
                f"{previous_idx}"
            )
        previous_idx = idx

    out = bytearray(struct.pack('!I', len(entries)))
    for idx, h in entries:
        out.extend(struct.pack('!I', idx))
        out.extend(h)
    return bytes(out)


def decode_manifest_response(payload: bytes) -> List[Tuple[int, bytes]]:
    """Decode a chain-manifest response payload.

    Returns:
        List of ``(block_index, block_hash)`` tuples in ascending
        index order.

    Raises:
        ValueError: If the payload is malformed, truncated, exceeds
            ``MAX_MANIFEST_ENTRIES``, or contains non-strictly-ascending
            indices.
    """
    if len(payload) < 4:
        raise ValueError(
            f"manifest response payload too short: {len(payload)}"
        )
    (count,) = struct.unpack('!I', payload[:4])
    if count > MAX_MANIFEST_ENTRIES:
        raise ValueError(
            f"entry count {count} exceeds {MAX_MANIFEST_ENTRIES}"
        )
    expected = 4 + count * MANIFEST_ENTRY_SIZE
    if len(payload) < expected:
        raise ValueError(
            f"manifest response payload truncated: {len(payload)} < {expected}"
        )

    entries: List[Tuple[int, bytes]] = []
    offset = 4
    previous_idx = -1
    for i in range(count):
        (idx,) = struct.unpack('!I', payload[offset:offset + 4])
        offset += 4
        h = bytes(payload[offset:offset + BLOCK_HASH_SIZE])
        offset += BLOCK_HASH_SIZE
        if idx <= previous_idx:
            raise ValueError(
                f"entries[{i}] index {idx} not greater than previous "
                f"{previous_idx}"
            )
        entries.append((idx, h))
        previous_idx = idx
    return entries


def encode_block_by_hash_request(block_hash: bytes) -> bytes:
    """Encode a block-by-hash request payload (the raw 32-byte hash).

    Raises:
        ValueError: If ``block_hash`` is not ``BLOCK_HASH_SIZE`` bytes.
    """
    if len(block_hash) != BLOCK_HASH_SIZE:
        raise ValueError(
            f"block hash is {len(block_hash)} bytes, "
            f"expected {BLOCK_HASH_SIZE}"
        )
    return bytes(block_hash)


def decode_block_by_hash_request(payload: bytes) -> bytes:
    """Decode a block-by-hash request payload and return the hash.

    Raises:
        ValueError: If ``payload`` is not exactly ``BLOCK_HASH_SIZE`` bytes.
    """
    if len(payload) != BLOCK_HASH_SIZE:
        raise ValueError(
            f"block-by-hash request payload is {len(payload)} bytes, "
            f"expected {BLOCK_HASH_SIZE}"
        )
    return bytes(payload)


def encode_block_by_hash_response(block: Optional[Block]) -> bytes:
    """Encode a block-by-hash response payload.

    An empty payload signals NOT_FOUND — the server has no canonical
    block matching the requested hash.
    """
    if block is None:
        return b""
    return block.to_network()


def decode_block_by_hash_response(payload: bytes) -> Optional[Block]:
    """Decode a block-by-hash response payload.

    An empty payload is interpreted as NOT_FOUND and yields ``None``.
    Any other error (malformed block bytes) propagates from
    ``Block.from_network``.
    """
    if not payload:
        return None
    return Block.from_network(payload)
