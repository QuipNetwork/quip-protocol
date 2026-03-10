"""Binary Encoding/Decoding for Rainbow Table.

This module provides efficient binary serialization of the rainbow table,
supporting both v1 (legacy) and v2 (with canonical keys and minor relationships)
formats.
"""

from __future__ import annotations

from typing import Dict, List

from ..polynomial import TuttePolynomial, encode_varuint, decode_varuint
from .core import RainbowTable, MinorEntry


# =============================================================================
# BINARY ENCODING
# =============================================================================

def encode_rainbow_table_binary(table: RainbowTable) -> bytes:
    """Encode rainbow table to compact binary v2 format.

    v2 Format:
        Header:
            [magic: 4 bytes]    = "RTBL"
            [version: 1 byte]   = 2
            [flags: 1 byte]     bit 0: has_minor_rels, bit 1: structural_minors_computed
            [num_entries: varuint]

        Entry Section (per entry):
            [canonical_key: 32 bytes]       <- raw SHA256
            [name_len: varuint] [name: bytes]
            [node_count: varuint]
            [edge_count: varuint]
            [spanning_trees: varuint]
            [poly_len: varuint] [poly_bytes: bytes]

        Minor Relationships Section (if flag bit 0):
            [num_majors: varuint]
            Per major:
                [major_index: varuint]
                [num_minors: varuint]
                [minor_index: varuint] x num_minors
    """
    import hashlib

    result = bytearray()

    # Magic header
    result.extend(b"RTBL")
    result.append(2)  # version

    # Flags
    has_minors = bool(table.minor_relationships)
    flags = 0
    if has_minors:
        flags |= 0x01
    if table._structural_minors_computed:
        flags |= 0x02
    result.append(flags)

    # Number of entries
    entries_ordered = list(table.entries.items())
    result.extend(encode_varuint(len(entries_ordered)))

    # Build key -> index map for minor relationships
    key_to_index: Dict[str, int] = {}
    for idx, (key, _entry) in enumerate(entries_ordered):
        key_to_index[key] = idx

    # Entry section
    for key, entry in entries_ordered:
        # Canonical key as raw 32-byte SHA256
        result.extend(bytes.fromhex(key))

        # Name
        name_bytes = entry.name.encode('utf-8')
        result.extend(encode_varuint(len(name_bytes)))
        result.extend(name_bytes)

        # Metadata
        result.extend(encode_varuint(entry.node_count))
        result.extend(encode_varuint(entry.edge_count))
        result.extend(encode_varuint(entry.spanning_trees))

        # Polynomial as bitstring
        poly_bytes = entry.polynomial.to_bytes()
        result.extend(encode_varuint(len(poly_bytes)))
        result.extend(poly_bytes)

    # Minor relationships section
    if has_minors:
        # Filter to only majors with valid indices
        valid_majors = []
        for major_key, minor_keys in table.minor_relationships.items():
            if major_key not in key_to_index:
                continue
            valid_minor_indices = []
            for mk in minor_keys:
                if mk in key_to_index:
                    valid_minor_indices.append(key_to_index[mk])
            if valid_minor_indices:
                valid_majors.append((key_to_index[major_key], valid_minor_indices))

        result.extend(encode_varuint(len(valid_majors)))
        for major_idx, minor_indices in valid_majors:
            result.extend(encode_varuint(major_idx))
            result.extend(encode_varuint(len(minor_indices)))
            for mi in minor_indices:
                result.extend(encode_varuint(mi))

    return bytes(result)


def decode_rainbow_table_binary(data: bytes) -> RainbowTable:
    """Decode binary rainbow table (supports v1 and v2 formats).

    Returns a fully populated RainbowTable with entries and minor_relationships.
    """
    offset = 0

    # Magic header
    if data[offset:offset + 4] != b"RTBL":
        raise ValueError("Invalid magic header -- not a rainbow table binary")
    offset += 4

    version = data[offset]
    offset += 1

    if version == 1:
        return _decode_binary_v1(data, offset)
    elif version == 2:
        return _decode_binary_v2(data, offset)
    else:
        raise ValueError(f"Unsupported binary version: {version}")


def _decode_binary_v1(data: bytes, offset: int) -> RainbowTable:
    """Decode v1 binary format (no canonical keys, no minor relationships)."""
    table = RainbowTable()

    num_entries, offset = decode_varuint(data, offset)

    for _ in range(num_entries):
        # Name
        name_len, offset = decode_varuint(data, offset)
        name = data[offset:offset + name_len].decode('utf-8')
        offset += name_len

        # Metadata
        node_count, offset = decode_varuint(data, offset)
        edge_count, offset = decode_varuint(data, offset)
        spanning_trees, offset = decode_varuint(data, offset)

        # Polynomial
        poly_len, offset = decode_varuint(data, offset)
        poly_bytes = data[offset:offset + poly_len]
        offset += poly_len
        polynomial = TuttePolynomial.from_bytes(poly_bytes)

        # v1 has no canonical key stored -- we can't reconstruct it without
        # the graph, so use a placeholder based on name
        import hashlib
        canonical_key = hashlib.sha256(f"v1:{name}".encode()).hexdigest()

        entry = MinorEntry(
            name=name,
            polynomial=polynomial,
            node_count=node_count,
            edge_count=edge_count,
            canonical_key=canonical_key,
            spanning_trees=spanning_trees,
            num_terms=polynomial.num_terms(),
        )

        table.entries[canonical_key] = entry
        table.name_index[name] = canonical_key

    table._sort_by_complexity()
    return table


def _decode_binary_v2(data: bytes, offset: int) -> RainbowTable:
    """Decode v2 binary format (with canonical keys and minor relationships)."""
    table = RainbowTable()

    # Flags
    flags = data[offset]
    offset += 1
    has_minors = bool(flags & 0x01)
    table._structural_minors_computed = bool(flags & 0x02)

    # Number of entries
    num_entries, offset = decode_varuint(data, offset)

    # Read entries, build index -> key mapping
    index_to_key: List[str] = []

    for _ in range(num_entries):
        # Canonical key: 32 raw bytes -> hex string
        canonical_key = data[offset:offset + 32].hex()
        offset += 32

        # Name
        name_len, offset = decode_varuint(data, offset)
        name = data[offset:offset + name_len].decode('utf-8')
        offset += name_len

        # Metadata
        node_count, offset = decode_varuint(data, offset)
        edge_count, offset = decode_varuint(data, offset)
        spanning_trees, offset = decode_varuint(data, offset)

        # Polynomial
        poly_len, offset = decode_varuint(data, offset)
        poly_bytes = data[offset:offset + poly_len]
        offset += poly_len
        polynomial = TuttePolynomial.from_bytes(poly_bytes)

        entry = MinorEntry(
            name=name,
            polynomial=polynomial,
            node_count=node_count,
            edge_count=edge_count,
            canonical_key=canonical_key,
            spanning_trees=spanning_trees,
            num_terms=polynomial.num_terms(),
        )

        table.entries[canonical_key] = entry
        table.name_index[name] = canonical_key
        index_to_key.append(canonical_key)

    # Minor relationships section
    if has_minors:
        num_majors, offset = decode_varuint(data, offset)
        for _ in range(num_majors):
            major_idx, offset = decode_varuint(data, offset)
            num_minors, offset = decode_varuint(data, offset)
            minor_keys = []
            for _ in range(num_minors):
                minor_idx, offset = decode_varuint(data, offset)
                if minor_idx < len(index_to_key):
                    minor_keys.append(index_to_key[minor_idx])
            if major_idx < len(index_to_key) and minor_keys:
                table.minor_relationships[index_to_key[major_idx]] = minor_keys

    table._sort_by_complexity()
    return table


def load_binary_rainbow_table(path: str) -> RainbowTable:
    """Load rainbow table from binary file."""
    with open(path, 'rb') as f:
        data = f.read()
    return decode_rainbow_table_binary(data)


def save_binary_rainbow_table(table: RainbowTable, path: str) -> int:
    """Save rainbow table to binary format, return size in bytes."""
    data = encode_rainbow_table_binary(table)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


# =============================================================================
# MULTIGRAPH LOOKUP TABLE BINARY ENCODING
# =============================================================================

def encode_multigraph_lookup_table(cache: Dict[str, 'TuttePolynomial']) -> bytes:
    """Encode multigraph lookup table to compact binary format.

    Format:
        Header:
            [magic: 4 bytes]    = "MGLT"
            [version: 1 byte]   = 1
            [num_entries: varuint]

        Entry Section (per entry):
            [canonical_key: 32 bytes]       <- raw SHA256
            [poly_len: varuint] [poly_bytes: bytes]
    """
    result = bytearray()

    # Magic header
    result.extend(b"MGLT")
    result.append(1)  # version

    # Number of entries
    result.extend(encode_varuint(len(cache)))

    # Entry section
    for key, poly in cache.items():
        # Canonical key as raw 32-byte SHA256
        result.extend(bytes.fromhex(key))

        # Polynomial as binary
        poly_bytes = poly.to_bytes()
        result.extend(encode_varuint(len(poly_bytes)))
        result.extend(poly_bytes)

    return bytes(result)


def decode_multigraph_lookup_table(data: bytes) -> Dict[str, 'TuttePolynomial']:
    """Decode multigraph lookup table from binary format.

    Returns dict of canonical_key -> TuttePolynomial.
    """
    from ..polynomial import TuttePolynomial

    offset = 0

    # Magic header
    if data[offset:offset + 4] != b"MGLT":
        raise ValueError("Invalid magic header -- not a multigraph lookup table binary")
    offset += 4

    version = data[offset]
    offset += 1
    if version != 1:
        raise ValueError(f"Unsupported multigraph lookup table version: {version}")

    # Number of entries
    num_entries, offset = decode_varuint(data, offset)

    cache: Dict[str, 'TuttePolynomial'] = {}
    for _ in range(num_entries):
        # Canonical key: 32 raw bytes -> hex string
        canonical_key = data[offset:offset + 32].hex()
        offset += 32

        # Polynomial
        poly_len, offset = decode_varuint(data, offset)
        poly_bytes = data[offset:offset + poly_len]
        offset += poly_len
        polynomial = TuttePolynomial.from_bytes(poly_bytes)

        cache[canonical_key] = polynomial

    return cache


def save_multigraph_lookup_table(cache: Dict[str, 'TuttePolynomial'], path: str) -> int:
    """Save multigraph lookup table to binary format, return size in bytes."""
    data = encode_multigraph_lookup_table(cache)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def load_multigraph_lookup_table(path: str) -> Dict[str, 'TuttePolynomial']:
    """Load multigraph lookup table from binary file."""
    with open(path, 'rb') as f:
        data = f.read()
    return decode_multigraph_lookup_table(data)


