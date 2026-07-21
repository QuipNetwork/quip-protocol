import struct


def encode_i32_le(values):
    return struct.pack(f"<{len(values)}i", *values)


def decode_i32_le(b):
    if len(b) % 4 != 0:
        raise ValueError("byte length not a multiple of 4")
    return list(struct.unpack(f"<{len(b) // 4}i", b))


def encode_spins(spins):
    return bytes(0x01 if s >= 0 else 0xFF for s in spins)


def decode_spins(b):
    out = []
    for byte in b:
        if byte == 0x01:
            out.append(1)
        elif byte == 0xFF:
            out.append(-1)
        else:
            raise ValueError(f"bad spin byte {byte}")
    return out
