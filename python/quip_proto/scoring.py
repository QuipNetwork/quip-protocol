import math


def _sign(s):
    return 1.0 if s > 0 else -1.0


def energy_milli(spins, h, j, edges):
    e = 0.0
    for i, s in enumerate(spins):
        if i < len(h):
            e += h[i] * _sign(s)
    for k, (u, v) in enumerate(edges):
        if k < len(j) and 0 <= u < len(spins) and 0 <= v < len(spins):
            e += j[k] * _sign(spins[u]) * _sign(spins[v])
    if not math.isfinite(e):
        return 1 << 62
    m = e * 1000.0
    # replicate Rust `(m as i64)` saturating cast
    if math.isnan(m):
        return 0
    if m >= 9223372036854775808.0:  # >= 2**63  -> i64::MAX
        return (1 << 63) - 1
    if m <= -9223372036854775808.0:  # <= -2**63 -> i64::MIN
        return -(1 << 63)
    return int(m)  # truncate toward zero


def _hamming_flip_invariant(a, b):
    n = len(a)
    raw = sum(1 for x, y in zip(a, b) if _sign(x) != _sign(y))
    return min(raw, n - raw)


def set_diversity(solutions):
    if len(solutions) < 2:
        return 0.0
    n = len(solutions[0])
    if n == 0:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(solutions)):
        for k in range(i + 1, len(solutions)):
            total += _hamming_flip_invariant(solutions[i], solutions[k]) / n
            pairs += 1
    return total / pairs
