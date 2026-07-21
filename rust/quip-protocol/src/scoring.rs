fn sign(s: i8) -> f64 {
    if s > 0 {
        1.0
    } else {
        -1.0
    }
}

pub fn energy_milli(spins: &[i8], h: &[f64], j: &[f64], edges: &[(usize, usize)]) -> i64 {
    let mut e = 0.0f64;
    for (i, &s) in spins.iter().enumerate() {
        if i < h.len() {
            e += h[i] * sign(s);
        }
    }
    for (k, &(u, v)) in edges.iter().enumerate() {
        if k < j.len() {
            e += j[k] * sign(spins[u]) * sign(spins[v]);
        }
    }
    if !e.is_finite() {
        return 1i64 << 62;
    }
    (e * 1000.0) as i64 // truncation toward zero, matches Python int(energy*1000)
}

pub fn hamming_flip_invariant(a: &[i8], b: &[i8]) -> u32 {
    let n = a.len();
    let raw = a
        .iter()
        .zip(b)
        .filter(|(x, y)| sign(**x) != sign(**y))
        .count();
    raw.min(n - raw) as u32
}

pub fn set_diversity(solutions: &[Vec<i8>]) -> f64 {
    if solutions.len() < 2 {
        return 0.0;
    }
    let n = solutions[0].len() as f64;
    let mut sum = 0.0f64;
    let mut pairs = 0u64;
    for i in 0..solutions.len() {
        for k in (i + 1)..solutions.len() {
            sum += hamming_flip_invariant(&solutions[i], &solutions[k]) as f64 / n;
            pairs += 1;
        }
    }
    sum / pairs as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_positive_sign_and_truncation() {
        // spins [+1,-1]; h=[1.0, -0.5]; edge (0,1) J=2.0
        // E = (1*1) + (-0.5*-1) + (2.0 * 1 * -1) = 1 + 0.5 - 2.0 = -0.5 -> -500 milli
        let e = energy_milli(&[1, -1], &[1.0, -0.5], &[2.0], &[(0, 1)]);
        assert_eq!(e, -500);
    }

    #[test]
    fn energy_truncates_toward_zero() {
        // E = 0.0015 -> int(1.5) -> 1 (truncation, not round to 2)
        let e = energy_milli(&[1], &[0.0015], &[], &[]);
        assert_eq!(e, 1);
    }

    #[test]
    fn diversity_flip_invariant() {
        // a and its exact inverse have flip-invariant distance 0 -> diversity 0
        assert_eq!(set_diversity(&[vec![1, 1, -1], vec![-1, -1, 1]]), 0.0);
        // a vs one-bit-flipped: min(1, 2)=1 over N=3 -> 1/3
        assert!((set_diversity(&[vec![1, 1, 1], vec![-1, 1, 1]]) - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(set_diversity(&[vec![1, 1]]), 0.0); // <2 solutions
    }
}
