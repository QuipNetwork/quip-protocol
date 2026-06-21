# RCS-VRF Subnet: Mathematical Reference

**Status:** Reference document capturing the mathematical content underpinning the protocol. Not a full derivation of every theorem — focused on the relationships and formulas the protocol depends on, with derivations included where they're short or illuminating.

**Scope:** F_XEB statistics, Haar-randomness and Porter-Thomas, min-entropy and the Leftover Hash Lemma, the Aaronson-Hung 2023 bound (and its Liu et al. 2025 correction), the heavy-output attack family, the VRF and hash-chain constructions.

**Notation:**
- `n` = number of qubits
- `N = 2^n` = dimension of the Hilbert space
- `m` = number of samples (shots)
- `M` = number of circuits in a multi-circuit protocol
- `p_C(x) = |⟨x|C|0^n⟩|²` = ideal probability of measuring bitstring x from circuit C
- `F` = fidelity of the quantum sampling
- `ε` = soundness parameter
- `H_∞` = min-entropy
- `H` = Shannon (von Neumann) entropy

---

## 1. The F_XEB statistic

### 1.1 Definition

Given m samples {x_i} from a quantum device running circuit C, the Linear Cross-Entropy Benchmarking (LXEB or F_XEB) score is:

```
F_XEB = (2^n / m) · Σ_{i=1}^{m} p_C(x_i) - 1
       = N · mean_i[p_C(x_i)] - 1
```

The ideal probabilities `p_C(x_i)` are computed classically (full state-vector simulation up to ~n=25, tensor-network contraction beyond).

### 1.2 Expected values under different sampling strategies

The expected value of F_XEB depends on the distribution the samples are drawn from.

**Uniform sampling (no quantum info):** if samples are drawn uniformly from {0,1}^n:
```
E[mean p(x_i)] = mean_x p(x) = 1/N
E[F_XEB]      = N · (1/N) - 1 = 0
```

**Quantum sampling at fidelity F=1 (ideal):** samples drawn from the ideal distribution p_C. The expected value of mean p(x_i) is the *collision probability* of p_C:
```
E[mean p(x_i)] = Σ_x p(x)²        (the collision probability of p_C)
```

For a Haar-random circuit (see §2):
```
E_Haar[Σ_x p(x)²] = 2/(N+1) ≈ 2/N
E[F_XEB]          = N · (2/N) - 1 = 1
```

**Quantum sampling at fidelity F < 1:** model the noisy state as a depolarizing mixture:
```
ρ_F = F · |ψ⟩⟨ψ| + (1-F) · I/N
```

The expected p(x_i) under this mixture is:
```
E[p(x_i)] = F · (2/N) + (1-F) · (1/N) = (1 + F) / N
F_XEB     = N · (1+F)/N - 1 = F
```

**Linear fidelity relation:** F_XEB ≈ F. This is the defining property that makes F_XEB a useful fidelity *measurement*, not just a yes/no test.

### 1.3 Variance and statistical convergence

For m samples, the standard deviation of F_XEB scales as ~1/√m. At m=1000 (our v0.x parameter), the shot-noise contribution to F_XEB std is ~3% of the mean.

Per-circuit variance (across different random circuits with same n, d) is typically larger than shot noise. Our depth sweep at n=8, d=10 measured noiseless F_XEB ≈ 1.0 ± 0.2 across multiple circuits — circuit-to-circuit variance dominates shot noise at this scale.

---

## 2. Haar-randomness and Porter-Thomas

### 2.1 Haar measure

The *Haar measure* is the unique probability distribution on the unitary group U(N) that's invariant under multiplication by any fixed unitary. It's the quantum analog of "uniform random" on the unitary group.

A Haar-random circuit is one whose unitary is sampled from the Haar measure. Properties (collision probability, output distribution statistics, etc.) computed under the Haar measure are reference values the protocol calibrates against.

### 2.2 Porter-Thomas distribution

For a Haar-random circuit acting on |0^n⟩, the output probabilities p(x) follow the Porter-Thomas distribution:

```
PDF: f(p) = N · exp(-N · p)
CDF: Pr[p(x) ≤ t] = 1 - exp(-N·t)
```

This is an exponential distribution with mean 1/N. Key moments:

```
E[p(x)] = 1/N              (mean — same as uniform)
Var[p(x)] = 1/N²            (variance — much larger than uniform's near-zero)
E[Σ_x p(x)²] = 2/(N+1)     (collision probability — twice uniform's 1/N)
```

The factor of 2 in the collision probability is what gives F_XEB = 1 for honest noiseless quantum sampling (see §1.2).

### 2.3 Why Porter-Thomas matters for the protocol

PT isn't a passive consequence of using random circuits — every quantitative claim in the protocol is calibrated to PT statistics:

| Protocol mechanism | What it needs from PT |
|--------------------|----------------------|
| F_XEB ≈ 1 for honest | E_PT[Σ p²] = 2/N |
| F_XEB = F for noisy | Linear mixture of PT and uniform |
| Quantum-classical hardness gap | PT's heavy outcomes are classically unpredictable |
| Min-entropy ≈ n - log n | PT's specific H_∞ structure |
| Heavy-output attack analysis | Defined relative to PT's heavy outcomes |

If the circuit distribution deviates from PT (e.g., undertrained circuit family, shallow depth), all these calibrations break. Empirically, our v0 depth sweep found noiseless F_XEB ≈ 3-7 at d=4 — direct evidence the d=4 circuit family wasn't producing PT statistics.

### 2.4 Approximate t-designs

A truly Haar-random unitary on n qubits requires exponentially many parameters to specify — no finite gate set can realize it exactly. RCS approximates Haar using finite circuit families. The relevant notion is *approximate t-design*:

> A circuit family is an ε-approximate t-design if its first t moments match Haar's first t moments to error ε.

For F_XEB (which depends on second moments — collision probabilities), we need approximately a *2-design*. This is enough for F_XEB to be a meaningful fidelity test.

For higher-order cryptographic claims, higher t-designs may be needed. The AH 2023 / Liu et al. 2025 analysis nominally requires sufficient depth that the circuit family is close to a 2-design.

### 2.5 Empirical convergence

The depth at which a circuit family becomes a good 2-design depends on connectivity:

- *1D brickwork (nearest-neighbor):* depth ~n needed for 2-design
- *All-to-all random pairing (Liu et al. recipe):* depth ~log n needed for 2-design

Our v0 depth sweep at n=8 found:
- d=4 noiseless F_XEB ≈ 3.1 (not close to Haar)
- d=8 noiseless F_XEB ≈ 1.06 ± 0.09 (close to Haar)
- d=10 noiseless F_XEB ≈ 1.11 ± 0.09 (close to Haar, matches Liu et al. choice)
- d=14 noiseless F_XEB ≈ 0.96 ± 0.05 (close to Haar)

We use d=10 with all-to-all connectivity as v0.x's default — confirmed empirically to put the circuit family close enough to Haar for PT statistics to hold.

---

## 3. F_XEB CDFs under realistic sampling

### 3.1 Uniform samples

For m samples drawn uniformly from {0,1}^n, the sum Σ p(x_i) follows an Erlang distribution of shape m. The CDF of F_XEB is (Liu et al. 2025 Eq. III.6):

```
Pr[F_XEB ≤ χ | uniform] = Γ̃(m, m·(χ+1))
```

where Γ̃ is the regularized lower-incomplete Gamma function.

### 3.2 Ideal quantum samples

For m samples drawn from the ideal PT distribution, Σ p(x_i) follows an Erlang distribution of shape 2m:

```
Pr[F_XEB ≤ χ | quantum] = Γ̃(2m, m·(χ+1))
```

### 3.3 Mixed samples (finite fidelity)

A fidelity-F sample is modeled as drawn from the PT distribution with probability F and uniform with probability (1-F). The CDF of F_XEB over m mixed samples is (Liu et al. 2025 Eq. III.12):

```
Pr[F_XEB ≤ χ | fidelity F] = Σ_{ℓ=0}^{m} Γ̃(m+ℓ, m·(χ+1)) · C(m,ℓ) · F^ℓ · (1-F)^{m-ℓ}
```

where C(m,ℓ) is the binomial coefficient.

This is the formula used to compute Q_min in the security analysis (§5.3).

---

## 4. Min-entropy and the Leftover Hash Lemma

### 4.1 Min-entropy definition

For a discrete distribution P over X:

```
H_∞(X) = -log₂(max_x P(X = x))
```

This is the worst-case unpredictability. If H_∞ = k, no specific outcome has probability greater than 2^{-k}.

**Operational meaning:** an attacker's best guessing strategy has success probability 2^{-H_∞}. Min-entropy directly bounds guessability, which is exactly what cryptographic applications need.

### 4.2 Min-entropy vs Shannon entropy

For a uniform distribution: H_∞ = H = log₂(N).

For non-uniform distributions: H_∞ ≤ H. The gap can be large.

Example: distribution with P(0000...0) = 0.5 and P(other) = 0.5/(N-1).
- H ≈ 1 + 0.5 · log₂(N-1) ≈ n/2 (looks decent)
- H_∞ = -log₂(0.5) = 1 (terrible — the attacker guesses correctly half the time)

The Leftover Hash Lemma (next subsection) requires *min-entropy*, not Shannon. If we used Shannon, extraction would fail on distributions like this — half the output would collapse to the same value.

### 4.3 Min-entropy for Haar / Porter-Thomas

For a single sample drawn from a Haar-random circuit:

```
H_∞(PT) ≈ n - log₂(n) - log₂(ln 2) ≈ n - log₂(n) - 0.529
```

This is the limiting min-entropy at large n. The derivation uses the fact that the maximum probability under PT is approximately `n · ln(2) / 2^n`.

For n=8: H_∞^Haar ≈ 8 - 3 - 0.53 ≈ 5.5 bits per sample.

### 4.4 Leftover Hash Lemma

**Theorem (LHL):** Let X be a random variable with H_∞(X) ≥ k. Let H be a 2-universal hash family from X to {0,1}^ℓ. For uniformly random h ← H, the joint distribution of (h, h(X)) is ε-close to uniform on H × {0,1}^ℓ, where:

```
ε = (1/2) · √(2^ℓ / 2^k)
```

Solving for ℓ in terms of ε:

```
ℓ ≤ k - 2·log₂(1/ε)
```

This is the maximum number of bits we can safely extract.

**Operational form:** given input with H_∞ ≥ k bits of min-entropy, a 2-universal hash extractor (like our Toeplitz) can produce up to ℓ ≤ k - 2·log₂(1/ε) bits that are ε-close to uniform.

### 4.5 Toeplitz extractor as a 2-universal hash

The Toeplitz extractor maps n input bits to ℓ output bits via T·x mod 2, where T is an ℓ × n Toeplitz matrix (constant along anti-diagonals). The matrix is generated from a fresh random seed of length ℓ + n - 1 bits.

The Toeplitz family is 2-universal: for any distinct x ≠ y, the probability over random T that T·x = T·y is at most 2^{-ℓ}. This is exactly the condition the LHL needs.

In our protocol:
- Input: 8000 bits (1000 samples × 8 qubits) from quantum measurement
- Seed: from drand pulse, expanded via SHAKE256 to needed length
- Output: 64 bits (in v0.x; production target depends on H_∞ bound)

The drand pulse serves as the LHL's "uniformly random hash selection" — crucially, drand is independent of the quantum samples, which is what the LHL requires.

---

## 5. The Aaronson-Hung bound

### 5.1 What the bound aims to prove

The AH 2023 protocol's goal: given that samples pass F_XEB at score ≥ χ, lower-bound the min-entropy of those samples (conditional on the circuit).

If proved, this lets us:
1. Run XEB to certify the samples are "good enough"
2. Compute a min-entropy bound from the F_XEB score
3. Apply the LHL to extract uniform bits

Without such a bound, passing F_XEB doesn't translate into a quantitative randomness claim.

### 5.2 The corrected single-round bound (von Neumann entropy)

AH 2023's original Theorem 5.8 claimed to lower-bound *min-entropy* directly. Liu et al. 2025 (the Quantinuum experimental paper) identified a flaw: the theorem as stated is impossible.

**Counterexample to original AH min-entropy claim** (Liu et al. 2025, SM Sec. I A):
> An algorithm that samples honestly 99% of the time and returns the all-zeros bitstring 1% of the time has H_∞ = -log₂(0.01) ≈ 6.6 bits regardless of n, and passes LXEB with high probability.

**Liu et al.'s fix:** bound *von Neumann entropy* (the standard Shannon-like entropy) in the single-round analysis, then derive smooth min-entropy at the protocol level via the Entropy Accumulation Theorem.

**Corrected single-round bound (Liu et al. 2025 Theorem 6):** under the LLHA hardness assumption, for any device A solving MLXEB_{b,k}(D) with probability q:

```
H(Z | C⃗) ≥ (B/2) · ((b·q - 1) / (b - 1)) - n^{-ω(1)}
```

where:
- Z = device output (k bitstrings)
- C⃗ = input circuits
- B = LLHA parameter (B ≤ n/2 by Grover; tighter bounds unknown for specific circuit families)
- b = LXEB threshold (corresponds to 1 + F_XEB)
- q = probability of passing LXEB

This bounds Shannon/von Neumann entropy, not min-entropy directly.

### 5.3 The protocol-level smooth min-entropy bound

**Theorem (Liu et al. 2025 Theorem 1):** Let Ω be the event that the protocol does not abort. Given εs ∈ (0, 1/4), the protocol either aborts with probability > 1 - 4εs, or:

```
H^{εs}_min(X_M | Ĩ_sn) ≥ Q_min · (n - 1) + log εs
```

where:
- X_M = the M bitstrings collected
- Ĩ_sn = classical side information
- Q_min = minimum quantum rounds the adversary must execute for the protocol to pass with probability ≥ 4εs
- (n - 1) = per-quantum-round smooth min-entropy contribution
- log εs = soundness tax

This is the bound the protocol actually relies on.

### 5.4 Computing Q_min

Q_min is the result of an optimization over adversary strategies. The adversary's optimal strategy uses Q quantum samples (honest) and (M-Q) classical samples (simulated at effective fidelity ϕ_classical = A·t_threshold / B, where A is adversary FLOPS, B is per-circuit simulation cost, t_threshold is the protocol's time threshold).

Using the mixed-fidelity CDF (§3.3), the adversary's success probability is:

```
ε_adv(Q, χ) = Pr[F_XEB ≥ χ | effective fidelity = ϕ_eff(Q)]
            = 1 - Σ_{ℓ=0}^{m} Γ̃(m+ℓ, m·(χ+1)) · C(m,ℓ) · ϕ_eff^ℓ · (1-ϕ_eff)^{m-ℓ}
```

where ϕ_eff(Q) = Q/M + ((M-Q)/M) · ϕ_classical (the effective average fidelity).

Q_min is then:
```
Q_min = argmin_Q { ε_adv(Q, χ) ≥ 4εs }
```

In words: the smallest Q such that the adversary can still pass with probability at least 4εs.

### 5.5 Honest-server simplification

For a server that does all M rounds honestly (no classical mixing), Q_min = M and the bound becomes:

```
H^{εs}_min ≥ M · (n - 1) + log εs
```

For our v0.3a parameters (n=8, M=1000, εs ≈ 2^-34):
```
H^{εs}_min ≥ 1000 · 7 + log(2^-34) ≈ 7000 - 34 = 6966 bits
```

After LHL with ε = 2^-32:
```
max extractable = 6966 - 64 = 6902 bits
```

**Caveat at v0.3a scale:** at n=8, classical simulation is trivial (microseconds on a laptop). For any reasonable adversary classical power A, ϕ_classical ≈ 1, forcing Q_min ≈ 0 and the bound to ≈ 0. The honest-server simplification only makes sense as a "what would the bound say if we trusted the server" exercise; real security at n=8 is not achievable.

The protocol's security activates at larger n where classical simulation costs become prohibitive. Liu et al. 2025 demonstrates this experimentally at n=56 (Frontier-scale verification cost).

### 5.6 Comparison with v0.3a's heuristic

v0.3a uses a simplified bound:
```
H_∞ ≈ m · F_XEB · (n - log₂(n) - log₂(ln 2))
```

At our parameters: ~1660 bits.

The Liu et al. honest-server bound: ~6966 bits.

**Differences:**
- *Magnitude:* v0.3a is ~4× conservative
- *Functional form:* v0.3a scales with F_XEB; the real bound has F_XEB entering only through Q_min, not as a per-sample multiplier
- *Per-quantum-sample entropy:* v0.3a uses (n - log n - log ln 2); the real bound uses (n - 1)

v0.3a's bound is *safe* (conservative) but has the wrong functional form. v0.3a-revised should adopt the honest-server bound or full Q_min computation (see ah_derivation.md).

---

## 6. The heavy-output attack

### 6.1 Why PT enables this attack

Porter-Thomas has a tail of "heavy" bitstrings — outcomes with probability much higher than 1/N. An attacker who can identify heavy outcomes (e.g., by classical simulation) can submit only those, achieving an F_XEB much higher than honest quantum sampling.

For the top-k strategy: attacker sorts bitstrings by p(x), submits the top k. Expected F_XEB:

```
F_XEB_top-k = (N/k) · Σ_{x ∈ top-k} p(x) - 1
```

For PT distribution, the top fraction k/N captures a much larger probability mass than k/N (heavy-tail effect). This makes F_XEB ≫ 1.

### 6.2 Empirical values from v0 spoofing experiment

At n=8, d=10, 5 trials, our v0 spoofing experiment measured:

| Attack strategy | Mean F_XEB | Std |
|----------------|------------|-----|
| Honest noiseless | 1.08 | 0.23 |
| Honest noisy (fidelity 0.5) | 0.30 | 0.08 |
| Uniform random (circuit-blind) | 0.005 | 0.009 |
| Sample from p (perfect classical) | 1.07 | 0.22 |
| Top-1 guess | 5.31 | 1.43 |
| Top-4 guess | 4.37 | 0.99 |
| Top-16 guess | 3.18 | 0.50 |

The top-k attacks produce F_XEB ≫ 1 — well above the honest noiseless range.

### 6.3 v0.1's two-sided defense

v0.1 added an upper bound: F_XEB must satisfy χ_low ≤ F_XEB ≤ χ_high. At our parameters, χ_low = 0.15, χ_high = 2.5.

Empirical pass rates (v0 vs v0.1):
- Honest noiseless: 100% → 100% (both)
- Honest noisy: 100% → 100% (both)
- Uniform random: 0% → 0% (caught by χ_low in both)
- Top-1 guess: 100% → 0% (now caught by χ_high)
- Top-4 guess: 100% → 0% (now caught by χ_high)
- Top-16 guess: 100% → 20% (mostly caught; F_XEB tail dips below χ_high in some trials)

The upper bound closes the heavy-output attack vector for top-1 and top-4 perfectly, and substantially for top-16.

### 6.4 The remaining attack: perfect classical sampling from p

The `sample from p` strategy (compute the full p(x) distribution, then sample from it) gives F_XEB ≈ 1.07 — indistinguishable from honest noiseless quantum. This attack:
- Requires O(2^n) compute to enumerate all amplitudes
- Is trivially feasible at n=8 (256 amplitudes)
- Is infeasible at n=56 (10^17 amplitudes, Frontier-scale)

The defense against this attack is *parameter scale*, not the F_XEB bounds. At v0.x's n=8, the protocol is not secure against this attack. At production scale, this attack becomes infeasible.

This is why the LLHA hardness assumption is required: it's the statement that no efficient classical algorithm exists for this task at appropriate parameter scales.

---

## 7. The VRF construction

### 7.1 v0.2 / v0.3a: Sign + Hash

For each participant i with Ed25519 keypair (sk_i, pk_i):

```
Proof:        π_i = Ed25519_sign(sk_i, α)
VRF output:   β_i = SHA3-256(π_i)
```

where α is the round's public input (epoch state, round number, etc.).

**Verification:** given (pk_i, α, β_i, π_i):
1. Check Ed25519_verify(pk_i, α, π_i) succeeds
2. Check SHA3-256(π_i) = β_i

**Properties:**
- Deterministic: same (sk_i, α) always gives same (π_i, β_i) (Ed25519 per RFC 8032 is deterministic)
- Verifiable: any party with pk_i can check both conditions
- Unpredictable without sk_i (under the assumption SHA3-256 is a good PRF over Ed25519 signatures)

**What this construction is NOT:**
- *Not full RFC 9381 ECVRF.* ECVRF has a more elaborate proof structure (involving a Pedersen-style commitment to the curve point) that gives stronger formal pseudorandomness guarantees.
- *Not formally proved pseudorandom.* The Sign + Hash construction's pseudorandomness depends on SHA3-256 destroying the algebraic structure of Ed25519 signatures — plausible but not the standardized choice.

For v0.2/v0.3a this is acceptable as a learning artifact. Production target: full RFC 9381 ECVRF.

### 7.2 H + SHAKE256 aggregation (architecture's design)

After collecting n_p VRF outputs (β_1, ..., β_{n_p}):

```
Concatenate:  c = β_1 || β_2 || ... || β_{n_p}
Hash:         seed = SHA3-256(c)          (H step, produces 32 bytes)
Expand:       bytes = SHAKE256(seed)      (XOF, produces arbitrary length)
```

The 32-byte seed is what LEG 3 uses as input to circuit generation. SHAKE256 expands it into all the bytes needed for random angles, gate selections, and pairings.

**Properties:**
- *Compression (H step):* reduces the 160-byte (n_p = 5, 32 bytes each) concatenation to 32 bytes
- *Expansion (SHAKE256):* produces as many bytes as needed downstream
- *Collision/preimage resistance:* SHA3-256 ensures no participant can predict the output by manipulating their contribution alone

The "H + SHAKE256" pattern is the standard cryptographic pattern of compress-then-expand. The architectural rationale for keeping both steps (rather than skipping H and going directly SHAKE256(c)) is collusion resistance: H provides a clean cryptographic boundary that's independent of how many participants there are or what their contributions look like.

---

## 8. Hash chains and the audit log

### 8.1 v0.x audit log structure

Each entry has the form:

```
entry_i = {
    leg:       string (which leg produced this entry)
    data:      arbitrary payload
    prev_hash: hash of entry_{i-1} (all zeros for first entry)
    hash:      SHA3-256(serialize(leg, data, prev_hash))
    timestamp: wall-clock time (informational only)
}
```

The chain property: each entry's `prev_hash` field stores the hash of the previous entry. Tampering with any entry breaks the chain at that point.

### 8.2 What single-chain hashing detects

For an attacker who modifies one entry's `data` field:
- The entry's recomputed hash doesn't match the stored `hash`
- The next entry's `prev_hash` no longer matches the previous entry's stored hash
- Multiple consistency checks fail

This catches *casual tampering* — single-point modifications without re-doing the chain math.

### 8.3 What single-chain hashing does NOT detect: cascading tamper

For a sophisticated attacker willing to redo work:
1. Modify entry j's data
2. Recompute entry j's hash
3. Update entry (j+1)'s prev_hash to match
4. Recompute entry (j+1)'s hash
5. Repeat for all subsequent entries

The result is an internally-consistent forged chain that passes all the single-chain checks.

In v0.3a, this is documented as a known limit (the deliberate negative test `test_cascading_tamper_is_not_caught_v0_3a` in test_v0_3a.py).

### 8.4 Multi-party hash graph defense

The architecture's LEG 7 calls for a hash *graph* (DAG) instead of a chain. Each party maintains its own per-party chain. Entries cross-reference entries in other parties' chains. Every entry is signed by its producing party.

To forge the graph, an attacker would need to:
1. Modify their own entry
2. Forge signatures from all parties whose entries reference theirs
3. Recursively forge signatures up the cross-reference graph

This requires compromising multiple independent parties' signing keys simultaneously — much harder than single-key cascading.

Bitcoin's defense against cascading tamper comes from proof-of-work + distributed consensus. Our protocol's defense (in the eventual LEG 7 design) comes from multi-party signatures + cross-reference structure. Both protect against cascading; they use different cryptographic primitives.

---

## 9. End-to-end security chain

Putting everything together, the protocol's security argument is:

1. **Circuit family is approximately Haar-random.** At d=10 with all-to-all connectivity, the circuit family is close enough to a 2-design for Porter-Thomas statistics to hold (validated empirically by our depth sweep).

2. **F_XEB measures fidelity.** Under PT, F_XEB = F (fidelity) for honest quantum sampling. Verified empirically: noiseless F_XEB ≈ 1, noisy F_XEB ≈ F.

3. **Two-sided XEB check rejects classical attacks.** χ_low rejects circuit-blind attacks (F_XEB ≈ 0); χ_high rejects heavy-output attacks (F_XEB ≫ 1). Empirically validated at v0.1.

4. **Passing F_XEB implies min-entropy.** Under the LLHA hardness assumption and the corrected Liu et al. 2025 analysis, passing F_XEB at score ≥ χ implies smooth min-entropy at least Q_min · (n-1) + log εs bits. At v0.x's small n, this only holds under the honest-server simplification.

5. **Leftover Hash Lemma extracts uniform bits.** Given input H_∞ ≥ k and soundness target ε, the Toeplitz extractor (with independent drand seed) produces up to k - 2·log₂(1/ε) bits ε-close to uniform.

6. **Audit log makes the chain verifiable.** Every cryptographic artifact (VRF contributions, circuit specification, F_XEB result, extractor output) is recorded with hash-chain integrity. Any party can independently verify each step.

Each step is a separate piece of math. Each step has assumptions and limits. The composition is what gives the final claim: *the output bits are unforgeable, unpredictable, and ε-close to uniform under the protocol's stated assumptions.*

---

## 10. Open mathematical questions

Items that need cryptographer (Mehdi) input before the spec is rigorous:

1. **VRF construction.** Sign+Hash (v0.x) vs RFC 9381 ECVRF vs threshold BLS. Affects formal pseudorandomness claims.

2. **AH bound to use.** Honest-server simplification vs full Q_min computation. At what parameter scales is each appropriate?

3. **LLHA parameter B.** Currently bounded by B ≤ n/2 (Grover). Are there tighter bounds for our specific circuit family?

4. **Extractor strength.** Toeplitz is 2-universal (classical). Liu et al. use "quantum-proof strong extractor." Do we need the stronger primitive given our threat model?

5. **Sample independence assumption.** The (n-1)-per-sample bound assumes independence. Is this realistic at our parameters, or do we need a correlated analysis?

6. **Smoothness parameter εs.** Liu et al. uses εs = ε_sou / 4. Is this ratio tunable for our protocol?

7. **Multi-shot vs single-shot model.** AH 2023 uses many samples per circuit; Liu et al. uses one. Which model does our protocol commit to?

These are all blocked on the Mehdi conversation. The ah_derivation.md document tees them up for that conversation.

---

## References

- **Aaronson, S. & Hung, S.-H.** *Certified Randomness from Quantum Supremacy.* arXiv:2303.01625 (STOC 2023). Original certified-randomness protocol from RCS.

- **Liu, M., Shaydulin, R., Niroula, P., DeCross, M., Hung, S.-H., et al.** *Certified randomness using a trapped-ion quantum processor.* arXiv:2503.20498 (Nature 2025). Experimental realization at n=56 + corrections to AH 2023.

- **Boixo, S. et al.** *Characterizing quantum supremacy in near-term devices.* Nat. Phys. 14, 595–600 (2018). XEB definition and analysis.

- **Porter, C. E. & Thomas, R. G.** *Fluctuations of Nuclear Reaction Widths.* Phys. Rev. 104, 483 (1956). Original Porter-Thomas distribution.

- **Haber, S. & Stornetta, W. S.** *How to Time-Stamp a Digital Document.* J. Cryptology 3, 99–111 (1991). Foundational paper on hash-chained tamper-evident logs.

- **Aaronson, S. & Zhang, Y.** *On quantum supremacy and certified randomness.* arXiv:2404.14493 (2024). Survey article relevant to AH 2023 follow-up.

- **RFC 9381.** *Verifiable Random Functions (VRFs).* IETF (2023). Standard ECVRF construction.

- **NIST FIPS 202.** *SHA-3 Standard.* NIST (2015). SHA3-256 and SHAKE256 specifications.
